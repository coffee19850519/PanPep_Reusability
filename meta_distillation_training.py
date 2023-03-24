import random
import time

import joblib
import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Memory_meta import Memory_Meta
from Memory_meta import Memory_module
from PepTCRdict import PepTCRdict

from utils import get_peptide_tcr, MLogger, Args, _split_parameters, Project_path, Data_output, K_fold, Train_dataset, Train_output_dir, Train_Round, Device, Aa_dict, Negative_dataset, Batch_size, Shuffle, Data_config, Model_config

from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def train_main(train_data, save_path, logger_file, task_num: int = 166):
    args = Args(C=Data_config['Train']['Meta_learning']['Model_parameter']['num_of_index'], L=Data_config['Train']['Meta_learning']['Model_parameter']['len_of_embedding'], R=Data_config['Train']['Meta_learning']['Model_parameter']['len_of_index'],
                meta_lr=Data_config['Train']['Meta_learning']['Model_parameter']['meta_lr'], update_lr=Data_config['Train']['Meta_learning']['Model_parameter']['inner_loop_lr'], update_step=Data_config['Train']['Meta_learning']['Model_parameter']['inner_update_step'],
                update_step_test=Data_config['Train']['Meta_learning']['Model_parameter']['inner_fine_tuning'], regular=Data_config['Train']['Meta_learning']['Model_parameter']['regular_coefficient'], epoch=Data_config['Train']['Meta_learning']['Trainer_parameter']['epoch'],
                distillation_epoch=Data_config['Train']['Disentanglement_distillation']['Trainer_parameter']['epoch'],
                num_of_tasks=task_num)
    # using GPU for training model
    device = torch.device(Device)

    # initialize the model
    model = Memory_Meta(args, Model_config).to(device)
    # output the trainable tensors
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(model)
    print('Total trainable tensors:', num)
    # loading training data
    Training_data = PepTCRdict(train_data, os.path.join(Project_path, Negative_dataset), 2, 3,
                               aa_dict_path=os.path.join(Project_path, Aa_dict), mode='train')

    ################ Meta learning #################
    # initialize the logger for saving the traininig log
    logger = logger_file
    logger.info('Meta learning start!')
    # training model until convergence
    for epoch in range(args.epoch):
        print(f"epoch:{epoch}")
        # construct the dataloader for the training dataset
        db = DataLoader(Training_data, Batch_size, Shuffle, num_workers=1, pin_memory=True)
        # list for the training acc of each peptide-specific learner
        epoch_train_acc = []
        # data iteration
        s = time.time()
        for step, (peptide_embedding, x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            peptide_embedding, x_spt, y_spt, x_qry, y_qry = peptide_embedding.to(device), x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # return the average acc of each step
            accs = model(peptide_embedding, x_spt, y_spt, x_qry, y_qry)
            # store the acc of each step
            epoch_train_acc.append(accs)
        e = time.time()
        # calculate the average acc of each epoch
        epoch_train_acc = np.array(epoch_train_acc)
        accs = epoch_train_acc.mean(axis=0)
        if logger:
            logger.info('Epoch:[{}/{}]\tTraining_acc:{:.5f}\tTime:{:.3f}s'.format(epoch + 1, args.epoch, accs[-1], e - s))
        # store the meta learner, peptide-specific learner, query set and loss in the last training epoch
        if epoch == args.epoch - 1:
            # store the peptide-specific learners
            # print(os.path.join(save_path, 'models.pkl'))
            joblib.dump(model.models, os.path.join(save_path, "models.pkl"))
            # store the logits of each peptide-specific task
            joblib.dump(model.prev_loss, os.path.join(save_path, "prev_loss.pkl"))
            # store the query set of each peptide-specific task
            joblib.dump(model.prev_data, os.path.join(save_path, "prev_data.pkl"))
            # store the meta learner
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        # reset the stored data
        model.reset()
    if logger:
        logger.info('Finish training!')

    ################ Disentanglement distillation #################
    # load the logits of each peptide-specific task
    prev_loss = joblib.load(os.path.join(save_path, "prev_loss.pkl"))
    # load the query set of each peptide-specific task
    prev_data = joblib.load(os.path.join(save_path, "prev_data.pkl"))
    # load the peptide-specific learners
    prev_models = joblib.load(os.path.join(save_path, "models.pkl"))
    # initialize the memory module
    if Device == 'cuda':
        model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cuda()
    else:
        model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cpu()
    # setting up the meta learner, peptide-specific learner, query set and loss in the last training epoch
    model.Memory_module.prev_loss = prev_loss
    model.Memory_module.prev_data = prev_data
    model.Memory_module.models = prev_models
    if logger:
        logger.info('\n')
        logger.info('Disentanglement distillation start!')
    # training model until convergence
    for d_epoch in range(args.distillation_epoch):
        print(f"epoch:{d_epoch}")
        # peptide-specific learner write into the memory
        model.Memory_module.writehead(model.Memory_module.models)
        # define the loss
        loss2 = 0
        # iteration on the peptide-specific tasks
        for o, (index_prev, x_prev, y_prev) in enumerate(model.Memory_module.prev_data):
            # calculate the weight based on the peptide embedding
            r = model.Memory_module(index_prev)[0]
            # calculate the logits
            logits = []
            # read the memory and ensemble the content memory
            for m, n in enumerate(model.Memory_module.memory.content_memory):
                # split the weight parameter into the model parameter
                weights_memory = _split_parameters(n.unsqueeze(0), model.net.parameters())
                # append the model ensembled logits
                logits.append(model.net(x_prev, weights_memory, bn_training=True))
            # calculate the distiilation cross entropy
            loss2 += torch.sum(torch.log(sum([r[k] * F.softmax(j) for k, j in enumerate(logits)])) * model.Memory_module.prev_loss[o] * -1)
        # calculate the mean loss
        loss2 /= o + 1
        if logger:
            logger.info('Epoch:[{}/{}]\tDistill_loss:{:.5f}\tTime:{:.3f}s'.format(d_epoch + 1, args.distillation_epoch, loss2.item(), e - s))
        model.Memory_module.optim.zero_grad()
        loss2.backward()
        model.Memory_module.optim.step()
        model.Memory_module.content_memory = model.Memory_module.memory.content_memory.detach()
    if logger:
        logger.info('Finish distillation!')
    # store the content memory and read head ("query")
    joblib.dump(model.Memory_module.memory.content_memory, os.path.join(save_path, "Content_memory.pkl"))
    joblib.dump(list(model.Memory_module.memory.parameters()), os.path.join(save_path, "Query.pkl"))


def KFold_data(k_fold, all_peptide, meta_dataset, output):
    kf = KFold(n_splits=k_fold, shuffle=True)
    kf_time = 1
    if not os.path.exists(output):
        os.makedirs(output)
    # save KFold result
    for train, test in kf.split(all_peptide):
        print('KFold:', str(kf_time))
        if not os.path.exists(os.path.join(output, 'kfold' + str(kf_time))):
            os.makedirs(os.path.join(output, 'kfold' + str(kf_time)))
        for i in range(len(train)):
            data_slice = meta_dataset.loc[meta_dataset['peptide'] == all_peptide[train[i]], :]
            if i == 0:
                train_data = data_slice
            else:
                train_data = pd.concat([train_data, data_slice])
        train_data.to_csv(os.path.join(output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv'),
                          index=False)
        for j in range(len(test)):
            data_slice = meta_dataset.loc[meta_dataset['peptide'] == all_peptide[test[j]], :]
            if j == 0:
                test_data = data_slice
            else:
                test_data = pd.concat([test_data, data_slice])
        test_data.to_csv(os.path.join(output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test.csv'),
                         index=False)
        kf_time += 1


if __name__ == '__main__':
    if not os.path.exists(os.path.join(Project_path, Data_output)):
        meta_dataset = pd.read_csv(os.path.join(Project_path, Train_dataset))
        pep_tcr_dict = get_peptide_tcr(meta_dataset, 'peptide', 'binding_TCR')
        peptide = list(meta_dataset.drop_duplicates(subset=['peptide'], keep='first', inplace=False)['peptide'])
        filter_peptide = [j for j in [i if len(meta_dataset[meta_dataset['peptide'] == i]) < 100 else None for i in peptide] if j != None]
        # filter_peptide = pep_tcr_dict.values()  # TODO
        all_data_dict = {'peptide': [], 'binding_TCR': [], 'label': []}
        for key, val in pep_tcr_dict.items():
            if key in filter_peptide:
                all_data_dict['peptide'].extend([key] * len(val))
                all_data_dict['binding_TCR'].extend(val)
                all_data_dict['label'].extend([1] * len(val))
        all_data_dict = pd.DataFrame(all_data_dict)
        KFold_data(K_fold, filter_peptide, all_data_dict, output=os.path.join(Project_path, Data_output))  # get KFold data (Just run once)
        # KFold_data(k_fold, peptide, meta_dataset, output=os.path.join(project_path, data_output))  # get KFold data (Just run once)
    for index in range(Train_Round):
        # seed
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        for kf_time in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):  # Here you can specify which fold or in config file ['current_fold']. (such as range(2, 3) is fold 2).
            # Create the output folder of the currently specified fold
            save_path = os.path.join(Project_path, Train_output_dir, 'Round' + str(index + 1), 'kfold' + str(kf_time))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # logger
            logger = MLogger(os.path.join(Project_path, Train_output_dir, 'Round' + str(index + 1), 'kfold' + str(kf_time), 'training.log'))
            logger.info('Round:' + str(index + 1) + ', KFold:' + str(kf_time))
            # training
            # get kfold training data
            train_data_path = os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv')
            train_data = pd.read_csv(train_data_path)
            train_main(train_data=train_data_path, save_path=save_path, logger_file=logger, task_num=len(Counter(train_data['peptide'])))
