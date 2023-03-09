import argparse
import logging
import random
import time

import joblib
import numpy as np
import os
import pandas as pd
import scipy.stats
import torch
import yaml
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Memory_meta import Memory_Meta
from Memory_meta import Memory_module
from PepTCRdict import PepTCRdict

from collections import Counter


class Args:
    def __init__(self, C, L, R, update_lr, update_step_test, update_step=3, meta_lr=0.001, regular=0, epoch=500, num_of_tasks=208):
        self.C = C
        self.L = L
        self.R = R
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.regular = regular
        self.epoch = epoch
        self.task_num = num_of_tasks


class DemoLogger:
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    def __init__(self, filename, verbosity=1):
        self.logger = logging.getLogger(__name__)
        logging.Logger.manager.loggerDict.pop(__name__)
        self.logger.handlers = []
        self.logger.removeHandler(self.logger.handlers)
        self.filename = filename
        self.verbosity = verbosity
        if not self.logger.handlers:
            self.handler = logging.FileHandler(self.filename, encoding="UTF-8")
            self.logger.setLevel(self.level_dict[self.verbosity])
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)
            self.sh = logging.StreamHandler()
            self.sh.setFormatter(self.formatter)
            self.logger.addHandler(self.sh)

    def info(self, message=None):
        self.logger.info(message)
        self.logger.removeHandler(self.logger.handlers)


# memory based net parameter reconstruction
def _split_parameters(x, memory_parameters):
    new_weights = []
    start_index = 0
    for i in range(len(memory_parameters)):
        end_index = np.prod(memory_parameters[i].shape)
        new_weights.append(x[:, start_index:start_index + end_index].reshape(memory_parameters[i].shape))
        start_index += end_index
    return new_weights


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def train_main(train_data, save_path, logger_file, task_num: int = 166):
    args = Args(C=config['Train']['Meta_learning']['Model_parameter']['num_of_index'], L=config['Train']['Meta_learning']['Model_parameter']['len_of_embedding'], R=config['Train']['Meta_learning']['Model_parameter']['len_of_index'],
                meta_lr=config['Train']['Meta_learning']['Model_parameter']['meta_lr'], update_lr=config['Train']['Meta_learning']['Model_parameter']['inner_loop_lr'], update_step=config['Train']['Meta_learning']['Model_parameter']['inner_update_step'],
                update_step_test=config['Train']['Meta_learning']['Model_parameter']['inner_fine_tuning'], regular=config['Train']['Meta_learning']['Model_parameter']['regular_coefficient'], epoch=config['Train']['Meta_learning']['Trainer_parameter']['epoch'],
                num_of_tasks=task_num)
    # using GPU for training model
    device = torch.device(config['Train']['Meta_learning']['Model_parameter']['device'])
    # configure the model architecture and parameters
    config_model = [
        ('self_attention', [[1, 5, 5], [1, 5, 5], [1, 5, 5]]),
        ('linear', [5, 5]),
        ('relu', [True]),
        ('conv2d', [16, 1, 2, 1, 1, 0]),
        ('relu', [True]),
        ('bn', [16]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [2, 608])
    ]
    # initialize the model
    model = Memory_Meta(args, config_model).to(device)
    # output the trainable tensors
    tmp = filter(lambda x: x.requires_grad, model.parameters())  # 过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(model)
    print('Total trainable tensors:', num)
    # loading training data
    Training_data = PepTCRdict(train_data, os.path.join(eval(config['Project_path']), config['dataset']['Negative_dataset']), 2, 3,
                               aa_dict_path=os.path.join(eval(config['Project_path']), eval(config['dataset']['aa_dict'])), mode='train')

    ################ Meta learning #################
    # initialize the logger for saving the traininig log
    logger = logger_file
    logger.info('Meta learning start!')
    # training model until convergence
    for epoch in range(args.epoch):
        print(f"epoch:{epoch}")
        # construct the dataloader for the training dataset
        db = DataLoader(Training_data, config['Train']['Meta_learning']['Sampling']['batch_size'], shuffle=config['Train']['Meta_learning']['Sampling']['sample_shuffle'], num_workers=1, pin_memory=True)
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
            joblib.dump(model.models, os.path.join(save_path,  "models.pkl"))
            # store the logits of each peptide-specific task
            joblib.dump(model.prev_loss, os.path.join(save_path,  "prev_loss.pkl"))
            # store the query set of each peptide-specific task
            joblib.dump(model.prev_data, os.path.join(save_path,  "prev_data.pkl"))
            # store the meta learner
            torch.save(model.state_dict(), os.path.join(save_path,  "model.pt"))
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
    if config['Train']['Meta_learning']['Model_parameter']['device'] == 'cuda':
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
    for d_epoch in range(config['Train']['Disentanglement_distillation']['Trainer_parameter']['epoch']):
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
            logger.info('Epoch:[{}/{}]\tDistill_loss:{:.5f}\tTime:{:.3f}s'.format(d_epoch + 1, config['Train']['Disentanglement_distillation']['Trainer_parameter']['epoch'], loss2.item(), e - s))
        model.Memory_module.optim.zero_grad()
        loss2.backward()
        model.Memory_module.optim.step()
        model.Memory_module.content_memory = model.Memory_module.memory.content_memory.detach()
    if logger:
        logger.info('Finish distillation!')
    # store the content memory and read head ("query")
    joblib.dump(model.Memory_module.memory.content_memory, os.path.join(save_path, "Content_memory.pkl"))
    joblib.dump(list(model.Memory_module.memory.parameters()), os.path.join(save_path, "Query.pkl"))


def KFold_data(k_fold, all_peptide, output):
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
    # load the cofig file
    config_file_path = os.path.join(os.path.abspath(''), 'Configs', 'TrainingConfig.yaml')
    config = load_config(config_file_path)

    project_path = eval(config['Project_path'])
    data_output = config['dataset']['data_output']
    if not os.path.exists(os.path.join(project_path, data_output)):
        k_fold = config['dataset']['k_fold']
        meta_dataset = pd.read_csv(os.path.join(project_path, eval(config['dataset']['Training_dataset'])))
        peptide = list(meta_dataset.drop_duplicates(subset=['peptide'], keep='first', inplace=False)['peptide'])
        KFold_data(k_fold, peptide, output=os.path.join(project_path, data_output))  # get KFold data (Just run once)
    Round = config['dataset']['Train_Round']  # The number of cross-validations
    for index in range(Round):
        # seed
        seed1 = random.randint(0, 10000)
        torch.manual_seed(seed1)
        torch.cuda.manual_seed_all(seed1)
        np.random.seed(seed1)
        random.seed(seed1)
        torch.cuda.manual_seed(seed1)

        for kf_time in range(config['dataset']['current_fold'][0], config['dataset']['current_fold'][1]):  # Here you can specify which fold or in config file ['current_fold']. (such as range(2, 3) is fold 2).
            # Create the output folder of the currently specified fold
            if not os.path.exists(os.path.join(project_path, 'Round' + str(index + 1), 'kfold' + str(kf_time))):
                os.makedirs(os.path.join(project_path, 'Round' + str(index + 1), 'kfold' + str(kf_time)))
            # logger
            logger = DemoLogger(os.path.join(project_path, 'Round' + str(index + 1), 'kfold' + str(kf_time), 'training.log'))
            logger.info('Round:' + str(index + 1) + ', KFold:' + str(kf_time))
            logger.info('Meta learning start!')
            # training
            # get kfold training data
            train_data = pd.read_csv(os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv'))
            train_main(train_data=os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv'),
                       save_path=os.path.join(project_path, 'Round' + str(index + 1), 'kfold' + str(kf_time)),
                       logger_file=logger, task_num=len(Counter(train_data['peptide'])))
