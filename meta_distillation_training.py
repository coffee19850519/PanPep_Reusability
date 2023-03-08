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

    def info(self, message=None):
        self.logger.info(message)
        self.logger.removeHandler(self.logger.handlers)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


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


# load the cofig file
# config_file_path = "./Configs/TrainingConfig.yaml"
config_file_path = r"G:\OneDrive - University of Missouri\PanPep_reusability\5fold_train-test\Configs\TrainingConfig.yaml"


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


config = load_config(config_file_path)


def train_main(train_data, save_path, logger_file, task_num: int = 166, config_file: str = config_file_path):
    # config = load_config(config_file)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number',
                           default=config['Train']['Meta_learning']['Trainer_parameter']['epoch'])
    argparser.add_argument('--task_num', type=int, help='a total number of tasks',
                           default=task_num)  ###
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate',
                           default=config['Train']['Meta_learning']['Model_parameter']['meta_lr'])
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate',
                           default=config['Train']['Meta_learning']['Model_parameter']['inner_loop_lr'])
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps',
                           default=config['Train']['Meta_learning']['Model_parameter']['inner_update_step'])
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning',
                           default=config['Train']['Meta_learning']['Model_parameter']['inner_fine_tuning'])
    argparser.add_argument('--C', type=int, help='Peptide clustering number',
                           default=config['Train']['Meta_learning']['Model_parameter']['num_of_index'])
    argparser.add_argument('--R', type=int, help='Peptide Index matrix vector length',
                           default=config['Train']['Meta_learning']['Model_parameter']['len_of_index'])
    argparser.add_argument('--L', type=int, help='Peptide embedding length',
                           default=config['Train']['Meta_learning']['Model_parameter']['len_of_embedding'])
    argparser.add_argument('--regular', type=float, help='The regular coefficient',
                           default=config['Train']['Meta_learning']['Model_parameter']['regular_coefficient'])
    args = argparser.parse_args()
    # using GPU for training model
    device = torch.device(config['Train']['Meta_learning']['Model_parameter']['device'])
    print(args)
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
    Training_data = PepTCRdict(train_data, config['dataset']['Negative_dataset'], 2, 3, mode='train')

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
            print(save_path + "/models.pkl")#TODO
            joblib.dump(model.models, save_path + "/models.pkl")#TODO
            # store the logits of each peptide-specific task
            joblib.dump(model.prev_loss, save_path + "/prev_loss.pkl")#TODO
            # store the query set of each peptide-specific task
            joblib.dump(model.prev_data, save_path + "/prev_data.pkl")#TODO
            # store the meta learner
            torch.save(model.state_dict(), save_path + "/model.pt")#TODO
        # reset the stored data
        model.reset()
    if logger:
        logger.info('Finish training!')

    ################ Disentanglement distillation #################
    # load the logits of each peptide-specific task
    prev_loss = joblib.load(save_path + "/prev_loss.pkl")#TODO
    # load the query set of each peptide-specific task
    prev_data = joblib.load(save_path + "/prev_data.pkl")#TODO
    # load the peptide-specific learners
    prev_models = joblib.load(save_path + "/models.pkl")#TODO
    # initialize the memory module
    model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cuda()
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
    joblib.dump(model.Memory_module.memory.content_memory, save_path + "/Content_memory.pkl")#TODO
    joblib.dump(list(model.Memory_module.memory.parameters()), save_path + "/Query.pkl")#TODO


def KFold_data(k_fold, output):
    kf = KFold(n_splits=k_fold, shuffle=True)
    kf_time = 1
    if not os.path.exists(output):
        os.makedirs(output)
    # save KFold result
    for train, test in kf.split(peptide):
        print('KFold:', str(kf_time))
        if not os.path.exists(output + '/kfold' + str(kf_time)):#TODO
            os.makedirs(output + '/kfold' + str(kf_time))#TODO
        for i in range(len(train)):
            data_slice = meta_dataset.loc[meta_dataset['peptide'] == peptide[train[i]], :]
            if i == 0:
                train_data = data_slice
            else:
                train_data = pd.concat([train_data, data_slice])
        train_data.to_csv(
            output + '/kfold' + str(kf_time) + '/KFold_' + str(kf_time) + '_train.csv', #TODO
            index=False)
        for j in range(len(test)):
            data_slice = meta_dataset.loc[meta_dataset['peptide'] == peptide[test[j]], :]
            if j == 0:
                test_data = data_slice
            else:
                test_data = pd.concat([test_data, data_slice])
        test_data.to_csv(
            output + '/kfold' + str(kf_time) + '/KFold_' + str(kf_time) + '_test.csv',
            index=False)
        kf_time += 1


if __name__ == '__main__':
    meta_dataset = pd.read_csv('./Requirements/meta_dataset.csv')
    peptide = list(meta_dataset.drop_duplicates(subset=['peptide'], keep='first', inplace=False)['peptide'])
    Round = config['dataset']['Train_Round']  # The number of cross-validations
    k_fold = config['dataset']['k_fold']
    data_output = config['dataset']['data_output']
    # KFold_data(k_fold, output=data_output)  # get KFold data (Just run once)
    for index in range(Round):
        # seed
        seed1 = random.randint(0, 10000)
        torch.manual_seed(seed1)
        torch.cuda.manual_seed_all(seed1)
        np.random.seed(seed1)
        random.seed(seed1)
        torch.cuda.manual_seed(seed1)

        for kf_time in range(2, 3):  # Here you can specify which fold
            if not os.path.exists('Round' + str(index + 1) + '/kfold' + str(kf_time)): #TODO
                os.makedirs('Round' + str(index + 1) + '/kfold' + str(kf_time)) #TODO
            logger = DemoLogger('Round' + str(index + 1) + '/kfold' + str(kf_time) + "/training.log")#TODO
            logger.info('Round:' + str(index + 1) + ', KFold:' + str(kf_time))
            logger.info('Meta learning start!')
            # training
            train_data = pd.read_csv(os.path.join(data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv'))
            train_main(os.path.join(data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv'),
                       'Round' + str(index + 1) + '/kfold' + str(kf_time), logger_file=logger, task_num=len(Counter(train_data['peptide'])))
            #TODO