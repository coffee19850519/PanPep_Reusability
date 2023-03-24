from typing import Union, List
import pandas as pd
import numpy as np
import logging
import yaml
import os
import torch
import joblib

from Requirements.Memory_meta_test import Memory_Meta
from Requirements.Memory_meta_test import Memory_module

import warnings

warnings.filterwarnings('ignore')

FilePath = Union[str, "PathLike[str]", pd.DataFrame]


def load_csv_like_file(file_name: FilePath, csv_encoding: str = 'utf-8', sep=','):
    '''
    read the csv file
    :param file_name:
    :param csv_encoding:
    :param sep:
    :return:
    '''
    if type(file_name) == pd.DataFrame:
        csv_file = file_name
    else:
        csv_file = pd.read_csv(file_name, encoding=csv_encoding, sep=sep)
    return csv_file


def get_peptide_tcr(src_csv: FilePath, PepColumnName: str = 'Peptide', CdrColumeName: str = 'Alpha CDR3', csv_encoding: str = 'utf-8', sep=','):
    '''
    Obtain peptide and tcr based on the column name and return them in dictionary
    :param src_csv:
    :param PepColumnName:
    :param CdrColumeName:
    :param csv_encoding:
    :param sep:
    :return:
    '''
    src_csv = load_csv_like_file(src_csv, csv_encoding=csv_encoding, sep=sep)
    PepTCRdict = {}
    for idx, pep in enumerate(src_csv[PepColumnName]):
        if pep not in PepTCRdict:
            PepTCRdict[pep] = []
        if (type(src_csv[CdrColumeName][idx]) == str) and (src_csv[CdrColumeName][idx] not in PepTCRdict[pep]):
            PepTCRdict[pep].append(src_csv[CdrColumeName][idx])
    return PepTCRdict


class Args:
    def __init__(self, C, L, R, update_lr, update_step_test, update_step=3, meta_lr=0.001, regular=0, epoch=500, distillation_epoch=800, num_of_tasks=208):
        self.C = C
        self.L = L
        self.R = R
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.regular = regular
        self.epoch = epoch
        self.distillation_epoch = distillation_epoch
        self.task_num = num_of_tasks


class MLogger:
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


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


# memory based net parameter reconstruction
def _split_parameters(x, memory_parameters):
    new_weights = []
    start_index = 0
    for i in range(len(memory_parameters)):
        end_index = np.prod(memory_parameters[i].shape)
        new_weights.append(x[:, start_index:start_index + end_index].reshape(memory_parameters[i].shape))
        start_index += end_index
    return new_weights


def get_num(dict1):
    num = 0
    for k, v in dict1.items():
        num += len(v)
    return num


def get_model(args, mdoel_config, model_path, device):
    '''
    get model
    '''
    # Initialize a new model
    model = Memory_Meta(args, mdoel_config).to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    # Load the memory block
    if str(device) == 'cuda':
        model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cuda()
    else:
        model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cpu()
    content = joblib.load(os.path.join(model_path, "Content_memory.pkl"))
    query = joblib.load(os.path.join(model_path, "Query.pkl"))
    # Load the content memory matrix and query matrix(read head)
    model.Memory_module.memory.content_memory = content
    model.Memory_module.memory.Query.weight = query[0]
    model.Memory_module.memory.Query.bias = query[1]
    return model


def aamapping(TCRSeq, encode_dim, aa_dict):
    """
    this function is used for encoding the TCR sequence

    Parameters:
        param TCRSeq: the TCR original sequence
        param encode_dim: the first dimension of TCR sequence embedding matrix

    Returns:
        this function returns a TCR embedding matrix;
        e.g. the TCR sequence of ASSSAA
        return: (6 + encode_dim - 6) x 5 embedding matrix, in which (encode_dim - 6) x 5 will be zero matrix

    Raises:
        KeyError - using 0 vector for replacing the original amino acid encoding
    """

    TCRArray = []
    if len(TCRSeq) > encode_dim:
        print('Length: ' + str(len(TCRSeq)) + ' over bound!')
        TCRSeq = TCRSeq[0:encode_dim]
    for aa_single in TCRSeq:
        try:
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            TCRArray.append(np.zeros(5, dtype='float64'))
    for i in range(0, encode_dim - len(TCRSeq)):
        TCRArray.append(np.zeros(5, dtype='float64'))
    return torch.FloatTensor(TCRArray)


# Sinusoidal position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)


def add_position_encoding(seq):
    """
    this function is used to add position encoding for the TCR embedding

    Parameters:
        param seq: the TCR embedding matrix

    Returns:
        this function returns a TCR embedding matrix containing position encoding
    """

    padding_ids = torch.abs(seq).sum(dim=-1) == 0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq


def task_embedding(pep, tcr_data, aa_dict):
    """
    this function is used to obtain the task-level embedding

    Parameters:
        param pep: peptide sequence
        param tcr_data: TCR and its label list in a pan-pep way;
        e.g. [[support TCRs],[support labels]] or [[support TCRs],[support labels],[query TCRs]]

    Returns:
        this function returns a peptide embedding, the embedding of support set, the labels of support set and the embedding of query set
    """
    # Obtain the TCRs of support set
    spt_TCRs = tcr_data[0]
    # Obtain the TCR labels of support set
    ypt = tcr_data[1]
    # Initialize the size of the Tensor for the support set and labels
    support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    support_y = np.zeros((1, len(ypt)), dtype=np.int)
    peptides = torch.FloatTensor(1, 75)
    # Determine whether there is a query set based on the length of input param2
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    # Initialize the size of the Tensor for the query set
    query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
    # Encoding for the peptide sequence
    peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))
    # Put the embedding of support set, labels and peptide embedding into the initialized tensor
    temp = torch.Tensor()
    for j in spt_TCRs:
        temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25, aa_dict))]).unsqueeze(0)])
    support_x[0] = temp
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()
    # Put the embedding of query set into the initialized tensor
    temp = torch.Tensor()
    if len(tcr_data) > 2:
        for j in qry_TCRs:
            temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25, aa_dict))]).unsqueeze(0)])
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
    return peptides, support_x, torch.LongTensor(support_y), query_x


def zero_task_embedding(pep, tcr_data, aa_dict):
        """
        this function is used to obtain the task-level embedding for the zero-shot setting

        Parameters:
            param pep: peptide sequence
            param tcr_data: TCR list
            e.g. [query TCRs]

        Returns:
            this function returns a peptide embedding and the embedding of query TCRs
        """
        # Obtain the TCRs of support set
        spt_TCRs = tcr_data
        # Initialize the size of the Tensor for the query set and peptide encoding
        query_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
        peptides = torch.FloatTensor(1, 75)
        # Encoding for the peptide sequence
        peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))
        # Put the embedding of query TCRs and peptide into the initialized tensor
        temp = torch.Tensor()
        for j in spt_TCRs:
            temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25, aa_dict))]).unsqueeze(0)])
        query_x[0] = temp
        peptides[0] = peptide_embedding.flatten()
        return peptides, query_x


# configure the model architecture and parameters
Model_config = [
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

# load the cofig file
config_file_path = os.path.join(os.path.abspath(''), 'Configs', 'TrainingConfig.yaml')
Data_config = load_config(config_file_path)

Project_path = eval(Data_config['Project_path'])
Train_output_dir = Data_config['Train']['Train_output_dir']
Test_output_dir = Data_config['Test']['Test_output_dir']
Data_output = Data_config['dataset']['data_output']
Train_dataset = eval(Data_config['dataset']['Training_dataset'])
Negative_dataset = Data_config['dataset']['Negative_dataset']
Zero_test_data = eval(Data_config['dataset']['Testing_zero_dataset'])
Train_Round = Data_config['dataset']['Train_Round']
K_fold = Data_config['dataset']['k_fold']

Aa_dict = eval(Data_config['dataset']['aa_dict'])
Device = Data_config['Train']['Meta_learning']['Model_parameter']['device']
Batch_size = Data_config['Train']['Meta_learning']['Sampling']['batch_size']
Shuffle = Data_config['Train']['Meta_learning']['Sampling']['sample_shuffle']
