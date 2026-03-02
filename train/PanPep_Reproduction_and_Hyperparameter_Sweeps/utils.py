"""
Utility functions and classes for TCR-peptide binding prediction.

This module provides essential utilities including data processing, model configuration,
logging, and embedding functions for the meta-learning framework.
"""

import os
import random
import logging
import warnings
from typing import Union, List, Any, Callable
from collections import Counter, OrderedDict
import weakref

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

try:
    from train.Requirements.Memory_meta import Memory_Meta
    from train.Requirements.Memory_meta import Memory_module
except ImportError:
    # Fallback to project root import
    from Memory_meta import Memory_Meta
    from Memory_meta import Memory_module

warnings.filterwarnings('ignore')

FilePath = Union[str, "PathLike[str]", pd.DataFrame]


def load_csv_like_file(file_name: FilePath, csv_encoding: str = 'utf-8', sep=','):
    """Load CSV file or return DataFrame if already provided.

    Args:
        file_name: Path to CSV file or pandas DataFrame
        csv_encoding: Character encoding for CSV file
        sep: Column separator

    Returns:
        pandas.DataFrame: Loaded data
    """
    if type(file_name) == pd.DataFrame:
        csv_file = file_name
    else:
        csv_file = pd.read_csv(file_name, encoding=csv_encoding, sep=sep)
    return csv_file


def get_peptide_tcr(src_csv: FilePath, PepColumnName: str = 'Peptide', CdrColumeName: str = 'Alpha CDR3', LabelColumeName: str = None, MHCColunmName: str=None, csv_encoding: str = 'utf-8', sep=','):
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
            if not LabelColumeName:
                PepTCRdict[pep] = []
            if not MHCColunmName:
                PepTCRdict[pep] = [[], []]
            else:
                PepTCRdict[pep] = [[], [], []]
        if not LabelColumeName:
            if (type(src_csv[CdrColumeName][idx]) == str) and (src_csv[CdrColumeName][idx] not in PepTCRdict[pep]):
                PepTCRdict[pep].append([src_csv[CdrColumeName][idx], src_csv[MHCColunmName][idx]])
        if not MHCColunmName:
            if (type(src_csv[CdrColumeName][idx]) == str) and (src_csv[CdrColumeName][idx] not in PepTCRdict[pep][0]):
                PepTCRdict[pep][0].append(src_csv[CdrColumeName][idx])
                PepTCRdict[pep][1].append(src_csv[LabelColumeName][idx])
        else:
            if (type(src_csv[CdrColumeName][idx]) == str) and (src_csv[CdrColumeName][idx] not in PepTCRdict[pep][0]):
                PepTCRdict[pep][0].append(src_csv[CdrColumeName][idx])
                PepTCRdict[pep][1].append(src_csv[LabelColumeName][idx])
                # PepTCRdict[pep][2].append(src_csv[MHCColunmName][idx])
    return PepTCRdict


def add_negative_data(positive_data, negative_data, ratio=1):
    """Add negative samples to positive data with specified ratio.

    Args:
        positive_data: DataFrame containing positive samples
        negative_data: Array or file path containing negative samples
        ratio: Ratio of negative to positive samples

    Returns:
        dict: Combined positive and negative data
    """
    if type(negative_data) is str:
        negative_data = np.loadtxt(negative_data, dtype=str)
    peptide_ = Counter(positive_data['peptide'])
    negative_ = {}
    # print('All:', len(peptide_))
    positive = 0
    for pep, num in peptide_.items():
        # negative_[pep] = [[], [], []]
        negative_[pep] = [[], []]
        negative_[pep][0].extend(positive_data[positive_data['peptide'] == pep]['binding_TCR'].array)
        negative_[pep][1].extend(positive_data[positive_data['peptide'] == pep]['label'].array)
        # negative_[pep][2].extend(positive_data[positive_data['peptide'] == pep]['HMC2_name'].array)
        positive += num

    selected_query_idx = np.random.choice(len(negative_data), int(positive * ratio), replace=False)
    selected_query_TCRs = negative_data[selected_query_idx]
    befor_num = 0
    for i, j in negative_.items():
        pep_num = len(j[0])
        negative_[i][0].extend(selected_query_TCRs[befor_num: befor_num + pep_num])
        negative_[i][1].extend([0] * pep_num)
        # negative_[i][2].extend(['Negative'] * pep_num)
        befor_num += pep_num

    # all_data_dict = {'peptide': [], 'binding_TCR': [], 'label': [], 'HMC2_name': []}
    all_data_dict = {'peptide': [], 'binding_TCR': [], 'label': []}
    for key, val in negative_.items():
        all_data_dict['peptide'].extend([key] * len(val[0]))
        all_data_dict['binding_TCR'].extend(val[0])
        all_data_dict['label'].extend(val[1])
        # all_data_dict['HMC2_name'].extend(val[2])
    # all_data_dict = pd.DataFrame(all_data_dict)
    return all_data_dict


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
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")

    def __init__(self, filename, verbosity=1, name=__name__):
        self.logger = logging.getLogger(name)
        # logging.Logger.manager.loggerDict.pop(__name__)
        self.logger.handlers = []
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
    if os.path.exists(os.path.join(model_path, 'model.pt')):
        # Load the pretrained model
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
    else:
        print("Do not find 'model.pt', only can do zero-shot!")
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
    support_y = np.zeros((1, len(ypt)), dtype=np.int64)
    peptides = torch.FloatTensor(1, 75)
    # Determine whether there is a query set based on the length of input param2
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
        qry_TCRs = list(dict.fromkeys(qry_TCRs))
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

def get_query_data(all_ranking_data, k_shot_data, k_shot):

    F_data = [[], []]
    F_data[0].extend([j for j in all_ranking_data[0] if j not in k_shot_data[0]])
    index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
    index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
    F_data[1].extend([1] * (len(index_p) - k_shot))
    F_data[1].extend([0] * (len(index_n) - k_shot))

    return F_data


def save_kshot_data(all_ranking_data, k_shot, pep, result):

    data = [[], []]

    index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
    index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
    positive_support_idx = random.sample(index_p, k_shot)
    negative_support_idx = random.sample(index_n, k_shot)

    # Choose support
    for idx in range(k_shot):
        data[0].append(all_ranking_data[0][positive_support_idx[idx]])
        data[1].append(1)
        data[0].append(all_ranking_data[0][negative_support_idx[idx]])
        data[1].append(0)

    output = pd.DataFrame({'tcr': data[0], 'label': data[1]})
    output.to_csv(os.path.join(result, "k_shot_" + pep + ".csv"), index=False)

    return data

def read_kshot_data(all_ranking_data, k_shot, pep, result):

    data = [[], []]

    data[0] = list(pd.read_csv(os.path.join(result, "k_shot_" + pep + ".csv"))['tcr'])
    data[1] = list(pd.read_csv(os.path.join(result, "k_shot_" + pep + ".csv"))['label'])

    return data


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


class RemovableHandle(object):
    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def get_train_data(peptide, *args, **kwargs):
    """Collect training data for meta-learning phase.

    This function aggregates all TCR sequences used during meta-training
    for a specific peptide, including both support and query sets.

    Args:
        peptide: Peptide sequence identifier
        *args: Variable arguments containing TCR sequences
        **kwargs: Additional data dictionary to update

    Returns:
        dict: Updated data dictionary with peptide-TCR mappings
              Format: {peptide: [list of all TCRs used in training]}
    """
    all_data = {}
    if kwargs:
        all_data = kwargs
    assert not (peptide in all_data.keys()), 'Peptide already exists!'
    tcr_ = []
    try:
        for tcr in args:
            if type(tcr) != list:
                tcr = list(tcr)
            tcr_.extend(tcr)
        all_data[peptide] = tcr_
    except Exception as e:
        print(e)
    return all_data


def merge_dict(*dicts):
    """Merge multiple dictionaries and remove duplicates from values.

    Args:
        *dicts: Variable number of dictionaries to merge

    Returns:
        dict: Merged dictionary with deduplicated list values
    """
    result = {}
    for dict in dicts:
        if len(result.keys()) == 0:
            result.update(dict)
        else:
            for k, v in dict.items():
                if k in result.keys():
                    result[k] = list(set(result[k] + v))
                else:
                    result[k] = v
    return result


def merge_all_TCR(pep_tcr: dict, new_dict: dict):
    for k, v in pep_tcr.items():
        pass
    pass


def generate_selected_idx(n, generator=None):
    """Generate random indices from 0 to n-1.

    Args:
        n: Upper bound for index generation
        generator: Optional torch random generator

    Yields:
        int: Random indices in shuffled order
    """
    if generator is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = generator
    a = torch.randperm(n, generator=generator)
    for i in range(n):
        yield a[i]


def return_m_from_n(m, n=None, generate_idx=None):
    """Return m items from a generator of length n.

    Args:
        m: Number of items to return
        n: Total length (required if generate_idx is None)
        generate_idx: Optional index generator

    Returns:
        list: List of m randomly selected indices

    Raises:
        AssertionError: If m > n or required parameters are missing
    """
    assert ((n is not None) and (m <= n)) or ((generate_idx is not None) and (m <= generate_idx.gi_frame.f_locals['n']))
    if generate_idx is None:
        generate_idx = generate_selected_idx(n)
    return_list = []
    for i in range(m):
        return_list.append(next(generate_idx).item())
    return return_list


# configure the model architecture and parameters
Model_config = [   # PanPep_Reproduction
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

Model_config_large_128 = [ 
    ('self_attention', [[1,  128,  5],[1,  128,  5], [1,  128,  5]]),
    ('linear', [128, 128]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 19456])
]

Model_config_large_16 = [
    ('self_attention', [[1,  16,  5],[1,  16,  5], [1,  16,  5]]),
    ('linear', [16, 16]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 2432])
]

Model_config_large_32 = [
    ('self_attention', [[1,32,  5],[1,32,  5], [1, 32,  5]]),
    ('linear', [32,32]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2,4864])
]

Model_config_attention5_conv3_large_16 = [
    ('res_attention_block', [[8, 16, 5], [8, 16,16], [8, 16, 16],[8, 16,16],[8, 16, 16]]), 
    ('linear', [16, 16]),
    ('relu', [True]),
    ('res_block', [[16, 1, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1]]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 2560])
]

Model_config_attention8 = [
    ('self_attention', [[8, 5, 5], [8, 5, 5], [8, 5, 5]]),
    ('linear', [5, 5]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]

Model_config_attention16 = [
    ('self_attention', [[16, 5, 5], [16, 5, 5], [16, 5, 5]]),
    ('linear', [5, 5]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]


Model_config_attention_stack5 = [
    ('res_attention_block', [[1, 5, 5], [1, 5, 5], [1, 5, 5],[1, 5, 5],[1, 5, 5]]), 
    ('linear', [5, 5]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]


Model_config_attention_stack8 = [
    ('res_attention_block', [[1, 5, 5], [1, 5, 5], [1, 5, 5],[1, 5, 5],[1, 5, 5],[1, 5, 5], [1, 5, 5], [1, 5, 5]]), 
    ('linear', [5, 5]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]

Model_config_conv_stack3 = [
    ('self_attention', [[1, 5, 5], [1, 5, 5], [1, 5, 5]]),
    ('linear', [5, 5]),
    ('relu', [True]),
    ('res_block', [[16, 1, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1]]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 640])
]

Model_config_conv_stack6 = [
    ('self_attention', [[1, 5, 5], [1, 5, 5], [1, 5, 5]]),
    ('linear', [5, 5]),
    ('relu', [True]),
    ('res_block', [[16, 1, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1]]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 640])
]

BASE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(BASE_DIR, "Configs")

Data_config  = load_config(os.path.join(CONFIG_DIR, "TrainingConfig.yaml"))


Project_path = eval(Data_config['Project_path'])
Train_output_dir = Data_config['Train']['Train_output_dir']
global Train_output_dir4other_update
Train_output_dir4other_update = Data_config['Train']['Train_output_dir4other_update_step']
if Train_output_dir.endswith('_'):
    data2save_path = 'save_train_data_'
    # Train_output_dir4other_update += '_'

else:
    data2save_path = 'save_train_data'
Test_output_dir = Data_config['Test']['Test_output_dir']
Data_output = Data_config['dataset']['data_output']
Train_dataset = eval(Data_config['dataset']['Training_dataset'])
Negative_dataset = Data_config['dataset']['Negative_dataset']
Zero_test_data = eval(Data_config['dataset']['Testing_zero_dataset'])
Testing_zero_remove_dataset = eval(Data_config['dataset']['Testing_zero_remove_dataset'])
# Majority_train_data = eval(Data_config['Train']['Majority']['Training_dataset'])
Majority_test_data = eval(Data_config['Train']['Majority']['Test_dataset'])
Majority_test_dataset_label = eval(Data_config['Train']['Majority']['Test_dataset_label'])
Train_Round = Data_config['dataset']['Train_Round']
K_fold = Data_config['dataset']['k_fold']

Aa_dict = os.path.join(Project_path, eval(Data_config['dataset']['aa_dict']))
Device = Data_config['Train']['Meta_learning']['Model_parameter']['device']
Batch_size = Data_config['Train']['Meta_learning']['Sampling']['batch_size']
Shuffle = Data_config['Train']['Meta_learning']['Sampling']['sample_shuffle']

Support = Data_config['Train']['Meta_learning']['Sampling']['support']
Query = Data_config['Train']['Meta_learning']['Sampling']['query']
