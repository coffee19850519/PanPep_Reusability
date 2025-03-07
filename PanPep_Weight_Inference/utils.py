import random
from typing import Union, List
import pandas as pd
import numpy as np
import logging
import yaml
import os
import torch
import joblib

from collections import Counter
import weakref
from collections import OrderedDict
from typing import Any, Callable

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
    '''
    将正、负样本加入新的csv中（个数与正样本一致）
    Args:
        positive_data:
        negative_data:

    Returns:

    '''
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
        #print('Length: ' + str(len(TCRSeq)) + ' over bound!')
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


# def task_embedding(pep, tcr_data, aa_dict):
#     """
#     this function is used to obtain the task-level embedding

#     Parameters:
#         param pep: peptide sequence
#         param tcr_data: TCR and its label list in a pan-pep way;
#         e.g. [[support TCRs],[support labels]] or [[support TCRs],[support labels],[query TCRs]]

#     Returns:
#         this function returns a peptide embedding, the embedding of support set, the labels of support set and the embedding of query set
#     """
#     # Obtain the TCRs of support set
#     spt_TCRs = tcr_data[0]
#     # Obtain the TCR labels of support set
#     ypt = tcr_data[1]
#     # Initialize the size of the Tensor for the support set and labels
#     support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
#     support_y = np.zeros((1, len(ypt)), dtype=np.int64)
#     peptides = torch.FloatTensor(1, 75)
#     # Determine whether there is a query set based on the length of input param2
#     if len(tcr_data) > 2:
#         qry_TCRs = tcr_data[2]
#         qry_TCRs = list(dict.fromkeys(qry_TCRs))
#     else:
#         qry_TCRs = ['None']
#     # Initialize the size of the Tensor for the query set
#     query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
#     # Encoding for the peptide sequence
#     peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))
#     # Put the embedding of support set, labels and peptide embedding into the initialized tensor
#     temp = torch.cat([
#         torch.cat([peptide_embedding, add_position_encoding(aamapping(tcr, 25, aa_dict))]).unsqueeze(0)
#         for tcr in spt_TCRs
#     ])
#     support_x[0] = temp
#     support_y[0] = np.array(ypt)
#     peptides[0] = peptide_embedding.flatten()
#     # Put the embedding of query set into the initialized tensor
#     temp = torch.Tensor()
#     if len(tcr_data) > 2:
#         tcr_embeddings = torch.stack([
#             add_position_encoding(aamapping(tcr, 25, aa_dict))
#             for tcr in qry_TCRs
#         ])
        
#         # 扩展peptide_embedding以匹配batch size
#         peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
        
#         # 一次性连接所有数据
#         temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
#         query_x[0] = temp
#     else:
#         query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
#     return peptides, support_x, torch.LongTensor(support_y), query_x

def task_embedding(pep, tcr_data, aa_dict, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    从预计算的字典中获取task-level embedding，如果字典不存在则实时计算

    Parameters:
        param pep: peptide序列
        param tcr_data: TCR及其标签列表
        param aa_dict: 氨基酸编码字典
        param peptide_encoding_dict: 预计算的peptide编码字典 {peptide序列: 编码tensor}
        param tcr_encoding_dict: 预计算的TCR编码字典 {TCR序列: 编码tensor}

    Returns:
        返回peptide embedding，support set embedding，support set labels和query set embedding
        如果support set为空，对应的support_x和support_y返回None
    """
    spt_TCRs = tcr_data[0]
    ypt = tcr_data[1]
    

    peptides = torch.FloatTensor(1, 75)
    
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    # 从peptide字典获取或计算peptide编码
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))

    # 检查support set是否为空
    if not spt_TCRs or not ypt:
        peptides[0] = peptide_embedding.flatten()
        
        # 处理query set
        if len(tcr_data) > 2:
            tcr_embeddings = []
            for tcr in qry_TCRs:
                if tcr_encoding_dict and tcr in tcr_encoding_dict:
                    tcr_embed = tcr_encoding_dict[tcr]
                else:
                    tcr_embed = add_position_encoding(aamapping(tcr, 25, aa_dict))
                tcr_embeddings.append(tcr_embed)
            
            tcr_embeddings = torch.stack(tcr_embeddings)
            peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
            temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
            query_x[0] = temp
        else:
            query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
            
        return peptides, None, None, query_x

    # 如果support set不为空，继续处理
    support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    support_y = np.zeros((1, len(ypt)), dtype=np.int64)
    
    # 处理support set
    temp = []
    for tcr in spt_TCRs:
        if tcr_encoding_dict and tcr in tcr_encoding_dict:
            tcr_embed = tcr_encoding_dict[tcr]
        else:
            tcr_embed = add_position_encoding(aamapping(tcr, 25, aa_dict))
        temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
    
    support_x[0] = torch.cat(temp)
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()

    # 处理query set
    if len(tcr_data) > 2:
        tcr_embeddings = []
        for tcr in qry_TCRs:
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25, aa_dict))
            tcr_embeddings.append(tcr_embed)
        
        tcr_embeddings = torch.stack(tcr_embeddings)
        peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
        temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
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

def get_query_data_more_data(all_ranking_data, k_shot_data, k_shot,update_step_test):

    F_data = [[], []]
    F_data[0].extend([j for j in all_ranking_data[0] if j not in k_shot_data[0]])
    index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
    index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
    F_data[1].extend([1] * (len(index_p) - k_shot))
    F_data[1].extend([0] * (len(index_n) - (update_step_test*k_shot)))

    return F_data
def load_kshot_more_data(all_ranking_data, k_shot, pep, data_dir, result, update_step_test):
    """
    加载k-shot数据：
    - 保持CSV中的所有正样本和负样本
    - 从all_ranking_data中随机选择额外的负样本(需要的总数-2)
    """
    data = [[], []]
    
    # 构建文件路径
    file_name = f"k_shot_{pep}.csv"
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 读取CSV中的所有样本
    df = pd.read_csv(file_path)
    
    # 添加CSV中的正样本
    positive_samples = df[df['label'] == 1]
    data[0].extend(positive_samples['tcr'].tolist())
    data[1].extend([1] * len(positive_samples))
    
    # 添加CSV中的负样本
    negative_samples = df[df['label'] == 0]
    data[0].extend(negative_samples['tcr'].tolist())
    data[1].extend([0] * len(negative_samples))
    
    # 计算需要额外选择的负样本数量
    additional_negative_needed = update_step_test * k_shot - len(negative_samples)
    
    # 从all_ranking_data中选择额外的负样本
    if additional_negative_needed > 0:
        index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
        negative_support_idx = random.sample(index_n, additional_negative_needed)
        
        # 添加额外选择的负样本
        for idx in negative_support_idx:
            data[0].append(all_ranking_data[0][idx])
            data[1].append(0)
    
    # 保存更新后的数据
    output = pd.DataFrame({'tcr': data[0], 'label': data[1]})
    output.to_csv(os.path.join(result, "k_shot_" + pep + ".csv"), index=False)
    
    return data
def save_kshot_data_more_data(all_ranking_data, k_shot, pep, result,update_step_test):

    data = [[], []]

    index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
    index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
    positive_support_idx = random.sample(index_p, k_shot)
    negative_support_idx = random.sample(index_n, update_step_test*k_shot)

    # Choose support
    for idx in positive_support_idx:
        data[0].append(all_ranking_data[0][idx])
        data[1].append(1)
    
    # 添加所有选择的负样本
    for idx in negative_support_idx:
        data[0].append(all_ranking_data[0][idx])
        data[1].append(0)

    output = pd.DataFrame({'tcr': data[0], 'label': data[1]})
    output.to_csv(os.path.join(result, "k_shot_" + pep + ".csv"), index=False)
    
    return data

def load_kshot_data(pep, data_dir):
    """
    从指定目录加载特定肽段的k-shot数据
    
    参数:
    pep: 肽段名称（如 'CLAVHECFV'）
    data_dir: 包含k-shot数据文件的目录路径
    
    返回:
    data: 列表 [TCR序列列表, 标签列表]
    """
    # 初始化数据结构
    data = [[], []]  # [TCR列表, 标签列表]
    
    # 构建文件路径
    file_name = f"k_shot_{pep}.csv"
    file_path = os.path.join(data_dir, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 按照原格式构建数据
    data[0] = df['tcr'].tolist()  # TCR序列列表
    data[1] = df['label'].tolist()  # 标签列表
    
    return data


# def save_kshot_data(all_ranking_data, k_shot, pep, result,chain_type):

#     data = [[], []]

#     index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
#     index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
#     positive_support_idx = random.sample(index_p, k_shot)
#     negative_support_idx = random.sample(index_n, k_shot)

#     # Choose support
#     for idx in range(k_shot):
#         data[0].append(all_ranking_data[0][positive_support_idx[idx]])
#         data[1].append(1)
#         data[0].append(all_ranking_data[0][negative_support_idx[idx]])
#         data[1].append(0)

#     output = pd.DataFrame({'tcr': data[0], 'label': data[1]})
#     output.to_csv(os.path.join(result, "k_shot_" + pep + ".csv"), index=False)

#     return data

def save_kshot_data(all_ranking_data, k_shot, pep, result, chain_type=None):
    """
    保存k-shot数据，如果提供了chain_type，文件名中会包含该信息
    
    参数:
        all_ranking_data: 包含TCR序列和标签的数据
        k_shot: 每类样本的数量
        pep: 肽段名称
        result: 结果保存路径
        chain_type: 可选参数，指定链类型（alpha或beta）
    """
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
    
    # 根据是否提供chain_type决定文件名
    if chain_type:
        filename = f"k_shot_{pep}_{chain_type}.csv"
    else:
        filename = f"k_shot_{pep}.csv"
    
    output.to_csv(os.path.join(result, filename), index=False)

    return data



def read_kshot_data(all_ranking_data, k_shot, pep, result):

    data = [[], []]

    data[0] = list(pd.read_csv(os.path.join(result, "k_shot_" + pep + ".csv"))['tcr'])
    data[1] = list(pd.read_csv(os.path.join(result, "k_shot_" + pep + ".csv"))['label'])

    return data


def zero_task_embedding(pep, tcr_data, aa_dict, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    从预计算的字典中获取zero-shot的task-level embedding，如果字典不存在则实时计算

    Parameters:
        param pep: peptide序列
        param tcr_data: TCR列表
        param aa_dict: 氨基酸编码字典
        param peptide_encoding_dict: 预计算的peptide编码字典 {peptide序列: 编码tensor}
        param tcr_encoding_dict: 预计算的TCR编码字典 {TCR序列: 编码tensor}

    Returns:
        返回peptide embedding和query TCRs的embedding
    """
    spt_TCRs = tcr_data
    query_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    peptides = torch.FloatTensor(1, 75)
    qry_TCRs = tcr_data[2] if len(tcr_data) > 2 else ['None']
    # 从peptide字典获取或计算peptide编码
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
        qry_TCRs = list(dict.fromkeys(qry_TCRs))
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))
    
    # 处理query TCRs
    temp = torch.cat([
        torch.cat([
            peptide_embedding,
            tcr_encoding_dict[tcr] if tcr_encoding_dict and tcr in tcr_encoding_dict 
            else add_position_encoding(aamapping(tcr, 25, aa_dict))
        ]).unsqueeze(0)
        for tcr in spt_TCRs
    ])
    
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
    """
    返回所有在meta-training训练阶段用到的 peptide-TCR , 格式为：
            键为: peptide
            值为: k_shot 个 positive TCR, k_query 个 positive TCR;
                 k_shot 个 negative TCR, k_query 个 negative TCR
    Args:
        peptide:
        *args:
        **kwargs:

    Returns:

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
    '''
    合并字典并去重
    :param dicts:
    :return:
    '''
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
    """
    随机生成长度为n的tensor列表（内部是0到n-1数字的乱序），返回一个生成器
    Args:
        n:
        generator:

    Returns:

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
    """
    从长度为n的生成器中返回m项 (n未指定时，生成器不��空；生成器未指定时，n不能为空) (且m<=n)
    Args:
        m:
        n:
        generate_idx:

    Returns:

    """
    assert ((n is not None) and (m <= n)) or ((generate_idx is not None) and (m <= generate_idx.gi_frame.f_locals['n']))
    if generate_idx is None:
        generate_idx = generate_selected_idx(n)
    return_list = []
    for i in range(m):
        return_list.append(next(generate_idx).item())
    return return_list


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
# Model_config = [
# ('self_attention', [[2, 5, 5], [2, 5, 5], [2, 5, 5]]),
# ('linear', [32, 5]),
# ('relu', [True]),
# ('conv2d', [32, 1, 2, 1, 1, 0]),
# ('relu', [True]),
# ('bn', [32]),
# ('max_pool2d', [2, 2, 0]),
# ('flatten', []),
# ('linear', [2, 9728])]
# # Modified architecture
# 1:
# Model_config = [
# ('self_attention', [[2, 5, 5], [2, 5, 5], [2, 5, 5]]),
# ('linear', [32, 5]),
# ('relu', [True]),
# ('conv2d', [32, 1, 2, 1, 1, 0]),
# ('relu', [True]),
# ('bn', [32]),
# ('max_pool2d', [2, 2, 0]),
# ('flatten', []),
# ('linear', [2, 39936])]
# 2:
# Model_config = [
# ('self_attention', [[8, 5, 5], [8, 5, 5], [8, 5, 5]]),
# ('linear', [48, 5]),
# ('relu', [True]),
# ('conv2d', [48, 1, 2, 1, 1, 0]),
# ('relu', [True]),
# ('bn', [48]),
# ('max_pool2d', [2, 2, 0]),
# ('flatten', []),
# ('linear', [2, 89856])]
















# Model_config = [
#     ('self_attention', [[3, 5, 5], [3, 5, 5], [3, 5, 5]]),
#     ('linear', [48, 5]),
#     ('relu', [True]),
#     ('conv2d', [48, 1, 2, 1, 1, 0]),
#     ('relu', [True]),
#     ('bn', [48]),
#     ('max_pool2d', [1, 1, 0]),
#     ('flatten', []),
#     ('linear', [2, 89856])
# ]
# Model_config = [
#     ('self_attention', [[2, 5, 5], [2, 5, 5], [2, 5, 5]]),  # 6-10
#     ('linear', [5, 5]),
#     ('relu', [True]),
#     ('conv2d', [32, 1, 2, 1, 1, 0]),
#     ('relu', [True]),
#     ('bn', [32]),
#     # ('max_pool2d', [1, 1, 0]),
#     ('flatten', []),
#     ('linear', [2, 6240])
# ]
# Model_config = [
#     ('self_attention', [[2, 5, 5], [2, 5, 5], [2, 5, 5]]),
#     ('linear', [32, 5]),
#     ('relu', [True]),
#     ('conv2d', [32, 1, 2, 1, 1, 0]),
#     ('relu', [True]),
#     ('bn', [32]),
#     # ('max_pool2d', [1, 1, 0]),
#     ('flatten', []),
#     ('linear', [2, 39936])
# ]
# Model_config = [
#     ('self_attention', [[8, 5, 5], [8, 5, 5], [8, 5, 5]]),  # 6-10
#     ('linear', [5, 5]),
#     ('relu', [True]),
#     ('conv2d', [32, 1, 2, 1, 1, 0]),
#     ('relu', [True]),
#     ('bn', [32]),
#     # ('max_pool2d', [1, 1, 0]),
#     ('flatten', []),
#     ('linear', [2, 6240])
# ]
# load the cofig file
config_file_path = os.path.join(os.path.dirname(__file__), 'Configs', 'TrainingConfig.yaml')
Data_config = load_config(config_file_path)

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
