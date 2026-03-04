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
    Add positive and negative samples into a new CSV
    (negative sample count matches positive sample count).
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


def get_model(args, model_config, model_path, device):
    '''
    get model
    '''
    # Initialize a new model
    model = Memory_Meta(args, model_config).to(device)
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
        
#         # Expand peptide_embedding to match batch size
#         peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
        
#         # Concatenate all data at once
#         temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
#         query_x[0] = temp
#     else:
#         query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
#     return peptides, support_x, torch.LongTensor(support_y), query_x

def task_embedding(pep, tcr_data, aa_dict, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    Get task-level embeddings from precomputed dictionaries.
    If a dictionary entry is missing, compute the embedding on the fly.

    Parameters:
        param pep: peptide sequence
        param tcr_data: TCR list and corresponding labels
        param aa_dict: amino-acid encoding dictionary
        param peptide_encoding_dict: precomputed peptide encoding dictionary {peptide sequence: encoded tensor}
        param tcr_encoding_dict: precomputed TCR encoding dictionary {TCR sequence: encoded tensor}

    Returns:
        Returns peptide embedding, support set embedding, support set labels, and query set embedding.
        If support set is empty, support_x and support_y are returned as None.
    """
    spt_TCRs = tcr_data[0]
    ypt = tcr_data[1]
    

    peptides = torch.FloatTensor(1, 75)
    
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    # Get peptide encoding from the dictionary or compute it.
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))

    # Check whether the support set is empty.
    if not spt_TCRs or not ypt:
        peptides[0] = peptide_embedding.flatten()
        
        # Process query set.
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

    # Continue only if the support set is not empty.
    support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    support_y = np.zeros((1, len(ypt)), dtype=np.int64)
    
    # Process support set.
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

    # Process query set.
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

# ## Optimization, difference
# def get_query_data(all_ranking_data, k_shot_data, k_shot):

#     F_data = [[], []]
#     F_data[0].extend([j for j in all_ranking_data[0] if j not in k_shot_data[0]])
#     index_p = [k for k, v in enumerate(all_ranking_data[1]) if v == 1 or v == "1"]
#     index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
#     F_data[1].extend([1] * (len(index_p) - k_shot))
#     F_data[1].extend([0] * (len(index_n) - k_shot))

#     return F_data
def get_query_data(all_ranking_data, k_shot_data, k_shot):
    mask = ~np.isin(all_ranking_data[0], k_shot_data[0])
    
    # Select elements directly with a mask.
    F_data = [
        np.array(all_ranking_data[0])[mask].tolist(),
        np.array(all_ranking_data[1])[mask].tolist()
    ]
    
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
    data = [[], []]

    file_name = f"k_shot_{pep}.csv"
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can't find file: {file_path}")

    df = pd.read_csv(file_path)

    positive_samples = df[df['label'] == 1]
    data[0].extend(positive_samples['tcr'].tolist())
    data[1].extend([1] * len(positive_samples))

    negative_samples = df[df['label'] == 0]
    data[0].extend(negative_samples['tcr'].tolist())
    data[1].extend([0] * len(negative_samples))

    additional_negative_needed = update_step_test * k_shot - len(negative_samples)

    if additional_negative_needed > 0:
        index_n = [k for k, v in enumerate(all_ranking_data[1]) if v == 0 or v == "0"]
        negative_support_idx = random.sample(index_n, additional_negative_needed)

        for idx in negative_support_idx:
            data[0].append(all_ranking_data[0][idx])
            data[1].append(0)

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
    
    # Add all selected negative samples.
    for idx in negative_support_idx:
        data[0].append(all_ranking_data[0][idx])
        data[1].append(0)

    output = pd.DataFrame({'tcr': data[0], 'label': data[1]})
    output.to_csv(os.path.join(result, "k_shot_" + pep + ".csv"), index=False)
    
    return data

def load_support_data(pep, data_dir):

    data = [[], []]

    file_name = f"k_shot_{pep}.csv"
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can't find file: {file_path}")

    df = pd.read_csv(file_path)

    data[0] = df['tcr'].tolist()
    data[1] = df['label'].tolist()
    
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
def sample_multi_round_support_data(positive_tcrs, neg_background, neg_reshuffling,
                                     k_shot, update_step_test, pep, result_dir):
    """
    Sample independent support sets for each inner-loop finetuning step.
    Odd steps (0, 2, 4, ...) draw negatives from background library;
    even steps (1, 3, 5, ...) draw negatives from reshuffling library.

    Args:
        positive_tcrs: list of positive TCR sequences for this peptide
        neg_background: list of background negative TCR sequences (pre-filtered, no positives)
        neg_reshuffling: list of reshuffling negative TCR sequences (pre-filtered, no positives)
        k_shot: number of positives (and negatives) per round
        update_step_test: total number of inner-loop steps
        pep: peptide string (for file naming)
        result_dir: directory to save the CSV log

    Returns:
        (rounds_list, all_support_tcrs_set)
        rounds_list: list of [tcr_list, label_list] per round
        all_support_tcrs_set: set of all TCR sequences used across all rounds
    """
    positive_set = set(positive_tcrs)

    # Pre-filter negative libraries to exclude positives
    bg_filtered = [t for t in neg_background if t not in positive_set]
    rs_filtered = [t for t in neg_reshuffling if t not in positive_set]

    rounds_list = []
    all_support_tcrs = set()
    csv_rows = []

    # Positives are sampled once and shared across all rounds
    pos_sampled = random.sample(positive_tcrs, k_shot)

    for r in range(update_step_test):
        # Alternate negative source: even index → background, odd index → reshuffling
        if r % 2 == 0:
            neg_pool = bg_filtered
            neg_source = 'background'
        else:
            neg_pool = rs_filtered
            neg_source = 'reshuffling'

        neg_sampled = random.sample(neg_pool, k_shot)

        tcr_list = pos_sampled + neg_sampled
        label_list = [1] * k_shot + [0] * k_shot

        rounds_list.append([tcr_list, label_list])
        all_support_tcrs.update(tcr_list)

        for tcr, label in zip(tcr_list, label_list):
            csv_rows.append({'round': r, 'tcr': tcr, 'label': label, 'source': neg_source if label == 0 else 'positive'})

    # Save for reproducibility
    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(result_dir, f"k_shot_multi_round_{pep}.csv"), index=False)

    return rounds_list, all_support_tcrs


def get_query_data_multi_round(all_ranking_data, all_support_tcrs):
    """
    Build query data by excluding all support TCRs used across all rounds.

    Args:
        all_ranking_data: [tcr_list, label_list] for this peptide
        all_support_tcrs: set of TCR sequences to exclude

    Returns:
        [filtered_tcr_list, filtered_label_list]
    """
    mask = ~np.isin(all_ranking_data[0], list(all_support_tcrs))
    F_data = [
        np.array(all_ranking_data[0])[mask].tolist(),
        np.array(all_ranking_data[1])[mask].tolist()
    ]
    return F_data
def save_support_data(all_ranking_data, k_shot, pep, result, chain_type=None):
    """
    Save k-shot data. If chain_type is provided, include it in the filename.
    
    Args:
        all_ranking_data: data containing TCR sequences and labels
        k_shot: number of samples per class
        pep: peptide name
        result: output path
        chain_type: optional chain type (alpha or beta)
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
    Get zero-shot task-level embeddings from precomputed dictionaries.
    If a dictionary entry is missing, compute the embedding on the fly.

    Parameters:
        param pep: peptide sequence
        param tcr_data: TCR list
        param aa_dict: amino-acid encoding dictionary
        param peptide_encoding_dict: precomputed peptide encoding dictionary {peptide sequence: encoded tensor}
        param tcr_encoding_dict: precomputed TCR encoding dictionary {TCR sequence: encoded tensor}

    Returns:
        Returns peptide embedding and query TCR embeddings.
    """
    spt_TCRs = tcr_data
    query_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    peptides = torch.FloatTensor(1, 75)
    qry_TCRs = tcr_data[2] if len(tcr_data) > 2 else ['None']
    # Get peptide encoding from the dictionary or compute it.
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
        qry_TCRs = list(dict.fromkeys(qry_TCRs))
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))
    
    # Process query TCRs.
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
    Return all peptide-TCR pairs used in the meta-training phase, in the format:
            key: peptide
            value: k_shot positive TCRs, k_query positive TCRs;
                   k_shot negative TCRs, k_query negative TCRs
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
    Merge dictionaries and remove duplicates.
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
    Randomly generate a length-n tensor permutation (values 0 to n-1) and return it as a generator.
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


def load_multi_round_support_data(pep, support_dir):
    """
    Load pre-saved multi-round support data from a CSV file.

    Args:
        pep: peptide string
        support_dir: directory containing k_shot_multi_round_{pep}.csv files

    Returns:
        (rounds_list, all_support_tcrs)
        rounds_list: list of [tcr_list, label_list] per round
        all_support_tcrs: set of all TCR sequences used across all rounds
    """
    file_path = os.path.join(support_dir, f"k_shot_multi_round_{pep}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Can't find multi-round support file: {file_path}")

    df = pd.read_csv(file_path)
    rounds_list = []
    all_support_tcrs = set()

    for r in sorted(df['round'].unique()):
        round_df = df[df['round'] == r]
        tcr_list = round_df['tcr'].tolist()
        label_list = round_df['label'].tolist()
        rounds_list.append([tcr_list, label_list])
        all_support_tcrs.update(tcr_list)

    return rounds_list, all_support_tcrs

    
def return_m_from_n(m, n=None, generate_idx=None):
    """
    Return m items from a length-n generator.
    If n is not provided, generate_idx must be provided.
    If generate_idx is not provided, n must be provided.
    Also requires m <= n.
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
Model_config_attention8= [
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

Model_config_attention16= [
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

Model_config_attention5_conv3_large = [
    ('res_attention_block', [[8, 16, 5], [8, 16,16], [8, 16, 16],[8, 16,16],[8, 16, 16]]), 
    ('linear', [16, 16]),
    ('relu', [True]),
    ('res_block', [[16, 1, 3, 1, 1], [16, 16, 3, 1, 1], [16, 16, 3, 1, 1]]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 2560])
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
