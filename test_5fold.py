import os
from collections import Counter
import pandas as pd
import numpy as np
import joblib
import argparse
import torch
from typing import Union, List
from pathlib import Path
from Requirements.Memory_meta_test import Memory_Meta
from Requirements.Memory_meta_test import Memory_module
from meta_distillation_training import load_config, Args

import warnings

warnings.filterwarnings('ignore')

FilePath = Union[str, "PathLike[str]", pd.DataFrame]
# Set the 'cuda' used for GPU testing
# device = torch.device('cuda')


def get_num(dict1):
    num = 0
    for k, v in dict1.items():
        num += len(v)
    return num


def aamapping(TCRSeq, encode_dim):
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


def task_embedding(pep, tcr_data):
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
    peptide_embedding = add_position_encoding(aamapping(pep, 15))

    # Put the embedding of support set, labels and peptide embedding into the initialized tensor
    temp = torch.Tensor()
    for j in spt_TCRs:
        temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
    support_x[0] = temp
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()

    # Put the embedding of query set into the initialized tensor
    temp = torch.Tensor()
    if len(tcr_data) > 2:
        for j in qry_TCRs:
            temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    return peptides, support_x, torch.LongTensor(support_y), query_x


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


def load_control_set(HealthyTCRFile):
    '''
    # load the control TCR set as negative.
    Args:
        HealthyTCRFile:

    Returns:

    '''
    HealthyTCR = np.loadtxt(HealthyTCRFile, dtype=str)
    return HealthyTCR


def change_dict2test_struct(ori_dict, HealthyTCRFile, k_shot=2, ratio=1):
    '''
    Convert the input Peptide_TCR dictionary into k_shot positive and k_shot negative samples, the rest is query
     (the key is Peptide, the value is [[positive TCR, negative TCR], [positive label, negative label], [query], []]).
    Args:
        ori_dict:
        HealthyTCRFile:
        k_shot:
        ratio: The ratio of positive and negative samples of the query set.

    Returns:

    '''
    HealthyTCR = load_control_set(HealthyTCRFile=HealthyTCRFile)
    print('Support size:', 2 * 2 * len(ori_dict.keys()), 'Query size:', int((ratio + 1) * (get_num(ori_dict) - 2 * len(ori_dict.keys()))))
    F_data = {}
    for i, j in ori_dict.items():
        if i not in F_data:
            F_data[i] = [[], [], [], []]
        selected_spt_idx = np.random.choice(len(j), k_shot, replace=False)
        for idx in selected_spt_idx:  # positive support
            F_data[i][0].append(j[idx])
            F_data[i][1].append(1)
        selected_heal_idx = np.random.choice(len(HealthyTCR), k_shot, replace=False)  # negative support
        selected_heal_TCRs = HealthyTCR[selected_heal_idx]
        for heal_TCR in selected_heal_TCRs:
            F_data[i][0].append(heal_TCR)
            F_data[i][1].append(0)
        for query_idx in range(len(j)):  # positive query
            if (query_idx not in selected_spt_idx) and (j[query_idx] not in F_data[i][0]):
                F_data[i][2].append(j[query_idx])
        selected_query_idx = np.random.choice(len(HealthyTCR), int((len(j) - k_shot) * ratio), replace=False)
        selected_query_TCRs = HealthyTCR[selected_query_idx]
        for query_TCR in selected_query_TCRs:
            F_data[i][2].append(query_TCR)
    return F_data


def test_5fold_few_shot(model, test_data, output_file):
    '''
    Few-shot test. Store the results in a csv.
    '''
    ends = []
    for i in test_data:
        # Convert the input into the embeddings
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i, test_data[i])
        # Support set is used for fine-tune the model and the query set is used to test the performance
        end = model.finetunning(peptide_embedding[0].to(device), x_spt[0].to(device), y_spt[0].to(device), x_qry[0].to(device))
        ends += list(end[0])
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in test_data:
        output_peps += [i] * len(test_data[i][2])
        output_tcrs += test_data[i][2]
    output = pd.DataFrame({'Peptide': output_peps, 'CDR3': output_tcrs, 'Score': ends})
    output.to_csv(output_file, index=False)


def test_5fold_zero_shot(model, test_data, output_file):
    '''
    Zero-shot test. Store the results in a csv.
    '''

    def task_embedding(pep, tcr_data):
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
        # spt_TCRs = tcr_data[2]
        spt_TCRs = tcr_data
        # Initialize the size of the Tensor for the query set and peptide encoding
        query_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
        peptides = torch.FloatTensor(1, 75)
        # Encoding for the peptide sequence
        peptide_embedding = add_position_encoding(aamapping(pep, 15))
        # Put the embedding of query TCRs and peptide into the initialized tensor
        temp = torch.Tensor()
        for j in spt_TCRs:
            temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
        query_x[0] = temp
        peptides[0] = peptide_embedding.flatten()
        return peptides, query_x

    # The variable "starts" is a list used for storing the predicted score for the unseen peptide-TCR pairs
    starts = []
    for i in test_data:
        # Convert the input into the embeddings
        all_test_data = test_data[i][0] + test_data[i][2]
        peptide_embedding, x_spt = task_embedding(i, all_test_data)
        # Memory block is used for predicting the binding scores of the unseen peptide-TCR pairs
        start = model.meta_forward_score(peptide_embedding.to(device), x_spt.to(device))
        starts += list(torch.Tensor.cpu(start[0]).numpy())
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in test_data:
        all_test_data = test_data[i][0] + test_data[i][2]
        output_peps += [i] * len(all_test_data)
        output_tcrs += all_test_data
    # Store the predicted result and output the result as .csv file
    output = pd.DataFrame({'Peptide': output_peps, 'CDR3': output_tcrs, 'Score': starts})
    output.to_csv(output_file, index=False)


def get_model(args, mdoel_config, data_config, model_path, device=device):
    '''
    get model
    '''
    # Initialize a new model
    model = Memory_Meta(args, mdoel_config).to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    # Load the memory block
    if data_config['Train']['Meta_learning']['Model_parameter']['device'] == 'cuda':
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


if __name__ == '__main__':
    args = Args(C=3, L=75, R=3, update_lr=0.01, update_step_test=3)
    # This is the model parameters
    mdoel_config = [
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
    data_config = load_config(config_file_path)
    device = torch.device(data_config['Train']['Meta_learning']['Model_parameter']['device'])
    # Load the Atchley_factors for encoding the amino acid
    aa_dict = joblib.load(os.path.join(eval(data_config['Project_path']), eval(data_config['dataset']['aa_dict'])))

    project_path = eval(data_config['Project_path'])
    Round = data_config['dataset']['Train_Round']
    k_fold = data_config['dataset']['k_fold']
    data_output = data_config['dataset']['data_output']
    for r_idx in range(1, (Round + 1)):
        print('Testing Round:', r_idx)
        round_dir = os.path.join(project_path, 'Round' + str(r_idx))
        for kfold_dir in os.listdir(round_dir):
            kfold_idx = str(kfold_dir).split('kfold')[-1]
            print('--kfload:', kfold_idx)
            test_data = os.path.join(project_path, data_output, 'kfold' + kfold_idx, 'KFold_' + kfold_idx + '_test.csv')
            F_data = change_dict2test_struct(get_peptide_tcr(test_data, 'peptide', 'binding_TCR'), HealthyTCRFile=os.path.join(project_path, data_config['dataset']['Negative_dataset']), ratio=1)
            print('Support size:', sum([len(j) for j in (F_data[k][0] for k in list(F_data.keys()))]), 'Query size:', sum([len(j) for j in (F_data[k][2] for k in list(F_data.keys()))]))
            model = get_model(args, mdoel_config, data_config, model_path=os.path.join(round_dir, kfold_dir))
            test_5fold_few_shot(model, F_data, output_file=os.path.join(round_dir, kfold_dir, 'Few-shot_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_test.csv'))
            test_5fold_zero_shot(model, F_data, output_file=os.path.join(round_dir, kfold_dir, 'Zero-shot_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_test.csv'))
