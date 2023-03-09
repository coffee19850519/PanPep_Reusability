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
from meta_distillation_training import load_config
from pep_distance import seqAligneEluli, seqAligneCosine
from test_5fold import Args, aamapping, task_embedding, load_csv_like_file, get_peptide_tcr, load_control_set, change_dict2test_struct, add_position_encoding, get_model, test_5fold_few_shot
import warnings

warnings.filterwarnings('ignore')


def get_test_peptide_similarity(train_peptide, test_peptide, threshold=1.4):
    """
    选择test中每个peptide和train中的所有peptide相似度 小于threshold 的peptide
    Select the peptide whose similarity between each peptide in test and all peptide in train is less than threshold
    Args:
        train_peptide:
        test_peptide:
        threshold:

    Returns: 以字典的形式返回小于阈值的 peptide和所有对应的TCR
             Returns the pentide and all corresponding TCRs that are less than the threshold value in the form of a dictionary

    """
    max_dist = []
    for test_p, test_t in test_peptide.items():
        pep_dis = []
        for train_p, train_t in train_peptide.items():
            pep_dis.append(seqAligneCosine(test_p, train_p))  # Which is in front and which is behind
        max_dist.append(max(pep_dis))
    choosed_p = {}
    for idx, num in enumerate(max_dist):
        if num < threshold:
            choosed_p[list(test_peptide.keys())[idx]] = test_peptide[list(test_peptide.keys())[idx]]
    return choosed_p


if __name__ == '__main__':
    args = Args(C=3, L=75, R=3, update_lr=0.01, update_step_test=3)
    # This is the model parameters
    config = [
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
    # get device
    device = torch.device(data_config['Train']['Meta_learning']['Model_parameter']['device'])
    # Load the Atchley_factors for encoding the amino acid
    aa_dict = joblib.load(os.path.join(eval(data_config['Project_path']), eval(data_config['dataset']['aa_dict'])))
    project_path = eval(data_config['Project_path'])  # project path
    Round = data_config['dataset']['Train_Round']
    k_fold = data_config['dataset']['k_fold']
    data_output = data_config['dataset']['data_output']  # output path
    for r_idx in range(1, (Round + 1)):
        print('Testing Round:', r_idx)
        round_dir = os.path.join(project_path, 'Round' + str(r_idx))
        for kfold_dir in os.listdir(round_dir):
            kfold_idx = str(kfold_dir).split('kfold')[-1]
            print('--kfload:', kfold_idx)
            # get test data
            test_data_path = os.path.join(project_path, data_output, 'kfold' + kfold_idx, 'KFold_' + kfold_idx + '_test.csv')
            Test_data = get_peptide_tcr(test_data_path, 'peptide', 'binding_TCR')
            # get train data
            train_data_path = os.path.join(project_path, data_output, 'kfold' + kfold_idx, 'KFold_' + kfold_idx + '_train.csv')
            Train_data = get_peptide_tcr(train_data_path, 'peptide', 'binding_TCR')
            threshold = 1.4
            # filter peptide
            choose_p = get_test_peptide_similarity(Train_data, Test_data, threshold=threshold)
            F_data = change_dict2test_struct(choose_p, HealthyTCRFile=os.path.join(project_path, data_config['dataset']['Negative_dataset']), ratio=1)
            print('Support size:', sum([len(j) for j in (F_data[k][0] for k in list(F_data.keys()))]), 'Query size:', sum([len(j) for j in (F_data[k][2] for k in list(F_data.keys()))]))
            model = get_model(args, config, data_config, model_path=os.path.join(round_dir, kfold_dir), device=device)
            test_5fold_few_shot(model, F_data, output_file=os.path.join(round_dir, kfold_dir, 'Few-shot_Threshold' + str(threshold) + '_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_test.csv'), aa_dict=aa_dict, device=device)


