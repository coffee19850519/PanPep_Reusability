import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from test_5fold import get_peptide_tcr, load_csv_like_file, FilePath, get_num
import os
from meta_distillation_training import load_config
from collections import Counter
from typing import Union, List
import math


def get_peptide_tcr_result(src_csv: FilePath, PepColumnName: str = 'Peptide', CdrColumeName: str = 'Alpha CDR3',
                           ResultColName: str = 'Score', csv_encoding: str = 'utf-8', sep=','):
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
            PepTCRdict[pep] = [[], []]
        if (type(src_csv[CdrColumeName][idx]) == str) and (src_csv[CdrColumeName][idx] not in PepTCRdict[pep]):
            PepTCRdict[pep][0].append(src_csv[CdrColumeName][idx])
            PepTCRdict[pep][1].append(src_csv[ResultColName][idx])
    return PepTCRdict


def get_metrics(result_path, label_path):
    '''
    Get the roc-auc and pr-auc.
    Args:
        result_path:
        label_path:

    Returns:

    '''
    result = get_peptide_tcr_result(src_csv=result_path, PepColumnName='Peptide', CdrColumeName='CDR3', ResultColName='Score')
    y_label = get_peptide_tcr(src_csv=label_path, PepColumnName='peptide', CdrColumeName='binding_TCR')
    result_list = []
    y_label_list = []
    for k, v in result.items():
        for idx, cdr in enumerate(v[0]):
            if cdr in y_label[k]:
                result_list.append(result[k][1][idx])
                y_label_list.append(1)
            else:
                result_list.append(result[k][1][idx])
                y_label_list.append(0)
    roc_auc = roc_auc_score(y_label_list, result_list)
    precision, recall, _thresholds = precision_recall_curve(y_label_list, result_list)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc


if __name__ == '__main__':
    project_path = r'G:\OneDrive - University of Missouri\PanPep_reusability\5fold_train-test' #TODO
    data_config = load_config(os.path.join(project_path, 'Configs', 'TrainingConfig.yaml'))
    Round = data_config['dataset']['Train_Round']
    kfold = data_config['dataset']['k_fold']
    data_output = data_config['dataset']['data_output']
    for round in range(1, Round + 1):
        F_roc_list, F_pr_list, Z_roc_list, Z_pr_list = [], [], [], []
        for k in range(2, kfold + 1):
            F_result_path = pd.read_csv(project_path + '\Round' + str(round) + '\kfold' + str(k) + '\Few-shot_Result_Round_' + str(round) + '_kfold_' + str(k) + '_test.csv') #TODO
            Z_result_path = pd.read_csv(project_path + '\Round' + str(round) + '\kfold' + str(k) + '\Zero-shot_Result_Round_' + str(round) + '_kfold_' + str(k) + '_test.csv') #TODO
            y_label_path = pd.read_csv(os.path.join(project_path, data_output, 'kfold' + str(k), 'KFold_' + str(k) + '_test.csv'))
            F_roc, F_pr = get_metrics(F_result_path, y_label_path)
            F_roc_list.append(F_roc)
            F_pr_list.append(F_pr)
            Z_roc, Z_pr = get_metrics(Z_result_path, y_label_path)
            Z_roc_list.append(Z_roc)
            Z_pr_list.append(Z_pr)
            print(F_roc, F_pr, Z_roc, Z_pr)
        mean_F_roc = sum(F_roc_list) / len(F_roc_list)
        mean_F_pr = sum(F_pr_list) / len(F_pr_list)
        mean_Z_roc = sum(Z_roc_list) / len(Z_roc_list)
        mean_Z_pr = sum(Z_pr_list) / len(Z_pr_list)
        print(mean_F_roc, mean_F_pr, mean_Z_roc, mean_Z_pr)
