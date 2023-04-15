import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
from utils import get_peptide_tcr, Project_path, Data_config, Data_output, Train_Round, FilePath, load_csv_like_file, Train_output_dir, Test_output_dir, Zero_test_data


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


def get_metrics(result_path, label_path, result_PepColumnName='Peptide', result_CdrColumeName='CDR3', result_ResultColName='Score', label_PepColumnName='peptide', label_CdrColumeName='binding_TCR', label_ResultColName='label'):
    '''
    Get the roc-auc and pr-auc.
    Args:
        result_path:
        label_path:

    Returns:

    '''
    result = get_peptide_tcr_result(src_csv=result_path, PepColumnName=result_PepColumnName, CdrColumeName=result_CdrColumeName, ResultColName=result_ResultColName)
    y_label = get_peptide_tcr_result(src_csv=label_path, PepColumnName=label_PepColumnName, CdrColumeName=label_CdrColumeName, ResultColName=label_ResultColName)
    result_list = []
    y_label_list = []
    for k, v in result.items():
        for idx, cdr in enumerate(v[0]):
            # assert idx == y_label[k][0].index(cdr)
            result_list.append(result[k][1][idx])
            y_label_list.append(y_label[k][1][y_label[k][0].index(cdr)])
    roc_auc = roc_auc_score(y_label_list, result_list)
    precision, recall, _thresholds = precision_recall_curve(y_label_list, result_list)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc


if __name__ == '__main__':
    for index in range(Train_Round):
        F_roc_list, F_pr_list = [], []
        for kf_time in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):
            F_result_path = pd.read_csv(os.path.join(os.path.abspath(""),
                                                     'Round' + str(index + 1), 'kfold' + str(kf_time), Data_config['Train']['General']['Test_result_path'], 'KFold_' + str(kf_time) + '_test_general_result.csv'))
            F_y_label_path = pd.read_csv(os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test_general_data.csv'))
            F_roc, F_pr = get_metrics(F_result_path, F_y_label_path)
            F_roc_list.append(F_roc)
            F_pr_list.append(F_pr)
            print('Round:', index, 'K:', kf_time, '--General: ROC-AUC', F_roc, 'PR-AUC', F_pr)

        mean_F_roc = sum(F_roc_list) / len(F_roc_list)
        mean_F_pr = sum(F_pr_list) / len(F_pr_list)
        print('Round:', index, 'average', '\n--General: ROC-AUC', mean_F_roc, 'PR-AUC', mean_F_pr)
