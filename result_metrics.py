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


class Args:
    def __init__(self, f, zf=False, z=False):
        self.f = f
        self.zf = zf
        self.z = z


if __name__ == '__main__':
    """
    F_xxx: few-shot model used in few-shot data
    Z_F_xxx: zero-shot model used in few-shot data
    Z_xxx: zero-shot model used in zero-shot data
    """
    args = Args(f=True, zf=True, z=True)
    for round in range(1, Train_Round + 1):
        F_roc_list, F_pr_list, Z_F_roc_list, Z_F_pr_list, Z_roc_list, Z_pr_list = [], [], [], [], [], []
        for k in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):
            if args.f:
                # few-shot result metrics
                F_result_path = pd.read_csv(os.path.join(Project_path, Train_output_dir, 'Round' + str(round), 'kfold' + str(k),
                                                         Test_output_dir, 'Few-shot_Result_Round_' + str(round) + '_kfold_' + str(k) + '_test.csv'))
                F_y_label_path = pd.read_csv(os.path.join(Project_path, Data_output, 'kfold' + str(k), 'KFold_' + str(k) + '_test.csv'))
                F_roc, F_pr = get_metrics(F_result_path, F_y_label_path)
                F_roc_list.append(F_roc)
                F_pr_list.append(F_pr)
                print('Round:', round, 'K:', k, '--Few-shot: ROC-AUC', F_roc, 'PR-AUC', F_pr)
            if args.zf:
                # zero-shot model used in few-shot result metrics
                Z_F_result_path = pd.read_csv(os.path.join(Project_path, Train_output_dir, 'Round' + str(round), 'kfold' + str(k),
                                                           Test_output_dir, 'Zero-shot_Result_Round_' + str(round) + '_kfold_' + str(k) + '_test.csv'))
                F_y_label_path = pd.read_csv(os.path.join(Project_path, Data_output, 'kfold' + str(k), 'KFold_' + str(k) + '_test.csv'))
                Z_F_roc, Z_F_pr = get_metrics(Z_F_result_path, F_y_label_path)
                Z_F_roc_list.append(Z_F_roc)
                Z_F_pr_list.append(Z_F_pr)
                print('Round:', round, 'K:', k, '--Zero-shot model test few-shot data: ROC-AUC', Z_F_roc, 'PR-AUC', Z_F_pr)
            if args.z:
                # zero-shot result metrics
                Z_result_path = pd.read_csv(os.path.join(Project_path, Train_output_dir, 'Round' + str(round),
                                                         'kfold' + str(k), Test_output_dir, 'Zero-shot_dataset_Result_Round_' + str(round) + '_kfold_' + str(k) + '_test.csv'))
                Z_y_label_path = pd.read_csv(os.path.join(Project_path, Zero_test_data))
                Z_roc, Z_pr = get_metrics(Z_result_path, Z_y_label_path)
                Z_roc_list.append(Z_roc)
                Z_pr_list.append(Z_pr)
                print('Round:', round, 'K:', k, '--Zero-shot: ROC-AUC', Z_roc, 'PR-AUC', Z_pr)
        if args.f:
            mean_F_roc = sum(F_roc_list) / len(F_roc_list)
            mean_F_pr = sum(F_pr_list) / len(F_pr_list)
            print('Round:', round, 'average', '\n--Few-shot: ROC-AUC', mean_F_roc, 'PR-AUC', mean_F_pr)
        if args.zf:
            mean_Z_F_roc = sum(Z_F_roc_list) / len(Z_F_roc_list)
            mean_Z_F_pr = sum(Z_F_pr_list) / len(Z_F_pr_list)
            print('Round:', round, 'average', '\n--Zero-shot model test few-shot data: ROC-AUC', mean_Z_F_roc, 'PR-AUC', mean_Z_F_pr)
        if args.z:
            mean_Z_roc = sum(Z_roc_list) / len(Z_roc_list)
            mean_Z_pr = sum(Z_pr_list) / len(Z_pr_list)
            print('Round:', round, 'average', '\n--Zero-shot: ROC-AUC', mean_Z_roc, 'PR-AUC', mean_Z_pr)
