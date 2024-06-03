"""
获取每个round中每个fold中的test和train的peptide-TCRs的相似度
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr
from utils import load_csv_like_file, data2save_path, Test_output_dir, Support, Query, merge_dict, get_peptide_tcr, \
    MLogger, Args, _split_parameters, Project_path, Data_output, K_fold, Train_dataset, Train_output_dir, Train_Round, \
    Device, Aa_dict, Negative_dataset, Batch_size, Shuffle, Data_config, Model_config

from result_metrics import get_metrics

total_width, n = 1, 2  # 柱状图总宽度，有几组数据
width = total_width / n  # 单个柱状图的宽度
length = 10
# all_x = []
# for i in range(3):
#     all_x.append(np.array([idx * width * 3 + i * width + 1 for idx in range(length)]))
pep_similarity_ = np.load('task_similarity.npy', allow_pickle=True)
# 去掉主对角线上的元素（因为是自身对自身的相似度值为1）
pep_similarity = pep_similarity_[~np.eye(pep_similarity_.shape[0], dtype=bool)].reshape(pep_similarity_.shape[0],
                                                                                        -1)
# 所有peptide的名称
pep_index = np.loadtxt('tasks.txt', dtype=str)
# 所有正样本TCR之间的相似度
pst_TCRs_ = np.load('TCRs.npy', allow_pickle=True)
# 去掉主对角线上的元素（因为是自身对自身的相似度值为1）
pst_TCRs = pst_TCRs_[~np.eye(pst_TCRs_.shape[0], dtype=bool)].reshape(pst_TCRs_.shape[0], -1)
# 所有正样本TCR的名称
pst_TCRs_index = np.loadtxt('TCRs.txt', dtype=str)


def return_not_num(array, num=1):
    """
    将array去掉值为num的数并返回
    Args:
        array:
        num:

    Returns:

    """
    # if array.shape:
    #     return np.delete(array, np.where(array == num))
    # else:
    return np.delete(array, np.where(array == num))


def get_peptide_simi(Round, Kfold):
    pep_similarity_ = np.load('task_similarity.npy', allow_pickle=True)
    # 去掉主对角线上的元素（因为是自身对自身的相似度值为1）
    pep_similarity = pep_similarity_[~np.eye(pep_similarity_.shape[0], dtype=bool)].reshape(pep_similarity_.shape[0],
                                                                                            -1)
    # print(pep_similarity.shape)
    pep_index = np.loadtxt('tasks.txt', dtype=str)
    simi = []
    test_d = load_csv_like_file(os.path.join(Project_path, Train_output_dir, 'Round' + str(Round), 'kfold' + str(Kfold),
                                             'KFold_' + str(Kfold) + '_test_all_test_data.csv'))
    # train_d = load_csv_like_file(os.path.join(Project_path, Train_output_dir, 'Round' + str(Round), 'kfold' + str(Kfold), 'KFold_' + str(Kfold) + '_train_all_train_data.csv'))
    test_pep = test_d.keys()
    # train_pep = train_d.keys()
    for pep in test_pep:
        idx = list(pep_index).index(pep)
        simi.append(max(pep_similarity[idx]))
    return simi
    # test_l = []
    # train_l = []
    # for k in range(1, 6):
    #     for r in range(1, 6):
    #         test_p = load_csv_like_file(os.path.join(Project_path, Train_output_dir, 'Round' + str(r), 'kfold' + str(k), 'KFold_' + str(k) + '_test_all_test_data.csv'))
    #         train_p = load_csv_like_file(os.path.join(Project_path, Train_output_dir, 'Round' + str(r), 'kfold' + str(k), 'KFold_' + str(k) + '_train_all_train_data.csv'))
    #         test_l.append(sorted(test_p))
    #         train_l.append(sorted(train_p))
    # if test_l:
    #     print(1)


def get_pep_tcr_simi(Round, Kfold=5):
    global data2save_path, Train_output_dir
    # 指定round、fold下test TCR与对应train TCR的相似度
    ngt_simi = np.load(os.path.join(Project_path, 'Similarity', 'Round' + str(Round), 'kfold' + str(Kfold),
                                    'Round' + str(Round) + '_kfold' + str(Kfold) + '_simimlarity.npy'),
                       allow_pickle=True)
    # 指定round、fold下test TCR的名称
    if Round >= 6:
        data2save_path = 'save_train_data_'
        Train_output_dir = 'Result_save_train_data_'
        Round -= 5
    ngt_simi_index = np.loadtxt(os.path.join(Project_path, data2save_path, 'Round' + str(Round), 'kfold' + str(Kfold),
                                             'Round' + str(Round) + '_kfold' + str(Kfold) + '_test.txt'), dtype=str)
    # test TCR的结果
    test_d = load_csv_like_file(
        os.path.join(Project_path, Train_output_dir, 'Round' + str(Round), 'kfold' + str(Kfold), Test_output_dir,
                     'Few-shot_Result_Round_' + str(Round) + '_kfold_' + str(
                         Kfold) + '_kshot2_kquery3_ulimit_None_test.csv'))
    # test_d_label_path = pd.read_csv(
    #     os.path.join(Project_path, Data_output, 'kfold' + str(Kfold), 'KFold_' + str(Kfold) + '_test.csv'))
    # F_roc, F_pr = get_metrics(test_d, test_d_label_path)
    simi = {}  # {task_name: [[pep_simi, TCR_simi], [ROC, PR]]}
    for pep, (tcr, score) in get_peptide_tcr(test_d, 'Peptide', 'CDR3', 'Score').items():
        # pep: task name, tcr: 该任务下的TCRs (list[positive, negative])
        if pep not in simi:
            simi[pep] = [[], []]
        idx = list(pep_index).index(pep)
        pep_simi = max(pep_similarity[idx])

        pst_index = [i for i, x in enumerate(pst_TCRs_index) for t in tcr if x == t]
        ngt_index = [i for i, x in enumerate(ngt_simi_index) for t in tcr if x == t]
        # simi[pep][0].extend([max(j) for j in [pst_TCRs[i] for i in pst_index]])
        # simi[pep][0].extend([max(j) for j in [ngt_simi[i] for i in ngt_index]])
        # simi[pep][1].extend(score)
        simi[pep][0].append((pep_simi,
                             np.mean([max(j) for j in [pst_TCRs[i] for i in pst_index]] + [max(j) for j in
                                                                                           [ngt_simi[i] for i in
                                                                                            ngt_index]])))
        task_label = [1, 1, 1, 0, 0, 0]
        roc_auc = roc_auc_score(task_label, score)
        precision, recall, _thresholds = precision_recall_curve(task_label, score)
        pr_auc = auc(recall, precision)
        simi[pep][1].append((roc_auc, pr_auc))
        # num = 0
        # for t in tcr:
        #     if pep not in simi:
        #         simi[pep] = []
        #     if num >= Query:
        #         simi[pep].append(max(return_not_num(ngt_simi[list(ngt_simi_index).index(t)])))
        #         num += 1
        #     else:
        #         simi[pep].append(max(return_not_num(pst_TCRs[list(pst_TCRs_index).index(t)])))
        #         num += 1
    return simi


def get_pep_tcr_simi_zero_shot(Round, Kfold=5):
    global data2save_path, Train_output_dir
    # 指定round、fold下test TCR与对应train TCR的相似度
    ngt_simi = np.load(os.path.join(Project_path, 'Similarity', 'Round' + str(Round), 'kfold' + str(Kfold),
                                    'Round' + str(Round) + '_kfold' + str(Kfold) + '_simimlarity.npy'),
                       allow_pickle=True)
    # 指定round、fold下test TCR的名称
    if Round >= 6:
        data2save_path = 'save_train_data_'
        Train_output_dir = 'Result_save_train_data_'
        Round -= 5
    ngt_simi_index = np.loadtxt(os.path.join(Project_path, data2save_path, 'Round' + str(Round), 'kfold' + str(Kfold),
                                             'Round' + str(Round) + '_kfold' + str(Kfold) + '_test.txt'), dtype=str)
    # test TCR的结果
    test_d = load_csv_like_file(
        os.path.join(Project_path, Train_output_dir, 'Round' + str(Round), 'kfold' + str(Kfold), Test_output_dir,
                     'Zero-shot_dataset_Result_Round_' + str(Round) + '_kfold_' + str(
                         Kfold) + '_test.csv'))
    # test_d_label_path = pd.read_csv(
    #     os.path.join(Project_path, Data_output, 'kfold' + str(Kfold), 'KFold_' + str(Kfold) + '_test.csv'))
    # F_roc, F_pr = get_metrics(test_d, test_d_label_path)
    simi = {}  # {task_name: [[pep_simi, TCR_simi], [ROC, PR]]}
    for pep, (tcr, score) in get_peptide_tcr(test_d, 'Peptide', 'CDR3', 'Score').items():
        # pep: task name, tcr: 该任务下的TCRs (list[positive, negative])
        if pep not in simi:
            simi[pep] = [[], []]
        idx = list(pep_index).index(pep)
        pep_simi = max(pep_similarity[idx])

        pst_index = [i for i, x in enumerate(pst_TCRs_index) for t in tcr if x == t]
        ngt_index = [i for i, x in enumerate(ngt_simi_index) for t in tcr if x == t]
        # simi[pep][0].extend([max(j) for j in [pst_TCRs[i] for i in pst_index]])
        # simi[pep][0].extend([max(j) for j in [ngt_simi[i] for i in ngt_index]])
        # simi[pep][1].extend(score)
        simi[pep][0].append((pep_simi,
                             np.mean([max(j) for j in [pst_TCRs[i] for i in pst_index]] + [max(j) for j in
                                                                                           [ngt_simi[i] for i in
                                                                                            ngt_index]])))
        task_label = [1, 1, 1, 0, 0, 0]
        roc_auc = roc_auc_score(task_label, score)
        precision, recall, _thresholds = precision_recall_curve(task_label, score)
        pr_auc = auc(recall, precision)
        simi[pep][1].append((roc_auc, pr_auc))
        # num = 0
        # for t in tcr:
        #     if pep not in simi:
        #         simi[pep] = []
        #     if num >= Query:
        #         simi[pep].append(max(return_not_num(ngt_simi[list(ngt_simi_index).index(t)])))
        #         num += 1
        #     else:
        #         simi[pep].append(max(return_not_num(pst_TCRs[list(pst_TCRs_index).index(t)])))
        #         num += 1
    return simi


def get_5fold_simi_pep_tcr_score(Round, Kfold):
    if type(Round) is tuple:
        start_R = Round[0]
        end_R = Round[1]
    else:
        start_R = 1
        end_R = Round + 1
    all_simi_pep_tcr_score = {}
    for r in range(start_R, end_R):
        pep_tcr_simi = []
        score = []
        for k in range(1, Kfold + 1):
            simi = get_pep_tcr_simi(r, k)
            pep_tcr_simi.append((np.mean([simi[pep][0][0][0] for pep in simi.keys()]),
                                 np.mean([simi[pep][0][0][1] for pep in simi.keys()])))
            score.append((np.mean([simi[pep][1][0][0] for pep in simi.keys()]),
                          np.mean([simi[pep][1][0][1] for pep in simi.keys()])))
        if str(r) not in all_simi_pep_tcr_score:
            all_simi_pep_tcr_score[str(r)] = [[], []]
        all_simi_pep_tcr_score[str(r)][0].extend(pep_tcr_simi)
        all_simi_pep_tcr_score[str(r)][1].extend(score)
    return all_simi_pep_tcr_score


def get_pep_tcr_simi_roc_pr(pep_tcr_simi_score):
    pep_simi = [pep_tcr_simi_score[k][0][i][0] for k in pep_tcr_simi_score.keys() for i in range(5)]
    tcr_simi = [pep_tcr_simi_score[k][0][i][1] for k in pep_tcr_simi_score.keys() for i in range(5)]
    roc = [pep_tcr_simi_score[k][1][i][0] for k in pep_tcr_simi_score.keys() for i in range(5)]
    pr = [pep_tcr_simi_score[k][1][i][1] for k in pep_tcr_simi_score.keys() for i in range(5)]
    return pep_simi, tcr_simi, roc, pr


def draw_5fold_simi_score(pep_tcr_simi_score, save=False):
    # https://blog.csdn.net/Leige_Smart/article/details/79583470
    label = ['Peptide', 'TCR', 'ROC', 'PR']
    pep_simi, tcr_simi, roc, pr = get_pep_tcr_simi_roc_pr(pep_tcr_simi_score)
    # pep_simi = [pep_tcr_simi_score[k][0][i][0] for k in pep_tcr_simi_score.keys() for i in range(5)]
    # tcr_simi = [pep_tcr_simi_score[k][0][i][1] for k in pep_tcr_simi_score.keys() for i in range(5)]
    # roc = [pep_tcr_simi_score[k][1][i][0] for k in pep_tcr_simi_score.keys() for i in range(5)]
    # pr = [pep_tcr_simi_score[k][1][i][1] for k in pep_tcr_simi_score.keys() for i in range(5)]
    total_width, n = 1, 2  # 柱状图总宽度，有几组数据
    width = total_width / n  # 单个柱状图的宽度
    length = 5
    # all_data = [pep_simi, tcr_simi]
    all_x = []
    gap = 4
    num_bar = 3
    for i in range(num_bar):
        # all_x.append(np.array([idx * 2 * width + (length * 2 * width + width * 2) * i for idx in range(length)]))
        all_x.append(np.array([idx * width * gap + i * width + gap for idx in range(length)]))
    # all_x_ = []
    # [all_x_.extend(i) for i in all_x]
    # xticks_ = xticks * len(pep_simi)
    # plt.xticks(all_x_, xticks_, rotation=30)

    # for i in range(len(all_data)):
    #     plt.bar(all_x[i], all_data[i], width=width, label=label[i])
    # if text:
    #     for i in range(len(all_data)):
    #         for a, b in zip(all_x[i], all_data[i]):
    #             plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    # plt.legend()
    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.bar(all_x[0], pep_simi, color='blue', label=label[0])
    plt.bar(all_x[2], tcr_simi, color='yellow', label=label[1])
    for a, b in zip(all_x[0], pep_simi):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    for a, b in zip(all_x[2], tcr_simi):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    ax1.legend(loc=2)
    ax1.set_ylim([0, 1.1])  # 设置y轴取值范围
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper left")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(all_x[1], roc, 'or-', label=label[2])
    ax2.plot(all_x[1], pr, 'or-', label=label[3], color='black')
    for a, b in zip(all_x[1], roc):
        plt.text(a, b - 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    for a, b in zip(all_x[1], pr):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    ax2.legend(loc=1)
    ax2.set_ylim([0, 1.1])
    # ax2.set_ylabel('score')
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper right")
    # ax1.set_xlim(0, 70)
    plt.xticks(all_x[1], ['KFold1', 'KFold2', 'KFold3', 'KFold4', 'KFold5'], rotation=30)
    if save:
        plt.savefig(save)
    plt.show()


def get_all_simi_pep_tcr_score(Round, Kfold):
    if type(Round) is tuple:
        start_R = Round[0]
        end_R = Round[1]
    else:
        start_R = 1
        end_R = Round + 1
    all_simi_pep_tcr_score = {}
    for r in range(start_R, end_R):
        pep_tcr_simi = []
        score = []
        for k in range(1, Kfold + 1):
            simi = get_pep_tcr_simi(r, k)
            pep_tcr_simi.append((np.mean([simi[pep][0][0][0] for pep in simi.keys()]),
                                 np.mean([simi[pep][0][0][1] for pep in simi.keys()])))
            score.append((np.mean([simi[pep][1][0][0] for pep in simi.keys()]),
                          np.mean([simi[pep][1][0][1] for pep in simi.keys()])))
        if str(r) not in all_simi_pep_tcr_score:
            all_simi_pep_tcr_score[str(r)] = [[], []]
        all_simi_pep_tcr_score[str(r)][0].append((np.mean([pep_tcr_simi[i][0] for i in range(len(pep_tcr_simi))]),
                                                  np.mean([pep_tcr_simi[i][1] for i in range(len(pep_tcr_simi))])))
        all_simi_pep_tcr_score[str(r)][1].append(
            (np.mean([score[i][0] for i in range(len(score))]), np.mean([score[i][1] for i in range(len(score))])))
    return all_simi_pep_tcr_score


def draw_simi_score(pep_tcr_simi_score, text=False):
    # https://blog.csdn.net/Leige_Smart/article/details/79583470
    label = ['Peptide', 'TCR', 'ROC', 'PR']
    pep_simi = [pep_tcr_simi_score[k][0][0][0] for k in pep_tcr_simi_score.keys()]
    tcr_simi = [pep_tcr_simi_score[k][0][0][1] for k in pep_tcr_simi_score.keys()]
    roc = [pep_tcr_simi_score[k][1][0][0] for k in pep_tcr_simi_score.keys()]
    pr = [pep_tcr_simi_score[k][1][0][1] for k in pep_tcr_simi_score.keys()]
    total_width, n = 1, 2  # 柱状图总宽度，有几组数据
    width = total_width / n  # 单个柱状图的宽度
    length = len(pep_tcr_simi_score.keys())
    # all_data = [pep_simi, tcr_simi]
    all_x = []
    gap = 4
    num_bar = 3
    for i in range(num_bar):
        # all_x.append(np.array([idx * 2 * width + (length * 2 * width + width * 2) * i for idx in range(length)]))
        all_x.append(np.array([idx * width * gap + i * width + gap for idx in range(length)]))
    # all_x_ = []
    # [all_x_.extend(i) for i in all_x]
    # xticks_ = xticks * len(pep_simi)
    # plt.xticks(all_x_, xticks_, rotation=30)

    # for i in range(len(all_data)):
    #     plt.bar(all_x[i], all_data[i], width=width, label=label[i])
    # if text:
    #     for i in range(len(all_data)):
    #         for a, b in zip(all_x[i], all_data[i]):
    #             plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    # plt.legend()
    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.bar(all_x[0], pep_simi, color='blue', label=label[0])
    plt.bar(all_x[2], tcr_simi, color='yellow', label=label[1])
    for a, b in zip(all_x[0], pep_simi):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    for a, b in zip(all_x[2], tcr_simi):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    ax1.legend(loc=2)
    ax1.set_ylim([0, 1.1])  # 设置y轴取值范围
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper left")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(all_x[1], roc, 'or-', label=label[2])
    ax2.plot(all_x[1], pr, 'or-', label=label[3], color='black')
    for a, b in zip(all_x[1], roc):
        plt.text(a, b - 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    for a, b in zip(all_x[1], pr):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    ax2.legend(loc=1)
    ax2.set_ylim([0, 1.1])
    # ax2.set_ylabel('score')
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper right")
    # ax1.set_xlim(0, 70)
    # plt.xticks(all_x[1], ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'], rotation=30)
    plt.show()


if __name__ == '__main__':
    # draw zero-shot similarity # 不能跑 没有计算zero的负样本和训练数据的相似度
    # get_pep_tcr_simi_zero_shot(1, 5)
    #  draw few-shot similarity
    # for roundi in range(1, 11):
    #     draw_5fold_simi_score(get_5fold_simi_pep_tcr_score((roundi, roundi + 1), 5), save='round_' + str(roundi) + '.jpg')

    # pearsonr相关系数
    pep_simi, tcr_simi, roc, pr = [], [], [], []
    for roundi in range(1, 11):
        pep_simi_, tcr_simi_, roc_, pr_ = get_pep_tcr_simi_roc_pr(get_5fold_simi_pep_tcr_score((roundi, roundi + 1), 5))
        pep_simi.extend(pep_simi_)
        tcr_simi.extend(tcr_simi_)
        roc.extend(roc_)
        pr.extend(pr_)
    p_pr = pearsonr(pep_simi, roc)
    p_pp = pearsonr(pep_simi, pr)
    p_tr = pearsonr(tcr_simi, roc)
    p_tp = pearsonr(tcr_simi, pr)
    p_ptr = pearsonr([tcr_simi[i] + pep_simi[i] for i in range(50)], roc)
    p_ptp = pearsonr([tcr_simi[i] + pep_simi[i] for i in range(50)], pr)
    print(p_pr, '\n', p_pp, '\n', p_tr, '\n', p_tp, '\n', p_ptr, '\n', p_ptp)
