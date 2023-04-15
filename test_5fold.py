import os
from collections import Counter
import random
import pandas as pd
import numpy as np
import joblib
import torch

from utils import Args, get_peptide_tcr, get_num, get_model, Model_config, Device, Project_path, Aa_dict, Data_output, Train_output_dir, Test_output_dir, Train_Round, Negative_dataset, task_embedding, add_position_encoding, aamapping, zero_task_embedding

import warnings

warnings.filterwarnings('ignore')


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
    # yield from a (The function is the same as that of the following statement,
    # but the following method is used to return fewer than n statements faster)
    for i in range(n):
        yield a[i]


def return_m_from_n(m, n=None, generate_idx=None):
    """
    从长度为n的生成器中返回m项 (n未指定时，生成器不为空；生成器未指定时，n不能为空) (且m<=n)
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


def change_dict2test_struct(ori_dict, HealthyTCR, k_shot=2, k_query=None, lower_limit=None, upper_limit=None, ratio=1, has_nega=False):
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
    if lower_limit is None:
        lower_limit = k_shot + 1
    if upper_limit is None:
        upper_limit = 100000
    if type(HealthyTCR) is str:
        HealthyTCR = np.loadtxt(HealthyTCR, dtype=str)
    # print('Support size:', k_shot * len(ori_dict.keys()), 'Query size:', int((ratio + 1) * (get_num(ori_dict) - k_shot * len(ori_dict.keys()))))
    if not has_nega:
        F_data = {}
        for i, j in ori_dict.items():
            if (len(j) < lower_limit) or (len(j) > upper_limit):
                continue
            if k_query is None or k_query == 'all':
                k_query = len(j) - k_shot
            if type(k_query) is int:
                if len(j) < k_shot + k_query:
                    continue
            if i not in F_data:
                F_data[i] = [[], [], [], []]
            # Positive
            positive_generate_idx = generate_selected_idx(len(j))
            # Choose positive support
            for idx in range(k_shot):
                F_data[i][0].append(j[next(positive_generate_idx)])
                F_data[i][1].append(1)
            # Choose positive query
            for idx in range(k_query):
                F_data[i][2].append(j[next(positive_generate_idx)])
            # Negative
            negative_generate_idx = generate_selected_idx(len(HealthyTCR))
            # Choose negative support
            selected_n_spt_idx = return_m_from_n(k_shot, generate_idx=negative_generate_idx)
            selected_heal_TCRs = HealthyTCR[selected_n_spt_idx]
            F_data[i][0].extend(selected_heal_TCRs)
            F_data[i][1].extend([0] * len(selected_heal_TCRs))
            # Choose negative query
            selected_n_query_idx = return_m_from_n(k_query * ratio, generate_idx=negative_generate_idx)
            selected_query_TCRs = HealthyTCR[selected_n_query_idx]
            F_data[i][2].extend(selected_query_TCRs)
        return F_data
    else:
        F_data = {}
        num = 0
        for i, j in ori_dict.items():
            num += 1
            # print(num)
            tcr = j[0]
            label = j[1]
            if (len(tcr) < 2 * lower_limit) or (len(tcr) > 2 * upper_limit):
                continue
            if k_query is None or k_query == 'all':
                k_query = (len(tcr) - 2 * k_shot) / 2
            if type(k_query) is int:
                if len(tcr) < (k_shot + k_query) * 2:
                    continue
            if i not in F_data:
                F_data[i] = [[], [], [], []]
            index_p = [k for k, v in enumerate(label) if v == 1]
            index_n = [k for k, v in enumerate(label) if v == 0]
            # Positive
            positive_support_idx = random.sample(index_p, k_shot)
            # Negative
            negative_support_idx = random.sample(index_n, k_shot)
            # Choose support
            for idx in range(k_shot):
                F_data[i][0].append(tcr[positive_support_idx[idx]])
                F_data[i][1].append(1)
                F_data[i][0].append(tcr[negative_support_idx[idx]])
                F_data[i][1].append(0)
            # Choose query
            selected_idx_p = positive_support_idx
            selected_idx_n = negative_support_idx
            for idx in range(k_query):
                positive_query_idx = random.sample(index_p, 1)
                while positive_query_idx[0] in selected_idx_p:
                    positive_query_idx = random.sample(index_p, 1)
                selected_idx_p.extend(positive_query_idx)
                F_data[i][2].append(tcr[positive_query_idx[0]])
                negative_query_idx = random.sample(index_n, 1)
                while negative_query_idx[0] in selected_idx_n:
                    negative_query_idx = random.sample(index_n, 1)
                selected_idx_n.extend(negative_query_idx)
                F_data[i][2].append(tcr[negative_query_idx[0]])
        return F_data


def test_5fold_few_shot(model, test_data, output_file, aa_dict, device):
    '''
    Few-shot test. Store the results in a csv.
    '''
    ends = []
    for i in test_data:
        # Convert the input into the embeddings
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i, test_data[i], aa_dict)
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


def test_5fold_zero_shot(model, test_data, output_file, aa_dict, device):
    '''
    Zero-shot test. Store the results in a csv.
    '''
    # The variable "starts" is a list used for storing the predicted score for the unseen peptide-TCR pairs
    starts = []
    for i in test_data:
        # Convert the input into the embeddings
        all_test_data = test_data[i][0] + test_data[i][2]
        peptide_embedding, x_spt = zero_task_embedding(i, all_test_data, aa_dict)
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


def test_5fold_zero_shot_finetune(model, test_data, output_file, aa_dict, device):
    '''
    Zero-shot test. Store the results in a csv.
    '''
    ends = []
    for i in test_data:
        # Convert the input into the embeddings
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i, test_data[i], aa_dict)
        # Support set is used for fine-tune the model and the query set is used to test the performance
        end = model.zero_model_test_few_data(peptide_embedding.to(device), x_spt.to(device), y_spt.to(device), x_qry.to(device))
        ends += list(end[0])
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in test_data:
        all_test_data = test_data[i][0] + test_data[i][2]
        output_peps += [i] * len(all_test_data)
        output_tcrs += all_test_data
    # Store the predicted result and output the result as .csv file
    output = pd.DataFrame({'Peptide': output_peps, 'CDR3': output_tcrs, 'Score': ends})
    output.to_csv(output_file, index=False)


if __name__ == '__main__':
    k_shot_list = [2]  # 5, 4, 3, 2
    k_query = 3
    upper_limit = 10  # None
    args = Args(C=3, L=75, R=3, update_lr=0.01, update_step_test=3)
    device = torch.device(Device)
    # Load the Atchley_factors for encoding the amino acid
    aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))
    HealthyTCRFile = np.loadtxt(os.path.join(Project_path, Negative_dataset), dtype=str)
    for r_idx in range(1, (Train_Round + 1)):
        print('Testing Round:', r_idx)
        round_dir = os.path.join(Project_path, Train_output_dir, 'Round' + str(r_idx))
        for kfold_dir in os.listdir(round_dir):
            for k_shot in k_shot_list:
                kfold_idx = str(kfold_dir).split('kfold')[-1]
                print('--kfload:', kfold_idx)
                test_data = os.path.join(Project_path, Data_output, 'kfold' + kfold_idx, 'KFold_' + kfold_idx + '_test.csv')
                F_data = change_dict2test_struct(get_peptide_tcr(test_data, 'peptide', 'binding_TCR'),
                                                 HealthyTCR=HealthyTCRFile, k_shot=k_shot, k_query=k_query,
                                                 lower_limit=None, upper_limit=upper_limit, ratio=1)
                print('Support size:', sum([len(j) for j in (F_data[k][0] for k in list(F_data.keys()))]), 'Query size:', sum([len(j) for j in (F_data[k][2] for k in list(F_data.keys()))]))
                model = get_model(args, Model_config, model_path=os.path.join(Project_path, Train_output_dir, round_dir, kfold_dir), device=device)
                test_output_dir = os.path.join(round_dir, kfold_dir, Test_output_dir)
                if not os.path.exists(test_output_dir):
                    os.makedirs(test_output_dir)
                test_5fold_few_shot(model, F_data, output_file=os.path.join(test_output_dir, 'Few-shot_Result_Round_' + str(r_idx) +
                                                                            '_kfold_' + kfold_idx + '_kshot' + str(k_shot) + '_kquery' +
                                                                            str(k_query) + '_ulimit_' + str(upper_limit) + '_test.csv'), aa_dict=aa_dict, device=device)
                # test_5fold_zero_shot(model, F_data, output_file=os.path.join(test_output_dir, 'Zero-shot_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_kshot'+str(k_shot)+'_test.csv'), aa_dict=aa_dict, device=device)
                # test_5fold_zero_shot_finetune(model, F_data, output_file=os.path.join(test_output_dir, 'Zero-shot_fine-tune_few_data_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_kshot'+str(k_shot)+'_test.csv'), aa_dict=aa_dict, device=device)
