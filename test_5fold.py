import os
from collections import Counter
import pandas as pd
import numpy as np
import joblib
import torch

from utils import Args, get_peptide_tcr, get_num, get_model, Model_config, Device, Project_path, Aa_dict, Data_output, Train_output_dir, Test_output_dir, Train_Round, Negative_dataset, task_embedding, add_position_encoding, aamapping, zero_task_embedding

import warnings

warnings.filterwarnings('ignore')


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
    HealthyTCR = np.loadtxt(HealthyTCRFile, dtype=str)
    # print('Support size:', k_shot * len(ori_dict.keys()), 'Query size:', int((ratio + 1) * (get_num(ori_dict) - k_shot * len(ori_dict.keys()))))
    F_data = {}
    for i, j in ori_dict.items():
        if len(j) < k_shot + 1:
            continue
        if i not in F_data:
            F_data[i] = [[], [], [], []]
        selected_spt_idx = np.random.choice(len(j), k_shot, replace=False)
        for idx in selected_spt_idx:  # positive support
            F_data[i][0].append(j[idx])
            F_data[i][1].append(1)
        selected_heal_idx = np.random.choice(len(HealthyTCR), k_shot, replace=False)  # negative support
        selected_heal_TCRs = HealthyTCR[selected_heal_idx]
        F_data[i][0].extend(selected_heal_TCRs)
        F_data[i][1].extend([0] * len(selected_heal_TCRs))
        for query_idx in range(len(j)):  # positive query
            if (query_idx not in selected_spt_idx) and (j[query_idx] not in F_data[i][0]):
                F_data[i][2].append(j[query_idx])
        selected_query_idx = np.random.choice(len(HealthyTCR), int((len(j) - k_shot) * ratio), replace=False)
        selected_query_TCRs = HealthyTCR[selected_query_idx]
        F_data[i][2].extend(selected_query_TCRs)
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
    k_shot_list = [5, 4, 3, 2, 1]
    args = Args(C=3, L=75, R=3, update_lr=0.01, update_step_test=3)
    device = torch.device(Device)
    # Load the Atchley_factors for encoding the amino acid
    aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))
    for r_idx in range(1, (Train_Round + 1)):
        print('Testing Round:', r_idx)
        round_dir = os.path.join(Project_path, Train_output_dir, 'Round' + str(r_idx))
        for kfold_dir in os.listdir(round_dir):
            for k_shot in k_shot_list:
                kfold_idx = str(kfold_dir).split('kfold')[-1]
                print('--kfload:', kfold_idx)
                test_data = os.path.join(Project_path, Data_output, 'kfold' + kfold_idx, 'KFold_' + kfold_idx + '_test.csv')
                F_data = change_dict2test_struct(get_peptide_tcr(test_data, 'peptide', 'binding_TCR'), HealthyTCRFile=os.path.join(
                    Project_path, Negative_dataset), k_shot=k_shot, ratio=1)
                print('Support size:', sum([len(j) for j in (F_data[k][0] for k in list(F_data.keys()))]), 'Query size:', sum([len(j) for j in (F_data[k][2] for k in list(F_data.keys()))]))
                model = get_model(args, Model_config, model_path=os.path.join(Project_path, Train_output_dir, round_dir, kfold_dir), device=device)
                test_output_dir = os.path.join(round_dir, kfold_dir, Test_output_dir)
                if not os.path.exists(test_output_dir):
                    os.makedirs(test_output_dir)
                test_5fold_few_shot(model, F_data, output_file=os.path.join(test_output_dir, 'Few-shot_Result_Round_' + str(r_idx) +
                                                                            '_kfold_' + kfold_idx + '_kshot' + str(k_shot) + '_test.csv'), aa_dict=aa_dict, device=device)
                # test_5fold_zero_shot(model, F_data, output_file=os.path.join(test_output_dir, 'Zero-shot_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_kshot'+str(k_shot)+'_test.csv'), aa_dict=aa_dict, device=device)
                # test_5fold_zero_shot_finetune(model, F_data, output_file=os.path.join(test_output_dir, 'Zero-shot_fine-tune_few_data_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_kshot'+str(k_shot)+'_test.csv'), aa_dict=aa_dict, device=device)
