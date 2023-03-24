import os
import numpy as np
import pandas as pd
import torch
import joblib
# from multiprocessing import Pool, cpu_count, Process
# # 多进程
# pool = Pool(cpu_count())

from utils import Args, get_peptide_tcr, get_model, Model_config, Device, Project_path, Aa_dict, Train_Round, Train_output_dir, Test_output_dir, Negative_dataset, Zero_test_data, aamapping, add_position_encoding, zero_task_embedding


def change_dict2test_struct(ori_dict, HealthyTCRFile, ratio=1):
    HealthyTCR = np.loadtxt(HealthyTCRFile, dtype=str)
    F_data = {}
    positive = 0
    for i, j in ori_dict.items():
        if i not in F_data:
            F_data[i] = [[], []]
        F_data[i][0].extend(j)
        positive += len(j)
    selected_query_idx = np.random.choice(len(HealthyTCR), int(positive * ratio), replace=False)
    selected_query_TCRs = HealthyTCR[selected_query_idx]
    befor_num = 0
    for i, j in ori_dict.items():
        F_data[i][0].extend(selected_query_TCRs[befor_num: befor_num + len(j)])
        befor_num += len(j)

    return F_data


def test_zero_shot(model, test_data, output_file, aa_dict, device):
    '''
    Zero-shot test. Store the results in a csv.
    '''
    # The variable "starts" is a list used for storing the predicted score for the unseen peptide-TCR pairs
    starts = []
    for i in test_data:
        # Convert the input into the embeddings
        all_test_data = test_data[i][0]
        peptide_embedding, x_spt = zero_task_embedding(i, all_test_data, aa_dict)
        # Memory block is used for predicting the binding scores of the unseen peptide-TCR pairs
        start = model.meta_forward_score(peptide_embedding.to(device), x_spt.to(device))
        starts += list(torch.Tensor.cpu(start[0]).numpy())
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in test_data:
        all_test_data = test_data[i][0]
        output_peps += [i] * len(all_test_data)
        output_tcrs += all_test_data
    # Store the predicted result and output the result as .csv file
    output = pd.DataFrame({'Peptide': output_peps, 'CDR3': output_tcrs, 'Score': starts})
    output.to_csv(output_file, index=False)


if __name__ == '__main__':
    args = Args(C=3, L=75, R=3, update_lr=0.01, update_step_test=3)
    device = torch.device(Device)
    # Load the Atchley_factors for encoding the amino acid
    aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))
    for r_idx in range(1, (Train_Round + 1)):
        print('Testing Round:', r_idx)
        round_dir = os.path.join(Project_path, Train_output_dir, 'Round' + str(r_idx))
        for kfold_dir in os.listdir(round_dir):
            kfold_idx = str(kfold_dir).split('kfold')[-1]
            print('--kfload:', kfold_idx)
            test_data = os.path.join(Project_path, Zero_test_data)
            F_data = change_dict2test_struct(get_peptide_tcr(test_data, 'peptide', 'binding_TCR'), HealthyTCRFile=os.path.join(Project_path, Negative_dataset), ratio=1)
            test_output_dir = os.path.join(round_dir, kfold_dir, Test_output_dir)
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            model = get_model(args, Model_config, model_path=os.path.join(Project_path, Train_output_dir, round_dir, kfold_dir), device=device)
            test_zero_shot(model, F_data, output_file=os.path.join(test_output_dir, 'Zero-shot_dataset_Result_Round_' + str(r_idx) + '_kfold_' + kfold_idx + '_test.csv'), aa_dict=aa_dict, device=device)
