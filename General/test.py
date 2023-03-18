import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from General_Model import Memory_Meta
from collections import Counter

from function import PepTCRdict, device, train_epoch, valid_epoch, data_dict, add_position_encoding, aamapping
from general import config


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


argparser = argparse.ArgumentParser()
argparser.add_argument('--positive_input', type=str, help='the path to the positive input data file (*.csv)', default='./Requirements/base_dataset.csv')
argparser.add_argument('--negative_input', type=str, help='the path to the negative input data file (*.csv)', default='./Control_dataset.txt')
argparser.add_argument('--epoch', type=int, help='epoch number', default=500)  # TODO
argparser.add_argument('--task_num', type=int, help='a total number of tasks', default=12)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
argparser.add_argument('--C', type=int, help='Peptide clustering number', default=3)
argparser.add_argument('--R', type=int, help='Peptide Index matrix vector length', default=3)
argparser.add_argument('--L', type=int, help='Peptide embedding length', default=75)
argparser.add_argument('--regular', type=float, help='The regular coefficient', default=0)
args = argparser.parse_args()

# Initialize a new model
model = Memory_Meta(args, config).to(device)

test_data = pd.read_csv('test.csv')
# Test_data = PepTCRdict(test_data, aa_dict_path='./Requirements/dic_Atchley_factors.pkl', mode='train')
# Test_db = torch.utils.data.DataLoader(Test_data, 64, shuffle=True, num_workers=1, pin_memory=True)
peptides = test_data['peptide']
TCRs = test_data['binding_TCR']
labels = test_data['label']

# Test_data = PepTCRdict(test_data, aa_dict_path='./Requirements/dic_Atchley_factors.pkl', mode='train')
# Test_db = torch.utils.data.DataLoader(Test_data, 64, shuffle=True, num_workers=1, pin_memory=True)
F_data = {}
for i, j in enumerate(peptides):
    if j not in F_data:
        F_data[j] = [[], [], [], []]
    if labels[i] != 'Unknown':
        F_data[j][0].append(TCRs[i])
        F_data[j][1].append(labels[i])

    # If the label is unknown, we put the TCRs into the query set
    else:
        F_data[j][2].append(TCRs[i])

# The variable "ends" is a list used for storing the predicted score for the 'Unknown' peptide-TCR pairs
ends = []

for i in F_data:
    # Convert the input into the embeddings
    peptide_embedding, x_spt = task_embedding(i, F_data[i][0])
    # Support set is used for fine-tune the model and the query set is used to test the performance
    end = model.test(x_spt[0].to(device))
    ends += list(end)

# Store the predicted result and output the result as .csv file

output = pd.DataFrame({'Peptide': peptides, 'CDR3': TCRs, 'Score': ends})
output.to_csv('./test_result.csv', index=False)
