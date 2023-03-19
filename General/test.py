import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from General_Model import Memory_Meta
from collections import Counter

from function import PepTCRdict, train_epoch, valid_epoch, add_negative_data, project_path, data_config, device, data_output, add_position_encoding, aamapping

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


# Initialize a new model
model = Memory_Meta(config).to(device)

# model_path = os.path.join(os.path.abspath(""), 'best.pt')
# model.load_state_dict(torch.load(model_path))
if not os.path.exists(os.path.join(os.path.abspath(""), data_config['Train']['General']['Test_result_path'])):
    os.makedirs(os.path.join(os.path.abspath(""), data_config['Train']['General']['Test_result_path']))
for kf_time in range(data_config['dataset']['current_fold'][0], data_config['dataset']['current_fold'][1]):
    genetal_test_path = os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test_general_data.csv')
    test_data = pd.read_csv(genetal_test_path)
    peptides = test_data['peptide']
    TCRs = test_data['binding_TCR']
    labels = test_data['label']
    F_data = {}
    for i, j in enumerate(peptides):
        if j not in F_data:
            F_data[j] = [[], []]
        if labels[i] != 'Unknown':
            F_data[j][0].append(TCRs[i])
            F_data[j][1].append(labels[i])
    model_path = os.path.join(os.path.abspath(""), 'KFold_' + str(kf_time) + 'best.pt')
    model.load_state_dict(torch.load(model_path))
    ends = []
    for i in F_data:
        # Convert the input into the embeddings
        peptide_embedding, x_spt = task_embedding(i, F_data[i][0])
        # Support set is used for fine-tune the model and the query set is used to test the performance
        end = model.test(x_spt[0].to(device))
        ends += list(end)

    # Store the predicted result and output the result as .csv file
    output = pd.DataFrame({'Peptide': peptides, 'CDR3': TCRs, 'Score': ends})
    output_file = os.path.join(os.path.abspath(""), data_config['Train']['General']['Test_result_path'], 'KFold_' + str(kf_time) + '_test_general_result.csv')
    output.to_csv(output_file, index=False)
