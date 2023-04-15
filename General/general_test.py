import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from General_Model import Memory_Meta
from collections import Counter
import joblib


from utils import Data_config, Project_path, Data_output, Device, Model_config, add_position_encoding, aamapping, Aa_dict, Train_Round


def task_embedding(pep, tcr_data, aa_dict):
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
    peptide_embedding = add_position_encoding(aamapping(pep, 15, aa_dict))

    # Put the embedding of query TCRs and peptide into the initialized tensor
    temp = torch.Tensor()
    for j in spt_TCRs:
        temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25, aa_dict))]).unsqueeze(0)])
    query_x[0] = temp
    peptides[0] = peptide_embedding.flatten()

    return peptides, query_x


# Initialize a new model
model = Memory_Meta(Model_config).to(Device)
# Load the Atchley_factors for encoding the amino acid
aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))

for index in range(Train_Round):
    for kf_time in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):
        try:
            genetal_test_path = os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test_general_data.csv')
            test_data = pd.read_csv(genetal_test_path)
        except Exception as e:
            print(e)
            print("Not find test file")
            continue
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
        model_path = os.path.join(os.path.abspath(""), 'Round' + str(index + 1), 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_best.pt')
        model.load_state_dict(torch.load(model_path))
        ends = []
        for i in F_data:
            # Convert the input into the embeddings
            peptide_embedding, x_spt = task_embedding(i, F_data[i][0], aa_dict)
            # Support set is used for fine-tune the model and the query set is used to test the performance
            end = model.test(x_spt[0].to(Device))
            ends += list(end)

        # Store the predicted result and output the result as .csv file
        output = pd.DataFrame({'Peptide': peptides, 'CDR3': TCRs, 'Score': ends})
        output_file = os.path.join(os.path.abspath(""), 'Round' + str(index + 1), 'kfold' + str(kf_time), Data_config['Train']['General']['Test_result_path'])
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        output.to_csv(os.path.join(output_file, 'KFold_' + str(kf_time) + '_test_general_result.csv'), index=False)
