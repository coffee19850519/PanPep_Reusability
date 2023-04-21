import os
import pandas as pd
import torch
from General_Model import Memory_Meta
from collections import Counter
import joblib


from utils import Data_config, Project_path, Data_output, Device, Model_config, add_position_encoding, aamapping, Aa_dict, Train_Round, get_peptide_tcr, Test_output_dir
from test_5fold import change_dict2test_struct


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

        # peptides = test_data['peptide']
        # TCRs = test_data['binding_TCR']
        # labels = test_data['label']
        # F_data = {}
        # for i, j in enumerate(peptides):
        #     if j not in F_data:
        #         F_data[j] = [[], []]
        #     if labels[i] != 'Unknown':
        #         F_data[j][0].append(TCRs[i])
        #         F_data[j][1].append(labels[i])

        F_data = change_dict2test_struct(get_peptide_tcr(test_data, 'peptide', 'binding_TCR', 'label'), None,
                                         k_shot=2, k_query=3, lower_limit=None, upper_limit=None, ratio=1, has_nega=True)
        Peptide = [key for key in F_data.keys() for i in range(6)]  # positive and negative total 6
        CDR3 = []
        [CDR3.extend(i) for i in [F_data[key][2] for key in F_data.keys()]]

        model_path = os.path.join(os.path.abspath(""), Data_config['Train']['General']['Train_output_dir'], 'Round' + str(index + 1), 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_best.pt')
        model.load_state_dict(torch.load(model_path))
        ends = []
        for i in F_data:
            # Convert the input into the embeddings
            peptide_embedding, x_qry = task_embedding(i, F_data[i][2], aa_dict)
            # Support set is used for fine-tune the model and the query set is used to test the performance
            end = model.test(x_qry[0].to(Device))
            ends += list(end)

        # Store the predicted result and output the result as .csv file
        output = pd.DataFrame({'Peptide': list(Peptide), 'CDR3': CDR3, 'Score': ends})
        output_file = os.path.join(os.path.abspath(""), Data_config['Train']['General']['Train_output_dir'], 'Round' + str(index + 1), 'kfold' + str(kf_time), Test_output_dir)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        output.to_csv(os.path.join(output_file, 'KFold_' + str(kf_time) + '_test_general_result.csv'), index=False)
