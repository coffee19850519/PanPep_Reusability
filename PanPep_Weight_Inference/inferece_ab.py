import joblib
import random
import argparse 
import torch, os
import numpy as np
import scipy.stats
import random, sys
import pandas as pd
import matplotlib.pyplot as plt
from Requirements.Memory_meta_test import Memory_Meta
from Requirements.Memory_meta_test import Memory_module

# The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--learning_setting', type=str, help='choosing the learning setting: few-shot, zero-shot,Meta-learner and majority',required=True)
argparser.add_argument('--input_dir', type=str, help='the directory containing input CSV files',required=True)
argparser.add_argument('--output_dir', type=str, help='the directory to save output CSV files',required=True)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
argparser.add_argument('--C', type=int, help='Number of bases', default=3)
argparser.add_argument('--R', type=int, help='Epitope Index matrix vector length', default=3)
argparser.add_argument('--L', type=int, help='Epitope embedding length', default=75)
argparser.add_argument('--fold', type=int, help='which fold to use (1-10)', required=True)
argparser.add_argument('--peptide_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/peptide_ab.npz',
                        help='Path to peptide encoding file')
argparser.add_argument('--tcr_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/tcr_ab.npz',
                        help='Path to TCR encoding file')
args = argparser.parse_args()

# Load the Atchley_factors for encoding the amino acid
aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")

def aamapping(TCRSeq, encode_dim):
    """
    Parameters:
        param TCRSeq: the TCR original sequence
        param encode_dim: the first dimension of TCR sequence embedding matrix
    Returns:
        this function returns a TCR embedding matrix;
        e.g. the TCR sequence of ASSSAA
        return: (6 + encode_dim - 6) x 5 embedding matrix, in which (encode_dim - 6) x 5 will be zero matrix
    Raises:
        KeyError - using 0 vector for replacing the original amino acid encoding
    """
    TCRArray = []
    if len(TCRSeq)>encode_dim:
        print('Length: '+str(len(TCRSeq))+' over bound!')
        TCRSeq=TCRSeq[0:encode_dim]
    for aa_single in TCRSeq:
        try:
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            TCRArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(TCRSeq)):
        TCRArray.append(np.zeros(5,dtype='float64'))
    return torch.FloatTensor(TCRArray) 

# Set the random seed
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.cuda.manual_seed(222)

# Sinusoidal position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)

def add_position_encoding(seq):
    """
    this function is used to add position encoding for the TCR embedding
    Parameters:
        param seq: the TCR embedding matrix
    Returns:
        this function returns a TCR embedding matrix containing position encoding
    """
    padding_ids = torch.abs(seq).sum(dim=-1)==0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq
def zero_task_embedding(pep, tcr_data, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    this function is used to obtain the task-level embedding for the zero-shot setting
     
    Parameters:
        param pep: peptide sequence
        param tcr_data: TCR list (query TCRs)
            
    Returns:
        this function returns a peptide embedding and the embedding of query TCRs
    """
    spt_TCRs = tcr_data
    query_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    peptides = torch.FloatTensor(1, 75)
    
    # Encoding the peptide sequence
    peptide_embedding = peptide_encoding_dict[pep] if peptide_encoding_dict and pep in peptide_encoding_dict else add_position_encoding(aamapping(pep, 15))
    
    # Process the query TCRs
    tcr_embeddings = []
    for tcr in spt_TCRs:
        tcr_embed = tcr_encoding_dict[tcr] if tcr_encoding_dict and tcr in tcr_encoding_dict else add_position_encoding(aamapping(tcr, 25))
        tcr_embeddings.append(tcr_embed)
    
    tcr_embeddings = torch.stack(tcr_embeddings)
    peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(spt_TCRs), -1, -1)
    temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
    query_x[0] = temp
    peptides[0] = peptide_embedding.flatten()
    
    return peptides, query_x
def task_embedding(pep, tcr_data, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    Creates task-level embeddings for peptide-TCR interaction prediction.
    
    Parameters:
        pep: str
            The peptide sequence to be embedded
        tcr_data: list
            A list containing TCR sequences and their labels in the format:
            [support_TCRs, support_labels, (optional)query_TCRs]
        peptide_encoding_dict: dict, optional
            Pre-computed peptide encodings {peptide_sequence: encoding_tensor}
        tcr_encoding_dict: dict, optional
            Pre-computed TCR encodings {TCR_sequence: encoding_tensor}
            
    Returns:
        tuple containing:
            - peptides: torch.FloatTensor
                Flattened peptide embedding (shape: [1, 75])
            - support_x: torch.FloatTensor or None
                Support set embeddings if available (shape: [1, n_support, 40, 5])
            - support_y: torch.LongTensor or None
                Support set labels if available (shape: [1, n_support])
            - query_x: torch.FloatTensor
                Query set embeddings (shape: [1, n_query, 40, 5])
                
    Note:
        If support set is empty, support_x and support_y will be None
    """
    # Extract TCR sequences and labels from input data
    spt_TCRs = tcr_data[0]  # Support set TCRs
    ypt = tcr_data[1]       # Support set labels
    peptides = torch.FloatTensor(1, 75)  # Initialize peptide tensor
    
    # Handle query TCRs if provided, otherwise use placeholder
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    # Get peptide embedding from dictionary or compute it
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15))

    # Handle case when support set is empty
    if not spt_TCRs or not ypt:
        peptides[0] = peptide_embedding.flatten()
        
        # Process query set if available
        if len(tcr_data) > 2:
            tcr_embeddings = []
            for tcr in qry_TCRs:
                # Get TCR embedding from dictionary or compute it
                if tcr_encoding_dict and tcr in tcr_encoding_dict:
                    tcr_embed = tcr_encoding_dict[tcr]
                else:
                    tcr_embed = add_position_encoding(aamapping(tcr, 25))
                tcr_embeddings.append(tcr_embed)
            
            # Stack TCR embeddings and combine with peptide embedding
            tcr_embeddings = torch.stack(tcr_embeddings)
            peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
            temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
            query_x[0] = temp
        else:
            query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
            
        return peptides, None, None, query_x

    # Process support set when available
    support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    support_y = np.zeros((1, len(ypt)), dtype=np.int64)
    
    # Create embeddings for support set TCRs
    temp = []
    for tcr in spt_TCRs:
        # Get TCR embedding from dictionary or compute it
        if tcr_encoding_dict and tcr in tcr_encoding_dict:
            tcr_embed = tcr_encoding_dict[tcr]
        else:
            tcr_embed = add_position_encoding(aamapping(tcr, 25))
        # Concatenate peptide and TCR embeddings
        temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
    
    support_x[0] = torch.cat(temp)
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()

    # Process query set if available
    if len(tcr_data) > 2:
        tcr_embeddings = []
        for tcr in qry_TCRs:
            # Get TCR embedding from dictionary or compute it
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25))
            tcr_embeddings.append(tcr_embed)
        
        # Stack TCR embeddings and combine with peptide embedding
        tcr_embeddings = torch.stack(tcr_embeddings)
        peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
        temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    return peptides, support_x, torch.LongTensor(support_y), query_x

def load_encodings(encoding_path):
    with np.load(encoding_path) as encodings:
        sequences = encodings['sequences']
        encoding_data = encodings['encodings']
        encoding_dict = {seq: torch.from_numpy(enc).float() for seq, enc in zip(sequences, encoding_data)}
    return encoding_dict

# This is the model parameters
config = [
    ('self_attention',[[1,5,5],[1,5,5],[1,5,5]]),
    ('linear', [5, 5]),
    ('relu',[True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]

# The paths of the trained models
alpha_path = f"./fold10alpha/fold{args.fold}/model.pt"
beta_path = f"./fold10beta/fold{args.fold}/model.pt"

# Set the 'cuda' used for GPU testing
device = torch.device('cuda')

# Initialize models for alpha and beta
model_alpha = Memory_Meta(args, config).to(device)
model_beta = Memory_Meta(args, config).to(device)

# Load the pretrained models
model_alpha.load_state_dict(torch.load(alpha_path))
model_beta.load_state_dict(torch.load(beta_path))

# Load the memory blocks
model_alpha.Memory_module = Memory_module(args, model_alpha.meta_Parameter_nums).cuda()
model_beta.Memory_module = Memory_module(args, model_beta.meta_Parameter_nums).cuda()

# Load the content memory matrix and query matrix(read head) for alpha
alpha_content = joblib.load(f"./fold10alpha/fold{args.fold}/Content_memory.pkl")
alpha_query = joblib.load(f"./fold10alpha/fold{args.fold}/Query.pkl")
model_alpha.Memory_module.memory.content_memory = alpha_content
model_alpha.Memory_module.memory.Query.weight = alpha_query[0]
model_alpha.Memory_module.memory.Query.bias = alpha_query[1]

# Load the content memory matrix and query matrix(read head) for beta
beta_content = joblib.load(f"./fold10beta/fold{args.fold}/Content_memory.pkl")
beta_query = joblib.load(f"./fold10beta/fold{args.fold}/Query.pkl")
model_beta.Memory_module.memory.content_memory = beta_content
model_beta.Memory_module.memory.Query.weight = beta_query[0]
model_beta.Memory_module.memory.Query.bias = beta_query[1]

# Load the encodings
peptide_encoding_dict = load_encodings(args.peptide_encoding)
tcr_encoding_dict = load_encodings(args.tcr_encoding)

def process_file(input_file, output_file, learning_setting):
    """
    Process a single CSV file and save results to output file
    """
    # Read the data from the csv file
    data = pd.read_csv(input_file)
    Epitopes = data['Epitope']
    alpha_tcrs = data['alpha']
    beta_tcrs = data['beta']
    labels = data['Label']  # Use Label for processing
    original_labels = data['label']  # Keep original labels for output
    
    if learning_setting == 'few-shot':
        # Construct the episode, the input for the panpep in the few-shot setting
        F_data_alpha = {}
        F_data_beta = {}
        for i,j in enumerate(Epitopes):
            if j not in F_data_alpha:
                F_data_alpha[j] = [[],[],[],[]]
                F_data_beta[j] = [[],[],[],[]]
            if labels[i] != 'Unknown':
                F_data_alpha[j][0].append(alpha_tcrs[i])
                F_data_alpha[j][1].append(labels[i])
                F_data_beta[j][0].append(beta_tcrs[i])
                F_data_beta[j][1].append(labels[i])
            else:
                F_data_alpha[j][2].append(alpha_tcrs[i])
                F_data_beta[j][2].append(beta_tcrs[i])
                F_data_alpha[j][3].append(i)  # Store the original index
                F_data_beta[j][3].append(i)  # Store the original index

        # Process each epitope
        ends_alpha = []
        ends_beta = []
        output_peps = []
        output_alpha_tcrs = []
        output_beta_tcrs = []
        output_indices = []
        
        for i in F_data_alpha:
            if F_data_alpha[i][2]:  # Check if there are query samples
                # Convert the input into the embeddings for alpha
                Epitope_embedding_alpha, x_spt_alpha, y_spt_alpha, x_qry_alpha = task_embedding(i, F_data_alpha[i], peptide_encoding_dict, tcr_encoding_dict)
                
                # Convert the input into the embeddings for beta
                Epitope_embedding_beta, x_spt_beta, y_spt_beta, x_qry_beta = task_embedding(i, F_data_beta[i], peptide_encoding_dict, tcr_encoding_dict)
                
                # Support set is used for fine-tune the model and the query set is used to test the performance
                end_alpha = model_alpha.finetunning(Epitope_embedding_alpha[0].to(device), x_spt_alpha[0].to(device), y_spt_alpha[0].to(device), x_qry_alpha[0].to(device))
                end_beta = model_beta.finetunning(Epitope_embedding_beta[0].to(device), x_spt_beta[0].to(device), y_spt_beta[0].to(device), x_qry_beta[0].to(device))
                
                ends_alpha += list(end_alpha[0])
                ends_beta += list(end_beta[0])
                output_peps += [i]*len(F_data_alpha[i][2])
                output_alpha_tcrs += F_data_alpha[i][2]
                output_beta_tcrs += F_data_beta[i][2]
                output_indices += F_data_alpha[i][3]
                
                torch.cuda.empty_cache()
        
        # Calculate average scores
        avg_scores = [(a + b) / 2 for a, b in zip(ends_alpha, ends_beta)]
        
        # Get the original labels for the output
        output_labels = [original_labels[idx] for idx in output_indices]
        
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'label': output_labels,
            'score_alpha': ends_alpha,
            'score_beta': ends_beta,
            'score': avg_scores
        })
        output.to_csv(output_file, index=False)
    
    elif learning_setting == 'zero-shot':
        # Construct the episode, the input for the panpep in the zero-shot setting
        Z_data_alpha = {}
        Z_data_beta = {}
        for i,j in enumerate(Epitopes):
            if j not in Z_data_alpha:
                Z_data_alpha[j] = [[],[],[],[]]
                Z_data_beta[j] = [[],[],[],[]]
            if labels[i] != 'Unknown':
                Z_data_alpha[j][0].append(alpha_tcrs[i])
                Z_data_alpha[j][1].append(labels[i])
                Z_data_beta[j][0].append(beta_tcrs[i])
                Z_data_beta[j][1].append(labels[i])
            else:
                Z_data_alpha[j][2].append(alpha_tcrs[i])
                Z_data_beta[j][2].append(beta_tcrs[i])
                Z_data_alpha[j][3].append(i)  # Store the original index
                Z_data_beta[j][3].append(i)  # Store the original index

        # Process each epitope
        starts_alpha = []
        starts_beta = []
        output_peps = []
        output_alpha_tcrs = []
        output_beta_tcrs = []
        output_indices = []
        for i in Z_data_alpha:
            if Z_data_alpha[i][2]:  # Check if there are query samples
                # Convert the input into the embeddings for alpha
                Epitope_embedding_alpha, x_qry_alpha = zero_task_embedding(i, Z_data_alpha[i][2], peptide_encoding_dict, tcr_encoding_dict)
                
                # Convert the input into the embeddings for beta
                Epitope_embedding_beta, x_qry_beta = zero_task_embedding(i, Z_data_beta[i][2], peptide_encoding_dict, tcr_encoding_dict)
                
                # Support set is used for fine-tune the model and the query set is used to test the performance
                end_alpha = model_alpha.meta_forward_score(Epitope_embedding_alpha.to(device), x_qry_alpha.to(device))
                end_beta = model_beta.meta_forward_score(Epitope_embedding_beta.to(device), x_qry_beta.to(device))
                
                # 确保数据在转换为numpy之前先移到CPU
                if torch.is_tensor(end_alpha[0]):
                    if end_alpha[0].is_cuda:
                        end_alpha_cpu = end_alpha[0].cpu()
                    else:
                        end_alpha_cpu = end_alpha[0]
                    starts_alpha += list(end_alpha_cpu.numpy())
                else:
                    starts_alpha += list(end_alpha[0])
                
                if torch.is_tensor(end_beta[0]):
                    if end_beta[0].is_cuda:
                        end_beta_cpu = end_beta[0].cpu()
                    else:
                        end_beta_cpu = end_beta[0]
                    starts_beta += list(end_beta_cpu.numpy())
                else:
                    starts_beta += list(end_beta[0])
                
                output_peps += [i]*len(Z_data_alpha[i][2])
                output_alpha_tcrs += Z_data_alpha[i][2]
                output_beta_tcrs += Z_data_beta[i][2]
                output_indices += Z_data_alpha[i][3]
                
                torch.cuda.empty_cache()
        
        # Calculate average scores
        avg_scores = [(a + b) / 2 for a, b in zip(starts_alpha, starts_beta)]
        
        # Get the original labels for the output
        output_labels = [original_labels[idx] for idx in output_indices]
        
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'label': output_labels,
            'score_alpha': starts_alpha,
            'score_beta': starts_beta,
            'score': avg_scores
        })
        output.to_csv(output_file, index=False)
    
    elif learning_setting == 'majority':
        # Construct the episode, the input for the panpep in the majority setting
        G_data_alpha = {}
        G_data_beta = {}
        for i,j in enumerate(Epitopes):
            if j not in G_data_alpha:
                G_data_alpha[j] = [[],[],[],[]]
                G_data_beta[j] = [[],[],[],[]]
            if labels[i] != 'Unknown':
                G_data_alpha[j][0].append(alpha_tcrs[i])
                G_data_alpha[j][1].append(labels[i])
                G_data_beta[j][0].append(beta_tcrs[i])
                G_data_beta[j][1].append(labels[i])
            else:
                G_data_alpha[j][2].append(alpha_tcrs[i])
                G_data_beta[j][2].append(beta_tcrs[i])
                G_data_alpha[j][3].append(i)  # Store the original index
                G_data_beta[j][3].append(i)  # Store the original index

        # Process each epitope
        ends_alpha = []
        ends_beta = []
        output_peps = []
        output_alpha_tcrs = []
        output_beta_tcrs = []
        output_indices = []
        
        for i in G_data_alpha:
            if G_data_alpha[i][2]:  # Check if there are query samples
                # Convert the input into the embeddings for alpha
                Epitope_embedding_alpha, x_spt_alpha, y_spt_alpha, x_qry_alpha = task_embedding(i, G_data_alpha[i], peptide_encoding_dict, tcr_encoding_dict)
                
                # Convert the input into the embeddings for beta
                Epitope_embedding_beta, x_spt_beta, y_spt_beta, x_qry_beta = task_embedding(i, G_data_beta[i], peptide_encoding_dict, tcr_encoding_dict)
                
                # Support set is used for fine-tune the model and the query set is used to test the performance
                end_alpha = model_alpha.finetunning(Epitope_embedding_alpha[0].to(device), x_spt_alpha[0].to(device), y_spt_alpha[0].to(device), x_qry_alpha[0].to(device))
                end_beta = model_beta.finetunning(Epitope_embedding_beta[0].to(device), x_spt_beta[0].to(device), y_spt_beta[0].to(device), x_qry_beta[0].to(device))
                
                ends_alpha += list(end_alpha[0])
                ends_beta += list(end_beta[0])
                output_peps += [i]*len(G_data_alpha[i][2])
                output_alpha_tcrs += G_data_alpha[i][2]
                output_beta_tcrs += G_data_beta[i][2]
                output_indices += G_data_alpha[i][3]
                
                torch.cuda.empty_cache()
        
        # Calculate average scores
        avg_scores = [(a + b) / 2 for a, b in zip(ends_alpha, ends_beta)]
        
        # Get the original labels for the output
        output_labels = [original_labels[idx] for idx in output_indices]
        
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'label': output_labels,
            'score_alpha': ends_alpha,
            'score_beta': ends_beta,
            'score': avg_scores
        })
        output.to_csv(output_file, index=False)
    elif learning_setting == 'Meta-learner':
        # Construct the episode, the input for the panpep in the zero-shot setting
        Z_data_alpha = {}
        Z_data_beta = {}
        for i,j in enumerate(Epitopes):
            if j not in Z_data_alpha:
                Z_data_alpha[j] = [[],[],[],[]]
                Z_data_beta[j] = [[],[],[],[]]
            if labels[i] != 'Unknown':
                Z_data_alpha[j][0].append(alpha_tcrs[i])
                Z_data_alpha[j][1].append(labels[i])
                Z_data_beta[j][0].append(beta_tcrs[i])
                Z_data_beta[j][1].append(labels[i])
            else:
                Z_data_alpha[j][2].append(alpha_tcrs[i])
                Z_data_beta[j][2].append(beta_tcrs[i])
                Z_data_alpha[j][3].append(i)  # Store the original index
                Z_data_beta[j][3].append(i)  # Store the original index

        # Process each epitope
        starts_alpha = []
        starts_beta = []
        output_peps = []
        output_alpha_tcrs = []
        output_beta_tcrs = []
        output_indices = []
        
        for i in Z_data_alpha:
            if Z_data_alpha[i][2]:  # Check if there are query samples
                # Convert the input into the embeddings for alpha
                Epitope_embedding_alpha, x_qry_alpha = zero_task_embedding(i, Z_data_alpha[i][2], peptide_encoding_dict, tcr_encoding_dict)
                
                # Convert the input into the embeddings for beta
                Epitope_embedding_beta, x_qry_beta = zero_task_embedding(i, Z_data_beta[i][2], peptide_encoding_dict, tcr_encoding_dict)
                
                # Memory block is used for predicting the binding scores of the unseen Epitope-TCR pairs
                start_alpha = model_alpha.inference_with_params(x_qry_alpha[0].to(device), model_alpha.net)
                start_beta = model_beta.inference_with_params(x_qry_beta[0].to(device), model_beta.net)
                
                starts_alpha += list((start_alpha[0]))
                starts_beta += list((start_beta[0]))
                output_peps += [i]*len(Z_data_alpha[i][2])
                output_alpha_tcrs += Z_data_alpha[i][2]
                output_beta_tcrs += Z_data_beta[i][2]
                output_indices += Z_data_alpha[i][3]
                
                torch.cuda.empty_cache()
        
        # Calculate average scores
        avg_scores = [(a + b) / 2 for a, b in zip(starts_alpha, starts_beta)]
        
        # Get the original labels for the output
        output_labels = [original_labels[idx] for idx in output_indices]
        
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'label': output_labels,
            'score_alpha': starts_alpha,
            'score_beta': starts_beta,
            'score': avg_scores
        })
        output.to_csv(output_file, index=False)

# Main processing loop
def main():
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all CSV files from input directory
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    # Process each file with the same learning setting
    for input_file in input_files:
        print(f"Processing {input_file} using {args.learning_setting} setting...")
        input_path = os.path.join(args.input_dir, input_file)
        output_path = os.path.join(args.output_dir, input_file)
        process_file(input_path, output_path, args.learning_setting)
        print(f"Finished processing {input_file}")

if __name__ == "__main__":
    main()