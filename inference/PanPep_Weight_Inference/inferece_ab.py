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
argparser.add_argument('--save_alpha_dir', type=str, default='alpha_all', help='directory to save finetuned alpha model (majority)')
argparser.add_argument('--save_beta_dir', type=str, default='beta_all', help='directory to save finetuned beta model (majority)')
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
    Parameters:
        param pep: peptide序列
        param tcr_data: TCR及其标签列表
        param peptide_encoding_dict: 预计算的peptide编码字典 {peptide序列: 编码tensor}
        param tcr_encoding_dict: 预计算的TCR编码字典 {TCR序列: 编码tensor}
    Returns:
        返回peptide embedding，support set embedding，support set labels和query set embedding
        如果support set为空，对应的support_x和support_y返回None
    """
    spt_TCRs = tcr_data[0]
    ypt = tcr_data[1]
    peptides = torch.FloatTensor(1, 75)
    
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    # 从peptide字典获取或计算peptide编码
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15))

    # 检查support set是否为空
    if not spt_TCRs or not ypt:
        peptides[0] = peptide_embedding.flatten()
        
        # 处理query set
        if len(tcr_data) > 2:
            tcr_embeddings = []
            for tcr in qry_TCRs:
                if tcr_encoding_dict and tcr in tcr_encoding_dict:
                    tcr_embed = tcr_encoding_dict[tcr]
                else:
                    tcr_embed = add_position_encoding(aamapping(tcr, 25))
                tcr_embeddings.append(tcr_embed)
            
            tcr_embeddings = torch.stack(tcr_embeddings)
            peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
            temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
            query_x[0] = temp
        else:
            query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
            
        return peptides, None, None, query_x

    # 如果support set不为空，继续处理
    support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
    support_y = np.zeros((1, len(ypt)), dtype=np.int64)
    
    # 处理support set
    temp = []
    for tcr in spt_TCRs:
        if tcr_encoding_dict and tcr in tcr_encoding_dict:
            tcr_embed = tcr_encoding_dict[tcr]
        else:
            tcr_embed = add_position_encoding(aamapping(tcr, 25))
        temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
    
    support_x[0] = torch.cat(temp)
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()

    # 处理query set
    if len(tcr_data) > 2:
        tcr_embeddings = []
        for tcr in qry_TCRs:
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25))
            tcr_embeddings.append(tcr_embed)
        
        tcr_embeddings = torch.stack(tcr_embeddings)
        peptide_expanded = peptide_embedding.unsqueeze(0).expand(len(qry_TCRs), -1, -1)
        temp = torch.cat([peptide_expanded, tcr_embeddings], dim=1)
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)

    return peptides, support_x, torch.LongTensor(support_y), query_x



def task_embedding_majority(pep, tcr_data, peptide_encoding_dict=None, tcr_encoding_dict=None):
    """
    tcr_data: [train_alpha, train_label, val_alpha, val_label, test_alpha, test_index]
    Returns: Epitope_embedding, x_spt, y_spt, x_val, y_val, x_qry
    """
    # ============ peptide embedding ============
    if peptide_encoding_dict and pep in peptide_encoding_dict:
        peptide_embedding = peptide_encoding_dict[pep]
    else:
        peptide_embedding = add_position_encoding(aamapping(pep, 15))
    peptides = torch.FloatTensor(1, 75)
    peptides[0] = peptide_embedding.flatten()

    train_TCRs = tcr_data[0]
    train_labels = tcr_data[1]
    if train_TCRs and train_labels:
        x_spt = torch.FloatTensor(1, len(train_TCRs), 40, 5)
        temp = []
        for tcr in train_TCRs:
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25))
            temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
        x_spt[0] = torch.cat(temp)
        y_spt = torch.LongTensor(np.array([train_labels], dtype=np.int64))
    else:
        x_spt = None
        y_spt = None

    val_TCRs = tcr_data[2]
    val_labels = tcr_data[3]
    if val_TCRs and val_labels:
        x_val = torch.FloatTensor(1, len(val_TCRs), 40, 5)
        temp = []
        for tcr in val_TCRs:
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25))
            temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
        x_val[0] = torch.cat(temp)
        y_val = torch.LongTensor(np.array([val_labels], dtype=np.int64))
    else:
        x_val = None
        y_val = None

    test_TCRs = tcr_data[4]
    if test_TCRs:
        x_qry = torch.FloatTensor(1, len(test_TCRs), 40, 5)
        temp = []
        for tcr in test_TCRs:
            if tcr_encoding_dict and tcr in tcr_encoding_dict:
                tcr_embed = tcr_encoding_dict[tcr]
            else:
                tcr_embed = add_position_encoding(aamapping(tcr, 25))
            temp.append(torch.cat([peptide_embedding, tcr_embed]).unsqueeze(0))
        x_qry[0] = torch.cat(temp)
    else:
        x_qry = None

    return peptides, x_spt, y_spt, x_val, y_val, x_qry

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
    if 'label' in data.columns:
        original_labels = data['label']
    else:
        original_labels = pd.Series([0]*len(data))

    if 'set' in data.columns:
        sets = data['set']
    else:
        sets = pd.Series([""]*len(data))
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
            'Label': output_labels,
            'score_alpha': ends_alpha,
            'score_beta': ends_beta,
            'Score': avg_scores
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
            'Label': output_labels,
            'score_alpha': starts_alpha,
            'score_beta': starts_beta,
            'Score': avg_scores
        })
        output.to_csv(output_file, index=False)
    
    elif learning_setting == 'majority':
        G_data_alpha = {}
        G_data_beta = {}

        for i, j in enumerate(Epitopes):
            if j not in G_data_alpha:
                G_data_alpha[j] = [[], [], [], [], [], []]
                G_data_beta[j] = [[], [], [], [], [], []]
            if sets[i] == 'train':
                G_data_alpha[j][0].append(alpha_tcrs[i])
                G_data_alpha[j][1].append(labels[i])
                G_data_beta[j][0].append(beta_tcrs[i])
                G_data_beta[j][1].append(labels[i])
            elif sets[i] == 'val':
                G_data_alpha[j][2].append(alpha_tcrs[i])
                G_data_alpha[j][3].append(labels[i])
                G_data_beta[j][2].append(beta_tcrs[i])
                G_data_beta[j][3].append(labels[i])
            elif sets[i] == 'test':
                G_data_alpha[j][4].append(alpha_tcrs[i])
                G_data_beta[j][4].append(beta_tcrs[i])
                G_data_alpha[j][5].append(i)
                G_data_beta[j][5].append(i)

        output_peps, output_alpha_tcrs, output_beta_tcrs, output_indices = [], [], [], []
        starts_alpha, starts_beta = [], []

        for epitope in G_data_alpha:
            if not G_data_alpha[epitope][4]:
                continue

            Epitope_embedding_alpha, x_spt_alpha, y_spt_alpha, x_val_alpha, y_val_alpha, x_qry_alpha = \
                task_embedding_majority(epitope, G_data_alpha[epitope], peptide_encoding_dict, tcr_encoding_dict)
            Epitope_embedding_beta, x_spt_beta, y_spt_beta, x_val_beta, y_val_beta, x_qry_beta = \
                task_embedding_majority(epitope, G_data_beta[epitope], peptide_encoding_dict, tcr_encoding_dict)

            save_alpha_dir = os.path.join(args.save_alpha_dir, epitope)
            save_beta_dir  = os.path.join(args.save_beta_dir, epitope)
            os.makedirs(save_alpha_dir, exist_ok=True)
            os.makedirs(save_beta_dir, exist_ok=True)

            model_alpha.finetunning_majority(
                Epitope_embedding_alpha[0].to(device),
                x_spt_alpha[0].to(device),
                y_spt_alpha[0].to(device),
                x_val_alpha[0].to(device),
                y_val_alpha[0].to(device),
                x_qry_alpha[0].to(device),
                save_dir=save_alpha_dir
            )
            model_beta.finetunning_majority(
                Epitope_embedding_beta[0].to(device),
                x_spt_beta[0].to(device),
                y_spt_beta[0].to(device),
                x_val_beta[0].to(device),
                y_val_beta[0].to(device),
                x_qry_beta[0].to(device),
                save_dir=save_beta_dir
            )

            alpha_csv = os.path.join(save_alpha_dir, "step_final_trainval_loss.csv")
            beta_csv  = os.path.join(save_beta_dir,  "step_final_trainval_loss.csv")
            df_alpha = pd.read_csv(alpha_csv)
            df_beta  = pd.read_csv(beta_csv)

            if "step" in df_alpha.columns and "step" in df_beta.columns:
                df_alpha = df_alpha.set_index("step")
                df_beta = df_beta.set_index("step")
            else:
                df_alpha.index = range(len(df_alpha))
                df_beta.index = range(len(df_beta))

            merged_avg = pd.DataFrame(index=df_alpha.index)
            for col in df_alpha.columns:
                if col in df_beta.columns:
                    merged_avg[col] = (df_alpha[col] + df_beta[col]) / 2
            merged_avg = merged_avg.reset_index()
            if "step" in merged_avg.columns:
                merged_avg = merged_avg.rename(columns={"index": "step"})
            best_idx = merged_avg['val_loss'].idxmin()
            best_step = int(merged_avg.loc[best_idx, 'step'])

            pt_file_alpha = os.path.join(save_alpha_dir, f"step_{best_step}_finetuned_state_dict.pt")
            pt_file_beta  = os.path.join(save_beta_dir,  f"step_{best_step}_finetuned_state_dict.pt")
            model_alpha.net.load_state_dict(torch.load(pt_file_alpha, map_location=device))
            model_beta.net.load_state_dict(torch.load(pt_file_beta,  map_location=device))

            start_alpha = model_alpha.inference_with_params(x_qry_alpha[0].to(device), model_alpha.net)
            start_beta  = model_beta.inference_with_params(x_qry_beta[0].to(device), model_beta.net)

            starts_alpha += list(start_alpha[0])
            starts_beta  += list(start_beta[0])

            query_indices = G_data_alpha[epitope][5]
            num = len(query_indices)
            output_peps += [epitope] * num
            output_alpha_tcrs += G_data_alpha[epitope][4]
            output_beta_tcrs  += G_data_beta[epitope][4]
            output_indices += query_indices

            torch.cuda.empty_cache()

        avg_scores = [(a + b) / 2 for a, b in zip(starts_alpha, starts_beta)]
        output_labels = [labels[idx] for idx in output_indices]
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'Label': output_labels,
            'score_alpha': starts_alpha,
            'score_beta': starts_beta,
            'Score': avg_scores
        })
        output.to_csv(output_file, index=False)
        print(f"[INFO] Final (meta-learning) query output saved: {output_file}")




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
        print(starts_alpha)
        print(starts_beta)
        # Calculate average scores
        avg_scores = [(a + b) / 2 for a, b in zip(starts_alpha, starts_beta)]
        
        # Get the original labels for the output
        output_labels = [original_labels[idx] for idx in output_indices]
        
        output = pd.DataFrame({
            'Epitope': output_peps,
            'alpha': output_alpha_tcrs,
            'beta': output_beta_tcrs,
            'Label': output_labels,
            'score_alpha': starts_alpha,
            'score_beta': starts_beta,
            'Score': avg_scores
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