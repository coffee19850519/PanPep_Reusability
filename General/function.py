import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from collections import Counter
import os
import yaml
# from general import device
# # Set the 'cuda'
# device = torch.device('cuda')
# Load the Atchley_factors for encoding the amino acid
# aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")


# def aamapping(TCRSeq, encode_dim):
#     """
#     this function is used for encoding the TCR sequence
#
#     Parameters:
#         param TCRSeq: the TCR original sequence
#         param encode_dim: the first dimension of TCR sequence embedding matrix
#
#     Returns:
#         this function returns a TCR embedding matrix;
#         e.g. the TCR sequence of ASSSAA
#         return: (6 + encode_dim - 6) x 5 embedding matrix, in which (encode_dim - 6) x 5 will be zero matrix
#
#     Raises:
#         KeyError - using 0 vector for replacing the original amino acid encoding
#     """
#
#     TCRArray = []
#     if len(TCRSeq) > encode_dim:
#         print('Length: ' + str(len(TCRSeq)) + ' over bound!')
#         TCRSeq = TCRSeq[0:encode_dim]
#     for aa_single in TCRSeq:
#         try:
#             TCRArray.append(aa_dict[aa_single])
#         except KeyError:
#             TCRArray.append(np.zeros(5, dtype='float64'))
#     for i in range(0, encode_dim - len(TCRSeq)):
#         TCRArray.append(np.zeros(5, dtype='float64'))
#     return torch.FloatTensor(TCRArray)
#
#
# # Sinusoidal position encoding
# position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
# position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
# position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
# position_encoding = torch.from_numpy(position_encoding)
#
#
# def add_position_encoding(seq):
#     """
#     this function is used to add position encoding for the TCR embedding
#
#     Parameters:
#         param seq: the TCR embedding matrix
#
#     Returns:
#         this function returns a TCR embedding matrix containing position encoding
#     """
#
#     padding_ids = torch.abs(seq).sum(dim=-1) == 0
#     seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
#     return seq
#
#
# def task_embedding(pep, tcr_data):
#     """
#     this function is used to obtain the task-level embedding
#
#     Parameters:
#         param pep: peptide sequence
#         param tcr_data: TCR and its label list in a pan-pep way;
#         e.g. [[support TCRs],[support labels]] or [[support TCRs],[support labels],[query TCRs]]
#
#     Returns:
#         this function returns a peptide embedding, the embedding of support set, the labels of support set and the embedding of query set
#     """
#
#     # 默认每个 Peptide 都具有有标签的 CDR3
#     # Obtain the TCRs of support set
#     spt_TCRs = tcr_data[0]  # 有标签的CDR3名称
#
#     # Obtain the TCR labels of support set
#     ypt = tcr_data[1]  # 标签
#
#     # Initialize the size of the Tensor for the support set and labels
#     support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
#     support_y = np.zeros((1, len(ypt)), dtype=np.int)
#     peptides = torch.FloatTensor(1, 75)
#
#     # Determine whether there is a query set based on the length of input param2
#     if len(tcr_data) > 2:
#         qry_TCRs = tcr_data[2]  # 没标签的CDR3名称，组成 query
#     else:
#         qry_TCRs = ['None']
#
#     # Initialize the size of the Tensor for the query set
#     query_x = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
#
#     # Encoding for the peptide sequence
#     peptide_embedding = add_position_encoding(aamapping(pep, 15))
#
#     # Put the embedding of support set, labels and peptide embedding into the initialized tensor
#     temp = torch.Tensor()
#     for j in spt_TCRs:
#         temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
#     support_x[0] = temp
#     support_y[0] = np.array(ypt)
#     peptides[0] = peptide_embedding.flatten()
#
#     # Put the embedding of query set into the initialized tensor
#     temp = torch.Tensor()
#     if len(tcr_data) > 2:
#         for j in qry_TCRs:
#             temp = torch.cat(
#                 [temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
#         query_x[0] = temp
#     else:
#         query_x[0] = torch.FloatTensor(1, len(qry_TCRs), 25 + 15, 5)
#
#     return peptides, support_x, torch.LongTensor(support_y), query_x
#
#
# def labeled_embedding(pep, tcr_data):
#     """
#     this function is used to obtain the task-level embedding
#
#     Parameters:
#         param pep: peptide sequence
#         param tcr_data: TCR and its label list in a pan-pep way;
#         e.g. [[support TCRs],[support labels]] or [[support TCRs],[support labels],[query TCRs]]
#
#     Returns:
#         this function returns a peptide embedding, the embedding of support set, the labels of support set and the embedding of query set
#     """
#
#     # 默认每个 Peptide 都具有有标签的 CDR3
#     # Obtain the TCRs of support set
#     spt_TCRs = tcr_data[0]  # 有标签的CDR3名称
#     # Obtain the TCR labels of support set
#     ypt = tcr_data[1]  # 标签
#     # Initialize the size of the Tensor for the support set and labels
#     support_x = torch.FloatTensor(1, len(spt_TCRs), 25 + 15, 5)
#     support_y = np.zeros((1, len(ypt)), dtype=np.int)
#     peptides = torch.FloatTensor(1, 75)
#     # Encoding for the peptide sequence
#     peptide_embedding = add_position_encoding(aamapping(pep, 15))
#     # Put the embedding of support set, labels and peptide embedding into the initialized tensor
#     temp = torch.Tensor()
#     for j in spt_TCRs:
#         temp = torch.cat([temp, torch.cat([peptide_embedding, add_position_encoding(aamapping(j, 25))]).unsqueeze(0)])
#     support_x[0] = temp
#     support_y[0] = np.array(ypt)
#     peptides[0] = peptide_embedding.flatten()
#     return peptides, support_x, torch.LongTensor(support_y)


class AvgMeter:  # 保存loss相关的类
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class PepTCRdict(torch.utils.data.Dataset):
    def __init__(self, PepTCRdictFile, aa_dict_path, encode_dim=25):
        self.PepTCRdict = PepTCRdictFile
        self.encode_dim = encode_dim  # 每一个TCR对应的embedding维度

        # load the atchley_factor matrix
        self.aa_dict = joblib.load(aa_dict_path)
        self.position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)

    def aamapping(self, TCRSeq, encode_dim):
        """
        this function is used for encoding the TCR sequence

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
        if len(TCRSeq) > encode_dim:
            # print('Length: '+str(len(TCRSeq))+' over bound!')
            TCRSeq = TCRSeq[0:encode_dim]
        for aa_single in TCRSeq:
            try:
                TCRArray.append(self.aa_dict[aa_single])
            except KeyError:
                print('Not proper aaSeqs: ' + TCRSeq)
                TCRArray.append(np.zeros(5, dtype='float64'))
        for i in range(0, encode_dim - len(TCRSeq)):
            TCRArray.append(np.zeros(5, dtype='float64'))
        return torch.FloatTensor(np.array(TCRArray))

    def add_position_encoding(self, seq):
        """
        this function is used to add position encoding for the TCR embedding

        Parameters:
            param seq: the TCR embedding matrix

        Returns:
            this function returns a TCR embedding matrix containing position encoding
        """
        padding_ids = torch.abs(seq).sum(dim=-1) == 0
        seq[~padding_ids] += self.position_encoding[:seq[~padding_ids].size()[-2]]
        return seq

    def __getitem__(self, item):
        # extract the TCRs based on the item
        peptide, TCRs, label = self.PepTCRdict.iloc[item]

        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        support_x = torch.cat([peptide_embedding, self.add_position_encoding(self.aamapping(TCRs, self.encode_dim))])
        support_y = label

        peptide_embedding = peptide_embedding.flatten()

        return peptide_embedding, support_x, torch.LongTensor([support_y])

    def __len__(self):
        return len(self.PepTCRdict)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def split_data(positive_data, negative_data, new_data_name='all_data.csv'):
    '''
    将正、负样本加入新的csv中（个数与正样本一致）
    Args:
        positive_data:
        negative_data:
        new_data_name:

    Returns:

    '''
    positive_data = pd.read_csv(positive_data)
    negative_data = np.loadtxt(negative_data, dtype=str)
    peptide_ = Counter(positive_data['peptide'])
    negative_ = {}
    for pep, num in peptide_.items():
        negative_[pep] = [[], []]
        negative_[pep][0].extend(positive_data[positive_data['peptide'] == pep]['binding_TCR'].array)
        negative_[pep][1].extend(positive_data[positive_data['peptide'] == pep]['label'].array)
        selected_heal_idx = np.random.choice(len(negative_data), num, replace=False)
        negative_[pep][0].extend(negative_data[selected_heal_idx])
        negative_[pep][1].extend([0] * num)
    all_data_dict = {'peptide': [], 'binding_TCR': [], 'label': []}
    for key, val in negative_.items():
        all_data_dict['peptide'].extend([key] * len(val[0]))
        all_data_dict['binding_TCR'].extend(val[0])
        all_data_dict['label'].extend(val[1])
    all_data_dict = pd.DataFrame(all_data_dict)
    all_data_dict.to_csv(new_data_name, index=False)


def data_dict(data):
    data_peptides = data['peptide']
    data_TCRs = data['binding_TCR']
    data_labels = data['label']

    data_loader = {}
    for i, j in enumerate(list(data_peptides.axes[0])):

        if data_peptides[j] not in data_loader:
            data_loader[data_peptides[j]] = [[], [], [], []]
        # Labeled data 有标签的数据
        if data_labels[j] != 'Unknown':
            data_loader[data_peptides[j]][0].append(data_TCRs[j])
            data_loader[data_peptides[j]][1].append(data_labels[j])
    return data_loader


# load the cofig file
config_file_path = os.path.join(os.path.abspath('').split(__file__.split(os.sep)[-2])[0], 'Configs', 'TrainingConfig.yaml')
data_config = load_config(config_file_path)
project_path = eval(data_config['Project_path']).split(__file__.split(os.sep)[-2])[0]
device = data_config['Train']['General']['device']


def train_epoch(model, train_loader, optimizer):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))  # 显示进度条
    for step, (peptide_embedding, x_spt, y_spt) in enumerate(tqdm_object):
        setsz, h, w = x_spt.size()
        peptide_embedding, x_spt, y_spt = peptide_embedding.to(device), x_spt.to(device), y_spt.flatten().to(device)

        loss = model(x_spt, y_spt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), setsz)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))  # 显示进度条
    for step, (peptide_embedding, x_spt, y_spt) in enumerate(tqdm_object):
        setsz, h, w = x_spt.size()
        peptide_embedding, x_spt, y_spt = peptide_embedding.to(device), x_spt.to(device), y_spt.flatten().to(device)
        loss = model(x_spt, y_spt)
        loss_meter.update(loss.item(), setsz)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter
