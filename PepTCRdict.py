import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib

from collections import OrderedDict

from utils import RemovableHandle


class PepTCRdict(Dataset):
    def __init__(self, PepTCRdictFile, HealthyTCRFile, k_shot, k_query, aa_dict_path, encode_dim=25, mode='train', k=5):
        self._hooks = None
        self.all_tasks = {}  # keys: peptide, values: [all positive, all negative]
        # load the known binding TCR set
        tmp = pd.read_csv(PepTCRdictFile)
        self.PepTCRdict = {}
        for idx, pep in enumerate(tmp['peptide']):
            if pep not in self.PepTCRdict:
                self.PepTCRdict[pep] = []
            self.PepTCRdict[pep].append(tmp['binding_TCR'][idx])
        del tmp
        # load binding dataset from *.pkl file directly
        # self.PepTCRdict = joblib.load(PepTCRdictFile)

        # load the control TCR set
        self.HealthyTCR = np.loadtxt(HealthyTCRFile, dtype=str)

        # set the number of instances of each class in support set
        self.k_shot = k_shot

        # set the number of instances of each class in query set
        self.k_query = k_query

        # set the embedding dimension of each TCR
        self.encode_dim = encode_dim  # 每一个TCR对应的embedding维度

        # load the atchley_factor matrix
        self.aa_dict = joblib.load(aa_dict_path)
        # filtering the peptides based on the known binding TCRs
        FilteredDict = {}
        for i in self.PepTCRdict:
            if len(self.PepTCRdict[i]) >= k_shot + k_query:
                FilteredDict[i] = self.PepTCRdict[i]
        print(f"There are {len(FilteredDict)} peptides can be used for training")

        # convert the data format
        TCRsList = []
        for i in FilteredDict:
            TCRsList.append(FilteredDict[i] + [i])

        tmp_dict = {}
        for i in TCRsList:
            tmp_dict[i[-1]] = i[:-1]
        self.PepTCRdict = tmp_dict

        # set the position encoding
        self.position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)

        # self.path = paths

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
            print('Length: '+str(len(TCRSeq))+' over bound!')
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

    def register_hook(self, hook):
        if self._hooks is None:
            self._hooks = OrderedDict()
        handle = RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook

    def set_all_tasks(self, tasks):
        self.all_tasks = tasks

    def __getitem__(self, item):

        # initialize four tensors for support set, support labels, query set and query labels
        support_x = torch.FloatTensor(self.k_shot * 2, self.encode_dim + 15, 5)
        support_y = np.zeros((self.k_shot * 2), dtype=np.int)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int)

        # extract the TCRs based on the item
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs = self.PepTCRdict[peptide]
        np.random.shuffle(TCRs)

        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # select the control TCRs based on the size of k_shot
        selected_res_TCRs = TCRs[:self.k_shot]
        selected_heal_idx = np.random.choice(len(self.HealthyTCR), self.k_shot)
        selected_heal_TCRs = self.HealthyTCR[selected_heal_idx]

        # embed TCRs based on the atchley factor and position encoding
        for i, seq in enumerate(selected_res_TCRs):
            support_x[i] = torch.cat([peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat([peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # select the TCRs for the query set
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]
        selected_heal_idx_query = np.random.choice(len(self.HealthyTCR), self.k_query)
        selected_heal_TCRs_query = self.HealthyTCR[selected_heal_idx_query]
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat([peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat([peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

            # flatten the peptide encoding used for embedding the task
        peptide_embedding = peptide_embedding.flatten()

        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query, selected_heal_TCRs, selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)
        return peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}

# TestData = PepTCRdict("/home/gaoyicheng/pep_tcr_with_gyl/TCRBagger/Requirements/dict_pepTcr_Result.pkl",\
#     "/home/gaoyicheng/pep_tcr_with_gyl/TCRBagger/HealthyTCR/FilteredOutput.txt",1,1,mode = 'train',k=5)
