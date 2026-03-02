import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib

from collections import OrderedDict, Counter

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
        # print(f"There are {len(self.PepTCRdict)} peptides can be used for training")
        # load binding dataset from *.pkl file directly
        # self.PepTCRdict = joblib.load(PepTCRdictFile)

        # load the control TCR set using fast method
        print(f"Loading background library from {HealthyTCRFile}...")
        with open(HealthyTCRFile, 'r') as f:
            self.HealthyTCR = np.array([line.strip() for line in f])
        print(f"Loaded {len(self.HealthyTCR)} TCRs from background library")

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

        # Pre-compute positive TCR sets for each peptide (for deduplication)
        print("Pre-computing positive TCR sets for deduplication...")
        self.positive_sets = {}
        for peptide, tcrs in self.PepTCRdict.items():
            self.positive_sets[peptide] = set(tcrs)
        print(f"Pre-computed {len(self.positive_sets)} positive sets")

        # set the position encoding
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)

        # self.path = paths

    def _sample_negatives_with_dedup(self, positive_set, n_needed, exclude_set=None):
        """Sample negative TCRs with deduplication.

        Args:
            positive_set: Set of positive TCRs to exclude
            n_needed: Number of negative samples needed
            exclude_set: Optional set of TCRs to exclude (e.g., already sampled)

        Returns:
            Array of sampled negative TCRs
        """
        if exclude_set is None:
            exclude_set = set()

        selected_tcrs = []
        selected_set = set()

        max_attempts = n_needed * 10
        attempts = 0

        while len(selected_tcrs) < n_needed and attempts < max_attempts:
            batch_size = min((n_needed - len(selected_tcrs)) * 3, 1000)
            sampled_indices = np.random.choice(len(self.HealthyTCR), batch_size, replace=True)

            for idx in sampled_indices:
                tcr = self.HealthyTCR[idx]
                if tcr not in positive_set and tcr not in exclude_set and tcr not in selected_set:
                    selected_tcrs.append(tcr)
                    selected_set.add(tcr)
                    if len(selected_tcrs) >= n_needed:
                        break
            attempts += batch_size

        return np.array(selected_tcrs)

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
            print('Length: ' + str(len(TCRSeq)) + ' over bound!')
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
        support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int64)

        # extract the TCRs based on the item
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs = self.PepTCRdict[peptide]
        np.random.shuffle(TCRs)

        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # Get pre-computed positive set for this peptide
        positive_set = self.positive_sets[peptide]

        # select the control TCRs based on the size of k_shot with deduplication
        selected_res_TCRs = TCRs[:self.k_shot]
        selected_heal_TCRs = self._sample_negatives_with_dedup(positive_set, self.k_shot)

        # embed TCRs based on the atchley factor and position encoding
        for i, seq in enumerate(selected_res_TCRs):
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # select the TCRs for the query set with deduplication
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]
        exclude_set = set(selected_heal_TCRs)
        selected_heal_TCRs_query = self._sample_negatives_with_dedup(positive_set, self.k_query, exclude_set)
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

            # flatten the peptide encoding used for embedding the task
        peptide_embedding = peptide_embedding.flatten()

        # Prepare sequence lists for support and query sets (in same order as embeddings)
        support_tcr_seqs = list(selected_res_TCRs) + list(selected_heal_TCRs)
        query_tcr_seqs = list(selected_res_TCRs_query) + list(selected_heal_TCRs_query)

        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query, selected_heal_TCRs,
                                 selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)
        return (peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y),
                peptide, support_tcr_seqs, query_tcr_seqs)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}

class PepTCRdict_old(Dataset):
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
        # print(f"There are {len(self.PepTCRdict)} peptides can be used for training")
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
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
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
            print('Length: ' + str(len(TCRSeq)) + ' over bound!')
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
        support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int64)

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
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # select the TCRs for the query set
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]
        selected_heal_idx_query = np.random.choice(len(self.HealthyTCR), self.k_query)
        selected_heal_TCRs_query = self.HealthyTCR[selected_heal_idx_query]
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

            # flatten the peptide encoding used for embedding the task
        peptide_embedding = peptide_embedding.flatten()
        support_tcr_seqs = list(selected_res_TCRs) + list(selected_heal_TCRs)
        query_tcr_seqs = list(selected_res_TCRs_query) + list(selected_heal_TCRs_query)
        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query, selected_heal_TCRs,
                                 selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)
        return (peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y),
                peptide, support_tcr_seqs, query_tcr_seqs)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}



# TestData = PepTCRdict("/home/gaoyicheng/pep_tcr_with_gyl/TCRBagger/Requirements/dict_pepTcr_Result.pkl",\
#     "/home/gaoyicheng/pep_tcr_with_gyl/TCRBagger/HealthyTCR/FilteredOutput.txt",1,1,mode = 'train',k=5)

class PepTCRdict_new(Dataset):
    """
    The class is used for reshuffling Negative samples adding strategy
    """
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

        tmp = pd.read_csv(HealthyTCRFile)
        self.HealthyTCRs = {}
        for idx, pep in enumerate(tmp['peptide']):
            if pep not in self.HealthyTCRs:
                self.HealthyTCRs[pep] = []
            self.HealthyTCRs[pep].append(tmp['binding_TCR'][idx])
        del tmp

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

        # Negative samples
        FilteredDict = {}
        for i in self.HealthyTCRs:
            if len(self.HealthyTCRs[i]) >= k_shot + k_query:
                FilteredDict[i] = self.HealthyTCRs[i]

        # convert the data format
        TCRsList = []
        for i in FilteredDict:
            TCRsList.append(FilteredDict[i] + [i])

        tmp_dict = {}
        for i in TCRsList:
            tmp_dict[i[-1]] = i[:-1]
        self.HealthyTCRs = tmp_dict

        # set the position encoding
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
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
            print('Length: ' + str(len(TCRSeq)) + ' over bound!')
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
        support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int64)

        # extract the TCRs based on the item
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs = self.PepTCRdict[peptide]
        np.random.shuffle(TCRs)

        Negative_TCRs = self.HealthyTCRs[peptide]
        np.random.shuffle(Negative_TCRs)

        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # select the control TCRs based on the size of k_shot
        selected_res_TCRs = TCRs[:self.k_shot]
        selected_heal_TCRs = Negative_TCRs[:self.k_shot]

        # embed TCRs based on the atchley factor and position encoding
        for i, seq in enumerate(selected_res_TCRs):
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # select the TCRs for the query set
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]
        selected_heal_TCRs_query = Negative_TCRs[self.k_shot:self.k_shot + self.k_query]
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

            # flatten the peptide encoding used for embedding the task
        peptide_embedding = peptide_embedding.flatten()

        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query, selected_heal_TCRs,
                                 selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)
        return peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}


class PepTCRdict_ranking(Dataset):
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
        # print(f"There are {len(self.PepTCRdict)} peptides can be used for training")
        # load binding dataset from *.pkl file directly
        # self.PepTCRdict = joblib.load(PepTCRdictFile)

        # load the control TCR set
        self.HealthyTCR = list(pd.read_csv(HealthyTCRFile)['binding_TCR'])

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
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
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
            print('Length: ' + str(len(TCRSeq)) + ' over bound!')
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
        support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int64)

        # extract the TCRs based on the item
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs = self.PepTCRdict[peptide]
        np.random.shuffle(TCRs)

        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # select the control TCRs based on the size of k_shot
        selected_res_TCRs = TCRs[:self.k_shot]
        HealthyTCRs = list(set(self.HealthyTCR).difference(set(TCRs)))
        np.random.shuffle(HealthyTCRs)
        selected_heal_TCRs = HealthyTCRs[:self.k_shot]

        # embed TCRs based on the atchley factor and position encoding
        for i, seq in enumerate(selected_res_TCRs):
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # select the TCRs for the query set
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]
        selected_heal_TCRs_query = HealthyTCRs[self.k_shot:self.k_shot + self.k_query]
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

            # flatten the peptide encoding used for embedding the task
        peptide_embedding = peptide_embedding.flatten()

        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query, selected_heal_TCRs,
                                 selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)
        return peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}



class PepTCRdict4using_save_data(torch.utils.data.Dataset):
    def __init__(self, PepTCRdictFile, aa_dict_path, k_shot=2, k_query=3, encode_dim=25):
        self.PepTCRdict = PepTCRdictFile
        # set the number of instances of each class in support set
        self.k_shot = k_shot
        # set the number of instances of each class in query set
        self.k_query = k_query
        self.encode_dim = encode_dim  # 每一个TCR对应的embedding维度

        # load the atchley_factor matrix
        self.aa_dict = joblib.load(aa_dict_path)
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
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

    def set_all_tasks(self, tasks):
        self.all_tasks = tasks

    def __getitem__(self, item):
        # initialize four tensors for support set, support labels, query set and query labels
        support_x = torch.FloatTensor(self.k_shot * 2, self.encode_dim + 15, 5)
        # support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        # query_y = np.zeros((self.k_query * 2), dtype=np.int64)
        # extract the TCRs based on the item
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs, label = self.PepTCRdict[peptide]
        # peptide atchley factor embedding added with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # embed TCRs based on the atchley factor and position encoding
        for i, seq in enumerate(TCRs[: self.k_shot * 2]):
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
        support_y = np.array(label[: self.k_shot * 2], dtype=np.int64)
        for i, seq in enumerate(TCRs[self.k_shot * 2:]):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
        query_y = np.array(label[self.k_shot * 2:], dtype=np.int64)

        # support_x = torch.cat(
        #     [peptide_embedding, self.add_position_encoding(self.aamapping(TCRs[: self.k_shot * 2], self.encode_dim))])
        # support_y = label[: self.k_shot]
        #
        # query_x = torch.cat(
        #     [peptide_embedding, self.add_position_encoding(self.aamapping(TCRs[self.k_shot * 2:], self.encode_dim))])
        # query_y = label[self.k_shot:]

        peptide_embedding = peptide_embedding.flatten()

        return peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        return len(self.PepTCRdict)


def accord_epoch_get_few_data(few_data_path, train_pep, epoch, support=2, query=3, train=True):
    if type(few_data_path) == pd.DataFrame:
        few_data = few_data_path
    else:
        few_data = pd.read_csv(few_data_path)
    peptide = list(few_data.columns)
    negative_ = {}
    num = 0
    for pep in peptide:
        if train:
            if pep in train_pep:
                num += 1
                negative_[pep] = [[], []]
                # 正样本的support集
                pst_spt = few_data[pep][2 * (support + query) * epoch: 2 * (support + query) * epoch + support]
                negative_[pep][0].extend(pst_spt)
                negative_[pep][1].extend([1] * len(pst_spt))
                # 负样本的support集
                ngt_spt = few_data[pep][
                          2 * (support + query) * epoch + (support + query): 2 * (support + query) * epoch + (
                                  support + query) + support]
                negative_[pep][0].extend(ngt_spt)
                negative_[pep][1].extend([0] * len(ngt_spt))
                # 正样本的query集
                pst_query = few_data[pep][
                            2 * (support + query) * epoch + support: 2 * (support + query) * epoch + (support + query)]
                negative_[pep][0].extend(pst_query)
                negative_[pep][1].extend([1] * len(pst_query))
                # 负样本的query集
                ngt_query = few_data[pep][2 * (support + query) * epoch + (support + query) + support: 2 * (
                        support + query) * epoch + 2 * (support + query)]
                negative_[pep][0].extend(ngt_query)
                negative_[pep][1].extend([0] * len(ngt_query))
        else:
            if pep not in train_pep:
                num += 1
                negative_[pep] = [[], []]
                # 正样本的support集
                pst_spt = few_data[pep][2 * (support + query) * epoch: 2 * (support + query) * epoch + support]
                negative_[pep][0].extend(pst_spt)
                negative_[pep][1].extend([1] * len(pst_spt))
                # 负样本的support集
                ngt_spt = few_data[pep][
                          2 * (support + query) * epoch + (support + query): 2 * (support + query) * epoch + (
                                  support + query) + support]
                negative_[pep][0].extend(ngt_spt)
                negative_[pep][1].extend([0] * len(ngt_spt))
                # 正样本的query集
                pst_query = few_data[pep][
                            2 * (support + query) * epoch + support: 2 * (support + query) * epoch + (support + query)]
                negative_[pep][0].extend(pst_query)
                negative_[pep][1].extend([1] * len(pst_query))
                # 负样本的query集
                ngt_query = few_data[pep][2 * (support + query) * epoch + (support + query) + support: 2 * (
                        support + query) * epoch + 2 * (support + query)]
                negative_[pep][0].extend(ngt_query)
                negative_[pep][1].extend([0] * len(ngt_query))

    all_data_dict = {'peptide': [], 'binding_TCR': [], 'label': []}
    for key, val in negative_.items():
        all_data_dict['peptide'].extend([key] * len(val[0]))
        all_data_dict['binding_TCR'].extend(val[0])
        all_data_dict['label'].extend(val[1])
    all_data_dict = pd.DataFrame(all_data_dict)
    print(num)
    return all_data_dict
