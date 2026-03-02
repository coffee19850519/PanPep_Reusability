"""Alternating sampling strategy for TCR-peptide binding prediction.

This module implements a sampling strategy that alternates between two
background TCR libraries based on epoch parity (odd/even).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
from collections import OrderedDict
from utils import RemovableHandle


class PepTCRdict_alternating_source(Dataset):
    """Dataset class with alternating source sampling strategy.

    Odd epochs: Use background_draw library
    Even epochs: Use reshuffling library

    Args:
        PepTCRdictFile: Path to peptide-TCR binding data
        background_draw_file: Path to background-draw TCR library
        reshuffling_file: Path to reshuffling TCR library
        k_shot: Number of positive/negative samples per class in support set
        k_query: Number of positive/negative samples per class in query set
        aa_dict_path: Path to Atchley factors dictionary
        encode_dim: Embedding dimension for TCR sequences
        mode: Training mode
        k: Parameter for filtering
    """

    def __init__(self, PepTCRdictFile, background_draw_file, reshuffling_file,
                 k_shot, k_query, aa_dict_path, encode_dim=25, mode='train', k=5):
        self._hooks = None
        self.all_tasks = {}
        self.current_epoch = 1  # Default to epoch 1 (odd - use Library A)

        # Load the known binding TCR set
        tmp = pd.read_csv(PepTCRdictFile)
        self.PepTCRdict = {}
        for idx, pep in enumerate(tmp['peptide']):
            if pep not in self.PepTCRdict:
                self.PepTCRdict[pep] = []
            self.PepTCRdict[pep].append(tmp['binding_TCR'][idx])
        del tmp

        # Load both background TCR libraries using fast method
        print(f"Loading background_draw library from {background_draw_file}...")
        with open(background_draw_file, 'r') as f:
            self.background_draw = np.array([line.strip() for line in f])
        print(f"Loaded {len(self.background_draw)} TCRs from background_draw")

        print(f"Loading reshuffling library from {reshuffling_file}...")
        with open(reshuffling_file, 'r') as f:
            self.reshuffling = np.array([line.strip() for line in f])
        print(f"Loaded {len(self.reshuffling)} TCRs from reshuffling")

        # Set sampling parameters
        self.k_shot = k_shot
        self.k_query = k_query
        self.encode_dim = encode_dim

        # Load the Atchley factor matrix
        self.aa_dict = joblib.load(aa_dict_path)

        # Filter peptides based on the number of known binding TCRs
        FilteredDict = {}
        for i in self.PepTCRdict:
            if len(self.PepTCRdict[i]) >= k_shot + k_query:
                FilteredDict[i] = self.PepTCRdict[i]
        print(f"There are {len(FilteredDict)} peptides can be used for training")

        # Convert the data format
        TCRsList = []
        for i in FilteredDict:
            TCRsList.append(FilteredDict[i] + [i])

        tmp_dict = {}
        for i in TCRsList:
            tmp_dict[i[-1]] = i[:-1]
        self.PepTCRdict = tmp_dict

        # Set the position encoding
        self.position_encoding = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)

    def _sample_negatives_with_dedup(self, library, positive_set, n_needed, exclude_set=None):
        """Sample negative TCRs with deduplication using set-based filtering.

        Args:
            library: Background TCR library (numpy array)
            positive_set: Set of positive TCRs to exclude
            n_needed: Number of negative samples needed
            exclude_set: Optional set of TCRs to exclude (e.g., already sampled)

        Returns:
            Array of sampled negative TCRs
        """
        if exclude_set is None:
            exclude_set = set()

        selected_tcrs = []
        selected_set = set()  # Track already selected to avoid duplicates

        # Sample with replacement (much faster for large arrays), then deduplicate
        max_attempts = n_needed * 100  # Safety limit
        attempts = 0

        while len(selected_tcrs) < n_needed and attempts < max_attempts:
            # Use replace=True for speed (critical for 57M+ arrays)
            batch_size = min((n_needed - len(selected_tcrs)) * 3, 1000)
            sampled_indices = np.random.choice(len(library), batch_size, replace=True)

            for idx in sampled_indices:
                tcr = library[idx]
                # Check using set (O(1) lookup)
                if tcr not in positive_set and tcr not in exclude_set and tcr not in selected_set:
                    selected_tcrs.append(tcr)
                    selected_set.add(tcr)
                    if len(selected_tcrs) >= n_needed:
                        break
            attempts += batch_size

        return np.array(selected_tcrs)

    def set_epoch(self, epoch):
        """Set the current epoch for alternating sampling.

        Args:
            epoch: Current training epoch (1-indexed)
        """
        self.current_epoch = epoch
        if epoch % 2 == 1:
            print(f"Epoch {epoch} (odd): Using background_draw for negative sampling")
        else:
            print(f"Epoch {epoch} (even): Using reshuffling for negative sampling")

    def _get_current_library(self):
        """Get the current background TCR library based on epoch parity.

        Returns:
            Current background TCR library (background_draw for odd epochs, reshuffling for even epochs)
        """
        if self.current_epoch % 2 == 1:
            return self.background_draw
        else:
            return self.reshuffling

    def _filter_available_TCR(self, tcr_library, positive_tcrs):
        """Filter out positive TCRs from the background library to avoid false negatives.

        Args:
            tcr_library: Background TCR library (numpy array)
            positive_tcrs: List of positive TCRs for current peptide

        Returns:
            Filtered TCR library without positive TCRs
        """
        positive_set = set(positive_tcrs)
        # Use numpy boolean indexing for faster filtering
        mask = np.array([tcr not in positive_set for tcr in tcr_library])
        return tcr_library[mask]

    def aamapping(self, TCRSeq, encode_dim):
        """Encode TCR sequence using Atchley factors.

        Args:
            TCRSeq: Original TCR amino acid sequence
            encode_dim: Target embedding dimension

        Returns:
            TCR embedding matrix of shape (encode_dim, 5)
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
        """Add position encoding to TCR embedding.

        Args:
            seq: TCR embedding matrix

        Returns:
            TCR embedding matrix with position encoding
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
        # Initialize tensors for support and query sets
        support_x = torch.FloatTensor(self.k_shot * 2, self.encode_dim + 15, 5)
        support_y = np.zeros((self.k_shot * 2), dtype=np.int64)
        query_x = torch.FloatTensor(self.k_query * 2, self.encode_dim + 15, 5)
        query_y = np.zeros((self.k_query * 2), dtype=np.int64)

        # Extract TCRs for current peptide
        peptide = list(self.PepTCRdict.keys())[item]
        TCRs = self.PepTCRdict[peptide]
        np.random.shuffle(TCRs)

        # Peptide embedding with position encoding
        peptide_embedding = self.aamapping(peptide, 15)
        peptide_embedding = self.add_position_encoding(peptide_embedding)

        # Get current library based on epoch
        current_library = self._get_current_library()
        positive_set = set(TCRs)

        # === SUPPORT SET ===
        selected_res_TCRs = TCRs[:self.k_shot]

        # Sample negatives with set-based deduplication
        selected_heal_TCRs = self._sample_negatives_with_dedup(
            current_library, positive_set, self.k_shot
        )

        # Embed support set TCRs
        for i, seq in enumerate(selected_res_TCRs):
            support_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs):
            support_x[i + len(selected_res_TCRs)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            support_y[i + len(selected_res_TCRs)] = 0

        # === QUERY SET ===
        selected_res_TCRs_query = TCRs[self.k_shot:self.k_shot + self.k_query]

        # Sample negatives excluding those used in support set
        exclude_set = set(selected_heal_TCRs)
        selected_heal_TCRs_query = self._sample_negatives_with_dedup(
            current_library, positive_set, self.k_query, exclude_set
        )

        # Embed query set TCRs
        for i, seq in enumerate(selected_res_TCRs_query):
            query_x[i] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i] = 1
        for i, seq in enumerate(selected_heal_TCRs_query):
            query_x[i + len(selected_res_TCRs_query)] = torch.cat(
                [peptide_embedding, self.add_position_encoding(self.aamapping(seq, self.encode_dim))])
            query_y[i + len(selected_res_TCRs_query)] = 0

        # Flatten peptide embedding for task embedding
        peptide_embedding = peptide_embedding.flatten()
        support_tcr_seqs = list(selected_res_TCRs) + list(selected_heal_TCRs)
        query_tcr_seqs = list(selected_res_TCRs_query) + list(selected_heal_TCRs_query)
        if self._hooks:
            for hook in self._hooks.values():
                all_tasks = hook(peptide, selected_res_TCRs, selected_res_TCRs_query,
                                selected_heal_TCRs, selected_heal_TCRs_query, **self.all_tasks)
                self.set_all_tasks(all_tasks)

        return (peptide_embedding, support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y),
                peptide, support_tcr_seqs, query_tcr_seqs)

    def __len__(self):
        return len(self.PepTCRdict)

    def reset(self):
        self.all_tasks = {}