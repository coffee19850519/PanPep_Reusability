"""
Utility functions and classes for TCR-peptide binding prediction.

This module provides essential utilities including data processing, model configuration,
logging, and embedding functions for the meta-learning framework.
"""

import os
import random
import logging
import warnings
from typing import Union, List, Any, Callable
from collections import Counter, OrderedDict
import weakref

import joblib
import numpy as np
import pandas as pd
import yaml
import torch

warnings.filterwarnings('ignore')

FilePath = Union[str, "PathLike[str]", pd.DataFrame]

def aamapping(TCRSeq, encode_dim, aa_dict):
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
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            TCRArray.append(np.zeros(5, dtype='float64'))
    for i in range(0, encode_dim - len(TCRSeq)):
        TCRArray.append(np.zeros(5, dtype='float64'))
    return torch.FloatTensor(TCRArray)


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

    padding_ids = torch.abs(seq).sum(dim=-1) == 0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq
