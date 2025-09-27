#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-Peptide Binding Training Module

This module implements the training pipeline for TCR-peptide binding prediction
using meta-learning approach with cross-validation.
"""
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from meta_distillation_training import train_main
from utils import MLogger, Project_path, get_train_data

warnings.filterwarnings('ignore')

# Configuration
data_idx = 0
# Set random seed for reproducibility
seed = 44
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# Training configuration
train_data_files = ['fold_3_train.csv']

def main():
    """Main training function for TCR-peptide binding prediction."""
    for i, train_file in enumerate(train_data_files):
        print(f"Starting KFold training {i+1}")

        # Create output directory
        save_path = os.path.join(Project_path, 'fold10alpha', 'fold3')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Initialize logger
        logger = MLogger(os.path.join(save_path, 'training.log'))
        logger.info('Starting training process...')

        # Load training data
        train_data_path = os.path.join(
            Project_path, 'new_data', 'HLA-A-beta-original-github',
            '10fold_alpha', train_file
        )
        current_train_data = pd.read_csv(train_data_path)
        unique_peptides_count = len(current_train_data['peptide'].unique())

        # Start training
        train_main(
            train_data=train_data_path,
            save_path=save_path,
            logger_file=logger,
            task_num=unique_peptides_count,
            hook=get_train_data,
            save_train_data=True,
            strategy='mode2'
        )

if __name__ == '__main__':
    main()
