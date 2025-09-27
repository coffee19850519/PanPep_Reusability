"""Meta-learning training module with knowledge distillation.

This module implements the core training pipeline for TCR-peptide binding prediction
using meta-learning with disentanglement distillation approach.
"""

import random
import time
import warnings
from collections import Counter

import joblib
import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Memory_meta import Memory_Meta
from Memory_meta import Memory_module
from PepTCRdict import PepTCRdict, PepTCRdict_new, PepTCRdict_ranking

from utils import (
    get_train_data, Support, Query, merge_dict, get_peptide_tcr, MLogger, Args,
    _split_parameters, Project_path, Data_output, K_fold, Train_dataset,
    Train_output_dir, Train_Round, Device, Aa_dict, Negative_dataset,
    Batch_size, Shuffle, Data_config, Model_config
)

warnings.filterwarnings('ignore')


def train_main(train_data, save_path, logger_file, task_num: int = 166, hook=None,
               ranking_tcr_file=None, save_train_data=False, strategy='mode2',
               negative_data=None):
    """Main training function for meta-learning with distillation.

    Args:
        train_data: Path to training data file or DataFrame
        save_path: Directory to save model outputs
        logger_file: Logger instance for training logs
        task_num: Number of peptide-specific tasks
        hook: Optional hook function for data processing
        ranking_tcr_file: Optional ranking TCR file path
        save_train_data: Whether to save training data
        strategy: Training strategy ('mode1', 'mode2', or 'ranking')
        negative_data: Path to negative data file
    """
    args = Args(C=Data_config['Train']['Meta_learning']['Model_parameter']['num_of_index'],
                L=Data_config['Train']['Meta_learning']['Model_parameter']['len_of_embedding'],
                R=Data_config['Train']['Meta_learning']['Model_parameter']['len_of_index'],
                meta_lr=Data_config['Train']['Meta_learning']['Model_parameter']['meta_lr'], update_lr=Data_config['Train']['Meta_learning']['Model_parameter']['inner_loop_lr'], update_step=Data_config['Train']['Meta_learning']['Model_parameter']['inner_update_step'],
                update_step_test=Data_config['Train']['Meta_learning']['Model_parameter']['inner_fine_tuning'], regular=Data_config['Train']['Meta_learning']['Model_parameter']['regular_coefficient'], epoch=Data_config['Train']['Meta_learning']['Trainer_parameter']['epoch'],
                distillation_epoch=Data_config['Train']['Disentanglement_distillation']['Trainer_parameter']['epoch'],
                num_of_tasks=task_num)
    # Initialize device and model
    device = torch.device(Device)
    print("Data configuration:", Data_config)

    # Initialize the meta-learning model
    model = Memory_Meta(args, Model_config).to(device)
    print("Model configuration:", Model_config)

    # Count trainable parameters
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(model)
    print('Total trainable tensors:', num)
    # Load training data based on strategy
    aa_dict_path = os.path.join(Project_path, Aa_dict)

    if strategy == "mode2":
        Training_data = PepTCRdict(
            train_data,
            os.path.join(Project_path, Negative_dataset),
            Support, Query,
            aa_dict_path=aa_dict_path,
            mode='train'
        )
    elif strategy == "mode1":
        Training_data = PepTCRdict_new(
            train_data, negative_data, Support, Query,
            aa_dict_path=aa_dict_path, mode='train'
        )
    elif strategy == "ranking":
        Training_data = PepTCRdict_ranking(
            train_data, negative_data, Support, Query,
            aa_dict_path=aa_dict_path, mode='train'
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if hook:
        all_train_data: dict = Training_data.all_tasks
        Training_data.register_hook(hook)
    args.task_num = len(Training_data.PepTCRdict)
    # Meta-learning phase
    logger = logger_file
    logger.info('Starting meta-learning phase')
    logger.info(f'Distillation parameters: C={args.C}, R={args.R}')
    # Training loop
    for epoch in range(args.epoch):
        print(f"Epoch: {epoch + 1}/{args.epoch}")

        # Create data loader
        dataloader = DataLoader(
            Training_data, Batch_size, Shuffle,
            num_workers=0, pin_memory=True
        )

        epoch_train_acc = []
        start_time = time.time()

        # Process batches
        for step, (peptide_embedding, x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
            # Move data to device
            peptide_embedding = peptide_embedding.to(device)
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            # Forward pass
            accs = model(peptide_embedding, x_spt, y_spt, x_qry, y_qry)
            epoch_train_acc.append(accs)

        # Update training data if hook is provided
        if hook:
            if len(all_train_data) == 0:
                all_train_data = Training_data.all_tasks
            else:
                all_train_data = {
                    k: v + all_train_data[k]
                    for k, v in Training_data.all_tasks.items()
                    if k in all_train_data.keys()
                }

        # Calculate epoch metrics
        end_time = time.time()
        epoch_train_acc = np.array(epoch_train_acc)
        avg_acc = epoch_train_acc.mean(axis=0)

        if logger:
            logger.info(
                f'Epoch: [{epoch + 1}/{args.epoch}]\t'
                f'Training Accuracy: {avg_acc[-1]:.5f}\t'
                f'Time: {end_time - start_time:.3f}s'
            )
        # Save model artifacts at the end of training
        if epoch == args.epoch - 1:
            # Save peptide-specific learners
            joblib.dump(model.models, os.path.join(save_path, "models.pkl"))

            # Save loss and data from previous tasks
            joblib.dump(model.prev_loss, os.path.join(save_path, "prev_loss.pkl"))
            joblib.dump(model.prev_data, os.path.join(save_path, "prev_data.pkl"))

            # Save meta learner state
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

            # Save training data if requested
            if save_train_data and hook is not None:
                base_name = os.path.splitext(os.path.basename(train_data))[0]
                all_data_name = f'{base_name}_all_train_data.csv'
                distill_data_name = f'{base_name}_distillation_train_data.csv'

                pd.DataFrame(all_train_data).to_csv(
                    os.path.join(save_path, all_data_name), index=False
                )
                pd.DataFrame(Training_data.all_tasks).to_csv(
                    os.path.join(save_path, distill_data_name), index=False
                )

        # Reset model and data for next epoch
        model.reset()
        Training_data.reset()
    if logger:
        logger.info('Finish training!')

    # Disentanglement distillation phase
    logger.info('Starting disentanglement distillation phase')

    # Load saved artifacts from meta-learning
    prev_loss = joblib.load(os.path.join(save_path, "prev_loss.pkl"))
    prev_data = joblib.load(os.path.join(save_path, "prev_data.pkl"))
    prev_models = joblib.load(os.path.join(save_path, "models.pkl"))

    # Initialize memory module
    memory_module = Memory_module(args, model.meta_Parameter_nums)
    if Device == 'cuda':
        model.Memory_module = memory_module.cuda()
    else:
        model.Memory_module = memory_module.cpu()
    # Configure memory module with saved data
    model.Memory_module.prev_loss = prev_loss
    model.Memory_module.prev_data = prev_data
    model.Memory_module.models = prev_models
    # Distillation training loop
    for d_epoch in range(args.distillation_epoch):
        print(f"Distillation Epoch: {d_epoch + 1}/{args.distillation_epoch}")

        # Write peptide-specific learners to memory
        model.Memory_module.writehead(model.Memory_module.models)

        total_loss = 0

        # Process each peptide-specific task
        for task_idx, (index_prev, x_prev, y_prev) in enumerate(model.Memory_module.prev_data):
            # Calculate peptide-query similarity weights
            similarity_weights = model.Memory_module(index_prev)[0]

            # Generate logits from memory ensemble
            logits = []
            for memory_idx, content in enumerate(model.Memory_module.memory.content_memory):
                # Reconstruct model parameters from memory
                weights_memory = _split_parameters(
                    content.unsqueeze(0),
                    model.net.parameters()
                )
                # Calculate logits using reconstructed weights
                logits.append(model.net(x_prev, weights_memory, bn_training=True))

            # Compute distillation loss
            weighted_softmax = sum([
                similarity_weights[k] * F.softmax(logit)
                for k, logit in enumerate(logits)
            ])
            task_loss = torch.sum(
                torch.log(weighted_softmax) * model.Memory_module.prev_loss[task_idx] * -1
            )
            total_loss += task_loss

        # Average loss across tasks
        avg_loss = total_loss / (task_idx + 1)

        if logger:
            logger.info(
                f'Distillation Epoch: [{d_epoch + 1}/{args.distillation_epoch}]\t'
                f'Loss: {avg_loss.item():.5f}'
            )

        # Backward pass and optimization
        model.Memory_module.optim.zero_grad()
        avg_loss.backward()
        model.Memory_module.optim.step()
        model.Memory_module.content_memory = model.Memory_module.memory.content_memory.detach()
    if logger:
        logger.info('Distillation training completed')

    # Save final memory components
    joblib.dump(
        model.Memory_module.memory.content_memory,
        os.path.join(save_path, "Content_memory.pkl")
    )
    joblib.dump(
        list(model.Memory_module.memory.parameters()),
        os.path.join(save_path, "Query.pkl")
    )


def create_kfold_data(k_fold, all_peptides, meta_dataset, output_dir):
    """Create K-fold cross-validation splits for peptide data.

    Args:
        k_fold: Number of folds for cross-validation
        all_peptides: List of unique peptides
        meta_dataset: Complete dataset as pandas DataFrame
        output_dir: Directory to save the fold splits
    """
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(all_peptides), 1):
        print(f'Creating fold {fold_idx}')

        fold_dir = os.path.join(output_dir, f'kfold{fold_idx}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        # Create training data for this fold
        train_peptides = [all_peptides[i] for i in train_indices]
        train_data = meta_dataset[
            meta_dataset['peptide'].isin(train_peptides)
        ].copy()
        train_data.to_csv(
            os.path.join(fold_dir, f'KFold_{fold_idx}_train.csv'),
            index=False
        )

        # Create test data for this fold
        test_peptides = [all_peptides[i] for i in test_indices]
        test_data = meta_dataset[
            meta_dataset['peptide'].isin(test_peptides)
        ].copy()
        test_data.to_csv(
            os.path.join(fold_dir, f'KFold_{fold_idx}_test.csv'),
            index=False
        )


def main():
    """Main function for training with K-fold cross-validation."""
    # Create K-fold data splits if they don't exist
    data_output_path = os.path.join(Project_path, Data_output)
    if not os.path.exists(data_output_path):
        print("Creating K-fold data splits...")
        meta_dataset = pd.read_csv(os.path.join(Project_path, Train_dataset))
        unique_peptides = meta_dataset['peptide'].drop_duplicates().tolist()

        create_kfold_data(
            K_fold, unique_peptides, meta_dataset,
            output_dir=data_output_path
        )

    # Run training for multiple rounds
    for round_idx in range(Train_Round):
        # Set random seed for reproducibility
        seed = round_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Train on specified folds
        fold_range = Data_config['dataset']['current_fold']
        for fold_idx in range(fold_range[0], fold_range[1]):
            print(f"Training Round {round_idx + 1}, Fold {fold_idx}")

            # Create output directory
            save_path = os.path.join(
                Project_path, Train_output_dir,
                f'Round{round_idx + 1}', f'kfold{fold_idx}'
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Initialize logger
            logger = MLogger(os.path.join(save_path, 'training.log'))
            logger.info(f'Round: {round_idx + 1}, K-Fold: {fold_idx}')

            # Get training data path
            train_data_path = os.path.join(
                Project_path, Data_output, f'kfold{fold_idx}',
                f'KFold_{fold_idx}_train.csv'
            )

            # Start training
            train_main(
                train_data=train_data_path,
                save_path=save_path,
                logger_file=logger,
                hook=get_train_data,
                save_train_data=True
            )

if __name__ == '__main__':
    main()
