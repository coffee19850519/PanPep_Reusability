import numpy as np
import pickle
from Loader import SignedPairsDataset, get_index_dicts
from Trainer import ERGOLightning
from torch.utils.data import DataLoader
from argparse import Namespace
import torch
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
import glob
import argparse
from tqdm import tqdm

def read_input_file(datafile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    
    def is_valid_amino_acid_seq(seq):
        if pd.isna(seq):
            return False
        return all([aa in amino_acids for aa in seq])
    
    try:
        data = pd.read_csv(datafile)
        valid_indices = []
        
        for index in range(len(data)):
            if not is_valid_amino_acid_seq(data['TRB'][index]):
                continue
                
            sample = {}
            sample['tcra'] = data['TRA'][index]
            sample['tcrb'] = data['TRB'][index]
            sample['va'] = data['TRAV'][index]
            sample['ja'] = data['TRAJ'][index]
            sample['vb'] = data['TRBV'][index]
            sample['jb'] = data['TRBJ'][index]
            sample['t_cell_type'] = data['T-Cell-Type'][index]
            sample['peptide'] = data['Peptide'][index]
            sample['mhc'] = data['MHC'][index]
            sample['sign'] = 0

            if not is_valid_amino_acid_seq(sample['peptide']):
                continue

            if not is_valid_amino_acid_seq(sample['tcra']):
                sample['tcra'] = 'UNK'
                
            all_pairs.append(sample)
            valid_indices.append(index)

        filtered_data = data.iloc[valid_indices].reset_index(drop=True)
        return all_pairs, filtered_data
    except Exception as e:
        print(f"Error reading file {datafile}: {str(e)}")
        return None, None

def load_model(hparams, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = ERGOLightning(hparams)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_model(dataset):
    if dataset == 'vdjdb':
        version = '1veajht'
    elif dataset == 'mcpas':
        version = '1meajht'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    try:
        checkpoint_path = os.path.join('Models', 'version_' + version, 'checkpoints')
        files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
        if not files:
            raise FileNotFoundError("No checkpoint files found")
        checkpoint_path = os.path.join(checkpoint_path, files[0])

        args_path = os.path.join('Models', 'version_' + version, 'meta_tags.csv')
        with open(args_path, 'r') as file:
            lines = file.readlines()
            args = {}
            for line in lines[1:]:
                key, value = line.strip().split(',')
                if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
                    args[key] = value
                else:
                    args[key] = eval(value)

        hparams = Namespace(**args)
        model = load_model(hparams, checkpoint_path)
        train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
        
        return model, train_pickle
    except Exception as e:
        print(f"Error getting model: {str(e)}")
        return None, None

def get_train_dicts(train_pickle):
    try:
        with open(train_pickle, 'rb') as handle:
            train = pickle.load(handle)
        return get_index_dicts(train)
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return None

def process_file(dataset, input_file, output_dir):
    # output_dir = 'prediction1'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model, train_file = get_model(dataset)
    if model is None:
        print("Model loading failed")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    print("Loading training data...")
    train_dicts = get_train_dicts(train_file)
    if train_dicts is None:
        print("Training data loading failed")
        return
    
    try:
        file_name = os.path.basename(input_file)
        print(f"\nProcessing file: {file_name}")
        
        test_samples, dataframe = read_input_file(input_file)
        if test_samples is None or dataframe is None:
            print(f"Failed to process file {file_name}")
            return
            
        test_dataset = SignedPairsDataset(test_samples, train_dicts)
        
        batch_size = 1000
        loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda b: test_dataset.collate(
                b, 
                tcr_encoding=model.tcr_encoding_model,
                cat_encoding=model.cat_encoding
            )
        )
        
        outputs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Prediction progress"):
                batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
                output = model.validation_step(batch, 0)
                if output:
                    outputs.extend(output['y_hat'].cpu().numpy().tolist())

        dataframe['Score'] = np.array(outputs, dtype=np.float32)
        output_filename = f'{os.path.splitext(os.path.basename(input_file))[0]}_predicted.parquet'
        output_path = os.path.join(output_dir, output_filename)

        dataframe.to_parquet(output_path, compression='gzip', index=False)
        
        print(f"File processed, results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing file {os.path.basename(input_file)}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['vdjdb', 'mcpas'],
                      help='Dataset type (vdjdb or mcpas)')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory containing CSV files')
    args = parser.parse_args()
         
    process_file(args.dataset, args.input_file, args.output_dir)

if __name__ == '__main__':
    main()