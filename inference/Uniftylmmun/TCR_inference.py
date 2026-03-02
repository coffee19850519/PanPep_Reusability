#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCR-Peptide Binding Prediction Script
Predicts binding scores for TCR-peptide pairs
Modified to save results in Parquet format with compression
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import os
import argparse
from torch.utils.data import DataLoader
from fastparquet import write
from filelock import FileLock

# Import model and data processing functions
from models.TCR import Mymodel_tcr, data_process_tcr, MyDataSet_tcr

warnings.filterwarnings("ignore")

# Model configuration
SEED = 66
PEP_MAX_LEN = 15
TCR_MAX_LEN = 34
BATCH_SIZE = 8192

# Set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_size = len(np.load('../data/data_dict.npy', allow_pickle=True).item())


class TCRPredictor:
    def __init__(self, model_path):
        """Initialize TCR predictor with trained model"""
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = Mymodel_tcr().to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()
        print(f"Model loaded: {self.model_path}")
        return model

  
    def predict(self, data_path, output_path=None, batch_size=8192):
        """
        Predict TCR-peptide binding scores with flexible batch size
        - Uses configurable batch_size (default 8192)
        - Drops incomplete batches (drop_last=True)
        - Only processes complete batches
        - Now supports any batch size (16384, 32768, etc.)
        - Saves results in compressed Parquet format

        Args:
            data_path: Input CSV file path
            output_path: Output file path (auto-generated if None)
            batch_size: Batch size for prediction (default 8192, can be larger)

        Returns:
            DataFrame with original data and prediction scores
        """
        print(f"\nPredicting: {data_path}")
        start_time = time.time()

        # Validate batch_size (must be positive and reasonable)
        if batch_size <= 0:
            print(f"Warning: Invalid batch_size {batch_size}. Using default 8192.")
            batch_size = 8192
        elif batch_size > 65536:  # 64K is reasonable upper limit
            print(f"Warning: Very large batch_size {batch_size}. This may cause memory issues.")

        print(f"Using batch_size: {batch_size}")

        # Load data first to get sample count
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        original_data = pd.read_csv(data_path)

        # Validate required columns
        required_columns = ['tcr', 'peptide']
        missing = [col for col in required_columns if col not in original_data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Add default label if not present
        if 'label' not in original_data.columns:
            original_data['label'] = 0

        data_size = len(original_data)

        print(f"Using flexible batch method:")
        print(f"  Total samples: {data_size:,}")
        print(f"  Batch size: {batch_size:,}")

        # Calculate batch info
        full_batches = data_size // batch_size
        has_remainder = data_size % batch_size > 0
        total_batches = full_batches + (1 if has_remainder else 0)

        print(f"  Full batches: {full_batches}")
        print(f"  Has remainder batch: {'Yes' if has_remainder else 'No'}")
        print(f"  Total batches: {total_batches}")
        print(f"  Will process ALL {data_size:,} samples (no data dropped)")

        # Dynamically modify global batch_size in models.TCR module
        import models.TCR as TCR_module

        # Backup original batch_size and set new one
        original_global_batch_size = TCR_module.batch_size
        TCR_module.batch_size = batch_size

        try:
            # Process data using new batch size
            pep_inputs, tcr_inputs, labels = TCR_module.data_process_tcr(original_data)
            dataset = TCR_module.MyDataSet_tcr(pep_inputs, tcr_inputs, labels)

            # DataLoader with custom batch_size and drop_last=False to process all data
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

            # Generate output path
            if output_path is None:
                input_name = os.path.splitext(os.path.basename(data_path))[0]
                output_path = f"{input_name}_with_score.parquet"

            # Predict scores using original logic
            scores = []
            with torch.no_grad():
                for pep_batch, tcr_batch, _ in tqdm(data_loader, desc="Predicting"):
                    pep_batch = pep_batch.to(device)
                    tcr_batch = tcr_batch.to(device)

                    outputs, _ = self.model(pep_batch, tcr_batch)
                    batch_scores = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()
                    scores.extend(batch_scores.tolist())

        except Exception as e:
            print(f"Error during prediction: {e}")
            return pd.DataFrame()

        finally:
            # Restore original global batch_size
            TCR_module.batch_size = original_global_batch_size

        # Create result dataframe with ALL processed samples
        result_data = original_data.copy()
        result_data['score'] = scores

        # Optimize data types before saving to Parquet
        # Convert string columns to str type
        # for col in ['tcr', 'peptide']:
        #     if col in result_data.columns:
        #         result_data[col] = result_data[col].astype(str)
        
        # # Convert label to int8 (saves memory)
        # if 'label' in result_data.columns:
        #     result_data['label'] = result_data['label'].astype(np.int8)
        
        # # Convert score to float32 (sufficient precision, saves memory)
        # result_data['score'] = result_data['score'].astype(np.float32)
        
        print(f"Data types: {dict(result_data.dtypes)}")

        # Save to Parquet with compression and file lock
        lock_path = output_path + '.lock'
        file_lock = FileLock(lock_path)
        
        with file_lock:
            try:
                if not os.path.exists(output_path):
                    write(output_path, result_data, compression='gzip')
                    print(f"Created new Parquet file with compression")
                else:
                    write(output_path, result_data, append=True, compression='gzip')
                    print(f"Appended to existing Parquet file")
            except Exception as e:
                print(f"Error writing to file {output_path}: {e}")
                # Fallback to CSV if Parquet fails
                csv_path = output_path.replace('.parquet', '.csv')
                result_data.to_csv(csv_path, index=False)
                print(f"Saved as CSV instead: {csv_path}")
                raise

        # Print statistics
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s ({elapsed/len(result_data)*1000:.2f}ms per sample)")
        print(f"Score - Min: {result_data['score'].min():.4f}, "
              f"Max: {result_data['score'].max():.4f}, "
              f"Mean: {result_data['score'].mean():.4f}")

        # Calculate metrics if labels available
        self._calculate_metrics(result_data)

        print(f"Saved to: {output_path}")
        
        # Show file size comparison
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
        print(f"File size: {file_size:.2f} MB (compressed)")
        print(f"All {len(result_data):,} samples processed successfully (no data dropped)\n")
        
        return result_data

    def _calculate_metrics(self, result_data):
        """Calculate performance metrics if labels are available"""
        if 'label' not in result_data.columns:
            return
            
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            y_true = result_data['label'].values
            y_prob = result_data['score'].values
            
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
                y_pred = (y_prob > 0.5).astype(int)
                acc = accuracy_score(y_true, y_pred)
                print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Could not calculate metrics: {e}")

    def predict_batch(self, data_paths, output_dir=None, batch_size=8192):
        """Batch prediction for multiple files with flexible batch size"""
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []
        for data_path in data_paths:
            output_path = None
            if output_dir:
                input_name = os.path.splitext(os.path.basename(data_path))[0]
                output_path = os.path.join(output_dir, f"{input_name}_with_score.parquet")

            result = self.predict(data_path, output_path, batch_size)
            results.append(result)

        print(f"Batch processing completed: {len(results)} files")
        return results


def main():
    parser = argparse.ArgumentParser(description='TCR-Peptide Binding Prediction (Parquet format)')
    parser.add_argument('--model', '-m', type=str,
                       default='../trained_model/TCR_2/model_TCR.pkl',
                       help='Path to trained model')
    parser.add_argument('--input', '-i', type=str, nargs='+',
                       default=['../data/data_TCR/independent_set.csv'],
                       help='Input CSV file(s)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output Parquet file path (single file mode)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (batch mode)')
    parser.add_argument('--batch-size', '-b', type=int, default=8192,
                       help='Batch size for prediction (default: 8192, can be larger)')
    
    args = parser.parse_args()

    try:
        predictor = TCRPredictor(args.model)

        if len(args.input) == 1:
            predictor.predict(args.input[0], args.output, args.batch_size)
        else:
            predictor.predict_batch(args.input, args.output_dir, args.batch_size)

        print("Prediction completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())