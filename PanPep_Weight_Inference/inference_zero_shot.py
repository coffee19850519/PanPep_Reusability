import os
import sys
import time
from datetime import datetime
import joblib
import math
import numpy as np
import torch
import pandas as pd
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

from utils import (
    Args, get_model, Model_config, Project_path, Aa_dict,
    zero_task_embedding
)

def parse_args():
    parser = argparse.ArgumentParser(description='Zero-shot inference')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs separated by comma (e.g., "0,1,2")')
    parser.add_argument('--distillation', type=int, default=3, help='Distillation number')
    parser.add_argument('--batch_size', type=int, default=400000, help='Upper limit for batch size')
    parser.add_argument('--test_data', type=str, 
                       default='/public/home/wxy/Panpep1/zero_shot.csv',
                       help='Path to test data CSV')
    parser.add_argument('--negative_data', type=str,
                       default="/public/home/wxy/Panpep1/Control_dataset.txt",
                       help='Path to negative TCR data')
    parser.add_argument('--model_path', type=str,
                       default='/public/home/wxy/Panpep1/Requirements',
                       help='Path to model')
    parser.add_argument('--result_dir', type=str,
                       default='result/zero-test',
                       help='Directory for results')
    parser.add_argument('--peptide_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/peptide_b.npz',
                        help='Path to peptide encoding file')
    parser.add_argument('--tcr_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/peptide_b.npz',
                        help='Path to TCR encoding file')
    # parser.add_argument('--tcr_encoding', type=str,
    #                     default='/public/home/wxy/Panpep1/encoding/tcr_b.npz',
    #                     help='Path to TCR encoding file')
    return parser.parse_args()

def zero_shot_inference(peptide_encoding_dict, tcr_encoding_dict, config):
    total_start_time = time.time()
    gpu_ids = [int(gpu_id) for gpu_id in config.gpu.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    result_dir = os.path.join(Project_path, config.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    args = Args(C=config.distillation, L=75, R=config.distillation, update_lr=0.01, update_step_test=3)
    aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))
    test_data = pd.read_csv(config.test_data)
    test_data_pep = sorted(list(set(test_data['peptide'])))
    test_data_tcr_negative = pd.read_csv(config.negative_data)
    test_data_tcr_negative = np.array(test_data_tcr_negative).reshape(1, -1).tolist()[0]
    from multiprocessing import Process, Manager, Lock
    manager = Manager()
    file_lock = manager.Lock()
    
    def process_peptide_with_lock(pep, test_data, test_data_tcr_negative, model, aa_dict, args, config, device, result_dir, file_lock):
        pep_start_time = time.time()
        print(f"\nProcessing peptide: {pep} on device: {device}")
        
        csv_file_path = os.path.join(result_dir, f"{pep}.csv")

        with file_lock:
            file_exists = os.path.exists(csv_file_path)
            if file_exists:
                print(f"Skipping peptide {pep} - CSV file already exists")
                return
        
        positive_tcr = list(test_data[test_data['peptide'] == pep]['binding_TCR'])
        negative_tcr = list(set(test_data_tcr_negative).difference(set(positive_tcr)))
        all_tcrs = positive_tcr + negative_tcr
        all_labels = [1] * len(positive_tcr) + [0] * len(negative_tcr)
        
        batch_count = math.ceil(len(all_tcrs) / config.batch_size)
        
        breakpoint = 0
   
        for i in range(breakpoint, batch_count):
            batch_start_time = time.time()
            print(f"\nProcessing batch {i+1}/{batch_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(all_tcrs))
            current_tcrs = all_tcrs[start_idx:end_idx]
            current_labels = all_labels[start_idx:end_idx]
            
            peptide_embedding, x_qry = zero_task_embedding(
                pep, current_tcrs, aa_dict,
                peptide_encoding_dict, tcr_encoding_dict
            )
            
            with torch.no_grad():
                scores = model.meta_forward_score(
                    peptide_embedding.to(device),
                    x_qry.to(device)
                )
            scores = scores[0].cpu()
            output = pd.DataFrame({
                'CDR3': pd.Series(current_tcrs).astype(str),
                'Score': np.array(scores, dtype=np.float32),  # scores is now on CPU
                'Label': pd.Series(current_labels).astype(np.int8)
            })
            with file_lock:
                try:
                    parquet_file_path = os.path.splitext(csv_file_path)[0] + '.parquet'
                    
                    if not os.path.exists(parquet_file_path):
                        output.to_parquet(parquet_file_path, engine='pyarrow')
                    else:
                        existing_df = pd.read_parquet(parquet_file_path)
                        combined_df = pd.concat([existing_df, output], ignore_index=True)
                        combined_df.to_parquet(parquet_file_path, engine='pyarrow')
                except Exception as e:
                    print(f"Error writing to file {parquet_file_path}: {e}")
            
            batch_time = time.time() - batch_start_time
            print(f"batch processing time: {batch_time:.2f}s, Progress: {(i+1)/batch_count*100:.1f}%")
        
        pep_time = time.time() - pep_start_time
        print(f"\nPeptide {pep} processing time: {pep_time:.2f}s")
    
    def process_peptide_batch(gpu_id, peptide_batch, file_lock):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f'cuda:0')
        
        try:
            model = get_model(args, Model_config, model_path=config.model_path, device=device)
            model = model.to(device)
            
            for pep in peptide_batch:
                process_peptide_with_lock(pep, test_data, test_data_tcr_negative, model, aa_dict, args, config, device, result_dir, file_lock)
        except Exception as e:
            print(f"Error on GPU {gpu_id}: {e}")
    peptide_info = [(pep, len(pep)) for pep in test_data_pep]
    peptide_info.sort(key=lambda x: x[1], reverse=True)
    
    peptide_batches = [[] for _ in range(num_gpus)]
    for i, (pep, _) in enumerate(peptide_info):
        peptide_batches[i % num_gpus].append(pep)

    processes = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        p = Process(target=process_peptide_batch, args=(gpu_id, peptide_batches[gpu_idx], file_lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

def load_encodings(encoding_path):
    with np.load(encoding_path) as encodings:
        sequences = encodings['sequences']
        encoding_data = encodings['encodings']
        encoding_dict = {seq: torch.from_numpy(enc).float() for seq, enc in zip(sequences, encoding_data)}
    return encoding_dict

if __name__ == '__main__':
    config = parse_args()
    peptide_encoding_dict = load_encodings(config.peptide_encoding)
    tcr_encoding_dict = load_encodings(config.tcr_encoding)
    zero_shot_inference(peptide_encoding_dict, tcr_encoding_dict, config)