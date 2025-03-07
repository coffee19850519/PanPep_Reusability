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
    parser.add_argument('--upper_limit', type=int, default=160000, help='Upper limit for window size')
    parser.add_argument('--test_data', type=str, 
                       default='/public/home/wxy/PanPep_reusability_new/zero/zero-b.csv',
                       help='Path to test data CSV')
    parser.add_argument('--negative_data', type=str,
                       default="/public/home/wxy/PanPep_reusability_new/zero/pooling_tcrb.txt",
                       help='Path to negative TCR data')
    parser.add_argument('--model_path', type=str,
                       default='/public/home/wxy/Panpep1/Requirements',
                       help='Path to model')
    parser.add_argument('--result_dir', type=str,
                       default='resulalphabeta/panpep/zero-b',
                       help='Directory for results')
    parser.add_argument('--peptide_encoding', type=str,
                        default='/public/home/wxy/PanPep_reusability_new/new_data/HLA-A-beta-original-github/encodings/peptide_ab.npz',
                        help='Path to peptide encoding file')
    parser.add_argument('--tcr_encoding', type=str,
                        default='/public/home/wxy/PanPep_reusability_new/new_data/HLA-A-beta-original-github/encodings/tcr_ab.npz',
                        help='Path to TCR encoding file')
    return parser.parse_args()

def process_peptide(pep, test_data, test_data_tcr_negative, model, aa_dict, args, config, device, result_dir):
    pep_start_time = time.time()
    print(f"\nProcessing peptide: {pep} on device: {device}")
    
    csv_file_path = os.path.join(result_dir, f"{pep}.csv")
    if os.path.exists(csv_file_path):
        print(f"Skipping peptide {pep} - CSV file already exists")
        return
    
    positive_tcr = list(test_data[test_data['peptide'] == pep]['binding_TCR'])
    negative_tcr = list(set(test_data_tcr_negative).difference(set(positive_tcr)))
    all_tcrs = positive_tcr + negative_tcr
    all_labels = [1] * len(positive_tcr) + [0] * len(negative_tcr)
    
    window_count = math.ceil(len(all_tcrs) / config.upper_limit)
    
    breakpoint = 0
    if os.path.exists(csv_file_path):
        breakpoint = len(list(set(pd.read_csv(csv_file_path)['Window'])))
    
    for i in range(breakpoint, window_count):
        window_start_time = time.time()
        print(f"\nProcessing Window {i+1}/{window_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_idx = i * config.upper_limit
        end_idx = min((i + 1) * config.upper_limit, len(all_tcrs))
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
        
        output = pd.DataFrame({
            "Window": i,
            'CDR3': current_tcrs,
            'Score': scores[0].cpu().numpy(),
            'Label': current_labels
        })
        
        if not os.path.exists(csv_file_path):
            output.to_csv(csv_file_path, index=False, header=True)
        else:
            output.to_csv(csv_file_path, mode="a", index=False, header=False)
        
        window_time = time.time() - window_start_time
        print(f"Window time: {window_time:.2f}s, Progress: {(i+1)/window_count*100:.1f}%")
    
    pep_time = time.time() - pep_start_time
    print(f"\nPeptide {pep} time: {pep_time:.2f}s")

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
        
        window_count = math.ceil(len(all_tcrs) / config.upper_limit)
        
        breakpoint = 0
        with file_lock:
            if os.path.exists(csv_file_path):
                try:
                    breakpoint = len(list(set(pd.read_csv(csv_file_path)['Window'])))
                except:
                    breakpoint = 0
        
        for i in range(breakpoint, window_count):
            window_start_time = time.time()
            print(f"\nProcessing Window {i+1}/{window_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            start_idx = i * config.upper_limit
            end_idx = min((i + 1) * config.upper_limit, len(all_tcrs))
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
            
            output = pd.DataFrame({
                "Window": i,
                'CDR3': current_tcrs,
                'Score': scores[0].cpu().numpy(),
                'Label': current_labels
            })
            with file_lock:
                try:
                    if not os.path.exists(csv_file_path):
                        output.to_csv(csv_file_path, index=False, header=True)
                    else:
                        output.to_csv(csv_file_path, mode="a", index=False, header=False)
                except Exception as e:
                    print(f"Error writing to file {csv_file_path}: {e}")
            
            window_time = time.time() - window_start_time
            print(f"Window processing time: {window_time:.2f}s, Progress: {(i+1)/window_count*100:.1f}%")
        
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