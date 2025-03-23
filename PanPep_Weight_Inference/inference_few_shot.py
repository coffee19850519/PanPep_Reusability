import os
import sys
import time
from datetime import datetime
import joblib
import math
import numpy as np
import torch
import pandas as pd
import random, sys
import argparse
from multiprocessing import Process, Manager, Lock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

from utils import Args, get_model, Model_config, Project_path, Aa_dict, task_embedding, load_support_data, get_query_data, save_support_data

def parse_args():
    parser = argparse.ArgumentParser(description='Test ranking model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs separated by comma (e.g., "0,1,2")')
    parser.add_argument('--distillation', type=int, default=3, help='Distillation number')
    parser.add_argument('--upper_limit', type=int, default=20000, help='Upper limit for batch size')
    parser.add_argument('--support', type=int, default=2, help='K-shot value')
    parser.add_argument('--test_data', type=str, default='/public/home/wxy/Panpep1/few-shot.csv', 
                        help='Path to test data CSV')
    parser.add_argument('--negative_data', type=str, 
                        default="/public/home/wxy/Panpep1/Control_dataset.txt",
                        help='Path to negative TCR data')
    parser.add_argument('--model_path', type=str, 
                        default='/public/home/wxy/Panpep1/Requirements',
                        help='Path to model')
    parser.add_argument('--result_dir', type=str, 
                        default='result/few_reproduction111',
                        help='Directory for results')
    parser.add_argument('--support_dir', type=str, default=None,
                        help='Directory for k-shot data. If not provided, k-shot data will be generated.')
    parser.add_argument('--peptide_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/peptide_b.npz',
                        help='Path to peptide encoding file')
    parser.add_argument('--tcr_encoding', type=str,
                        default='/public/home/wxy/Panpep1/encoding/tcr_b.npz',
                        help='Path to TCR encoding file')
    return parser.parse_args()

def load_encodings(encoding_path):
    with np.load(encoding_path) as encodings:
        sequences = encodings['sequences']
        encoding_data = encodings['encodings']
        encoding_dict = {seq: torch.from_numpy(enc).float() for seq, enc in zip(sequences, encoding_data)}
    return encoding_dict

def process_peptide_with_lock(pep, test_data, test_data_tcr_negative, model, aa_dict, args, config, device, result_dir, file_lock, peptide_encoding_dict, tcr_encoding_dict):
    pep_start_time = time.time()
    print(f"\nProcessing peptide: {pep} on device: {device}")
    
    csv_file_path = os.path.join(result_dir, f"{pep}.csv")
    finetuned_model_path = os.path.join(result_dir, f"{pep}_finetuned_params.pt")

    with file_lock:
        file_exists = os.path.exists(csv_file_path)
        if file_exists:
            print(f"Skipping peptide {pep} - CSV file already exists")
            return
    
    positive_tcr = list(test_data[test_data['peptide'] == pep]['binding_TCR'])
    negative_tcr = list(set(test_data_tcr_negative).difference(set(positive_tcr)))
    print(f"Positive TCRs: {len(positive_tcr)}, Negative TCRs: {len(negative_tcr)}")

    all_tcrs = positive_tcr + negative_tcr
    all_labels = [1] * len(positive_tcr) + [0] * len(negative_tcr)
    all_ranking_data = {pep: [all_tcrs, all_labels]}

    batch_count = math.ceil((len(all_ranking_data[pep][1]) - 2*config.support) / config.upper_limit)
    print(f"Total batches: {batch_count}")
    
    breakpoint = 0
    if config.support_dir:
        print(f"Loading k-shot data from: {config.support_dir}")
        support_data = load_support_data(pep, config.support_dir)
    else:
        with file_lock:
            print(f"Generating new k-shot data")
            support_data = save_support_data(all_ranking_data[pep], config.support, pep, result_dir)

    all_query_data = get_query_data(all_ranking_data[pep], support_data, config.support)
    print(f"Query data size: {len(all_query_data[0])}")

    for i in range(breakpoint, batch_count):
        batch_start_time = time.time()
        print(f"\nProcessing batch {i+1}/{batch_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if i != batch_count - 1:
            current_slice = all_query_data[0][i * config.upper_limit : (i+1) * config.upper_limit]
            current_labels = all_query_data[1][i * config.upper_limit : (i+1) * config.upper_limit]
        else:
            current_slice = all_query_data[0][i * config.upper_limit:]
            current_labels = all_query_data[1][i * config.upper_limit:]
        
        F_data = [support_data[0], support_data[1], current_slice, current_labels]

        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(
            pep, F_data, aa_dict, peptide_encoding_dict, tcr_encoding_dict
        )

        if i == 0:
            print("Finetuning model...")
            with file_lock:

                end, finetuned_net = model.finetunning(
                    peptide_embedding[0].to(device), 
                    x_spt[0].to(device), 
                    y_spt[0].to(device), 
                    x_qry[0].to(device),
                    return_params=True
                )
                torch.save(finetuned_net, finetuned_model_path)

        else:
            print("Using finetuned model for inference...")
            with torch.no_grad():
                end = model.inference_with_params(
                    x_qry[0].to(device), 
                    finetuned_net
                )

        output = pd.DataFrame({
            'CDR3': pd.Series(F_data[2]).astype(str),
            'Score': np.array(end[0], dtype=np.float32),
            'Label': pd.Series(F_data[3]).astype(np.int8)
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
    print(f"\nTotal time for peptide {pep}: {pep_time:.2f}s")

def few_shot_inference(peptide_encoding_dict, tcr_encoding_dict, config):
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

    manager = Manager()
    file_lock = manager.Lock()
    
    def process_peptide_batch(gpu_id, peptide_batch):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda:0')
        
        try:
            model = get_model(args, Model_config, model_path=config.model_path, device=device)
            model = model.to(device)
            
            for pep in peptide_batch:
                process_peptide_with_lock(pep, test_data, test_data_tcr_negative, model, aa_dict, 
                                         args, config, device, result_dir, file_lock,
                                         peptide_encoding_dict, tcr_encoding_dict)
        except Exception as e:
            print(f"Error processing on GPU {gpu_id}: {e}")
            import traceback
            traceback.print_exc()

    peptide_info = [(pep, len(pep)) for pep in test_data_pep]
    peptide_info.sort(key=lambda x: x[1], reverse=True)

    peptide_batches = [[] for _ in range(num_gpus)]
    for i, (pep, _) in enumerate(peptide_info):
        peptide_batches[i % num_gpus].append(pep)

    processes = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        p = Process(target=process_peptide_batch, args=(gpu_id, peptide_batches[gpu_idx]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/3600:.2f} hours)")

if __name__ == '__main__':
    config = parse_args()
    peptide_encoding_dict = load_encodings(config.peptide_encoding)
    tcr_encoding_dict = load_encodings(config.tcr_encoding)
    few_shot_inference(peptide_encoding_dict, tcr_encoding_dict, config)