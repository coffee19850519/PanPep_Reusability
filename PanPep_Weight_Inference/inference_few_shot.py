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
from multiprocessing import Process, Manager, Lock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

from utils import Args, get_model, Model_config, Project_path, Aa_dict, task_embedding, load_kshot_data, get_query_data, save_kshot_data

def parse_args():
    parser = argparse.ArgumentParser(description='Test ranking model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs separated by comma (e.g., "0,1,2")')
    parser.add_argument('--distillation', type=int, default=3, help='Distillation number')
    parser.add_argument('--upper_limit', type=int, default=200000, help='Upper limit for window size')
    parser.add_argument('--k_shot', type=int, default=2, help='K-shot value')
    parser.add_argument('--test_data', type=str, default='/public/home/wxy/PanPep_reusability_new/few/few-b.csv', 
                        help='Path to test data CSV')
    parser.add_argument('--negative_data', type=str, 
                        default="/public/home/wxy/PanPep_reusability_new/few/pooling_tcrb.txt",
                        help='Path to negative TCR data')
    parser.add_argument('--model_path', type=str, 
                        default='/public/home/wxy/Panpep1/Requirements',
                        help='Path to model')
    parser.add_argument('--result_dir', type=str, 
                        default='resulalphabeta/panpepfew-b',
                        help='Directory for results')
    parser.add_argument('--kshot_dir', type=str, default=None,
                        help='Directory for k-shot data. If not provided, k-shot data will be generated.')
    parser.add_argument('--peptide_encoding', type=str,
                        default='/public/home/wxy/PanPep_reusability_new/new_data/HLA-A-beta-original-github/encodings/peptide_ab.npz',
                        help='Path to peptide encoding file')
    parser.add_argument('--tcr_encoding', type=str,
                        default='/public/home/wxy/PanPep_reusability_new/new_data/HLA-A-beta-original-github/encodings/tcr_ab.npz',
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

    window_count = math.ceil((len(all_ranking_data[pep][1]) - 2*config.k_shot) / config.upper_limit)
    print(f"Total windows: {window_count}")
    
    breakpoint = 0
    with file_lock:
        if os.path.exists(csv_file_path):
            try:
                breakpoint = len(list(set(pd.read_csv(csv_file_path)['Window'])))
                print(f"Resuming from window {breakpoint}/{window_count}")
            except:
                breakpoint = 0
                print("Starting from window 0")
        else:
            print("Starting from window 0")

    if config.kshot_dir:
        print(f"Loading k-shot data from: {config.kshot_dir}")
        k_shot_data = load_kshot_data(pep, config.kshot_dir)
    else:
        with file_lock:
            print(f"Generating new k-shot data")
            k_shot_data = save_kshot_data(all_ranking_data[pep], config.k_shot, pep, result_dir)

    all_query_data = get_query_data(all_ranking_data[pep], k_shot_data, config.k_shot)
    print(f"Query data size: {len(all_query_data[0])}")

    for i in range(breakpoint, window_count):
        window_start_time = time.time()
        print(f"\nProcessing Window {i+1}/{window_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if i != window_count - 1:
            current_slice = all_query_data[0][i * config.upper_limit : (i+1) * config.upper_limit]
            current_labels = all_query_data[1][i * config.upper_limit : (i+1) * config.upper_limit]
        else:
            current_slice = all_query_data[0][i * config.upper_limit:]
            current_labels = all_query_data[1][i * config.upper_limit:]
        
        F_data = [k_shot_data[0], k_shot_data[1], current_slice, current_labels]

        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(
            pep, F_data, aa_dict, peptide_encoding_dict, tcr_encoding_dict
        )

        if i == 0:
            print("Finetuning model...")
            with file_lock:
                if not os.path.exists(finetuned_model_path):
                    end, finetuned_params = model.finetunning(
                        peptide_embedding[0].to(device), 
                        x_spt[0].to(device), 
                        y_spt[0].to(device), 
                        x_qry[0].to(device),
                        return_params=True
                    )
                    torch.save(finetuned_params, finetuned_model_path)
                else:
                    print("Using existing finetuned model parameters")
                    with torch.no_grad():
                        end = model.inference_with_params(
                            x_qry[0].to(device), 
                            torch.load(finetuned_model_path)
                        )
        else:
            print("Using finetuned model for inference...")
            with torch.no_grad():
                end = model.inference_with_params(
                    x_qry[0].to(device), 
                    torch.load(finetuned_model_path)
                )

        output = pd.DataFrame({
            "Window": i,
            'CDR3': F_data[2], 
            'Score': list(end[0]), 
            'Label': F_data[3]
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