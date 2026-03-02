import os
import sys
import time
from datetime import datetime
import joblib
import math
import numpy as np
import torch
import pandas as pd
import random
import argparse
from multiprocessing import Process, Manager, Lock
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

from utils import (
    Args, get_model, Model_config, Model_config_attention8, Model_config_attention_stack5,
    Model_config_conv_stack3, Model_config_multi_head_attention5_conv3, Model_config_attention5_conv3,
    Project_path, Aa_dict, task_embedding,
    load_support_data, get_query_data, save_support_data,
    sample_multi_round_support_data, load_multi_round_support_data, get_query_data_multi_round,Model_config_large,Model_config_large_16,Model_config_large_32,Model_config_attention5_conv3_large,Model_config_conv_stack6,Model_config_conv_stack8,Model_config_conv_stack12,Model_config_attention_stack10,Model_config_attention16,Model_config_attention24,Model_config_attention_stack6,Model_config_attention_stack7,Model_config_attention_stack8,Model_config_attention_stack9,Model_config_attention_stack12
)

MODEL_CONFIG_MAP = {
    'default': Model_config,
    'attention8': Model_config_attention8,
    'attention16': Model_config_attention16,
    'attention24': Model_config_attention24,
    'attention_stack5': Model_config_attention_stack5,
    'attention_stack10': Model_config_attention_stack10,
    'attention_stack6': Model_config_attention_stack6,
    'attention_stack8': Model_config_attention_stack8,
    'attention_stack12': Model_config_attention_stack12,
    'attention_stack9': Model_config_attention_stack9,
    'conv_stack3':  Model_config_conv_stack3,
    'conv_stack6':  Model_config_conv_stack6,
    'conv_stack8':  Model_config_conv_stack8,
    'conv_stack12':  Model_config_conv_stack12,
    'multi_head_attention5_conv3': Model_config_multi_head_attention5_conv3,
    'attention5_conv3': Model_config_attention5_conv3,
    'large': Model_config_large,
    'large16':Model_config_large_16,
    'large32':Model_config_large_32,
    'attention5_conv3_large':Model_config_attention5_conv3_large,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Test ranking model')
    parser.add_argument('--mode', type=str, default='mixed', choices=['single', 'mixed'],
                        help='Inference mode: "single" uses one negative source with single-round support; '
                             '"mixed" uses alternating negative sources with multi-round support.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs separated by comma (e.g., "0,1,2")')
    parser.add_argument('--model', type=str, default='attention5_conv3_large',
                        choices=['default', 'attention8', 'attention_stack5', 'conv_stack3',
                                'multi_head_attention5_conv3', 'attention5_conv3','large','large16','large32','attention5_conv3_large','attention16','attention24','attention_stack10','conv_stack6','conv_stack8','conv_stack12','attention_stack6','attention_stack8','attention_stack9','attention_stack12'],
                        help='Model configuration to use')
    parser.add_argument('--distillation', type=int, default=3, help='Distillation number')
    parser.add_argument('--batch_size', type=int, default=10000, help='Upper limit for batch size')
    parser.add_argument('--support', type=int, default=4, help='K-shot value')
    parser.add_argument('--test_data', type=str, default='/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/new_model/peptide_11times_or_more.csv',
                        help='Path to test data CSV')
    parser.add_argument('--negative_data', type=str,
                        default="/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/PanPep_Weight_Inference_attention/Combined_library_sample_0.1pct.txt",
                        help='Path to negative TCR data (used as query negatives in mixed mode, sole source in single mode) ')
    parser.add_argument('--negative_data_background', type=str, default='/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/PanPep_train/Control_dataset.txt',
                        help='[mixed mode only] Path to background negative TCR library (used for support set odd steps)')
    parser.add_argument('--negative_data_reshuffling', type=str, default='/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/PanPep_train/reshuffling.txt',
                        help='[mixed mode only] Path to reshuffling negative TCR library (used for support set even steps)')
    parser.add_argument('--model_path', type=str,
                        default='/fs/ess/PAS1475/Fei/code/PanPep_train/checkpoint/alternating_s4q',
                        help='Path to model')
    parser.add_argument('--result_dir', type=str,
                        default='result_alternating/few/alternating_s4q6',
                        help='Directory for results')
    parser.add_argument('--support_dir', type=str, default=None,
                        help='Directory containing pre-saved k-shot CSV files. If provided, load from here instead of generating.')
    parser.add_argument('--peptide_encoding', type=str,
                        default='/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/peptide_b.npz',
                        help='Path to peptide encoding file')
    parser.add_argument('--tcr_encoding', type=str,
                        default='/fs/ess/PAS1475/Fei/code/PanPep_Reusability-main/tcr_b.npz',
                        help='Path to TCR encoding file')
    parser.add_argument('--update_step_test', type=int, default=3,
                        help='Number of inner-loop finetuning steps during test')
    return parser.parse_args()

def load_encodings(encoding_path):
    with np.load(encoding_path) as encodings:
        sequences = encodings['sequences']
        encoding_data = encodings['encodings']
        encoding_dict = {seq: torch.from_numpy(enc).float() for seq, enc in zip(sequences, encoding_data)}
    return encoding_dict

def load_negative_library(path):
    df = pd.read_csv(path)
    return np.array(df).reshape(1, -1).tolist()[0]


# =============================================================================
# Single mode: one negative source, single-round support
# =============================================================================

def process_peptide_single(pep, test_data, test_data_tcr_negative, model, aa_dict, args, config, device, result_dir, file_lock, peptide_encoding_dict, tcr_encoding_dict):
    pep_start_time = time.time()
    print(f"\nProcessing peptide: {pep} on device: {device}")

    all_results = []
    finetuned_net = None

    csv_file_path = os.path.join(result_dir, f"{pep}.csv")
    parquet_file_path = os.path.join(result_dir, f"{pep}.parquet")
    finetuned_model_path = os.path.join(result_dir, f"{pep}_finetuned_params.pt")

    with file_lock:
        file_exists = os.path.exists(csv_file_path) or os.path.exists(parquet_file_path)
        if file_exists:
            print(f"Skipping peptide {pep} - result file already exists")
            return

    positive_tcr = list(test_data[test_data['peptide'] == pep]['binding_TCR'])
    negative_tcr = list(set(test_data_tcr_negative).difference(set(positive_tcr)))
    print(f"Positive TCRs: {len(positive_tcr)}, Negative TCRs: {len(negative_tcr)}")

    all_tcrs = positive_tcr + negative_tcr
    all_labels = [1] * len(positive_tcr) + [0] * len(negative_tcr)
    all_ranking_data = {pep: [all_tcrs, all_labels]}

    batch_count = math.ceil((len(all_ranking_data[pep][1]) - 2*config.support) / config.batch_size)
    print(f"Total batches: {batch_count}")

    breakpoint_idx = 0
    if config.support_dir:
        print(f"Loading k-shot data from: {config.support_dir}")
        support_data = load_support_data(pep, config.support_dir)
    else:
        with file_lock:
            print(f"Generating new k-shot data")
            support_data = save_support_data(all_ranking_data[pep], config.support, pep, result_dir)

    all_query_data = get_query_data(all_ranking_data[pep], support_data, config.support)
    print(f"Query data size: {len(all_query_data[0])}")

    for i in range(breakpoint_idx, batch_count):
        batch_start_time = time.time()
        print(f"\nProcessing batch {i+1}/{batch_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if i != batch_count - 1:
            current_slice = all_query_data[0][i * config.batch_size : (i+1) * config.batch_size]
            current_labels = all_query_data[1][i * config.batch_size : (i+1) * config.batch_size]
        else:
            current_slice = all_query_data[0][i * config.batch_size:]
            current_labels = all_query_data[1][i * config.batch_size:]

        F_data = [support_data[0], support_data[1], current_slice, current_labels]

        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(
            pep, F_data, aa_dict, peptide_encoding_dict, tcr_encoding_dict
        )

        if i == 0:
            print("Finetuning model...")
            end, finetuned_net = model.finetunning(
                peptide_embedding[0].to(device),
                x_spt[0].to(device),
                y_spt[0].to(device),
                x_qry[0].to(device),
                return_params=True
            )
            torch.save(finetuned_net, finetuned_model_path)

            output = pd.DataFrame({
                'CDR3': pd.Series(F_data[2]).astype(str),
                'Score': np.array(end[0], dtype=np.float32),
                'Label': pd.Series(F_data[3]).astype(np.int8)
            })
            all_results.append(output)

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
            all_results.append(output)

        batch_time = time.time() - batch_start_time
        print(f"batch processing time: {batch_time:.2f}s, Progress: {(i+1)/batch_count*100:.1f}%")

    if all_results:
        final_output = pd.concat(all_results, ignore_index=True)
        with file_lock:
            try:
                os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)
                final_output.to_parquet(parquet_file_path, compression='gzip')
                print(f"Successfully wrote {len(final_output)} records to {parquet_file_path}")
            except Exception as e:
                print(f"Error writing to {parquet_file_path}: {e}")

    pep_time = time.time() - pep_start_time
    print(f"\nTotal time for peptide {pep}: {pep_time:.2f}s")


# =============================================================================
# Mixed mode: alternating negative sources, multi-round support
# =============================================================================

def process_peptide_mixed(pep, test_data, negative_sources, model, aa_dict, args, config, device, result_dir, file_lock, peptide_encoding_dict, tcr_encoding_dict):
    pep_start_time = time.time()
    print(f"\nProcessing peptide: {pep} on device: {device}")

    all_results = []
    finetuned_net = None

    csv_file_path = os.path.join(result_dir, f"{pep}.csv")
    parquet_file_path = os.path.join(result_dir, f"{pep}.parquet")
    finetuned_model_path = os.path.join(result_dir, f"{pep}_finetuned_params.pt")

    with file_lock:
        file_exists = os.path.exists(csv_file_path) or os.path.exists(parquet_file_path)
        if file_exists:
            print(f"Skipping peptide {pep} - result file already exists")
            return

    # Build the query pool: positives + negatives from --negative_data (query library)
    positive_tcr = list(test_data[test_data['peptide'] == pep]['binding_TCR'])
    query_neg_library = negative_sources['query']
    negative_tcr = list(set(query_neg_library).difference(set(positive_tcr)))
    print(f"Positive TCRs: {len(positive_tcr)}, Query Negative TCRs: {len(negative_tcr)}")

    all_tcrs = positive_tcr + negative_tcr
    all_labels = [1] * len(positive_tcr) + [0] * len(negative_tcr)
    all_ranking_data = [all_tcrs, all_labels]

    # Load or sample multi-round support sets
    if config.support_dir:
        print(f"Loading k-shot data from: {config.support_dir}")
        rounds_list, all_support_tcrs = load_multi_round_support_data(pep, config.support_dir)
    else:
        rounds_list, all_support_tcrs = sample_multi_round_support_data(
            positive_tcrs=positive_tcr,
            neg_background=negative_sources['background'],
            neg_reshuffling=negative_sources['reshuffling'],
            k_shot=config.support,
            update_step_test=config.update_step_test,
            pep=pep,
            result_dir=result_dir,
        )
    print(f"Multi-round support: {len(rounds_list)} rounds, {len(all_support_tcrs)} unique support TCRs")

    # Embed each round's support set
    multi_support_tensors = []
    for r, round_data in enumerate(rounds_list):
        spt_data = [round_data[0], round_data[1]]
        _, x_spt_r, y_spt_r, _ = task_embedding(
            pep, spt_data, aa_dict, peptide_encoding_dict, tcr_encoding_dict
        )
        multi_support_tensors.append((x_spt_r[0].to(device), y_spt_r[0].to(device)))
        print(f"  Round {r}: {len(round_data[0])} support TCRs embedded")

    # Build query set excluding all support TCRs
    all_query_data = get_query_data_multi_round(all_ranking_data, all_support_tcrs)
    print(f"Query data size: {len(all_query_data[0])}")

    batch_count = math.ceil(len(all_query_data[0]) / config.batch_size)
    print(f"Total batches: {batch_count}")

    for i in range(batch_count):
        batch_start_time = time.time()
        print(f"\nProcessing batch {i+1}/{batch_count} for peptide {pep} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if i != batch_count - 1:
            current_slice = all_query_data[0][i * config.batch_size : (i+1) * config.batch_size]
            current_labels = all_query_data[1][i * config.batch_size : (i+1) * config.batch_size]
        else:
            current_slice = all_query_data[0][i * config.batch_size:]
            current_labels = all_query_data[1][i * config.batch_size:]

        # For task_embedding we use the first round's support just to build query embeddings
        F_data = [rounds_list[0][0], rounds_list[0][1], current_slice, current_labels]

        peptide_embedding, _, _, x_qry = task_embedding(
            pep, F_data, aa_dict, peptide_encoding_dict, tcr_encoding_dict
        )

        if i == 0:
            print("Finetuning model with multi-round support...")
            end, finetuned_net = model.finetunning(
                peptide_embedding[0].to(device),
                multi_support_tensors[0][0],
                multi_support_tensors[0][1],
                x_qry[0].to(device),
                return_params=True,
                multi_support=multi_support_tensors,
            )
            torch.save(finetuned_net, finetuned_model_path)

            scores = end[0]
            output = pd.DataFrame({
                'CDR3': pd.Series(F_data[2]).astype(str),
                'Score': np.array(scores, dtype=np.float32),
                'Label': pd.Series(F_data[3]).astype(np.int8)
            })
            all_results.append(output)
        else:
            print("Using finetuned model for inference...")
            with torch.no_grad():
                end = model.inference_with_params(
                    x_qry[0].to(device),
                    finetuned_net
                )
            scores = end[0]
            output = pd.DataFrame({
                'CDR3': pd.Series(F_data[2]).astype(str),
                'Score': np.array(scores, dtype=np.float32),
                'Label': pd.Series(F_data[3]).astype(np.int8)
            })
            all_results.append(output)

        batch_time = time.time() - batch_start_time
        print(f"batch processing time: {batch_time:.2f}s, Progress: {(i+1)/batch_count*100:.1f}%")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        csv_file_path = os.path.join(result_dir, f"{pep}.csv")
        try:
            final_df.to_csv(csv_file_path, index=False)
            print(f"Successfully saved results to {csv_file_path}")
        except Exception as e:
            print(f"Error saving results to {csv_file_path}: {e}")

    pep_time = time.time() - pep_start_time
    print(f"\nTotal time for peptide {pep}: {pep_time:.2f}s")


# =============================================================================
# Main inference entry point
# =============================================================================

def few_shot_inference(peptide_encoding_dict, tcr_encoding_dict, config):
    total_start_time = time.time()

    gpu_ids = [int(gpu_id) for gpu_id in config.gpu.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Using model configuration: {config.model}")
    print(f"Inference mode: {config.mode}")

    result_dir = os.path.join(Project_path, config.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    args = Args(C=config.distillation, L=75, R=config.distillation, update_lr=0.01,
                update_step_test=config.update_step_test)
    aa_dict = joblib.load(os.path.join(Project_path, Aa_dict))

    test_data = pd.read_csv(config.test_data)
    test_data_pep = sorted(list(set(test_data['peptide'])))

    if config.support_dir:
        print(f"support_dir enabled: will load k-shot data from {config.support_dir}")

    # Load negative libraries based on mode
    if config.mode == 'single':
        test_data_tcr_negative = load_negative_library(config.negative_data)
    else:  # mixed
        test_data_tcr_negative_query = load_negative_library(config.negative_data)
        neg_background = load_negative_library(config.negative_data_background)
        neg_reshuffling = load_negative_library(config.negative_data_reshuffling)
        negative_sources = {
            'query': test_data_tcr_negative_query,
            'background': neg_background,
            'reshuffling': neg_reshuffling,
        }

    manager = Manager()
    file_lock = manager.Lock()

    selected_model_config = MODEL_CONFIG_MAP[config.model]

    def process_peptide_batch(gpu_id, peptide_batch, model_config):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda:0')

        try:
            model = get_model(args, model_config, model_path=config.model_path, device=device)
            model = model.to(device)

            for pep in peptide_batch:
                if config.mode == 'single':
                    process_peptide_single(pep, test_data, test_data_tcr_negative, model, aa_dict,
                                           args, config, device, result_dir, file_lock,
                                           peptide_encoding_dict, tcr_encoding_dict)
                else:  # mixed
                    process_peptide_mixed(pep, test_data, negative_sources, model, aa_dict,
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
        p = Process(target=process_peptide_batch, args=(gpu_id, peptide_batches[gpu_idx], selected_model_config))
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
