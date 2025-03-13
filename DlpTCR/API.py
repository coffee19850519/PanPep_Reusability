import os
import argparse
import pandas as pd
import numpy as np
from Model_Predict_Feature_Extraction import *
from DLpTCR_server import *
import time

# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True,
                    help='Path to the input file')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device number(s) to use. e.g., "2" or "0,1,2"')
parser.add_argument('--sample_size', type=int, default=1000,
                    help='Number of samples to process in each chunk')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Number of samples to process in each batch')
args = parser.parse_args()
start = time.time()
input_file_path = args.input_file
job_dir_name = os.path.basename(input_file_path)[:-4]
sample_size = args.sample_size
batch_size = args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model_select = "B"  
user_dir = './newdata/' + str(job_dir_name) + '/'

user_dir_Exists = os.path.exists(user_dir)
if not user_dir_Exists: 
    os.makedirs(user_dir)
def validate_sequence(seq):
    if pd.isna(seq): 
        return None
    valid_aas = set('ARNDCQEGHILKMFPSTWYV')
    if not seq or not isinstance(seq, str): 
        return None
    if not set(seq).issubset(valid_aas): 
        return None
    return seq



def process_chunk(chunk_idx, batch_data, temp_dir, model_select, batch_size, gpu):
    batch_dir = os.path.join(temp_dir, f'batch_{chunk_idx}')
    os.makedirs(batch_dir, exist_ok=True)
    print(f"\nProcessing chunk {chunk_idx}")
    print(f"Input batch_data shape: {batch_data.shape}")
    try:
        result = deal_file(batch_data, model_select)


        if not isinstance(result, tuple) or len(result) not in [4, 5]:
            print(f"Invalid result format in chunk {chunk_idx+1}", "ERROR")
            return None
            
        if len(result) == 4:
            error_info, TCRA_cdr3, TCRB_cdr3, Epitope = result
            TCRB_pca_features = None
        else:
            error_info, TCRA_cdr3, TCRB_cdr3, Epitope, TCRB_pca_features = result
            
        print(f"Feature extraction completed for chunk {chunk_idx + 1}")
        batch_output = save_outputfile(
            batch_dir, model_select, batch_data,
            TCRA_cdr3, TCRB_cdr3, Epitope, 
            TCRB_pca_features, batch_size, gpu
        )
        if batch_output and os.path.exists(batch_output):
            print(f"Reading results from: {batch_output}")
            return pd.read_csv(batch_output)

    except Exception as e:
        print(f"Error processing chunk {chunk_idx+1}: {str(e)}", "ERROR")
        return None
    
    return None

def main():
    start = time.time()

    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}", "ERROR")
        return

    os.makedirs(user_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file_path, converters={'TCRB_CDR3': validate_sequence})
        full_input_file = df.dropna(subset=['TCRB_CDR3'])
        del df 
        
        total_samples = len(full_input_file)
        num_chunks = (total_samples + sample_size - 1) // sample_size
        print(f"Processing {total_samples} samples in {num_chunks} chunks")

        sorted_predictions = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * sample_size
            end_idx = min((chunk_idx + 1) * sample_size, total_samples)
            
            batch_data = full_input_file.iloc[start_idx:end_idx].reset_index(drop=True)
            result = process_chunk(chunk_idx, batch_data, user_dir, model_select, 
                                 batch_size, args.gpu)
            
            if result is not None:
                sorted_predictions.append(result)
                print(f"Successfully processed chunk {chunk_idx+1}/{num_chunks}")

            del batch_data
            if result is not None:
                del result

        if sorted_predictions:
            final_predictions = pd.concat(sorted_predictions, ignore_index=True)
            final_output_path = os.path.join(user_dir, 'final_predictions.csv')
            final_predictions.to_csv(final_output_path, index=False)
            print(f"Final results saved to: {final_output_path}")
        else:
            print("No valid predictions were generated", "WARNING")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}", "ERROR")
        raise
    finally:
        end = time.time()
        print(f"Total processing time: {end-start:.2f} seconds")

if __name__ == "__main__":
    main()