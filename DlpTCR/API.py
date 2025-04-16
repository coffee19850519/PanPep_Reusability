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
parser.add_argument('--sample_size', type=int, default=1000,
                    help='Number of samples to process in each chunk')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Number of samples to process in each batch')
parser.add_argument('--model', type=str, default='B',
                    help='model_select,A,B or AB')
args = parser.parse_args()
start = time.time()
input_file_path = args.input_file
job_dir_name = os.path.basename(input_file_path)[:-4]
sample_size = args.sample_size
batch_size = args.batch_size
model_select = args.model
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



def process_chunk(chunk_idx, batch_data, temp_dir, model_select, batch_size):
    batch_dir = os.path.join(temp_dir, f'batch_{chunk_idx}')
    os.makedirs(batch_dir, exist_ok=True)
    print(f"\nProcessing chunk {chunk_idx}")
    print(f"Input batch_data shape: {batch_data.shape}")
    try:
        result = deal_file(batch_data, model_select)
        error_info,TCRA_cdr3,TCRB_cdr3,Epitope,TCRA_pca_features,TCRB_pca_features = result
        print(f"Feature extraction completed for chunk {chunk_idx + 1}")
        batch_output = save_outputfile(
            batch_dir, model_select, batch_data,
            TCRA_cdr3, TCRB_cdr3, Epitope, TCRA_pca_features,
            TCRB_pca_features, batch_size,gpu_id='0'
        )
        if batch_output and os.path.exists(batch_output):
            print(f"Reading results from: {batch_output}")
            return pd.read_csv(batch_output)

    except Exception as e:
        print(f"Error processing chunk {chunk_idx+1}: {str(e)}", "ERROR")
        return None
    
    return None
def save_final_predictions(sorted_predictions, user_dir, model_select):
    if sorted_predictions:
        final_predictions = pd.concat(sorted_predictions, ignore_index=True)

        if model_select == 'AB':
            columns = ['TCRA_CDR3', 'TCRB_CDR3', 'Epitope', 'Predict', 
                       'Probability (TCRA_Epitope)', 'Probability (TCRB_Epitope)']
        elif model_select == 'B':
            columns = ['TCRB_CDR3', 'Epitope', 'Predict', 
                       'Probability (predicted as a positive sample)']
        else:
            columns = ['TCRA_CDR3', 'Epitope', 'Predict', 
                       'Probability (predicted as a positive sample)']

        missing_columns = [col for col in columns if col not in final_predictions.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}", "WARNING")
            return

        final_predictions = final_predictions[columns]

        final_output_path = os.path.join(user_dir, 'final_predictions.parquet')

        if 'TCRB_CDR3' in final_predictions.columns:
            final_predictions['TCRB_CDR3'] = final_predictions['TCRB_CDR3'].astype(str)
        if 'TCRA_CDR3' in final_predictions.columns:
            final_predictions['TCRA_CDR3'] = final_predictions['TCRA_CDR3'].astype(str)
        final_predictions['Epitope'] = final_predictions['Epitope'].astype(str)
        final_predictions['Predict'] = final_predictions['Predict'].astype(str)

        if model_select == 'AB':
            final_predictions['Probability (TCRA_Epitope)'] = final_predictions[
                'Probability (TCRA_Epitope)'].astype(np.float32)
            final_predictions['Probability (TCRB_Epitope)'] = final_predictions[
                'Probability (TCRB_Epitope)'].astype(np.float32)
        else:
            final_predictions['Probability (predicted as a positive sample)'] = final_predictions[
                'Probability (predicted as a positive sample)'].astype(np.float32)

        final_predictions.to_parquet(final_output_path, engine='pyarrow', index=False, compression='gzip')
        print(f"Final results saved to: {final_output_path}")
    else:
        print("No valid predictions were generated", "WARNING")


def main():
    start = time.time()

    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}", "ERROR")
        return

    os.makedirs(user_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file_path,
                         converters={'TCRA_CDR3': validate_sequence,'Epitope': validate_sequence})
        full_input_file = df.dropna(subset=['TCRA_CDR3','Epitope'])
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
                                 batch_size)
            
            if result is not None:
                sorted_predictions.append(result)
                print(f"Successfully processed chunk {chunk_idx+1}/{num_chunks}")

            del batch_data
            if result is not None:
                del result
        save_final_predictions(sorted_predictions, user_dir, model_select)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}", "ERROR")
        raise
    finally:
        end = time.time()
        print(f"Total processing time: {end-start:.2f} seconds")

if __name__ == "__main__":
    main()