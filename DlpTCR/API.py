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
parser.add_argument('--job_dir', type=str, required=True,
                    help='Job directory name')
parser.add_argument('--gpu', type=str, default='2',
                    help='GPU device number(s) to use. e.g., "2" or "0,1,2"')
parser.add_argument('--sample_size', type=int, default=1000,
                    help='Number of samples to process in each chunk')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Number of samples to process in each batch')
args = parser.parse_args()
start = time.time()
input_file_path = args.input_file
job_dir_name = args.job_dir
sample_size = args.sample_size
batch_size = args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model_select = "B"  
user_dir = './newdata/' + str(job_dir_name) + '/'

user_dir_Exists = os.path.exists(user_dir)
if not user_dir_Exists: 
    os.makedirs(user_dir)

full_input_file = pd.read_csv(input_file_path)
total_samples = len(full_input_file)
print(f"Total samples in input file: {total_samples}")

num_chunks= (total_samples + sample_size - 1) // sample_size
print(f"Processing data in {num_chunks} batches of size {sample_size}")

# temp_dir = user_dir + 'temp_batches/'
temp_dir = user_dir + os.path.basename(input_file_path)[:-4]+ '/'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

all_predictions = []
processed_batches = set()

for chunks_idx in range(num_chunks):
    print(f"\nProcessing chunk {chunks_idx+1}/{num_chunks}")

    start_idx = chunks_idx * sample_size
    end_idx = min((chunks_idx + 1) * sample_size, total_samples)

    batch_data = full_input_file.iloc[start_idx:end_idx].reset_index(drop=True)

    batch_dir = temp_dir + f'batch_{chunks_idx}/'
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)

    print(f"Processing chunk {chunks_idx+1} with {end_idx-start_idx} samples")

    if chunks_idx in processed_batches:
        print(f"Chunk {chunks_idx+1} already processed, skipping...")
        continue
    
    try:
        result = deal_file(batch_data, batch_dir, model_select)

        if len(result) == 4:
            error_info, TCRA_cdr3, TCRB_cdr3, Epitope = result
            TCRB_pca_features = None
        elif len(result) == 5:
            error_info, TCRA_cdr3, TCRB_cdr3, Epitope, TCRB_pca_features = result
        else:
            print(f"Unexpected return value from deal_file for chunk {chunks_idx+1}")
            continue

        if error_info != 0:
            print(f"Error in chunk {chunks_idx+1}: error code {error_info}")
            continue

        if TCRA_cdr3 is None or TCRB_cdr3 is None or Epitope is None:
            print(f"No valid data in chunk {chunks_idx+1}")
            continue

        print(f"Feature extraction has done for chunk {chunks_idx + 1}")

        batch_output = save_outputfile(batch_dir, model_select, batch_data, 
                                      TCRA_cdr3, TCRB_cdr3, Epitope, TCRB_pca_features, batch_size)

        if batch_output and os.path.exists(batch_output):
            batch_predictions = pd.read_csv(batch_output)
            print(f"Chunk {chunks_idx+1} predictions shape: {batch_predictions.shape}")
            all_predictions.append(batch_predictions)
            processed_batches.add(chunks_idx)
            print(f"Successfully processed chunk {chunks_idx+1}")

        del result,batch_output

    except Exception as e:
        print(f"Error Processing chunk {chunks_idx+1}: {str(e)}")
        continue

print(f"\nSuccessfully processed {len(processed_batches)} out of {num_chunks} batches")
print(f"Number of prediction dataframes: {len(all_predictions)}")

if all_predictions:
    print("\nMerging predictions in order...")

    sorted_predictions = []
    for chunks_idx in range(num_chunks):
        batch_output_path = temp_dir + f'batch_{chunks_idx}/TCRB_pred.csv'
        if os.path.exists(batch_output_path):
            batch_pred = pd.read_csv(batch_output_path)
            sorted_predictions.append(batch_pred)
            print(f"Added chunk {chunks_idx+1} to final results")

    if sorted_predictions:
        final_predictions = pd.concat(sorted_predictions, ignore_index=True)
        print(f"Final predictions shape: {final_predictions.shape}")
        final_output_path = user_dir + 'final_predictions.csv'
        final_predictions.to_csv(final_output_path, index=False)
        print(f"\nAll batches processed. Final results saved to: {final_output_path}")
    else:
        print("\nNo valid predictions to merge.")
else:
    print("\nNo valid predictions were generated from any chunk.")


end = time.time()
print(f"Total processing time: {end-start:.2f} seconds")






# import os
# import argparse
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
# from Model_Predict_Feature_Extraction import *
# from DLpTCR_server import *
# import time
# # 创建参数解析器
# parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', type=str, required=True,
#                     help='Path to the input file')
# parser.add_argument('--job_dir', type=str, required=True,
#                     help='Job directory name')
# parser.add_argument('--gpu', type=str, default='2',
#                     help='GPU device number(s) to use. e.g., "2" or "0,1,2"')

# args = parser.parse_args()
# start =time.time()
# input_file_path = args.input_file
# job_dir_name = args.job_dir
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# model_select = "B"  
# user_dir = './newdata/' + str(job_dir_name) + '/'

# user_dir_Exists = os.path.exists(user_dir)
# if not user_dir_Exists: 
#     os.makedirs(user_dir)
    
# error_info,TCRA_cdr3,TCRB_cdr3,Epitope,TCRB_pca_features= deal_file(input_file_path, user_dir, model_select)
# output_file_path = save_outputfile(user_dir, model_select , input_file_path,TCRA_cdr3,TCRB_cdr3,Epitope,TCRB_pca_features)
# end =time.time()
# print(end-start)