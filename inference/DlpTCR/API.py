"""
TCR-Epitope Binding Prediction API

This module provides a command-line interface for large-scale TCR-Epitope binding prediction.
It supports batch processing of sequences using either TCRA, TCRB, or both chains for prediction.

Features:
- Chunk-based processing for memory efficiency
- Parallel processing support
- Multiple model types (A: TCRA, B: TCRB, AB: Both chains)
- Automatic input validation
- Progress tracking and error handling
"""

import os
import argparse
import pandas as pd
import numpy as np
from Model_Predict_Feature_Extraction import *
from DLpTCR_server import *
import time

# Create argument parser for command line interface
parser = argparse.ArgumentParser(description='TCR-Epitope Binding Prediction Tool')
parser.add_argument('--input_file', type=str, required=True,
                    help='Path to the input CSV file containing sequences')
parser.add_argument('--sample_size', type=int, default=1000,
                    help='Number of samples to process in each chunk for memory efficiency')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size for model prediction')
parser.add_argument('--model', type=str, default='B',
                    help='Model selection: A (TCRA), B (TCRB), or AB (both chains)')

# Parse command line arguments
args = parser.parse_args()
start = time.time()
input_file_path = args.input_file
job_dir_name = os.path.basename(input_file_path)[:-4]
sample_size = args.sample_size
batch_size = args.batch_size
model_select = args.model
user_dir = './newdata/' + str(job_dir_name) + '/'

# Create output directory if it doesn't exist
user_dir_Exists = os.path.exists(user_dir)
if not user_dir_Exists: 
    os.makedirs(user_dir)

def validate_sequence(seq):
    """
    Validate amino acid sequences for TCR and epitope inputs.
    
    Performs the following checks:
    1. Not null/NaN
    2. Contains only valid amino acid characters
    3. Is a valid string
    
    Args:
        seq (str): Input amino acid sequence
        
    Returns:
        str or None: Validated sequence if valid, None if invalid
        
    Note:
        Valid amino acids are: ARNDCQEGHILKMFPSTWYV
    """
    if pd.isna(seq): 
        return None
    valid_aas = set('ARNDCQEGHILKMFPSTWYV')
    if not seq or not isinstance(seq, str): 
        return None
    if not set(seq).issubset(valid_aas): 
        return None
    return seq

def process_chunk(chunk_idx, batch_data, temp_dir, model_select, batch_size):
    """
    Process a single chunk of sequences through the prediction pipeline.
    
    This function handles:
    1. Feature extraction for the batch
    2. Model prediction
    3. Result saving
    
    Args:
        chunk_idx (int): Index of the current chunk
        batch_data (pd.DataFrame): Data for current batch
        temp_dir (str): Directory for temporary files
        model_select (str): Model type (A/B/AB)
        batch_size (int): Batch size for prediction
        
    Returns:
        pd.DataFrame or None: Prediction results if successful, None if failed
        
    Note:
        Creates a separate directory for each batch to avoid conflicts
    """
    # Create batch-specific directory
    batch_dir = os.path.join(temp_dir, f'batch_{chunk_idx}')
    os.makedirs(batch_dir, exist_ok=True)
    print(f"\nProcessing chunk {chunk_idx}")
    print(f"Input batch_data shape: {batch_data.shape}")
    
    try:
        # Extract features
        result = deal_file(batch_data, model_select)
        error_info, TCRA_cdr3, TCRB_cdr3, Epitope, TCRA_pca_features, TCRB_pca_features = result
        print(f"Feature extraction completed for chunk {chunk_idx + 1}")
        
        # Generate predictions
        batch_output = save_outputfile(
            batch_dir, model_select, batch_data,
            TCRA_cdr3, TCRB_cdr3, Epitope, TCRA_pca_features,
            TCRB_pca_features, batch_size, gpu_id='0'
        )
        
        # Read and return results if successful
        if batch_output and os.path.exists(batch_output):
            print(f"Reading results from: {batch_output}")
            return pd.read_csv(batch_output)

    except Exception as e:
        print(f"Error processing chunk {chunk_idx+1}: {str(e)}", "ERROR")
        return None
    
    return None

def save_final_predictions(sorted_predictions, user_dir, model_select):
    """
    Combine and save all batch predictions into a final output file.
    
    This function:
    1. Concatenates all batch predictions
    2. Validates and organizes columns based on model type
    3. Converts data types for efficient storage
    4. Saves results in parquet format with compression
    
    Args:
        sorted_predictions (list): List of DataFrames containing batch predictions
        user_dir (str): Directory to save final results
        model_select (str): Model type (A/B/AB) determining output format
        
    Note:
        Output columns vary by model type:
        - AB: TCRA_CDR3, TCRB_CDR3, Epitope, Predict, Probability (TCRA_Epitope), Probability (TCRB_Epitope)
        - B: TCRB_CDR3, Epitope, Predict, Probability
        - A: TCRA_CDR3, Epitope, Predict, Probability
    """
    if sorted_predictions:
        # Combine all predictions
        final_predictions = pd.concat(sorted_predictions, ignore_index=True)

        # Define columns based on model type
        if model_select == 'AB':
            columns = ['TCRA_CDR3', 'TCRB_CDR3', 'Epitope', 'Predict', 
                       'Probability (TCRA_Epitope)', 'Probability (TCRB_Epitope)']
        elif model_select == 'B':
            columns = ['TCRB_CDR3', 'Epitope', 'Predict', 
                       'Probability (predicted as a positive sample)']
        else:  # model_select == 'A'
            columns = ['TCRA_CDR3', 'Epitope', 'Predict', 
                       'Probability (predicted as a positive sample)']

        # Validate columns
        missing_columns = [col for col in columns if col not in final_predictions.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}", "WARNING")
            return

        # Select and order columns
        final_predictions = final_predictions[columns]
        final_output_path = os.path.join(user_dir, 'final_predictions.parquet')

        # Convert data types for efficient storage
        if 'TCRB_CDR3' in final_predictions.columns:
            final_predictions['TCRB_CDR3'] = final_predictions['TCRB_CDR3'].astype(str)
        if 'TCRA_CDR3' in final_predictions.columns:
            final_predictions['TCRA_CDR3'] = final_predictions['TCRA_CDR3'].astype(str)
        final_predictions['Epitope'] = final_predictions['Epitope'].astype(str)
        final_predictions['Predict'] = final_predictions['Predict'].astype(str)

        # Convert probability columns
        if model_select == 'AB':
            final_predictions['Probability (TCRA_Epitope)'] = final_predictions[
                'Probability (TCRA_Epitope)'].astype(np.float32)
            final_predictions['Probability (TCRB_Epitope)'] = final_predictions[
                'Probability (TCRB_Epitope)'].astype(np.float32)
        else:
            final_predictions['Probability (predicted as a positive sample)'] = final_predictions[
                'Probability (predicted as a positive sample)'].astype(np.float32)

        # Save to parquet format with compression
        final_predictions.to_parquet(final_output_path, engine='pyarrow', index=False, compression='gzip')
        print(f"Final results saved to: {final_output_path}")
    else:
        print("No valid predictions were generated", "WARNING")

def main():
    """
    Main execution function for the TCR-Epitope binding prediction pipeline.
    
    Workflow:
    1. Validates input file existence
    2. Creates output directory
    3. Reads and validates input sequences
    4. Processes data in chunks to manage memory
    5. Combines and saves final results
    
    The function handles:
    - Memory-efficient processing of large datasets
    - Progress tracking
    - Error handling and reporting
    - Timing of the complete process
    
    Note:
        Uses the following global variables:
        - input_file_path: Path to input CSV file
        - user_dir: Output directory path
        - model_select: Selected model type
        - sample_size: Chunk size for processing
        - batch_size: Batch size for model prediction
    """
    start = time.time()

    # Validate input file
    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}", "ERROR")
        return

    # Ensure output directory exists
    os.makedirs(user_dir, exist_ok=True)

    try:
        # Read and validate input data
        df = pd.read_csv(input_file_path,
                         converters={'TCRA_CDR3': validate_sequence,
                                   'Epitope': validate_sequence})
        full_input_file = df.dropna(subset=['TCRA_CDR3','Epitope'])
        del df  # Free memory

        # Calculate processing chunks
        total_samples = len(full_input_file)
        num_chunks = (total_samples + sample_size - 1) // sample_size
        print(f"Processing {total_samples} samples in {num_chunks} chunks")

        # Process data in chunks
        sorted_predictions = []
        for chunk_idx in range(num_chunks):
            # Extract chunk
            start_idx = chunk_idx * sample_size
            end_idx = min((chunk_idx + 1) * sample_size, total_samples)
            
            # Process chunk
            batch_data = full_input_file.iloc[start_idx:end_idx].reset_index(drop=True)
            result = process_chunk(chunk_idx, batch_data, user_dir, model_select, 
                                 batch_size)
            
            # Store results
            if result is not None:
                sorted_predictions.append(result)
                print(f"Successfully processed chunk {chunk_idx+1}/{num_chunks}")

            # Clean up memory
            del batch_data
            if result is not None:
                del result

        # Save final results
        save_final_predictions(sorted_predictions, user_dir, model_select)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}", "ERROR")
        raise
    finally:
        end = time.time()
        print(f"Total processing time: {end-start:.2f} seconds")

if __name__ == "__main__":
    main()