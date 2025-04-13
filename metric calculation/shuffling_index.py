import os
import pandas as pd
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_data(file_path):
    """
    Automatically choose reading method based on file extension
    
    Args:
        file_path: Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def load_allowed_negatives(txt_file):
    """
    Load allowed negative CDR3 sequences from file
    
    Args:
        txt_file: Path to text file containing allowed CDR3 sequences
        
    Returns:
        set: Set of allowed CDR3 sequences
    """
    try:
        with open(txt_file, 'r') as f:
            allowed_cdr3s = set(line.strip() for line in f if line.strip())
        logging.info(f"Loaded {len(allowed_cdr3s)} allowed negative CDR3 sequences")
        return allowed_cdr3s
    except Exception as e:
        logging.error(f"Error loading allowed negatives: {str(e)}")
        return set()

def find_data_files(base_folder):
    """
    Find all valid data files in the base folder
    
    Args:
        base_folder: Base directory to search for files
        
    Returns:
        list: List of paths to valid data files
    """
    return [str(path) for path in Path(base_folder).rglob('*') 
            if path.suffix in ['.csv', '.parquet'] and 'k_shot' not in path.name]

def process_file(args):
    """
    Process a single file to generate all 100 sampling indices
    
    Args:
        args: Tuple containing (file_path, allowed_negatives_file)
        
    Returns:
        list: List of sampling records for all 100 samples
    """
    file_path, allowed_negatives_file = args
    try:
        logging.info(f"Processing file: {file_path}")
        df = read_data(file_path)
        
        # Check required columns
        required_columns = ['CDR3', 'Label']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {file_path}")
        
        allowed_cdr3s = load_allowed_negatives(allowed_negatives_file)
        if not allowed_cdr3s:
            raise ValueError(f"Failed to load allowed negatives")
        
        # Pre-compute positive and negative indices once for this file
        pos_indices = df[df['Label'] == 1].index.values
        positive_cdr3s = set(df.loc[pos_indices, 'CDR3']) if len(pos_indices) > 0 else set()
        
        # Pre-filter valid negative samples
        valid_neg_indices = df[(
            (df['Label'] == 0) & 
            (df['CDR3'].isin(allowed_cdr3s)) & 
            (~df['CDR3'].isin(positive_cdr3s))
        )].index.values if len(pos_indices) > 0 else df[(
            (df['Label'] == 0) & 
            (df['CDR3'].isin(allowed_cdr3s))
        )].index.values
        
        records = []
        # Process all 100 samples sequentially for this file
        for sample_index in range(100):
            # Set random seed
            random_seed = int(time.time()/100) + sample_index
            np.random.seed(random_seed)
            
            sampled_pos = []
            sampled_neg = []
            
            # Handle positive samples
            if len(pos_indices) > 0:
                # Use all positive samples
                sampled_pos = pos_indices.tolist()
                
                # Select equal number of negative samples
                if len(valid_neg_indices) > 0:
                    np.random.seed(random_seed + 1)
                    neg_count = min(len(valid_neg_indices), len(sampled_pos))
                    sampled_neg = np.random.choice(valid_neg_indices, size=neg_count, replace=False).tolist()
            
            # If no positive samples but have allowed negatives
            elif len(allowed_cdr3s) > 0 and len(valid_neg_indices) > 0:
                np.random.seed(random_seed + 100)
                sample_size = 10000
                sampled_neg = np.random.choice(valid_neg_indices, size=sample_size, replace=False).tolist()

            records.append({
                'filename': f"{os.path.splitext(os.path.basename(file_path))[0]}_{sample_index}.csv",
                'original_file': os.path.basename(file_path),
                'sample_index': sample_index,
                'random_seed': random_seed,
                'label_1_indices': str(sampled_pos),
                'label_0_indices': str(sampled_neg)
            })
        
        logging.info(f"Created {len(records)} index records for {os.path.basename(file_path)}")
        return records
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return []

def generate_all_indices(src_folder, allowed_negatives_file, record_file, num_processes):
    """
    Generate sampling indices for all files
    
    Args:
        src_folder: Source folder containing data files
        allowed_negatives_file: File containing allowed negative sequences
        record_file: Output file for saving indices
        num_processes: Number of processes for parallel processing
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(record_file), exist_ok=True)
    
    # Find all valid data files
    data_files = find_data_files(src_folder)
    logging.info(f"Found {len(data_files)} data files")
    
    if not data_files:
        logging.error("No valid data files found in the source directory")
        return
    
    start_time = time.time()
    
    # Create process pool for parallel file processing
    with Pool(processes=num_processes) as pool:
        # Process all files in parallel
        all_results = list(tqdm(
            pool.imap(
                process_file, 
                [(f, allowed_negatives_file) for f in data_files],
                chunksize=1
            ),
            total=len(data_files),
            desc="Processing peptide files"
        ))
    
    # Flatten results from all files
    all_records = [record for file_records in all_results for record in file_records]

    if all_records:
        records_df = pd.DataFrame(all_records)
        records_df.to_csv(record_file, index=False)
        
        # Log summary statistics
        elapsed_time = time.time() - start_time
        logging.info(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        logging.info(f"Saved {len(all_records)} records to {record_file}")
        logging.info(f"Processed {len(data_files)} files with {len(all_records)/len(data_files):.0f} samples each")
        
        # Additional statistics
        successful_files = len([r for r in all_results if r])
        logging.info(f"Successfully processed files: {successful_files}/{len(data_files)}")
    else:
        logging.error("No records generated")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate sampling indices for peptide data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--src_folder',
                       required=True,
                       help='Source folder containing data files')
    
    parser.add_argument('--allowed_negatives_file',
                       required=True,
                       help='File containing allowed negative sequences')
    
    parser.add_argument('--record_file',
                       required=True,
                       help='Output file for saving indices')
    
    parser.add_argument('--num_processes',
                       type=int,
                       default=os.cpu_count(),
                       help='Number of processes for parallel processing')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log configuration
    logging.info("Starting sampling index generation with configuration:")
    logging.info(f"Source folder: {args.src_folder}")
    logging.info(f"Allowed negatives file: {args.allowed_negatives_file}")
    logging.info(f"Output file: {args.record_file}")
    logging.info(f"Number of processes: {args.num_processes}")
    
    # Generate indices
    generate_all_indices(
        src_folder=args.src_folder,
        allowed_negatives_file=args.allowed_negatives_file,
        record_file=args.record_file,
        num_processes=args.num_processes
    )

if __name__ == "__main__":
    main()