import os
import pandas as pd
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

def read_file(file_path):
    """
    Read file based on its extension
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def process_single_file(args):
    """
    Process a single file to create samples based on indices
    
    Args:
        args: Tuple containing (src_file, dst_base_folder, file_records)
        
    Returns:
        int: Number of samples created
    """
    try:
        src_file, dst_base_folder, file_records = args
        filename = os.path.basename(src_file)
        base_filename = os.path.splitext(filename)[0]
        
        # Read source file
        df = read_file(src_file)
        if 'CDR3' not in df.columns or 'Label' not in df.columns:
            raise ValueError(f"Required columns missing in {src_file}")
            
        total_samples = 0
        # Process all sampling records for this file
        for _, record in file_records.iterrows():
            try:
                # Get indices
                label_1_indices = eval(record['label_1_indices'])
                label_0_indices = eval(record['label_0_indices'])
                all_indices = label_1_indices + label_0_indices
                
                # Extract corresponding rows
                sampled_data = df.iloc[all_indices]
                
                # Extract sample index from filename
                sample_index = record['filename'].split('_')[-1].split('.')[0]
                
                # Create directory for each sample
                sample_folder = os.path.join(dst_base_folder, f"sample_{sample_index}")
                os.makedirs(sample_folder, exist_ok=True)
                
                # Create output filename
                dst_file = os.path.join(sample_folder, f"{base_filename}_{sample_index}.csv")
                
                # Save data as CSV
                sampled_data.to_csv(dst_file, index=False)
                total_samples += len(sampled_data)
                
            except Exception as e:
                logging.error(f"Error processing sample {record['filename']} for {filename}: {str(e)}")
                continue
                
        logging.info(f"Processed {filename}: created {total_samples} total samples")
        return total_samples
        
    except Exception as e:
        logging.error(f"Error processing file {src_file}: {str(e)}")
        return 0

def find_data_files(base_folder):
    """
    Find all CSV and Parquet files in the directory
    
    Args:
        base_folder: Base directory to search
        
    Returns:
        list: List of file paths
    """
    data_files = []
    for ext in ['*.csv', '*.parquet']:
        for path in Path(base_folder).rglob(ext):
            if 'k_shot' not in path.name:
                data_files.append(str(path))
    return data_files

def apply_indices(src_folder, dst_folder, record_file, num_processes):
    """
    Apply sampling indices to create new datasets
    
    Args:
        src_folder: Source folder containing data files
        dst_folder: Destination folder for output files
        record_file: File containing sampling indices
        num_processes: Number of processes for parallel processing
    """
    # Ensure destination folder exists
    os.makedirs(dst_folder, exist_ok=True)
    logging.info(f"Output directory: {dst_folder}")
    
    # Read indices record file
    records_df = pd.read_csv(record_file)
    logging.info(f"Loaded indices from: {record_file}")
    
    # Find all source files
    data_files = find_data_files(src_folder)
    csv_count = len([f for f in data_files if f.endswith('.csv')])
    parquet_count = len([f for f in data_files if f.endswith('.parquet')])
    
    logging.info(f"Found {len(data_files)} source files:")
    logging.info(f"  CSV files: {csv_count}")
    logging.info(f"  Parquet files: {parquet_count}")
    
    # Prepare process arguments
    process_args = []
    for src_file in data_files:
        filename = os.path.basename(src_file)
        base_filename = os.path.splitext(filename)[0]
        
        # Find matching records
        file_records = records_df[records_df['original_file'].apply(
            lambda x: os.path.splitext(x)[0] == base_filename
        )]
        
        if not file_records.empty:
            process_args.append((src_file, dst_folder, file_records))
    
    logging.info(f"Found {len(process_args)} files with matching records")
    
    # Process files using process pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="Processing files"
        ))
    
    total_samples = sum(results)
    logging.info(f"\nTotal samples created: {total_samples}")
    logging.info(f"Average samples per file: {total_samples/len(process_args):.1f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Apply sampling indices to create new datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--src_folder',
                       required=True,
                       help='Source folder containing data files')
    
    parser.add_argument('--dst_folder',
                       required=True,
                       help='Destination folder for output files')
    
    parser.add_argument('--record_file',
                       required=True,
                       help='File containing sampling indices')
    
    parser.add_argument('--num_processes',
                       type=int,
                       default=20,
                       help='Number of processes for parallel processing')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log configuration
    logging.info("Starting index application with configuration:")
    logging.info(f"Source folder: {args.src_folder}")
    logging.info(f"Destination folder: {args.dst_folder}")
    logging.info(f"Record file: {args.record_file}")
    logging.info(f"Number of processes: {args.num_processes}")
    
    # Apply indices
    apply_indices(
        src_folder=args.src_folder,
        dst_folder=args.dst_folder,
        record_file=args.record_file,
        num_processes=args.num_processes
    )

if __name__ == "__main__":
    main()