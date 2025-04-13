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
    Read data file based on its extension
    
    Args:
        file_path: Path object of the file to read
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def create_samples_for_file(file_path):
    """
    Create 100 random samples for a single file
    
    Args:
        file_path: Path to the input file
        
    Returns:
        list: List of dictionaries containing sampling information
    """
    try:
        logging.info(f"Processing {file_path}")
        df = read_data(Path(file_path))
        
        if 'CDR3' not in df.columns or 'Label' not in df.columns:
            raise ValueError(f"Required columns missing in {file_path}")
        
        filename = os.path.basename(file_path)
        records = []

        # Pre-compute label indices
        label_1_indices = df[df['Label'] == 1].index.values
        label_0_indices = df[df['Label'] == 0].index.values

        # Create 100 samples for the file
        for sample_index in range(100):
            try:
                random_seed = int(time.time()/10) + sample_index
                np.random.seed(random_seed)
                
                min_samples = min(len(label_1_indices), len(label_0_indices))

                if min_samples == 0:
                    # Handle cases with only positive or only negative samples
                    if len(label_1_indices) > 0:
                        sampled_label_1 = label_1_indices
                        sampled_label_0 = []
                    else:
                        sampled_label_1 = []
                        sampled_label_0 = np.random.choice(
                            label_0_indices, 
                            size=min(len(label_0_indices), 1000), 
                            replace=False
                        ).tolist() if label_0_indices.size > 0 else []
                else:
                    # Balanced sampling
                    sampled_label_1 = np.random.choice(label_1_indices, size=min_samples, replace=False).tolist()
                    sampled_label_0 = np.random.choice(label_0_indices, size=min_samples, replace=False).tolist()

                records.append({
                    'filename': f"{os.path.splitext(filename)[0]}_{sample_index}.csv",
                    'original_file': filename,
                    'sample_index': sample_index,
                    'random_seed': random_seed,
                    'label_1_indices': str(sampled_label_1),
                    'label_0_indices': str(sampled_label_0)
                })

            except Exception as e:
                logging.error(f"Error in sample {sample_index} for {filename}: {str(e)}")
                continue

        logging.info(f"Created {len(records)} index records for {filename}")
        return records
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return []

def find_data_files(base_folder):
    """
    Find all valid data files in the base folder
    
    Args:
        base_folder: Base directory to search for files
        
    Returns:
        list: List of paths to valid data files
    """
    data_files = []
    for path in Path(base_folder).rglob('*'):
        if path.suffix in ['.csv', '.parquet'] and 'k_shot' not in path.name:
            data_files.append(str(path))
    return data_files

def generate_all_indices(src_folder, record_file, num_processes):
    """
    Generate sampling indices for all files
    
    Args:
        src_folder: Source folder containing data files
        record_file: Output file for saving indices
        num_processes: Number of processes for parallel processing
    """
    # Ensure output directory exists
    record_path = Path(record_file)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {record_path.parent}")

    # Find all data files
    data_files = find_data_files(src_folder)
    logging.info(f"Found {len(data_files)} data files")
    
    if not data_files:
        logging.error("No valid data files found")
        return
    
    start_time = time.time()
    
    # Process files using process pool
    with Pool(processes=num_processes) as pool:
        all_records_nested = list(tqdm(
            pool.imap(create_samples_for_file, data_files),
            total=len(data_files),
            desc="Processing files"
        ))
    
    # Flatten nested records
    all_records = [record for records in all_records_nested for record in records]

    if all_records:
        records_df = pd.DataFrame(all_records)
        records_df.to_csv(record_file, index=False)
        
        # Log summary statistics
        elapsed_time = time.time() - start_time
        logging.info(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        logging.info(f"Saved {len(all_records)} total records to {record_file}")
        logging.info(f"Average samples per file: {len(all_records)/len(data_files):.1f}")
        
        # Additional statistics
        successful_files = len([records for records in all_records_nested if records])
        logging.info(f"Successfully processed files: {successful_files}/{len(data_files)}")
    else:
        logging.error("No records generated")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate random sampling indices for data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--src_folder',
                       required=True,
                       help='Source folder containing data files')
    
    parser.add_argument('--record_file',
                       required=True,
                       help='Output file for saving indices')
    
    parser.add_argument('--num_processes',
                       type=int,
                       default=64,
                       help='Number of processes for parallel processing')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log configuration
    logging.info("Starting random sampling index generation with configuration:")
    logging.info(f"Source folder: {args.src_folder}")
    logging.info(f"Output file: {args.record_file}")
    logging.info(f"Number of processes: {args.num_processes}")
    
    # Generate indices
    generate_all_indices(
        src_folder=args.src_folder,
        record_file=args.record_file,
        num_processes=args.num_processes
    )

if __name__ == "__main__":
    main()