import pandas as pd
import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time
from pathlib import Path

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def find_all_files(directory):
    """
    Recursively find all CSV and Parquet files in the directory
    
    Args:
        directory: Root directory to search
        
    Returns:
        list: List of tuples containing (root_path, filename)
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(('.csv', '.parquet')):
                files.append((root, file))
    return files

def find_all_ones(file_path):
    """
    Find all positions where label=1 in the file
    
    Args:
        file_path: Path to the input file
        
    Returns:
        list: List of tuples containing (normalized_position, order)
    """
    try:
        # Choose reading method based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:  # .parquet
            df = pd.read_parquet(file_path)
            
        total_rows = len(df)

        # Find indices where Label is 1
        all_ones_idx = df.index[df['Label'] == 1].to_numpy()
        
        if len(all_ones_idx) > 0:
            normalized_positions = all_ones_idx / total_rows
            orders = np.arange(1, len(all_ones_idx) + 1)
            return list(zip(normalized_positions, orders))
        else:
            return [(-1, 0)]
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return [(-1, 0)]

def process_single_file(args):
    """
    Process a single file to find label=1 positions
    
    Args:
        args: Tuple containing (file_info, input_base_dir)
            file_info: Tuple of (root, filename)
            input_base_dir: Base directory for relative path calculation
            
    Returns:
        list: List of records [directory, filename, position, order]
    """
    (root, filename), input_base_dir = args
    try:
        file_path = os.path.join(root, filename)
        # Get relative path
        rel_path = os.path.relpath(file_path, start=input_base_dir)
        
        # Split into directory path and filename
        directory = os.path.dirname(rel_path)
        
        # Find all positions of ones and their orders
        all_ones_positions = find_all_ones(file_path)
        
        # Return multiple records for each position
        return [[directory, filename, pos, order] for pos, order in all_ones_positions]
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")
        return [[directory, filename, -1, 0]]

def process_directories(input_dir, output_file, num_workers=None):
    """
    Process all files in directories in parallel and generate statistics
    
    Args:
        input_dir: Input directory containing data files
        output_file: Path to save the output CSV file
        num_workers: Number of worker processes to use
    """
    # Find all valid files
    input_files = find_all_files(input_dir)
    
    if not input_files:
        logging.warning(f"No CSV or Parquet files found in {input_dir} and its subdirectories")
        return
    
    total_files = len(input_files)
    logging.info(f"Found {total_files} files to process")
    
    # Set up parallel processing
    num_workers = num_workers or os.cpu_count()
    logging.info(f"Using {num_workers} worker processes")
    
    start_time = time.time()
    results = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_file, (file_info, input_dir)): file_info 
            for file_info in input_files
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), 
                         total=total_files, 
                         desc="Processing files"):
            try:
                result = future.result()
                if result:
                    results.extend(result)
            except Exception as e:
                logging.error(f"Task failed: {e}")
    
    # Create DataFrame from results
    if results:
        columns = ['Directory', 'Filename', 'Position', 'Order']
        results_df = pd.DataFrame(results, columns=columns)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        # Log summary statistics
        elapsed_time = time.time() - start_time
        logging.info(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        logging.info(f"Results saved to: {output_file}")
        logging.info(f"Total records processed: {len(results)}")
        logging.info(f"Average records per file: {len(results)/total_files:.2f}")
        
        # Additional statistics
        successful_files = len([r for r in results if r[2] != -1])
        logging.info(f"Successfully processed files: {successful_files}/{total_files}")
    else:
        logging.error("No valid results generated")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Find positions of label=1 in CSV and Parquet files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_dir',
                       required=True,
                       help='Directory containing input files')
    
    parser.add_argument('--output_file',
                       required=True,
                       help='Path to save the output CSV file')
    
    parser.add_argument('--workers',
                       type=int,
                       default=None,
                       help='Number of worker processes (default: CPU count)')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log configuration
    logging.info("Starting label position finding with configuration:")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Number of workers: {args.workers or 'CPU count'}")
    
    # Process directories
    process_directories(
        input_dir=args.input_dir,
        output_file=args.output_file,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()