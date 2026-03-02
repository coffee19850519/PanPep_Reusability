import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
import argparse
from pathlib import Path

def read_file(file_path, usecols):
    """Read file based on its extension"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, usecols=usecols, engine='c', memory_map=True)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
        return df[usecols]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def sort_single_file(file_path, output_dir):
    """
    Process a single file (preserving original format)
    
    Args:
        file_path (str): Path to the input file
        output_dir (str): Directory where sorted files will be saved
        
    Returns:
        tuple: (base_name, labels_array) or None if processing fails
    """
    try:
        # Only read required columns
        usecols = ['CDR3', 'Score', 'Label']
        df = read_file(file_path, usecols)
        
        # Maintain original folder structure
        rel_path = os.path.relpath(os.path.dirname(file_path), start=os.path.dirname(output_dir))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1]  # Get original file extension
        
        # Create corresponding output directory
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Sort the dataframe
        df_sorted = df.sort_values(by='Score',
                                 ascending=False,
                                 ignore_index=True,
                                 inplace=False)
        
        # Save file maintaining original format
        output_path = os.path.join(output_subdir, f'{base_name}_sorted{ext}')
        if ext == '.csv':
            df_sorted.to_csv(output_path, index=False)
        else:  # .parquet
            df_sorted.to_parquet(output_path)
        
        return base_name, df_sorted['Label'].to_numpy()
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def find_all_files(directory):
    """
    Recursively find all CSV and Parquet files in the directory
    
    Args:
        directory (str): Root directory to search
        
    Returns:
        list: List of file paths
    """
    files = []
    for ext in ['.csv', '.parquet']:
        files.extend([str(f) for f in Path(directory).rglob(f'*{ext}')])
    return files

def sort_files(input_dir, output_dir, num_processes=None):
    """
    Process all files in directories in parallel
    
    Args:
        input_dir (str): Input directory containing files to process
        output_dir (str): Directory where sorted files will be saved
        num_processes (int, optional): Number of processes to use
        
    Returns:
        dict: Dictionary mapping file names to their label arrays
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursively get all supported files
    files = find_all_files(input_dir)
    
    if not files:
        print(f"No CSV or Parquet files found in {input_dir}")
        return {}
    
    # Count file formats
    csv_count = len([f for f in files if f.endswith('.csv')])
    parquet_count = len([f for f in files if f.endswith('.parquet')])
    print(f"Found {len(files)} files to process:")
    print(f"  CSV files: {csv_count}")
    print(f"  Parquet files: {parquet_count}")
    
    # Create process pool
    with Pool(processes=num_processes) as pool:
        # Use partial to fix output_dir parameter
        sort_func = partial(sort_single_file, output_dir=output_dir)
        
        # Process all files in parallel
        results = pool.map(sort_func, files)
    
    # Filter out None results and create dictionary
    results_dict = {name: labels for result in results 
                   if result is not None 
                   for name, labels in [result]}
    
    return results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process and sort CSV and Parquet files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory path containing files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory path for sorted files')
    parser.add_argument('--num_cores', type=int, default=None,
                      help='Number of CPU cores to use, defaults to all available cores')

    args = parser.parse_args()
    
    # Get number of CPU cores
    import multiprocessing
    num_cores = args.num_cores or 64
    
    # Process using command line arguments
    results = sort_files(args.input_dir, 
                        args.output_dir,
                        num_processes=num_cores)
    
    # Print processing results
    print(f"\nCompleted processing {len(results)} files")