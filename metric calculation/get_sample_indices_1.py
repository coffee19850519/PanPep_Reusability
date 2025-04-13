import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def merge_csv_in_folder(folder_path):
    """
    Merge all CSV files in the specified folder
    
    Args:
        folder_path: Path to the folder containing CSV files
        
    Returns:
        bool: True if merge was successful, False otherwise
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        return False
    
    # Read and merge all CSV files
    all_dataframes = []
    for file in csv_files:
        try:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
    
    # Merge all dataframes
    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        output_path = os.path.join(folder_path, "merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Merged CSV files in folder {folder_path} to: {output_path}")
        return True
    return False

def process_all_folders(root_folder):
    """
    Recursively process all folders and merge CSV files
    
    Args:
        root_folder: Root directory to start processing from
    """
    merged_count = 0
    # Process all subfolders
    for root, dirs, files in os.walk(root_folder, topdown=False):
        # If current folder contains CSV files, merge them
        if any(file.endswith('.csv') for file in files):
            if merge_csv_in_folder(root):
                merged_count += 1
    
    print(f"\nMerged CSV files in {merged_count} folders")

def process_single_file(args):
    """
    Process a single parquet file and create sampled CSV files
    
    Args:
        args: Tuple containing (src_file, dst_base_folder, file_records, src_base_folder)
    """
    try:
        src_file, dst_base_folder, file_records, src_base_folder = args
        filename = os.path.basename(src_file)
        
        # Use Path objects for more reliable path handling
        src_path = Path(src_file)
        src_base_path = Path(src_base_folder)
        
        # Get relative path to maintain directory structure
        try:
            rel_path = src_path.parent.relative_to(src_base_path)
            rel_path_str = str(rel_path)
        except ValueError:
            rel_path_str = str(src_path.parent.name)
            print(f"Warning: {src_file} is not a subpath of {src_base_folder}, using basename")
        
        # Read source parquet file
        df = pd.read_parquet(src_file)
        total_samples = 0
        
        # Process all sampling records for this file
        for idx, record in enumerate(file_records.iterrows()):
            try:
                # Get indices
                label_1_indices = eval(record[1]['label_1_indices'])
                label_0_indices = eval(record[1]['label_0_indices'])
                all_indices = label_1_indices + label_0_indices
                
                # Extract corresponding rows
                sampled_data = df.iloc[all_indices]
                
                # Build save path while maintaining original directory structure
                if rel_path_str and rel_path_str != '.':
                    sample_folder = os.path.join(dst_base_folder, f"sample_{idx}", rel_path_str)
                else:
                    sample_folder = os.path.join(dst_base_folder, f"sample_{idx}")
                
                # Ensure directory exists
                os.makedirs(sample_folder, exist_ok=True)
                
                # Use original filename as base but change extension to .csv
                dst_file = os.path.join(sample_folder, 
                    os.path.basename(src_file).replace('.parquet', '.csv'))
                
                # Save data as CSV
                sampled_data.to_csv(dst_file, index=False)
                total_samples += len(sampled_data)
                
            except Exception as e:
                print(f"Error processing sample {idx} for {filename}: {str(e)}")
                continue
                
        print(f"Processed {filename} (in {rel_path_str}): created {total_samples} total samples")
        return total_samples
        
    except Exception as e:
        print(f"Error processing file {src_file}: {str(e)}")
        return 0

def find_parquet_files(base_folder):
    """
    Find all parquet files in the base folder, excluding those with 'k_shot' in the name
    
    Args:
        base_folder: Base directory to search for parquet files
    """
    parquet_files = []
    for path in Path(base_folder).rglob('*.parquet'):
        if 'k_shot' not in path.name:
            parquet_files.append(str(path))
    return parquet_files

def apply_indices(src_folder, dst_folder, record_file, num_processes, merge_after=True):
    """
    Apply sampling indices to create new datasets and optionally merge results
    
    Args:
        src_folder: Source folder containing parquet files
        dst_folder: Destination folder for output CSV files
        record_file: CSV file containing sampling indices
        num_processes: Number of processes for parallel processing
        merge_after: Whether to merge CSV files after sampling
    """
    # Ensure destination folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"Created base directory: {dst_folder}")
    
    # Read indices record file
    records_df = pd.read_csv(record_file)
    
    # Get all source files
    parquet_files = find_parquet_files(src_folder)
    print(f"Found {len(parquet_files)} source parquet files")
    
    # Prepare process arguments
    process_args = []
    for src_file in parquet_files:
        filename = os.path.basename(src_file)
        file_records = records_df[records_df['original_file'] == filename]
        if not file_records.empty:
            process_args.append((src_file, dst_folder, file_records, src_folder))
        else:
            print(f"No matching records found for file: {filename}")
    
    # Process files using process pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="Processing files"
        ))
    
    total_samples = sum(results)
    print(f"\nTotal samples created: {total_samples}")
    
    # Merge CSV files if requested
    if merge_after:
        print("\nStarting CSV file merging process...")
        process_all_folders(dst_folder)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Apply sampling indices and merge resulting CSV files'
    )
    
    parser.add_argument('--src_folder',
                       required=True,
                       help='Source folder containing parquet files')
    
    parser.add_argument('--dst_folder',
                       required=True,
                       help='Destination folder for output CSV files')
    
    parser.add_argument('--record_file',
                       required=True,
                       help='CSV file containing sampling indices')
    
    parser.add_argument('--num_processes',
                       type=int,
                       default=64,
                       help='Number of processes for parallel processing (default: 64)')
    
    parser.add_argument('--skip_merge',
                       action='store_true',
                       help='Skip merging CSV files after sampling')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    args = parse_arguments()
    
    apply_indices(
        src_folder=args.src_folder,
        dst_folder=args.dst_folder,
        record_file=args.record_file,
        num_processes=args.num_processes,
        merge_after=not args.skip_merge
    )

if __name__ == "__main__":
    main()