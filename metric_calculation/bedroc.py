import os
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from skfp.metrics import bedroc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def calculate_metrics(file_path):
    """
    Calculate BEDROC score for a single file.

    The file must contain at least the following columns:
        - 'Label'
        - 'Score'
    """
    try:
        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.parquet', '.par')):
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None

        labels = df['Label'].to_numpy()
        scores = df['Score'].to_numpy()

        try:
            # Directly compute BEDROC score with alpha=20
            bedroc = bedroc_score(labels, y_score=scores, alpha=20)
        except Exception as e:
            print(f"Error calculating BEDROC score for file {file_path}: {e}")
            return None

        print(f"\nFile: {os.path.basename(file_path)}")
        return {
            'Directory': os.path.dirname(file_path),
            'Filename': os.path.basename(file_path),
            'BEDROC_Score': bedroc,
            'Alpha': 20
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def process_file(args):
    """
    Wrapper for processing a single file.

    Args:
        args: tuple(file_path, temp_dir, root_dir)
    """
    file_path, temp_dir, root_dir = args
    try:
        results = calculate_metrics(file_path)
        if results is None:
            return None

        # Build output directory and file path
        rel_path = os.path.relpath(file_path, start=root_dir)
        output_dir = os.path.join(temp_dir, os.path.dirname(rel_path))
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"temp_{os.path.basename(file_path).split('.')[0]}.csv"
        )
        pd.DataFrame([results]).to_csv(output_file, index=False)

        return output_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def calculate_summary(output_dir):
    """
    Aggregate all detailed results and compute summary statistics per directory.

    Args:
        output_dir: Directory containing detailed result files
    """
    logging.info("Starting summary calculation...")

    # Collect all temp CSV files
    all_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.startswith('temp_') and file.endswith('.csv'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        logging.warning("No result files found.")
        return

    # Read all result files
    all_dfs = []
    logging.info(f"Reading {len(all_files)} result files...")
    for file in tqdm(all_files, desc="Reading detailed results"):
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")

    if not all_dfs:
        logging.error("No valid data available for summary.")
        return

    # Concatenate all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Group by directory and compute statistics
    summary_df = (
        combined_df.groupby('Directory')
        .agg({
            'BEDROC_Score': ['mean', 'std', 'count']
        })
        .reset_index()
    )

    # Flatten column names
    summary_df.columns = ['Directory', 'BEDROC_Mean', 'BEDROC_Std', 'File_Count']
    summary_df = summary_df.sort_values('BEDROC_Mean', ascending=False)

    # Save summary
    summary_file = os.path.join(output_dir, 'summary_by_directory.csv')
    summary_df.to_csv(summary_file, index=False)

    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"Total files processed: {len(combined_df)}")
    logging.info(f"Total unique directories: {len(summary_df)}")
    logging.info(f"Global mean BEDROC score: {summary_df['BEDROC_Mean'].mean():.4f}")

    # Print top 10 directories
    print("Top 10 directories by BEDROC score:")
    print(summary_df.head(10).to_string(index=False))

    return summary_file


def merge_by_directory(output_dir):
    """
    Merge all per-file CSV results into one file per top-level directory.

    Args:
        output_dir: Directory containing detailed result files
    """
    logging.info("Starting per-directory merge of detailed results...")

    # Collect all result files and group them by directory
    dir_files = {}
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.startswith('temp_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)

                # Get parent directory relative to output_dir
                rel_path = os.path.relpath(file_path, output_dir)
                parent_dir = os.path.dirname(rel_path)

                # Only keep the first-level directory name
                if parent_dir and '/' in parent_dir:
                    first_level = parent_dir.split('/')[0]
                else:
                    first_level = parent_dir

                # Ignore files directly in the root of output_dir
                if first_level:
                    dir_files.setdefault(first_level, []).append(file_path)

    if not dir_files:
        logging.warning("No directories found for merging.")
        return

    logging.info(f"Found {len(dir_files)} directories to merge.")

    merged_dir = os.path.join(output_dir, 'merged_by_directory')
    os.makedirs(merged_dir, exist_ok=True)

    # Merge files for each directory
    for dir_name, file_list in tqdm(dir_files.items(), desc="Merging directories"):
        try:
            dfs = []
            for file_path in file_list:
                try:
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                except Exception as e:
                    logging.error(f"Failed to read file {file_path}: {e}")

            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)

                # Sanitize directory name for filename
                safe_dir_name = dir_name.replace('/', '_').replace('\\', '_')
                output_file = os.path.join(merged_dir, f'{safe_dir_name}_merged.csv')
                merged_df.to_csv(output_file, index=False)

                logging.info(
                    f"Directory '{dir_name}': merged {len(dfs)} files -> {output_file}"
                )
        except Exception as e:
            logging.error(f"Error while merging directory {dir_name}: {e}")

    logging.info(f"All merged files saved in: {merged_dir}/")
    print(f"✅ Generated {len(dir_files)} merged files in: {merged_dir}/")

    return merged_dir


def process_directory(root_dir, output_dir="bedroc_results", num_processes=60):
    """
    Main processing function.

    - Recursively search for input files
    - Run BEDROC calculation in parallel
    - Generate summary statistics
    - Merge detailed results per directory
    """
    temp_dir = output_dir
    os.makedirs(temp_dir, exist_ok=True)

    file_list = []
    files_to_process = []

    # Walk through input directory and collect files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.csv', '.parquet', '.par')):
                file_path = os.path.join(dirpath, filename)
                file_list.append(file_path)

                # Check whether this file has already been processed
                rel_path = os.path.relpath(file_path, start=root_dir)
                output_subdir = os.path.join(temp_dir, os.path.dirname(rel_path))
                output_file = os.path.join(
                    output_subdir,
                    f"temp_{os.path.basename(file_path).split('.')[0]}.csv"
                )

                if not os.path.exists(output_file):
                    files_to_process.append(file_path)

    print(f"Total: {len(file_list)} files, To process: {len(files_to_process)} files")

    if not files_to_process:
        logging.info("All files have already been processed.")
        # Even if no new files, still compute summary and merged outputs
        calculate_summary(temp_dir)
        merge_by_directory(temp_dir)
        return

    # Parallel processing of remaining files
    with Pool(processes=num_processes) as pool:
        result_files = list(tqdm(
            pool.imap(process_file, [(f, temp_dir, root_dir) for f in files_to_process]),
            total=len(files_to_process),
            desc="Processing files"
        ))

    # Filter valid results
    result_files = [f for f in result_files if f is not None]
    if result_files:
        logging.info(f"Successfully processed {len(result_files)} files.")
        logging.info(f"Detailed results stored in: {temp_dir}/")

        # Compute summary statistics
        calculate_summary(temp_dir)

        # Merge detailed results by directory
        merge_by_directory(temp_dir)
    else:
        logging.warning("No valid results were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch compute BEDROC scores and generate summary reports.'
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to the input directory.'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='bedroc_results',
        help='Path to the output directory.'
    )
    parser.add_argument(
        '-n', '--num_processes', type=int, default=8,
        help='Number of parallel processes.'
    )

    args = parser.parse_args()

    logging.info(f"Input directory: {args.input}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Number of processes: {args.num_processes}")
    process_directory(args.input, args.output, args.num_processes)
