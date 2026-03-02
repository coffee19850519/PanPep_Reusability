import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import concurrent.futures

def calculate_metrics(csv_path):
    """Calculate ROC-AUC and PR-AUC for a single CSV file"""
    try:
        df = pd.read_csv(csv_path)
        y_true = df['label'].values
        y_scores = df['score'].values
        
        # Check if there are enough positive and negative samples
        if len(np.unique(y_true)) != 2:
            return {
                'folder_path': os.path.dirname(csv_path),
                'file_name': os.path.basename(csv_path),
                'roc_auc': None,
                'pr_auc': None,
                'status': 'insufficient samples'
            }
        
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        return {
            'folder_path': os.path.dirname(csv_path),
            'file_name': os.path.basename(csv_path),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'status': 'success'
        }
    except Exception as e:
        return {
            'folder_path': os.path.dirname(csv_path),
            'file_name': os.path.basename(csv_path),
            'roc_auc': None,
            'pr_auc': None,
            'status': f'error: {str(e)}'
        }

def find_all_csvs(root_dir):
    """Find paths of all CSV files"""
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_files(root_dir, output_path, max_workers=None):
    """Main processing function"""
    start_time = time.time()
    
    # Find all CSV files
    print("Finding CSV files...")
    csv_files = find_all_csvs(root_dir)
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files")
    
    # Process files using process pool
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(calculate_metrics, csv_file): csv_file 
                         for csv_file in csv_files}
        
        # Handle completed tasks
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            completed += 1
            csv_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                if completed % 100 == 0 or completed == total_files:
                    print(f"Processed {completed}/{total_files} files "
                          f"({(completed/total_files*100):.1f}%)")
            
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
    
    # Convert to DataFrame and save
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(['folder_path', 'file_name'])
        
        # Calculate summary metrics for each folder
        folder_metrics = []
        for folder in df_results['folder_path'].unique():
            folder_data = df_results[df_results['folder_path'] == folder]
            successful_data = folder_data[folder_data['status'] == 'success']
            
            if len(successful_data) > 0:
                folder_metrics.append({
                    'folder_path': folder,
                    'file_name': '[FOLDER_SUMMARY]',
                    'roc_auc': successful_data['roc_auc'].mean(),
                    'pr_auc': successful_data['pr_auc'].mean(),
                    'status': f'success (averaged over {len(successful_data)} files)'
                })
        
        # Add folder summaries to results
        if folder_metrics:
            df_folder_metrics = pd.DataFrame(folder_metrics)
            df_results = pd.concat([df_results, df_folder_metrics], ignore_index=True)
            
        # Sort results
        df_results = df_results.sort_values(['folder_path', 'file_name'])
        df_results.to_csv(output_path, index=False)
        
        # Print statistics
        successful = df_results['status'] == 'success'
        print(f"Failed: {sum(~successful)}")
    else:
        print("No valid results found")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate ROC-AUC and PR-AUC metrics for CSV files')
    
    parser.add_argument('--root_dir', 
                       required=True,
                       help='Root directory containing input CSV files')
    
    parser.add_argument('--output_file',
                       required=True,
                       help='Path to save the output CSV file with results')
    
    parser.add_argument('--max_workers',
                       type=int,
                       default=None,
                       help='Maximum number of worker processes (default: number of CPU cores)')
    
    return parser.parse_args()

def main():
    """Entry point of the script"""
    args = parse_arguments()
    process_files(
        root_dir=args.root_dir,
        output_path=args.output_file,
        max_workers=args.max_workers
    )

if __name__ == '__main__':
    main()