import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from skfp.metrics import bedroc_score
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

class BEDROCProcessor:
    """Class to handle BEDROC score calculation and result aggregation"""
    
    def __init__(self, 
                 root_dir: str, 
                 output_dir: str,
                 detailed_dir: str,
                 averaged_file: str,
                 num_processes: Optional[int] = None):
        """
        Initialize the BEDROC processor
        
        Args:
            root_dir: Root directory containing input files
            output_dir: Base directory for all outputs
            detailed_dir: Directory name for detailed results
            averaged_file: Filename for averaged results
            num_processes: Number of processes for parallel processing
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.detailed_dir = self.output_dir / detailed_dir
        self.averaged_file = self.output_dir / averaged_file
        self.num_processes = min(64, cpu_count()-2) if num_processes is None else num_processes
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.detailed_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Input directory: {self.root_dir}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Detailed results directory: {self.detailed_dir}")
        logging.info(f"Averaged results file: {self.averaged_file}")

    # ... [previous methods remain the same until calculate_averages] ...

    def calculate_averages(self, result_files: List[Path]) -> None:
        """
        Calculate and save averaged results
        
        Args:
            result_files: List of paths to detailed result files
        """
        all_dfs = []
        for file in tqdm(result_files, desc="Reading detailed results"):
            try:
                df = pd.read_csv(file)
                all_dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")

        if not all_dfs:
            logging.error("No data to average")
            return

        # Combine all results and calculate averages
        combined_df = pd.concat(all_dfs, ignore_index=True)
        averaged_df = (combined_df.groupby(['Directory', 'Alpha'])
                      .agg({'BEDROC_Score': 'mean'})
                      .reset_index()
                      .sort_values(['Directory', 'Alpha']))

        # Save averaged results
        averaged_df.to_csv(self.averaged_file, index=False)
        
        # Log summary statistics
        logging.info(f"Processed {len(result_files)} files successfully")
        logging.info(f"Averaged results saved to {self.averaged_file}")
        logging.info(f"Detailed results saved in {self.detailed_dir}")
        logging.info(f"Number of unique directories: {len(averaged_df['Directory'].unique())}")
        logging.info(f"Number of alpha values per directory: {len(averaged_df['Alpha'].unique())}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process BEDROC scores with detailed and averaged results')
    
    parser.add_argument('--root_dir', 
                       required=True,
                       help='Root directory containing input files')
    
    parser.add_argument('--output_dir',
                       default='results',
                       help='Base directory for all outputs (default: results)')
    
    parser.add_argument('--detailed_dir',
                       default='detailed_results',
                       help='Directory name for detailed results (default: detailed_results)')
    
    parser.add_argument('--averaged_file',
                       default='averaged_metrics.csv',
                       help='Filename for averaged results (default: averaged_metrics.csv)')
    
    parser.add_argument('--num_processes',
                       type=int,
                       default=None,
                       help='Number of processes for parallel processing (default: min(64, CPU_COUNT-2))')
    
    return parser.parse_args()

def main():
    """Main entry point of the script"""
    args = parse_arguments()
    
    processor = BEDROCProcessor(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        detailed_dir=args.detailed_dir,
        averaged_file=args.averaged_file,
        num_processes=args.num_processes
    )
    
    processor.process_all_files()

if __name__ == "__main__":
    main()