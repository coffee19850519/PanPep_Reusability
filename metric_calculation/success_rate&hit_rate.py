import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import multiprocessing as mp
import pyarrow.parquet as pq
import pyarrow as pa
import gc
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ProcessingConfig:
    """Configuration class for processing"""
    root_dir: str  # Root directory path
    top_k: int  # Top-k value for calculation
    batch_size: int  # Batch size
    num_gpus: int  # Number of GPUs
    output_file: str  # Output file name
    output_dir: Optional[str] = None  # Optional output directory
    gpu_id: Optional[int] = None  # Optional GPU ID

class FileProcessor:
    """File processing class"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = None

    def setup_device(self, gpu_id: int):
        """Set up the computation device"""
        self.config.gpu_id = gpu_id  # Set GPU ID
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.device = torch.device('cuda:0')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.device = torch.device('cpu')
        
        logging.info(f"Process {os.getpid()} using device: {self.device}")

    def process_file(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Process a single file"""
        try:
            # Check file validity
            if not file_path.exists():
                logging.warning(f"File not found: {file_path}")
                return None

            # Check if the file has required columns
            if file_path.suffix == '.csv':
                try:
                    with open(file_path, 'r') as f:
                        header = pd.read_csv(f, nrows=0).columns
                    if 'Label' not in header:
                        logging.warning(f"CSV file {file_path} missing 'Label' column, skipping.")
                        return None
                except pd.errors.EmptyDataError:
                    logging.warning(f"CSV file {file_path} is empty, skipping.")
                    return None
                
                df = pd.read_csv(file_path, usecols=['Label'])

            elif file_path.suffix == '.parquet':
                # Read metadata
                parquet_file = pq.ParquetFile(file_path)
                if 'Label' not in parquet_file.schema.names:
                    logging.warning(f"Parquet file {file_path} missing 'Label' column, skipping.")
                    return None
                df = parquet_file.read(columns=['Label']).to_pandas()
            
            else:
                logging.warning(f"Unsupported file type: {file_path.suffix}")
                return None

            labels = torch.tensor(df['Label'].values, dtype=torch.int8, device=self.device)
            total_hits = int(torch.sum(labels).item())

            if total_hits == 0:
                return None

            # Compute metrics
            chunk_size = min(len(labels), self.config.top_k)
            cum_hits = torch.cumsum(labels[:chunk_size], dim=0)
            
            success_rates = cum_hits.float() / float(total_hits)
            positions = torch.arange(1, chunk_size + 1, dtype=torch.float32, device=self.device)
            hit_rates = cum_hits.float() / positions

            return (
                success_rates.cpu().numpy(),
                hit_rates.cpu().numpy(),
                np.ones(chunk_size, dtype=np.int32)
            )

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None

        finally:
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    def process_files_concurrently(self, file_paths: List[Path]) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Process files concurrently using threading (shared GPU context)"""
        from concurrent.futures import ThreadPoolExecutor
        import math
        
        num_workers =  4
        chunk_size = math.ceil(len(file_paths) / num_workers)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(file_paths), chunk_size):
                chunk = file_paths[i:i+chunk_size]
                futures.append(executor.submit(self._process_chunk, chunk))
            
            for future in futures:
                results.extend(future.result())
        return results

    def _process_chunk(self, chunk: List[Path]) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Process a chunk of files"""
        return [self.process_file(fp) for fp in chunk]

    def process_directory_concurrently(self, directory: str) -> Optional[pd.DataFrame]:
        """Process multiple files in a single directory concurrently"""
        try:
            file_list = list(Path(directory).glob('*.[cp]*'))
            file_list = [f for f in file_list if f.suffix in ('.csv', '.parquet')]

            if not file_list:
                return None

            # Process files concurrently
            processor = FileProcessor(self.config)
            processor.setup_device(self.config.gpu_id)

            results = processor.process_files_concurrently(file_list)
            
            # Aggregate results
            sum_success = np.zeros(self.config.top_k, dtype=np.float32)
            sum_hit = np.zeros(self.config.top_k, dtype=np.float32)
            counts = np.zeros(self.config.top_k, dtype=np.int32)

            for result in results:
                if result:
                    s_rates, h_rates, cnt = result
                    length = len(s_rates)
                    sum_success[:length] += s_rates
                    sum_hit[:length] += h_rates
                    counts[:length] += cnt

            return self._create_result_df(directory, sum_success, sum_hit, counts)
        except Exception as e:
            logging.error(f"Error processing directory {directory}: {e}")
            return None

    def _create_result_df(self, directory: str, sum_success: np.ndarray, 
                         sum_hit: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
        """Create the result DataFrame"""
        valid = counts > 0
        if not np.any(valid):
            return None

        pos = np.nonzero(valid)[0] + 1
        avg_success = sum_success[valid] / counts[valid]
        avg_hit = sum_hit[valid] / counts[valid]

        return pd.DataFrame({
            'Directory': directory,
            'Top_K': pos,
            'Success_Rate': avg_success,
            'Hit_Rate': avg_hit,
        })

class DirectoryProcessor:
    """Directory processing class"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.temp_dir = Path("./temp_results")
        self.temp_dir.mkdir(exist_ok=True)

    def find_directories(self) -> List[str]:
        """Find all directories containing target files"""
        root_path = Path(self.config.root_dir)
        all_dirs = set()
        
        for ext in ['.csv', '.parquet']:
            for file_path in root_path.rglob(f'*{ext}'):
                all_dirs.add(str(file_path.parent))

        return sorted(list(all_dirs))

    def process_all(self):
        """Process all directories (optimize GPU allocation)"""
        directories = self.find_directories()
        if not directories:
            logging.error(f"No valid directories found in {self.config.root_dir}")
            return

        # Create process pool (one process per GPU)
        ctx = mp.get_context('spawn')
        num_processes = min(self.config.num_gpus, len(directories)) if self.config.num_gpus > 0 else mp.cpu_count()
        pool = ctx.Pool(processes=num_processes)
        
        try:
            # Create task queue (each GPU processes multiple directories)
            tasks = []
            for idx, directory in enumerate(directories):
                gpu_id = idx % self.config.num_gpus if self.config.num_gpus > 0 else -1
                tasks.append((directory, gpu_id))
            
            # Use imap_unordered to improve throughput
            results = []
            with tqdm(total=len(tasks), desc="Processing directories") as pbar:
                for result in pool.imap_unordered(self._process_single_directory_wrapper, tasks):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
            
            self._save_final_results(results)
            
        finally:
            pool.close()
            pool.join()

    def _process_single_directory_wrapper(self, args: Tuple[str, int]) -> Optional[pd.DataFrame]:
        """Wrapper to unpack arguments"""
        return self._process_single_directory(*args)

    def _process_single_directory(self, directory: str, gpu_id: int) -> Optional[pd.DataFrame]:
        """Process a single directory (without nested processes)"""
        try:
            processor = FileProcessor(self.config)
            processor.setup_device(gpu_id)
            return processor.process_directory_concurrently(directory)
        except Exception as e:
            logging.error(f"Failed to process directory {directory}: {e}")
            return None

    def _save_final_results(self, results: List[pd.DataFrame]):
        """Save final results"""
        if not results:
             logging.warning("No results to save.")
             return
        logging.info(f"Combining {len(results)} DataFrames...")
        final_df = pd.concat(results, ignore_index=True)
        
        output_dir = Path(self.config.output_dir) if self.config.output_dir else Path.cwd()
        csv_path = output_dir / f"{self.config.output_file}.csv"
        parquet_path = output_dir / f"{self.config.output_file}.parquet"

        final_df.to_csv(csv_path, index=False)
        table = pa.Table.from_pandas(final_df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        logging.info(f"Final results saved to {csv_path} and {parquet_path}")

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Compute average hit rate of directories')
    parser.add_argument('--root_dir', required=True, help='Root directory path')
    parser.add_argument('--top_k', type=int, default=11419896, help='Maximum top-k value to calculate')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size')
    parser.add_argument('--output', default='results', help='Output file name (without extension)')
    parser.add_argument('--output_dir', help='Output directory')
    args = parser.parse_args()

    config = ProcessingConfig(
        root_dir=args.root_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0, 
        output_file=args.output,
        output_dir=args.output_dir
    )

    logging.info(f"Configuration: {config}")

    if config.num_gpus == 0:
        logging.warning("No CUDA GPU detected or GPU usage not requested. Running on CPU.")

    processor = DirectoryProcessor(config)
    processor.process_all()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    logging.info("Multiprocessing start method set to 'spawn'.")
    
    main()