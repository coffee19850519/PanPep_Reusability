import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse
import random

class DataSampler:
    """Class to handle sampling of peptide data (positive and negative samples)"""
    def __init__(self, positive_path: str, negative_path: str, peptide: str = None):
        """Initialize with paths to positive and negative sample data and an optional peptide"""
        self.positive_data = None
        self.tcr_pool = None
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.peptide = peptide
        self._load_data()  # Load the data when the class is instantiated
    
    def _load_data(self) -> None:
        """Load the positive and negative data from the specified paths"""
        self.positive_data = pd.read_csv(self.positive_path)
        self.tcr_pool = pd.read_csv(self.negative_path)
        
        # Check for required columns in the positive data
        required_columns = ['Epitope', 'alpha', 'beta', 'Label']
        missing_columns = [col for col in required_columns if col not in self.positive_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in positive data: {missing_columns}")
        
        # Check for required columns in the negative data
        required_columns = ['tcra', 'tcrb']
        missing_columns = [col for col in required_columns if col not in self.tcr_pool.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in negative data: {missing_columns}")

        # Rename columns in the TCR pool for consistency
        self.tcr_pool = self.tcr_pool.copy()
        self.tcr_pool = self.tcr_pool.rename(columns={
            'tcra': 'alpha',
            'tcrb': 'beta'
        })
        
        print(f"Data loaded successfully: {len(self.positive_data)} positive samples, {len(self.tcr_pool)} TCR pairs")

    def get_negative_pool(self, peptide: str) -> pd.DataFrame:
        """Generate a negative pool for a specific peptide by excluding its positive pairs"""
        # Get the positive pairs for the current peptide
        peptide_positives = self.positive_data[self.positive_data['Epitope'] == peptide]
        positive_pairs = set(zip(peptide_positives['alpha'], peptide_positives['beta']))
        
        # Exclude positive pairs from the TCR pool
        tcr_pairs = set(zip(self.tcr_pool['alpha'], self.tcr_pool['beta']))
        filtered_pairs = tcr_pairs - positive_pairs
        
        # Create a DataFrame for the filtered negative pairs
        filtered_pairs_list = list(filtered_pairs)
        negative_pool = pd.DataFrame({
            'alpha': [pair[0] for pair in filtered_pairs_list],
            'beta': [pair[1] for pair in filtered_pairs_list],
            'Epitope': peptide,  # Set the current peptide
            'Label': 0
        })
        
        print(f"Negative pool for peptide {peptide}: {len(negative_pool)} pairs "
              f"(excluded {len(positive_pairs)} positive pairs)")
        return negative_pool

    def sample_majority(self, positive_ratio: float = 0.8, random_state: int = 42) -> pd.DataFrame:
        """Majority sampling mode: sample both positive and negative samples"""
        print("Starting majority sampling...")
        
        # Get the positive samples for the current peptide
        peptide_positives = self.positive_data[self.positive_data['Epitope'] == self.peptide]
        
        # Calculate the number of positive samples to sample
        n_pos_samples = int(len(peptide_positives) * positive_ratio)
        
        # Sample positive samples
        sampled_pos = peptide_positives.sample(
            n=n_pos_samples,
            random_state=random_state
        )
        
        # Get the negative pool and sample negative samples
        negative_pool = self.get_negative_pool(self.peptide)
        sampled_neg = negative_pool.sample(
            n=n_pos_samples,
            random_state=random_state
        )

        # Combine positive and negative samples
        combined = pd.concat([sampled_pos, sampled_neg])
        
        print(f"Majority sampling completed: {len(combined)} samples")
        return combined
    
    def sample_few_shot(self, n_samples: int = 2, random_state: int = 42) -> pd.DataFrame:
        """Few-shot sampling mode: sample a small number of positive and negative samples"""
        print("Starting few-shot sampling...")
        
        # Get the positive samples for the current peptide
        peptide_positives = self.positive_data[self.positive_data['Epitope'] == self.peptide]
        
        # Sample the positive samples
        sampled_pos = peptide_positives.sample(
            n=n_samples,
            random_state=random_state
        )
        
        # Get the negative pool and sample the negative samples
        negative_pool = self.get_negative_pool(self.peptide)
        sampled_neg = negative_pool.sample(
            n=n_samples,
            random_state=random_state
        )
        
        # Combine and shuffle the samples
        combined = pd.concat([sampled_pos, sampled_neg])
        
        print(f"Few-shot sampling completed: {len(combined)} samples")
        return combined

    def sample_zero_shot(self) -> pd.DataFrame:
        """Zero-shot sampling mode: do not sample any positive or negative samples"""
        print("Starting zero-shot sampling...")
        
        # Get the positive samples for the current peptide
        peptide_positives = self.positive_data[self.positive_data['Epitope'] == self.peptide]
        
        # Get the negative pool for the peptide
        negative_pool = self.get_negative_pool(self.peptide)
        
        # Return an empty DataFrame (no samples selected)
        empty_samples = pd.DataFrame(columns=['Epitope', 'alpha', 'beta', 'Label'])
        
        print("Zero-shot sampling completed: no samples selected")
        return empty_samples
    
    def save_finetune_data(self, samples: pd.DataFrame, output_dir: Path, peptide: str) -> None:
        """Save the final fine-tuned dataset, including selected and remaining samples"""
        print(f"Saving finetune data to directory: {output_dir}")
        
        # Create the finetune directory if it doesn't exist
        finetune_dir = output_dir
        finetune_dir.mkdir(exist_ok=True)
        
        # Get selected sample pairs (both positive and negative)
        selected_pairs = set(zip(samples['alpha'], samples['beta'])) if not samples.empty else set()
        
        # Prepare all positive samples for the current peptide
        all_positive = self.positive_data[self.positive_data['Epitope'] == peptide].copy()
        
        # Split the positive samples into selected and unselected
        selected_mask = all_positive.apply(lambda row: (row['alpha'], row['beta']) in selected_pairs, axis=1)
        selected_pos = all_positive[selected_mask].copy()
        unselected_pos = all_positive[~selected_mask].copy()
        
        # Set the labels for selected and unselected positive samples
        selected_pos['label'] = selected_pos['Label']
        unselected_pos['label'] = unselected_pos['Label']
        unselected_pos['Label'] = 'Unknown'  # Label unselected positive samples as 'Unknown'
        
        # Prepare selected negative samples
        selected_neg = samples[samples['Label'] == 0].copy() if not samples.empty else pd.DataFrame()
        if not selected_neg.empty:
            selected_neg['label'] = selected_neg['Label']  # Save the true label
        
        # Get the negative pool for the peptide
        peptide_negative_pool = self.get_negative_pool(peptide)
        
        # Get the remaining negative pairs
        all_negative_pairs = set(zip(peptide_negative_pool['alpha'], peptide_negative_pool['beta']))
        remaining_pairs = all_negative_pairs - selected_pairs
        
        # Create DataFrame for the remaining negative samples
        remaining_samples = pd.DataFrame({
            'alpha': [pair[0] for pair in remaining_pairs],
            'beta': [pair[1] for pair in remaining_pairs],
            'Epitope': peptide,
            'Label': 'Unknown',
            'label': 0
        })
        
        # Combine selected positive, unselected positive, selected negative, and remaining negative samples
        complete_dataset = pd.concat([selected_pos, unselected_pos, selected_neg, remaining_samples])
        
        # Save the final dataset
        complete_dataset.to_csv(finetune_dir / f"{peptide}.csv", index=False)
        print(f"Finetune data saved successfully for peptide {peptide}: {len(complete_dataset)} samples "
              f"({len(selected_pos)} selected positive, {len(unselected_pos)} unselected positive, "
              f"{len(selected_neg)} selected negative, {len(remaining_samples)} remaining negative)")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Peptide sequence sampling tool')

    # Define arguments for input files and sampling modes
    parser.add_argument('--positive_data', type=str, default='data/tcrab_majority.csv',
                       help='Path to positive sample data file')
    parser.add_argument('--tcr_pool', type=str, default='data/pooling_tcrab.csv',
                       help='Path to negative sample data file')

    # Define the sampling mode options
    parser.add_argument('--mode', type=str, choices=['majority', 'few_shot', 'zero_shot'], 
                       default='majority', help='Sampling mode: majority, few_shot, zero_shot')

    # Additional parameters for majority and few-shot modes
    parser.add_argument('--majority_ratio', type=float, default=0.8,
                       help='Positive sample ratio in majority mode, default 0.8')
    parser.add_argument('--majority_seed', type=int, default=42,
                       help='Random seed for majority mode, default 42')

    parser.add_argument('--few_shot', type=int, default=2,
                       help='Number of samples in few_shot mode, default 2')
    parser.add_argument('--few_shot_seed', type=int, default=42,
                       help='Random seed for few_shot mode, default 42')

    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory path, default output')

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return args

def main():
    """Main function to perform sampling and save the results"""
    args = parse_args()

    initial_sampler = DataSampler(args.positive_data, args.tcr_pool)
    all_peptides = initial_sampler.positive_data['Epitope'].unique()
    print(f"Found {len(all_peptides)} peptides: {', '.join(all_peptides)}")

    # Create subdirectories for each sampling mode
    if args.mode in ['majority']:
        (Path(args.output_dir) / 'majority').mkdir(parents=True, exist_ok=True)
    if args.mode in ['few_shot']:
        (Path(args.output_dir) / 'few_shot').mkdir(parents=True, exist_ok=True)
    if args.mode in ['zero_shot']:
        (Path(args.output_dir) / 'zero_shot').mkdir(parents=True, exist_ok=True)

    # Perform sampling for each peptide
    for peptide in all_peptides:
        sampler = DataSampler(args.positive_data, args.tcr_pool, peptide)

        if args.mode in ['majority']:
            majority_samples = sampler.sample_majority(
                positive_ratio=args.majority_ratio,
                random_state=args.majority_seed
            )
            output_dir = Path(args.output_dir) / 'majority'
            sampler.save_finetune_data(majority_samples, output_dir, peptide)
        
        if args.mode in ['few_shot']:
            few_shot_samples = sampler.sample_few_shot(
                n_samples=args.few_shot,
                random_state=args.few_shot_seed
            )
            output_dir = Path(args.output_dir) / 'few_shot'
            sampler.save_finetune_data(few_shot_samples, output_dir, peptide)

        if args.mode in ['zero_shot']:
            zero_shot_samples = sampler.sample_zero_shot()
            output_dir = Path(args.output_dir) / 'zero_shot'
            sampler.save_finetune_data(zero_shot_samples, output_dir, peptide)

if __name__ == "__main__":
    main()
