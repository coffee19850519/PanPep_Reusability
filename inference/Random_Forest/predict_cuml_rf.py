"""
Inference using trained cuML Random Forest model with pre-computed encodings.
CSV format: tcr, peptide, label
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import os
import sys
import time

import cudf
import fastparquet
from cuml.ensemble import RandomForestClassifier as cuRF

# Import encoding functions from utils (for fallback)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)
from utils import aamapping, add_position_encoding


def load_atchley_factors(aa_dict_path):
    """Load Atchley factor dictionary."""
    return joblib.load(aa_dict_path)


def load_encodings(encoding_path):
    """Load pre-computed sequence encodings from npz file."""
    print(f"Loading encodings from {encoding_path}...")
    with np.load(encoding_path) as encodings:
        sequences = encodings['sequences']
        encoding_data = encodings['encodings']
        encoding_dict = {seq: enc for seq, enc in zip(sequences, encoding_data)}
    print(f"Loaded {len(encoding_dict)} pre-encoded sequences")
    return encoding_dict


def encode_tcr_peptide_fast(tcr, peptide, tcr_encoding_dict, peptide_encoding_dict, aa_dict):
    """
    Fast encode TCR-peptide pair using pre-computed encodings.
    If sequence not in dict, fallback to real-time encoding.
    Returns: (encoded_feature, peptide_cached, tcr_cached)
    """
    # Try to get peptide encoding from pre-computed dict
    if peptide in peptide_encoding_dict:
        pep_emb = peptide_encoding_dict[peptide]
        peptide_cached = True
    else:
        # fallback: real-time encode peptide
        pep_emb_tensor = add_position_encoding(aamapping(peptide, 15, aa_dict))
        pep_emb = pep_emb_tensor.numpy()
        peptide_cached = False

    # Try to get TCR encoding from pre-computed dict
    if tcr in tcr_encoding_dict:
        tcr_emb = tcr_encoding_dict[tcr]
        tcr_cached = True
    else:
        # fallback: real-time encode TCR
        tcr_emb_tensor = add_position_encoding(aamapping(tcr, 25, aa_dict))
        tcr_emb = tcr_emb_tensor.numpy()
        tcr_cached = False

    combined = np.vstack([pep_emb, tcr_emb])
    return combined.flatten(), peptide_cached, tcr_cached


def main():
    parser = argparse.ArgumentParser(description='Predict using cuML RF model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with tcr,peptide,label')
    parser.add_argument('--output', type=str, default='predictions.parquet', help='Output predictions parquet')
    parser.add_argument('--append', action='store_true', help='Append to existing parquet file')
    parser.add_argument('--aa_dict', type=str, required=True, help='Path to amino acid factors dictionary (.joblib)')
    parser.add_argument('--peptide_encoding', type=str, required=True, help='Path to peptide encoding file (.npz)')
    parser.add_argument('--tcr_encoding', type=str, required=True, help='Path to TCR encoding file (.npz)')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for inference (default: 50000)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model_data = joblib.load(args.model)
    rf_model = model_data['model']
    print(rf_model)

    # Load Atchley factors (for fallback encoding)
    print(f"Loading Atchley factors from {args.aa_dict}...")
    aa_dict = load_atchley_factors(args.aa_dict)

    # Load pre-computed encodings
    peptide_encoding_dict = load_encodings(args.peptide_encoding)
    tcr_encoding_dict = load_encodings(args.tcr_encoding)

    # Get expected dimensions
    sample_peptide_enc = next(iter(peptide_encoding_dict.values()))
    sample_tcr_enc = next(iter(tcr_encoding_dict.values()))
    peptide_dim = sample_peptide_enc.size
    tcr_dim = sample_tcr_enc.size
    expected_dim = peptide_dim + tcr_dim
    print(f"Expected feature dimension: {expected_dim} (peptide: {peptide_dim}, tcr: {tcr_dim})")

    # Load test data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples")

    # Check for missing sequences (but don't exit, we have fallback)
    missing_peptides = set(df['peptide'].unique()) - set(peptide_encoding_dict.keys())
    missing_tcrs = set(df['tcr'].unique()) - set(tcr_encoding_dict.keys())

    if missing_peptides:
        print(f"WARNING: {len(missing_peptides)} peptides missing from encoding dict (will use fallback)")
        if len(missing_peptides) <= 10:
            print(f"Missing peptides: {missing_peptides}")
        else:
            print(f"First 10 missing peptides: {list(missing_peptides)[:10]}")

    if missing_tcrs:
        print(f"WARNING: {len(missing_tcrs)} TCRs missing from encoding dict (will use fallback)")
        if len(missing_tcrs) <= 10:
            print(f"Missing TCRs: {missing_tcrs}")
        else:
            print(f"First 10 missing TCRs: {list(missing_tcrs)[:10]}")

    if not missing_peptides and not missing_tcrs:
        print("All sequences found in encoding dictionaries ✓")
    else:
        print(f"Fallback encoding enabled for missing sequences")

    # Encode and predict in batches with streaming output
    print(f"Starting batch inference with batch size {args.batch_size}...")
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size

    # Track encoding sources
    cached_peptide_hits = 0
    cached_tcr_hits = 0
    fallback_peptides = 0
    fallback_tcrs = 0

    start_time = time.time()
    for batch_idx in range(total_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(df))

        batch_df = df.iloc[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")

        # Encode this batch (optimized iteration)
        X_list = []
        for tcr, peptide in zip(batch_df['tcr'], batch_df['peptide']):
            try:
                feature, peptide_cached, tcr_cached = encode_tcr_peptide_fast(
                    tcr, peptide,
                    tcr_encoding_dict, peptide_encoding_dict, aa_dict
                )
                X_list.append(feature)

                # Track cache hits
                if peptide_cached:
                    cached_peptide_hits += 1
                else:
                    fallback_peptides += 1
                if tcr_cached:
                    cached_tcr_hits += 1
                else:
                    fallback_tcrs += 1

            except Exception as e:
                print(f"ERROR encoding row {tcr}, {peptide}: {e}")
                # Use zero vector for failed encodings
                X_list.append(np.zeros(expected_dim, dtype='float32'))

        X_batch = np.array(X_list, dtype=np.float32)  # Use float32 to save memory
        print(f"  Batch encoded shape: {X_batch.shape}")

        # Sanity check
        if X_batch.shape[1] != expected_dim:
            print(f"ERROR: Encoded dimension {X_batch.shape[1]} != expected {expected_dim}")
            sys.exit(1)

        # Predict this batch with GPU memory management
        print(f"  Making predictions for batch...")
        X_cudf = cudf.DataFrame(X_batch)

        try:
            y_pred_proba_batch = rf_model.predict_proba(X_cudf).values.get()[:, 1]
        finally:
            # Always clean up GPU memory
            del X_cudf

        # Create result dataframe for this batch
        batch_result = batch_df.copy()
        batch_result['score'] = y_pred_proba_batch

        # Stream write to parquet
        append_mode = (batch_idx > 0) or (args.append and os.path.exists(args.output))

        try:
            fastparquet.write(args.output, batch_result, append=append_mode)
        except Exception as e:
            print(f"ERROR writing to parquet: {e}")
            # Fallback to CSV if parquet fails
            csv_output = args.output.replace('.parquet', '.csv')
            mode = 'a' if append_mode else 'w'
            header = not append_mode
            batch_result.to_csv(csv_output, mode=mode, header=header, index=False)
            print(f"Wrote to CSV instead: {csv_output}")

        # Clean up memory
        del X_batch, X_list, batch_result, y_pred_proba_batch

        # Progress reporting
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        samples_per_sec = end_idx / elapsed_time
        eta = (len(df) - end_idx) / samples_per_sec if samples_per_sec > 0 else 0

        print(f"  Batch time: {batch_time:.2f}s | "
              f"Total elapsed: {elapsed_time:.2f}s | "
              f"Speed: {samples_per_sec:.0f} samples/s | "
              f"ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time

    # Print encoding statistics
    total_peptides = cached_peptide_hits + fallback_peptides
    total_tcrs = cached_tcr_hits + fallback_tcrs

    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(df)} samples in {total_time:.2f}s")
    print(f"  Average speed: {len(df)/total_time:.0f} samples/s")
    print(f"  Output saved to: {args.output}")
    print(f"\n=== Encoding Statistics ===")
    print(f"Peptide encoding:")
    print(f"  Pre-computed cache hits: {cached_peptide_hits:,} ({cached_peptide_hits/total_peptides*100:.1f}%)")
    print(f"  Real-time encoding fallback: {fallback_peptides:,} ({fallback_peptides/total_peptides*100:.1f}%)")
    print(f"  Total: {total_peptides:,}")

    print(f"\nTCR encoding:")
    print(f"  Pre-computed cache hits: {cached_tcr_hits:,} ({cached_tcr_hits/total_tcrs*100:.1f}%)")
    print(f"  Real-time encoding fallback: {fallback_tcrs:,} ({fallback_tcrs/total_tcrs*100:.1f}%)")
    print(f"  Total: {total_tcrs:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
