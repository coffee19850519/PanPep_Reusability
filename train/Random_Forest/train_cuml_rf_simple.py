"""
Train cuML Random Forest using pre-sampled CSV data.
CSV format: tcr, peptide, label
"""

import numpy as np
import pandas as pd
import joblib
import argparse

from cuml.ensemble import RandomForestClassifier as cuRF
import cudf


def load_atchley_factors(aa_dict_path):
    """Load Atchley factor dictionary."""
    return joblib.load(aa_dict_path)


def aamapping(seq, aa_dict, encode_dim):
    """Encode amino acid sequence using Atchley factors."""
    result = []
    if len(seq) > encode_dim:
        seq = seq[:encode_dim]
    for aa in seq:
        if aa in aa_dict:
            result.append(aa_dict[aa])
        else:
            result.append(np.zeros(5, dtype='float64'))
    # Pad to encode_dim
    for _ in range(encode_dim - len(seq)):
        result.append(np.zeros(5, dtype='float64'))
    return np.array(result)


def add_position_encoding(seq):
    """Add position encoding to sequence embedding."""
    position_encoding = np.array(
        [[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    seq = seq.copy()
    padding_mask = np.abs(seq).sum(axis=-1) != 0
    seq[padding_mask] += position_encoding[:np.sum(padding_mask)]
    return seq


def encode_tcr_peptide(tcr, peptide, aa_dict):
    """Encode TCR-peptide pair into feature vector."""
    # Peptide embedding: 15 x 5
    pep_emb = aamapping(peptide, aa_dict, 15)
    pep_emb = add_position_encoding(pep_emb)

    # TCR embedding: 25 x 5
    tcr_emb = aamapping(tcr, aa_dict, 25)
    tcr_emb = add_position_encoding(tcr_emb)

    # Concatenate: 40 x 5, then flatten to 200
    combined = np.vstack([pep_emb, tcr_emb])
    return combined.flatten()


def main():
    parser = argparse.ArgumentParser(description='Train cuML RF on TCR-peptide data')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with tcr,peptide,label')
    parser.add_argument('--aa_dict', type=str, default='Requirements/dic_Atchley_factors.pkl',
                        help='Path to Atchley factors dictionary')
    parser.add_argument('--output', type=str, default='cuml_rf_model.pkl', help='Output model path')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples")

    # Load Atchley factors
    print(f"Loading Atchley factors from {args.aa_dict}...")
    aa_dict = load_atchley_factors(args.aa_dict)

    # Encode features
    print("Encoding TCR-peptide pairs...")
    X_list = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        feature = encode_tcr_peptide(row['tcr'], row['peptide'], aa_dict)
        X_list.append(feature)

    X = np.array(X_list)
    y = df['label'].values.astype(np.int32)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Positive: {np.sum(y)}, Negative: {np.sum(y == 0)}")

    # Train cuML RF
    print("Converting to cuDF...")
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)

    print("Training cuML Random Forest...")
    rf_model = cuRF()
    rf_model.fit(X_cudf, y_cudf)

    # Save model and aa_dict path for inference
    print(f"Saving model to {args.output}...")
    model_data = {
        'model': rf_model,
        'aa_dict_path': args.aa_dict
    }
    joblib.dump(model_data, args.output)

    print("Done!")


if __name__ == "__main__":
    main()

