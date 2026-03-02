#!/usr/bin/env python3
import os
import argparse
import pandas as pd


def validate_sequence(seq):
    if pd.isna(seq):
        return None
    if not isinstance(seq, str):
        return None
    seq = seq.strip()
    if not seq:
        return None
    valid_aas = set("ARNDCQEGHILKMFPSTWYV")
    if not set(seq).issubset(valid_aas):
        return None
    return seq


def create_tcr_epitope_csv(peptide_file, tcr_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Read and validate peptide sequences
    with open(peptide_file, "r") as f:
        peptide_seqs = f.readlines()

    valid_peptides = []
    for peptide_seq in peptide_seqs:
        peptide_seq = peptide_seq.strip().upper()
        validated = validate_sequence(peptide_seq)
        if validated:
            valid_peptides.append(validated)

    print(f"Valid peptides: {len(valid_peptides)} out of {len(peptide_seqs)}")

    # Read and validate TCR sequences
    df_tcr = pd.read_csv(tcr_file, delimiter="\r", header=None, names=["tcr"], engine="python")
    original_tcr_count = len(df_tcr)

    df_tcr["tcr"] = df_tcr["tcr"].astype(str).str.upper()
    df_tcr["tcr"] = df_tcr["tcr"].apply(validate_sequence)
    df_tcr = df_tcr.dropna(subset=["tcr"])

    print(f"Valid TCR sequences: {len(df_tcr)} out of {original_tcr_count}")

    # Generate CSV files for each valid peptide
    for peptide_seq in valid_peptides:
        df = df_tcr.copy(deep=True)
        df["peptide"] = peptide_seq
        df["label"] = 0
        df = df[["tcr", "peptide", "label"]]

        output_filename = f"{peptide_seq}.csv"
        df.to_csv(os.path.join(output_folder, output_filename), index=False)

    print(f"Generated {len(valid_peptides)} CSV files in {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-peptide CSV files with columns: tcr, peptide, label(0)."
    )
    parser.add_argument("--peptide-file", required=True, help="Peptide txt file (one peptide per line).")
    parser.add_argument("--tcr-file", required=True, help="TCR file path.")
    parser.add_argument("--output-folder", required=True, help="Output folder.")

    args = parser.parse_args()
    create_tcr_epitope_csv(args.peptide_file, args.tcr_file, args.output_folder)


if __name__ == "__main__":
    main()