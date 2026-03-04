#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


VALID_AAS = set("ARNDCQEGHILKMFPSTWYV")


def validate_sequence(seq) -> Optional[str]:
    if pd.isna(seq):
        return None
    if not isinstance(seq, str):
        return None
    seq = seq.strip().upper()
    if not seq:
        return None
    if not set(seq).issubset(VALID_AAS):
        return None
    return seq


def _read_lines(file_path: Path) -> List[str]:
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def _read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    # Try normal CSV first
    try:
        return pd.read_csv(file_path)
    except Exception:
        # Fallback for no-header files
        return pd.read_csv(file_path, header=None)


def _pick_column(df: pd.DataFrame, preferred: List[str]) -> pd.Series:
    columns = {c.lower(): c for c in df.columns}
    for c in preferred:
        if c.lower() in columns:
            return df[columns[c.lower()]]
    return df.iloc[:, 0]


def load_peptides(peptide_file: str) -> Tuple[List[str], int]:
    path = Path(peptide_file)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".list", ".fa", ".fasta"}:
        raw = _read_lines(path)
        total = len(raw)
        cleaned = [validate_sequence(x) for x in raw]
    else:
        df = _read_table(path)
        series = _pick_column(df, ["peptide", "epitope"])
        total = len(series)
        cleaned = [validate_sequence(x) for x in series.tolist()]

    valid = [x for x in cleaned if x is not None]
    return valid, total


def load_tcrs(tcr_file: str, chain: str = "beta") -> Tuple[List[str], int]:
    path = Path(tcr_file)
    suffix = path.suffix.lower()

    if chain == "alpha":
        preferred = ["tcra", "tra", "tcra_cdr3", "alpha", "cdr3a"]
    else:
        preferred = ["tcr", "tcrb", "trb", "binding_tcr", "tcrb_cdr3", "beta", "cdr3", "cdr3b"]

    if suffix in {".txt", ".list"}:
        raw = _read_lines(path)
        total = len(raw)
        cleaned = [validate_sequence(x) for x in raw]
    else:
        df = _read_table(path)
        series = _pick_column(df, preferred)
        total = len(series)
        cleaned = [validate_sequence(x) for x in series.tolist()]

    valid = [x for x in cleaned if x is not None]
    return valid, total


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_per_peptide_csv(df_builder, peptides: List[str], output_folder: str) -> int:
    ensure_dir(output_folder)
    for pep in peptides:
        df = df_builder(pep)
        df.to_csv(os.path.join(output_folder, f"{pep}.csv"), index=False)
    return len(peptides)


def create_for_random_forest(peptides: List[str], tcrb_list: List[str], output_folder: str) -> int:
    def _build(pep: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tcr": tcrb_list,
                "peptide": [pep] * len(tcrb_list),
                "label": [0] * len(tcrb_list),
            }
        )

    return write_per_peptide_csv(_build, peptides, output_folder)


def create_for_unifyimmun(peptides: List[str], tcrb_list: List[str], output_folder: str) -> int:
    # Input format is the same as Random_Forest: tcr, peptide, label
    return create_for_random_forest(peptides, tcrb_list, output_folder)


def create_for_dlptcr(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    tcra_list: Optional[List[str]] = None,
) -> int:
    if tcra_list is not None and len(tcra_list) == len(tcrb_list):
        tcra_column = tcra_list
    else:
        if tcra_list is not None and len(tcra_list) != len(tcrb_list):
            print(
                f"[DLpTCR_ext] Warning: TCRA count ({len(tcra_list)}) != TCRB count "
                f"({len(tcrb_list)}). Leaving TCRA_CDR3 empty."
            )
        tcra_column = [""] * len(tcrb_list)

    def _build(pep: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TCRA_CDR3": tcra_column,
                "TCRB_CDR3": tcrb_list,
                "Epitope": [pep] * len(tcrb_list),
            }
        )

    return write_per_peptide_csv(_build, peptides, output_folder)


def create_for_ergo2(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    tcra_list: Optional[List[str]] = None,
    mhc_value: str = "UNK",
    tcell_type_value: str = "UNK",
) -> int:
    if tcra_list is not None and len(tcra_list) == len(tcrb_list):
        tra_column = tcra_list
    else:
        if tcra_list is not None and len(tcra_list) != len(tcrb_list):
            print(
                f"[ERGO-II_ext] Warning: TCRA count ({len(tcra_list)}) != TCRB count "
                f"({len(tcrb_list)}). Leaving TRA empty."
            )
        tra_column = [""] * len(tcrb_list)

    def _build(pep: str) -> pd.DataFrame:
        n = len(tcrb_list)
        return pd.DataFrame(
            {
                "TRA": tra_column,
                "TRB": tcrb_list,
                "TRAV": [""] * n,
                "TRAJ": [""] * n,
                "TRBV": [""] * n,
                "TRBJ": [""] * n,
                "T-Cell-Type": [tcell_type_value] * n,
                "Peptide": [pep] * n,
                "MHC": [mhc_value] * n,
            }
        )

    return write_per_peptide_csv(_build, peptides, output_folder)


def create_legacy_tcr_epitope_csv(peptide_file: str, tcr_file: str, output_folder: str) -> None:
    peptides, total_peps = load_peptides(peptide_file)
    tcrs, total_tcrs = load_tcrs(tcr_file, chain="beta")

    print(f"Valid peptides: {len(peptides)} out of {total_peps}")
    print(f"Valid TCR sequences: {len(tcrs)} out of {total_tcrs}")

    created = create_for_random_forest(peptides, tcrs, output_folder)
    print(f"Generated {created} CSV files in {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-peptide input CSV files for inference modules:\n"
            "DLpTCR_ext, ERGO-II_ext, Random_Forest, UnifyImmun_ext."
        )
    )
    parser.add_argument(
        "--target",
        default="all",
        choices=["dlptcr", "ergo2", "rf", "unifyimmun", "all"],
        help=(
            "Which format to generate: "
            "dlptcr | ergo2 | rf | unifyimmun | all"
        ),
    )
    parser.add_argument("--peptide-file", required=True, help="Peptide file path (txt/csv/parquet).")
    parser.add_argument("--tcr-file", required=True, help="TCR beta file path (txt/csv/parquet).")
    parser.add_argument("--tcra-file", default=None, help="Optional TCR alpha file path for DLpTCR/ERGO.")
    parser.add_argument("--output-folder", required=True, help="Output folder.")
    parser.add_argument("--mhc", default="UNK", help="Default MHC value for ERGO-II input.")
    parser.add_argument("--tcell-type", default="UNK", help="Default T-Cell-Type value for ERGO-II input.")
    args = parser.parse_args()

    if args.target == "legacy":
        create_legacy_tcr_epitope_csv(args.peptide_file, args.tcr_file, args.output_folder)
        return

    peptides, total_peps = load_peptides(args.peptide_file)
    tcrb_list, total_tcrs = load_tcrs(args.tcr_file, chain="beta")
    tcra_list = None
    if args.tcra_file:
        tcra_list, total_tcras = load_tcrs(args.tcra_file, chain="alpha")
        print(f"Valid TCRA sequences: {len(tcra_list)} out of {total_tcras}")

    print(f"Valid peptides: {len(peptides)} out of {total_peps}")
    print(f"Valid TCRB sequences: {len(tcrb_list)} out of {total_tcrs}")
    if not peptides or not tcrb_list:
        raise ValueError("No valid peptide/TCRB sequences left after filtering.")

    ensure_dir(args.output_folder)

    if args.target in {"rf", "all"}:
        out = os.path.join(args.output_folder, "Random_Forest") if args.target == "all" else args.output_folder
        n = create_for_random_forest(peptides, tcrb_list, out)
        print(f"[Random_Forest] Generated {n} files in {out}")

    if args.target in {"unifyimmun", "all"}:
        out = os.path.join(args.output_folder, "UnifyImmun_ext") if args.target == "all" else args.output_folder
        n = create_for_unifyimmun(peptides, tcrb_list, out)
        print(f"[UnifyImmun_ext] Generated {n} files in {out}")

    if args.target in {"dlptcr", "all"}:
        out = os.path.join(args.output_folder, "DLpTCR_ext") if args.target == "all" else args.output_folder
        n = create_for_dlptcr(
            peptides=peptides,
            tcrb_list=tcrb_list,
            output_folder=out,
            tcra_list=tcra_list,
        )
        print(f"[DLpTCR_ext] Generated {n} files in {out}")

    if args.target in {"ergo2", "all"}:
        out = os.path.join(args.output_folder, "ERGO-II_ext") if args.target == "all" else args.output_folder
        n = create_for_ergo2(
            peptides=peptides,
            tcrb_list=tcrb_list,
            output_folder=out,
            tcra_list=tcra_list,
            mhc_value=args.mhc,
            tcell_type_value=args.tcell_type,
        )
        print(f"[ERGO-II_ext] Generated {n} files in {out}")


if __name__ == "__main__":
    main()
