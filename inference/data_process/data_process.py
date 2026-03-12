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
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(file_path, encoding="utf-8-sig", header=None)


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


def load_positive_csv(csv_file: str, chain: str = "beta") -> Tuple[List[str], set]:
    """Load a CSV with positive (peptide, TCR) pairs.

    Returns:
        peptide_list: deduplicated list of valid peptide sequences from the CSV.
        positive_pairs: set of (peptide, tcr) tuples that carry label=1.
    """
    df = _read_table(Path(csv_file))
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    pep_col = None
    for candidate in ["peptide", "epitope"]:
        if candidate in col_map:
            pep_col = col_map[candidate]
            break
    if pep_col is None:
        pep_col = df.columns[0]

    tcr_col = None
    if chain == "alpha":
        alpha_candidates = ["binding_TCR", "tcra", "tra", "tcra_cdr3", "cdr3a", "alpha"]
        for candidate in alpha_candidates:
            if candidate in col_map:
                tcr_col = col_map[candidate]
                break
    if tcr_col is None:
        for candidate in ["binding_TCR", "tcr", "tcrb", "trb", "tcrb_cdr3", "cdr3b", "cdr3"]:
            if candidate in col_map:
                tcr_col = col_map[candidate]
                break
    if tcr_col is None:
        tcr_col = df.columns[1]

    peps_raw = [validate_sequence(x) for x in df[pep_col].tolist()]
    tcrs_raw = [validate_sequence(x) for x in df[tcr_col].tolist()]

    positive_pairs: set = set()
    seen_peps: dict = {}  # preserve insertion order while deduplicating
    for pep, tcr in zip(peps_raw, tcrs_raw):
        if pep is None or tcr is None:
            continue
        positive_pairs.add((pep, tcr))
        seen_peps[pep] = None

    peptide_list = list(seen_peps.keys())
    return peptide_list, positive_pairs


def load_positive_csv_ab(csv_file: str) -> Tuple[List[str], set]:
    """Load a CSV with positive (peptide, alpha, beta) triples.

    Returns:
        peptide_list: deduplicated list of valid peptide sequences.
        positive_triples: set of (peptide, tcra, tcrb) tuples that carry label=1.
    """
    df = _read_table(Path(csv_file))
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    pep_col = None
    for c in ["peptide", "epitope"]:
        if c in col_map:
            pep_col = col_map[c]
            break
    if pep_col is None:
        pep_col = df.columns[0]

    tcra_col = None
    for c in ["alpha", "tcra", "tra", "tcra_cdr3", "cdr3a"]:
        if c in col_map:
            tcra_col = col_map[c]
            break
    if tcra_col is None:
        tcra_col = df.columns[1]

    tcrb_col = None
    for c in ["beta", "tcrb", "trb", "tcrb_cdr3", "cdr3b", "cdr3"]:
        if c in col_map:
            tcrb_col = col_map[c]
            break
    if tcrb_col is None:
        tcrb_col = df.columns[2]

    peps_raw  = [validate_sequence(x) for x in df[pep_col].tolist()]
    tcras_raw = [validate_sequence(x) for x in df[tcra_col].tolist()]
    tcrbs_raw = [validate_sequence(x) for x in df[tcrb_col].tolist()]

    positive_triples: set = set()
    seen_peps: dict = {}
    for pep, tcra, tcrb in zip(peps_raw, tcras_raw, tcrbs_raw):
        if pep is None or tcra is None or tcrb is None:
            continue
        positive_triples.add((pep, tcra, tcrb))
        seen_peps[pep] = None

    return list(seen_peps.keys()), positive_triples


def load_tcrab(tcr_file: str) -> Tuple[List[str], List[str], int]:
    """Load a CSV with paired alpha and beta TCR sequences.

    Returns:
        tcra_list, tcrb_list: parallel lists of valid paired sequences.
        total: raw row count before filtering.
    """
    df = _read_table(Path(tcr_file))
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    tcra_col = None
    for c in ["alpha", "tcra", "tra", "tcra_cdr3", "cdr3a"]:
        if c in col_map:
            tcra_col = col_map[c]
            break
    if tcra_col is None:
        tcra_col = df.columns[0]

    tcrb_col = None
    for c in ["beta", "tcrb", "trb", "tcrb_cdr3", "cdr3b", "cdr3"]:
        if c in col_map:
            tcrb_col = col_map[c]
            break
    if tcrb_col is None:
        tcrb_col = df.columns[1]

    total = len(df)
    tcras_raw = [validate_sequence(x) for x in df[tcra_col].tolist()]
    tcrbs_raw = [validate_sequence(x) for x in df[tcrb_col].tolist()]

    # Keep only rows where both chains are valid; deduplicate by (tcra, tcrb) pair
    seen: dict = {}
    for a, b in zip(tcras_raw, tcrbs_raw):
        if a is not None and b is not None:
            seen[(a, b)] = None

    tcra_list = [a for a, b in seen]
    tcrb_list = [b for a, b in seen]
    return tcra_list, tcrb_list, total


def load_tcrs(tcr_file: str, chain: str = "beta") -> Tuple[List[str], int]:
    path = Path(tcr_file)
    suffix = path.suffix.lower()

    if chain == "alpha":
        preferred = ["tcra", "tra", "tcra_cdr3", "alpha", "cdr3a","binding_TCR"]
    else:
        preferred = ["tcr", "tcrb", "trb","binding_TCR", "tcrb_cdr3", "beta", "cdr3", "cdr3b"]

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


def _build_tcr_list_for_pep(pep: str, base_tcrs: List[str], positive_pairs: Optional[set]) -> List[str]:
    """Merge base TCR list with this peptide's positive TCRs (deduplicated)."""
    if positive_pairs is None:
        return base_tcrs
    seen = set(base_tcrs)
    extra = [tcr for (p, tcr) in positive_pairs if p == pep and tcr not in seen]
    return base_tcrs + extra


def _build_tcrab_list_for_pep(
    pep: str,
    base_tcra: List[str],
    base_tcrb: List[str],
    positive_triples: Optional[set],
) -> Tuple[List[str], List[str]]:
    """Merge base (alpha, beta) pool with this peptide's positive pairs (deduplicated)."""
    seen = set(zip(base_tcra, base_tcrb))
    extra_a, extra_b = [], []
    if positive_triples:
        for p, a, b in positive_triples:
            if p == pep and (a, b) not in seen:
                extra_a.append(a)
                extra_b.append(b)
                seen.add((a, b))
    return base_tcra + extra_a, base_tcrb + extra_b


def _get_labels(pep: str, tcrb_list: List[str], positive_pairs: Optional[set]) -> List[int]:
    if positive_pairs is None:
        return [0] * len(tcrb_list)
    return [1 if (pep, tcr) in positive_pairs else 0 for tcr in tcrb_list]


def _get_labels_ab(
    pep: str,
    tcra_list: List[str],
    tcrb_list: List[str],
    positive_triples: Optional[set],
) -> List[int]:
    if positive_triples is None:
        return [0] * len(tcrb_list)
    return [1 if (pep, a, b) in positive_triples else 0 for a, b in zip(tcra_list, tcrb_list)]


def create_for_random_forest(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    positive_pairs: Optional[set] = None,
) -> int:
    def _build(pep: str) -> pd.DataFrame:
        per_pep_tcrs = _build_tcr_list_for_pep(pep, tcrb_list, positive_pairs)
        return pd.DataFrame(
            {
                "tcr": per_pep_tcrs,
                "peptide": [pep] * len(per_pep_tcrs),
                "label": _get_labels(pep, per_pep_tcrs, positive_pairs),
            }
        )

    return write_per_peptide_csv(_build, peptides, output_folder)


def create_for_unifyimmun(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    positive_pairs: Optional[set] = None,
) -> int:
    # Input format is the same as Random_Forest: tcr, peptide, label
    return create_for_random_forest(peptides, tcrb_list, output_folder, positive_pairs)


def create_for_dlptcr(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    tcra_list: Optional[List[str]] = None,
    positive_pairs: Optional[set] = None,
    chain: str = "beta",
) -> int:
    def _build(pep: str) -> pd.DataFrame:
        if chain == "ab":
            pa, pb = _build_tcrab_list_for_pep(pep, tcra_list or [], tcrb_list, positive_pairs)
            n = len(pb)
            labels = _get_labels_ab(pep, pa, pb, positive_pairs)
            return pd.DataFrame({"TCRA_CDR3": pa, "TCRB_CDR3": pb, "Epitope": [pep] * n, "label": labels})
        per_pep_tcrs = _build_tcr_list_for_pep(pep, tcrb_list, positive_pairs)
        n = len(per_pep_tcrs)
        labels = _get_labels(pep, per_pep_tcrs, positive_pairs)
        if chain == "alpha":
            return pd.DataFrame({"TCRA_CDR3": per_pep_tcrs, "TCRB_CDR3": [""] * n, "Epitope": [pep] * n, "label": labels})
        return pd.DataFrame({"TCRA_CDR3": [""] * n, "TCRB_CDR3": per_pep_tcrs, "Epitope": [pep] * n, "label": labels})

    return write_per_peptide_csv(_build, peptides, output_folder)


def create_for_ergo2(
    peptides: List[str],
    tcrb_list: List[str],
    output_folder: str,
    tcra_list: Optional[List[str]] = None,
    positive_pairs: Optional[set] = None,
    chain: str = "beta",
) -> int:
    def _build(pep: str) -> pd.DataFrame:
        if chain == "ab":
            tra_col, trb_col = _build_tcrab_list_for_pep(pep, tcra_list or [], tcrb_list, positive_pairs)
            labels = _get_labels_ab(pep, tra_col, trb_col, positive_pairs)
        else:
            per_pep_tcrs = _build_tcr_list_for_pep(pep, tcrb_list, positive_pairs)
            tra_col = [""] * len(per_pep_tcrs)
            trb_col = per_pep_tcrs
            labels = _get_labels(pep, per_pep_tcrs, positive_pairs)
        n = len(trb_col)
        return pd.DataFrame(
            {
                "TRA": tra_col,
                "TRB": trb_col,
                "TRAV": [""] * n,
                "TRAJ": [""] * n,
                "TRBV": [""] * n,
                "TRBJ": [""] * n,
                "T-Cell-Type": [""] * n,
                "Peptide": [pep] * n,
                "MHC": [""] * n,
                "label": labels,
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
            "DLpTCR_ext, ERGO-II_ext, Random_Forest, UnifyImmun_ext.\n\n"
            "Two input modes:\n"
            "  Classic : --peptide-file + --tcr-file (both txt/csv/parquet), all labels=0\n"
            "  Labeled : --positive-csv (csv with peptide/TCR positive pairs) + --tcr-file (txt);\n"
            "            peptides come from the CSV, TCR pool from the txt,\n"
            "            label=1 if (peptide,TCR) in CSV else 0."
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
    parser.add_argument(
        "--peptide-file",
        default=None,
        help="Peptide list (txt/csv/parquet). Used in classic mode.",
    )
    parser.add_argument(
        "--positive-csv",
        default=None,
        help=(
            "CSV with positive (peptide, TCR) pairs. "
            "Provides the peptide list and positive labels. "
            "Used together with --tcr-file in labeled mode."
        ),
    )
    parser.add_argument(
        "--tcr-file",
        required=True,
        help="TCR beta file (txt/csv/parquet). The full TCR pool to evaluate.",
    )
    parser.add_argument("--tcra-file", default=None, help="Optional TCR alpha file path for DLpTCR/ERGO.")
    parser.add_argument(
        "--chain",
        default="beta",
        choices=["alpha", "beta", "ab"],
        help="TCR chain type: alpha | beta (default) | ab (both chains, for DLpTCR/ERGO).",
    )
    parser.add_argument("--output-folder", required=True, help="Output folder.")
    args = parser.parse_args()

    if args.positive_csv and args.peptide_file:
        parser.error("Specify either --positive-csv or --peptide-file, not both.")
    if not args.positive_csv and not args.peptide_file:
        parser.error("One of --positive-csv or --peptide-file is required.")

    positive_pairs: Optional[set] = None
    tcra_list: Optional[List[str]] = None

    if args.chain == "ab":
        # --- ab mode ---
        if args.positive_csv:
            peptides, positive_pairs = load_positive_csv_ab(args.positive_csv)
            print(f"Valid peptides (from CSV): {len(peptides)}")
            print(f"Positive (peptide, TCRA, TCRB) triples: {len(positive_pairs)}")
        else:
            peptides, total_peps = load_peptides(args.peptide_file)
            print(f"Valid peptides: {len(peptides)} out of {total_peps}")
        tcra_list, tcr_list, total_tcrs = load_tcrab(args.tcr_file)
        print(f"Valid TCR ab pairs (from file): {len(tcr_list)} out of {total_tcrs}")
    else:
        # --- single-chain mode ---
        if args.positive_csv:
            peptides, positive_pairs = load_positive_csv(args.positive_csv, chain=args.chain)
            print(f"Valid peptides (from CSV): {len(peptides)}")
            print(f"Positive (peptide, TCR) pairs: {len(positive_pairs)}")
        else:
            peptides, total_peps = load_peptides(args.peptide_file)
            print(f"Valid peptides: {len(peptides)} out of {total_peps}")
        tcr_list, total_tcrs = load_tcrs(args.tcr_file, chain=args.chain)
        print(f"Valid TCR sequences (from file): {len(tcr_list)} out of {total_tcrs}")
        if args.tcra_file:
            tcra_list, total_tcras = load_tcrs(args.tcra_file, chain="alpha")
            print(f"Valid TCRA sequences: {len(tcra_list)} out of {total_tcras}")

    if not peptides or not tcr_list:
        raise ValueError("No valid peptide/TCR sequences left after filtering.")

    ensure_dir(args.output_folder)

    if args.target in {"rf", "all"}:
        out = os.path.join(args.output_folder, "Random_Forest") if args.target == "all" else args.output_folder
        n = create_for_random_forest(peptides, tcr_list, out, positive_pairs)
        print(f"[Random_Forest] Generated {n} files in {out}")

    if args.target in {"unifyimmun", "all"}:
        out = os.path.join(args.output_folder, "UnifyImmun_ext") if args.target == "all" else args.output_folder
        n = create_for_unifyimmun(peptides, tcr_list, out, positive_pairs)
        print(f"[UnifyImmun_ext] Generated {n} files in {out}")

    if args.target in {"dlptcr", "all"}:
        out = os.path.join(args.output_folder, "DLpTCR_ext") if args.target == "all" else args.output_folder
        n = create_for_dlptcr(
            peptides=peptides,
            tcrb_list=tcr_list,
            output_folder=out,
            tcra_list=tcra_list,
            positive_pairs=positive_pairs,
            chain=args.chain,
        )
        print(f"[DLpTCR_ext] Generated {n} files in {out}")

    if args.target in {"ergo2", "all"}:
        out = os.path.join(args.output_folder, "ERGO-II_ext") if args.target == "all" else args.output_folder
        n = create_for_ergo2(
            peptides=peptides,
            tcrb_list=tcr_list,
            output_folder=out,
            tcra_list=tcra_list,
            positive_pairs=positive_pairs,
            chain=args.chain,
        )
        print(f"[ERGO-II_ext] Generated {n} files in {out}")


if __name__ == "__main__":
    main()
