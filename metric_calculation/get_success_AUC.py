#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute partial AUC values by Directory from a CSV or Parquet file.

Supported input columns:
- Directory
- Top_K
- Success_Rate / Hit_Rate / other user-specified rate column

Examples
--------
python compute_partial_auc_by_directory.py \
    --input /path/to/result.parquet \
    --rate-type Success_Rate \
    --cutoffs 0.0001 0.001 0.01 0.05 0.1 0.2 1.0

python compute_partial_auc_by_directory.py \
    --input /path/to/result.csv \
    --rate-type Hit_Rate \
    --replace-zeros-for-hit-rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_data(file_path: str, columns=None, directory=None) -> pd.DataFrame:
    """
    Read data from a Parquet or CSV file.

    For Parquet files, try to filter by Directory using pyarrow filters
    to avoid loading the full table into memory.
    For CSV files, load then filter.
    """
    if file_path.endswith(".parquet"):
        try:
            if directory is None:
                return pd.read_parquet(file_path, columns=columns, engine="pyarrow")
            return pd.read_parquet(
                file_path,
                columns=columns,
                engine="pyarrow",
                filters=[("Directory", "==", directory)],
            )
        except Exception:
            df = pd.read_parquet(file_path, columns=columns)
            if directory is not None:
                df = df[df["Directory"] == directory]
            return df

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, usecols=columns)
        if directory is not None:
            df = df[df["Directory"] == directory]
        return df

    raise ValueError(f"Unsupported file format: {file_path}")


def discover_unique_directories(file_path: str) -> np.ndarray:
    """Load only the Directory column and return unique non-null values."""
    df_dir = read_data(file_path, columns=["Directory"])
    return df_dir["Directory"].dropna().unique()


def interpolate_y_at_cutoff(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    idx_end: int,
    x_cut: float,
) -> float:
    """
    Linearly interpolate y at x_cut.

    Parameters
    ----------
    x_sorted : np.ndarray
        Monotonically increasing x array.
    y_sorted : np.ndarray
        Corresponding y array.
    idx_end : int
        Result of np.searchsorted(x_sorted, x_cut, side="right").
    x_cut : float
        Cutoff value in the original x scale.
    """
    n = len(x_sorted)
    if n == 0:
        return np.nan
    if idx_end <= 0:
        return float(y_sorted[0])
    if idx_end >= n:
        return float(y_sorted[-1])

    x0, y0 = float(x_sorted[idx_end - 1]), float(y_sorted[idx_end - 1])
    x1, y1 = float(x_sorted[idx_end]), float(y_sorted[idx_end])

    if x1 == x0:
        return y0

    t = (x_cut - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def deduplicate_x_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate x values by averaging y values for repeated x.
    """
    unique_x, inv = np.unique(x, return_inverse=True)
    if len(unique_x) == len(x):
        return x, y

    y_sum = np.bincount(inv, weights=y)
    count = np.bincount(inv)
    unique_y = y_sum / count
    return unique_x, unique_y


def partial_auc_from_topk(
    topk_sorted: np.ndarray,
    y_sorted: np.ndarray,
    cutoff: float,
) -> float:
    """
    Compute partial AUC using the first top-x% region.

    Steps
    -----
    1. Normalize Top_K into [0, 1]:
         x_norm = (Top_K - min) / (max - min)
    2. Keep only points with x_norm <= cutoff
    3. If cutoff falls between two points, add one interpolated cutoff point
    4. Integrate using trapezoidal rule

    Returns
    -------
    float
        Partial AUC area.
    """
    x = np.asarray(topk_sorted, dtype=np.float64)
    y = np.asarray(y_sorted, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    xmin, xmax = x[0], x[-1]
    denom = xmax - xmin
    if denom <= 0 or not np.isfinite(denom):
        return np.nan

    cutoff = float(cutoff)
    if cutoff <= 0 or cutoff > 1:
        return np.nan

    x_cut = xmin + cutoff * denom
    idx_end = int(np.searchsorted(x, x_cut, side="right"))

    if idx_end < 2:
        return np.nan

    xs = x[:idx_end].copy()
    ys = y[:idx_end].copy()

    if idx_end < len(x) and xs[-1] < x_cut:
        y_cut = interpolate_y_at_cutoff(x, y, idx_end, x_cut)
        xs = np.append(xs, x_cut)
        ys = np.append(ys, y_cut)

    xs_norm = (xs - xmin) / denom
    xs_norm[-1] = min(xs_norm[-1], cutoff)

    xs_norm, ys = deduplicate_x_mean(xs_norm, ys)
    if len(xs_norm) < 2:
        return np.nan

    area = float(np.trapz(ys, xs_norm))
    return area


def format_cutoff_label(cutoff: float) -> str:
    """
    Convert cutoff to a stable column label.

    Examples
    --------
    0.0001 -> 0_01pct
    0.001  -> 0_1pct
    0.01   -> 1pct
    0.05   -> 5pct
    1.0    -> 100pct
    """
    pct = cutoff * 100
    s = f"{pct:.6f}".rstrip("0").rstrip(".")
    s = s.replace(".", "_")
    return f"{s}pct"


def process_one_directory(
    df: pd.DataFrame,
    directory: str,
    rate_type: str,
    cutoffs: tuple[float, ...],
    replace_zeros_for_hit_rate: bool,
) -> dict | None:
    """
    Compute all partial AUC values for one directory.
    """
    df = df[["Top_K", rate_type]].dropna()
    if df.empty:
        print(f"[{directory}] empty after dropna")
        return None

    df = df.sort_values("Top_K", kind="mergesort")
    topk = df["Top_K"].to_numpy(dtype=np.float64)
    y = df[rate_type].to_numpy(dtype=np.float64)

    if len(topk) < 2:
        print(f"[{directory!r}] insufficient points")
        return None

    if rate_type == "Hit_Rate" and replace_zeros_for_hit_rate:
        non_zero = y[y > 0]
        replacement = (non_zero.min() / 10.0) if len(non_zero) > 0 else 1e-10
        y = np.where(y == 0, replacement, y)

    print(
        f"[{directory!r}] "
        f"n={len(df)}  "
        f"Top_K range=({topk[0]:.0f}, {topk[-1]:.0f})  "
        f"y range=({np.nanmin(y):.6g}, {np.nanmax(y):.6g})"
    )

    row = {
        "Directory": directory,
        "n_rows": len(df),
    }

    for cutoff in cutoffs:
        area = partial_auc_from_topk(topk, y, cutoff)
        label = format_cutoff_label(cutoff)
        print(f"  AUC@{cutoff:.4%}: area={area:.6g}")
        row[f"AUC_{label}_area"] = area

    return row


def compute_directory_partial_aucs(
    file_path: str,
    rate_type: str = "Success_Rate",
    cutoffs: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 1.0),
    replace_zeros_for_hit_rate: bool = False,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute partial AUC values by Directory and save results to CSV.
    """
    required_cols = ["Directory", "Top_K", rate_type]
    rows = []

    if file_path.endswith(".csv"):
        df_all = read_data(file_path, columns=required_cols)
        df_all = df_all.dropna(subset=["Directory"])

        directories = df_all["Directory"].drop_duplicates().to_numpy()
        print(f"Found {len(directories)} directories.")
        for i, directory in enumerate(directories, 1):
            print(f"{i}. {directory!r}")

        for directory in directories:
            df_d = df_all[df_all["Directory"] == directory]
            row = process_one_directory(
                df_d,
                directory,
                rate_type,
                cutoffs,
                replace_zeros_for_hit_rate,
            )
            if row is not None:
                rows.append(row)

    else:
        directories = discover_unique_directories(file_path)
        print(f"Found {len(directories)} directories.")
        for i, directory in enumerate(directories, 1):
            print(f"{i}. {directory!r}")

        for directory in directories:
            df_d = read_data(file_path, columns=required_cols, directory=directory)
            if df_d.empty:
                print(f"[{directory}] empty")
                continue

            row = process_one_directory(
                df_d,
                directory,
                rate_type,
                cutoffs,
                replace_zeros_for_hit_rate,
            )
            if row is not None:
                rows.append(row)

    out_df = pd.DataFrame(rows)

    if output_path is None:
        input_path = Path(file_path)
        if file_path.endswith(".parquet"):
            output_path = str(
                input_path.with_name(
                    input_path.name.replace(
                        ".parquet",
                        f"_{rate_type.lower()}_partial_auc_by_directory.csv",
                    )
                )
            )
        else:
            output_path = str(
                input_path.with_name(
                    f"{input_path.stem}_{rate_type.lower()}_partial_auc_by_directory.csv"
                )
            )

    out_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")
    return out_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute partial AUC values by Directory from CSV/Parquet."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV or Parquet file.",
    )
    parser.add_argument(
        "--rate-type",
        default="Success_Rate",
        help="Column name of the target rate. Default: Success_Rate",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=float,
        default=[0.0001, 0.001, 0.01, 0.05, 0.10, 0.20, 1.0],
        help="List of cutoff values in [0, 1].",
    )
    parser.add_argument(
        "--replace-zeros-for-hit-rate",
        action="store_true",
        help="Replace zero values when rate_type == Hit_Rate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    compute_directory_partial_aucs(
        file_path=args.input,
        rate_type=args.rate_type,
        cutoffs=tuple(args.cutoffs),
        replace_zeros_for_hit_rate=args.replace_zeros_for_hit_rate,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()