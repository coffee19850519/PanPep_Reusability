# Metrics Calculation

The evaluation metrics pipeline is located in `./metric_calculation/` and supports both **classification** and **ranking** metrics. All scripts support CSV and Parquet input formats.

---

## Pipeline Overview

```
Inference Output (Parquet/CSV)
        │
        ▼
[1] Sample Index Generation
    ├── shuffling_index.py   (shuffling-based)
    ├── random_index.py      (random balanced)
    └── get_sample_indices_100.py(for CASE 1 & 2) / get_sample_indices_1.py(for CASE 3，4 & 5)
        │
        ▼
[2] Sort Predictions
    └── sort.py
        │
        ▼
[3] Calculate Metrics
    ├── AUC.py                    (classification)
    ├── Top_rank_percentile.py    (ranking)
    ├── bedroc.py                 (ranking)
    ├── success_rate&hit_rate.py  (ranking)
    └── get_success_AUC.py        (partial AUC of success/hit rate curves)
```

---

## Step 1: Sample Index Generation

### [`shuffling_index.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/shuffling_index.py) — Shuffling-Based Sampling (CASE 1 & 2)

Generates 100 sampling index records per file using a **shuffling strategy**: all positive samples are kept, and an equal number of negatives are drawn from sequences present in a provided background database (allowed negatives list).

```bash
python metric_calculation/shuffling_index.py \
    --src_folder <folder_with_prediction_files> \
    --allowed_negatives_file <path_to_background_tcr_list.txt> \
    --record_file <output_indices.csv> \
    --num_processes 64
```

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `--src_folder` | Folder containing prediction CSV/Parquet files |
| `--allowed_negatives_file` | Text file with one allowed CDR3 per line |
| `--record_file` | Output CSV recording sampling indices |
| `--num_processes` | Parallel workers (default: CPU count) |

**Required columns** in input files: `CDR3`, `Label`

---

### [`random_index.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/random_index.py) — Random Balanced Sampling (CASE 1 & 2)

Generates 100 balanced sampling index records per file by randomly selecting equal numbers of positive and negative samples.

```bash
python metric_calculation/random_index.py \
    --src_folder <folder_with_prediction_files> \
    --record_file <output_indices.csv> \
    --num_processes 64
```

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `--src_folder` | Folder containing prediction CSV/Parquet files |
| `--record_file` | Output CSV recording sampling indices |
| `--num_processes` | Parallel workers (default: 64) |

---

### [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) — Apply Indices (CASE 1 & 2)

Applies the recorded sampling indices to source Parquet files, producing 100 sampled CSV datasets per peptide. Optionally merges all sample CSVs within each output folder.

```bash
python metric_calculation/get_sample_indices_100.py \
    --src_folder <folder_with_parquet_files> \
    --dst_folder <output_folder> \
    --record_file <indices.csv> \
    --num_processes 64
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--src_folder` | Source folder with Parquet files | required |
| `--dst_folder` | Destination folder for sampled CSVs | required |
| `--record_file` | CSV file with sampling indices | required |
| `--num_processes` | Parallel workers | `64` |
| `--skip_merge` | Skip merging step | `False` |

Output structure: `dst_folder/sample_<N>/<peptide>.csv` for each of the 100 samples.

---

### [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) — Apply Indices (CASE 3, 4, 5)

Simplified version for training reproducibility cases. Applies one set of indices per file without the merging step.

```bash
python metric_calculation/get_sample_indices_1.py \
    --src_folder <folder_with_data_files> \
    --dst_folder <output_folder> \
    --record_file <indices.csv> \
    --num_processes 20
```

**Required columns** in source files: `CDR3`, `Label`

> **Note**: Use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) for CASE 3, 4, and 5 (training reproducibility). Use [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for CASE 1 and 2 (inference reproducibility).

---

## Step 2: Sort Predictions

### [`sort.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/sort.py) — Sort by Score

Sorts prediction files by descending score. Supports parallel processing of multiple files and preserves the original directory structure.

```bash
python metric_calculation/sort.py \
    --input_dir <folder_with_prediction_files> \
    --output_dir <sorted_output_folder> \
    --num_cores 64
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory (CSV or Parquet) | required |
| `--output_dir` | Output directory for sorted files | required |
| `--num_cores` | Parallel workers | `64` |

**Required columns**: `CDR3`, `Score`, `Label`

Output: `<output_dir>/<original_path>/<filename>_sorted.<ext>`

---

## Step 3: Calculate Metrics

### [`AUC.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/AUC.py) — Classification Metrics (ROC-AUC & PR-AUC)

Computes ROC-AUC and PR-AUC for all CSV files in a directory tree. Generates per-file results and per-folder averages.

```bash
python metric_calculation/AUC.py \
    --root_dir <folder_with_sampled_csvs> \
    --output_file <results.csv> \
    --max_workers 32
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root_dir` | Root directory containing sampled CSV files | required |
| `--output_file` | Output CSV with per-file and folder-averaged metrics | required |
| `--max_workers` | Parallel workers | CPU count |

**Required columns** in input CSVs: `label`, `score`

Output columns: `folder_path`, `file_name`, `roc_auc`, `pr_auc`, `status`
Folder-level averages are appended with `file_name = [FOLDER_SUMMARY]`.

---

### [`Top_rank_percentile.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/Top_rank_percentile.py) — Top-K Success & Hit Rate

Computes average **success rate** and **hit rate** at each rank position (Top-K), using GPU acceleration (falls back to CPU if unavailable).

```bash
python metric_calculation/Top_rank_percentile.py \
    --root_dir <sorted_predictions_folder> \
    --top_k 11419896 \
    --batch_size 150 \
    --output results \
    --output_dir <output_directory>
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root_dir` | Directory with sorted prediction files | required |
| `--top_k` | Maximum Top-K positions to calculate | `11419896` |
| `--batch_size` | Processing batch size | `150` |
| `--output` | Output filename (without extension) | `results` |
| `--output_dir` | Output directory | current directory |

**Required column**: `Label`

Output: `<output_dir>/<output>.csv` and `<output_dir>/<output>.parquet`
Output columns: `Directory`, `Top_K`, `Success_Rate`, `Hit_Rate`

---

### [`bedroc.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/bedroc.py) — BEDROC Score

Computes BEDROC (Boltzmann-Enhanced Discrimination of ROC) scores across multiple alpha values. BEDROC penalizes models that rank positives late, rewarding early enrichment.

```bash
python metric_calculation/bedroc.py \
    --root_dir <sorted_predictions_folder> \
    --output_dir <output_directory> \
    --detailed_dir detailed_results \
    --averaged_file averaged_metrics.csv \
    --num_processes 32
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root_dir` | Root directory with sorted prediction files | required |
| `--output_dir` | Output directory | `results` |
| `--detailed_dir` | Subdirectory for per-file results | `detailed_results` |
| `--averaged_file` | File for averaged results | `averaged_metrics.csv` |
| `--num_processes` | Parallel workers | `min(64, CPU-2)` |

Output columns in averaged file: `Directory`, `Alpha`, `BEDROC_Score`

---

### [`success_rate&hit_rate.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/success_rate%26hit_rate.py) — Per-Folder Success & Hit Rate

Computes cumulative success rate and hit rate curves aggregated per directory.

```bash
python metric_calculation/success_rate&hit_rate.py \
    --root_dir <sorted_predictions_folder> \
    --top_k 11419896 \
    --output results \
    --output_dir <output_directory>
```

Same interface as [`Top_rank_percentile.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/Top_rank_percentile.py). Uses GPU if available.

---

### [`get_success_AUC.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_success_AUC.py) — Partial AUC of Success/Hit Rate Curves

Computes **partial AUC** values under the Success Rate or Hit Rate curve for each directory, integrating over a specified fraction of the Top-K range. Useful for summarising early-enrichment performance as a single scalar per peptide/directory.

Reads the output of  [`success_rate&hit_rate.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/success_rate%26hit_rate.py) directly (CSV or Parquet).

```bash
python metric_calculation/get_success_AUC.py \
    --input <results.csv or results.parquet> \
    --rate-type Success_Rate \
    --cutoffs 0.0001 0.001 0.01 0.05 0.1 0.2 1.0 \
    --output <output.csv>
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Path to input CSV or Parquet file | required |
| `--rate-type` | Column name of the rate to integrate (`Success_Rate` or `Hit_Rate`) | `Success_Rate` |
| `--cutoffs` | One or more cutoff values in `(0, 1]` defining the Top-K fraction to integrate over | `0.0001 0.001 0.01 0.05 0.10 0.20 1.0` |
| `--replace-zeros-for-hit-rate` | Replace zero Hit Rate values with a small positive number before integration (log-scale friendlier) | `False` |
| `--output` | Output CSV path. Auto-generated from input filename if omitted | auto |

**Required input columns**: `Directory`, `Top_K`, and the column named by `--rate-type`

**How cutoffs work**: For each directory, `Top_K` is normalised to `[0, 1]` (min–max). The partial AUC is the area under the rate curve from `x=0` to `x=cutoff`, computed with the trapezoidal rule. A point is linearly interpolated at the cutoff boundary when needed.

**Output columns**:

| Column | Description |
|--------|-------------|
| `Directory` | Peptide / directory identifier |
| `n_rows` | Number of data points used |
| `AUC_<cutoff>_area` | Partial AUC area at each cutoff (e.g. `AUC_1pct_area`, `AUC_5pct_area`) |

Cutoff labels are formatted as `{pct}pct` (e.g. `0.01 → 1pct`, `0.001 → 0_1pct`, `0.0001 → 0_01pct`).

If `--output` is not specified, the output file is saved next to the input file with a name like `<stem>_success_rate_partial_auc_by_directory.csv`.

**Examples**:

```bash
# Partial AUC of Success Rate at standard cutoffs
python metric_calculation/get_success_AUC.py \
    --input results.parquet \
    --rate-type Success_Rate
```

---

## Column Name Reference

| Script | Key Input Columns | Key Output Columns |
|--------|------------------|--------------------|
| [`shuffling_index.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/shuffling_index.py) / [`random_index.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/random_index.py) | `CDR3`, `Label` | `label_1_indices`, `label_0_indices` |
| `get_sample_indices_*.py` | `CDR3`, `Label` | sampled CSV rows |
| [`sort.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/sort.py) | `CDR3`, `Score`, `Label` | sorted file |
| [`AUC.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/AUC.py) | `label`, `score` | `roc_auc`, `pr_auc` |
| [`Top_rank_percentile.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/Top_rank_percentile.py) | `Label` | `Success_Rate`, `Hit_Rate` |
| [`bedroc.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/bedroc.py) | `Label` | `BEDROC_Score` |
| [`success_rate&hit_rate.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/success_rate%26hit_rate.py) | `Label` | `Success_Rate`, `Hit_Rate` |
| [`get_success_AUC.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_success_AUC.py) | `Directory`, `Top_K`, `Success_Rate`/`Hit_Rate` | `AUC_<cutoff>_area` per cutoff |
