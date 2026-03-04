# Baseline Methods Manual

This document covers data preparation and inference usage for the following baseline / comparison methods:

- [Data Preparation](#data-preparation)
- [DLpTCR](#dlptcr)
- [ERGO-II](#ergo-ii)
- [UnifyImmun](#unifyimmun)
- [Random Forest](#random-forest)
- [UniPMT](#unipmt)

---

## Data Preparation

[`data_process.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/data_process/data_process.py) (located in `./inference/data_process/`) generates per-peptide input CSV files in the format required by each baseline method. It accepts a peptide list and a TCR beta pool, and optionally a TCR alpha pool, producing one CSV file per peptide.

### Usage

```bash
python data_process.py \
    --target <dlptcr|ergo2|rf|unifyimmun|all> \
    --peptide-file <path_to_peptide_file> \
    --tcr-file <path_to_tcrb_file> \
    --output-folder <output_directory> \
    [--tcra-file <path_to_tcra_file>]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target` | Format to generate: `dlptcr`, `ergo2`, `rf`, `unifyimmun`, or `all` | `all` |
| `--peptide-file` | Peptide file path (`.txt`, `.csv`, or `.parquet`) | required |
| `--tcr-file` | TCRβ file path (`.txt`, `.csv`, or `.parquet`) | required |
| `--tcra-file` | TCRα file path (optional; used by DLpTCR and ERGO-II) | — |
| `--output-folder` | Root output directory | required |

### Input File Formats

**Peptide file** — any of:
- Plain text (`.txt`): one peptide per line
- CSV/Parquet: column named `peptide` or `epitope` (first column used as fallback)

**TCR file** — any of:
- Plain text (`.txt`): one sequence per line
- CSV/Parquet: for TCRβ, column named `tcr`, `tcrb`, `trb`, `binding_TCR`, etc.; for TCRα, column named `tcra`, `tra`, `alpha`, etc.

All sequences are validated against the standard 20 amino acids (`ARNDCQEGHILKMFPSTWYV`). Invalid or missing sequences are silently filtered out.

### Output Structure

When `--target all`, the script creates one subdirectory per method:

```
<output-folder>/
├── DLpTCR_ext/
│   ├── <peptide1>.csv
│   └── <peptide2>.csv
├── ERGO-II_ext/
│   ├── <peptide1>.csv
│   └── ...
├── Random_Forest/
│   └── ...
└── UnifyImmun_ext/
    └── ...
```

Each CSV contains the full TCR pool paired with that peptide, with label `0` (unlabeled). When targeting a single method, files are written directly to `--output-folder`.

### Example

```bash
# Generate input for all methods at once
python data_process.py \
    --target all \
    --peptide-file ./data/peptides.txt \
    --tcr-file ./data/tcr_pool.txt \
    --output-folder ./input_data

# Generate only ERGO-II input, with alpha chain
python data_process.py \
    --target ergo2 \
    --peptide-file ./data/peptides.csv \
    --tcr-file ./data/tcrb_pool.txt \
    --tcra-file ./data/tcra_pool.txt \
    --output-folder ./input_data/ergo2
```

---

## DLpTCR

### Overview

DLpTCR is a deep learning model for TCR-epitope binding prediction. It supports three chain configurations: **TCRA (A)**, **TCRB (B)**, and **paired TCRαβ (AB)**. Each configuration uses an ensemble of three architectures (FULL, CNN, RESNET).

### Setup

1. Clone the DLpTCR repository ([GitHub](https://github.com/JiangBioLab/DLpTCR)):

   ```bash
   git clone https://github.com/JiangBioLab/DLpTCR.git
   ```

2. Copy the files from `./inference/DLpTCR_ext/` into the cloned repository:

   ```
   ./DLpTCR/code/
   ```

### Checkpoints

[Download Checkpoints](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/checkpoint/DLpTCR?csf=1&web=1&e=LuQ4XB)

### Scripts

Located in `./inference/DLpTCR_ext/`:

| File | Description |
|------|-------------|
| [`API.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/DLpTCR_ext/API.py) | Main entry point — chunk-based batch inference pipeline |
| [`DLpTCR_server.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/DLpTCR_ext/DLpTCR_server.py) | Model loading, prediction, and result saving |
| [`Model_Predict_Feature_Extraction.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/DLpTCR_ext/Model_Predict_Feature_Extraction.py) | Feature extraction (PCA, one-hot, chemical encoding) |

### Input Format

Use `data_process.py --target dlptcr` to generate the required input. CSV files contain:

| Model | Required Columns |
|-------|-----------------|
| A | `TCRA_CDR3`, `Epitope` |
| B | `TCRB_CDR3`, `Epitope` |
| AB | `TCRA_CDR3`, `TCRB_CDR3`, `Epitope` |

If `TCRA_CDR3` has no data, the column is left empty.

### Usage

```bash
python API.py \
    --input_file <path_to_csv> \
    --model <A|B|AB> \
    --sample_size 1000 \
    --batch_size 1000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_file` | Path to input CSV file | required |
| `--model` | Chain selection: `A`, `B`, or `AB` | `B` |
| `--sample_size` | Chunk size for memory-efficient processing | `1000` |
| `--batch_size` | Batch size for model prediction | `1000` |

### Output

Results saved to `./newdata/<input_filename>/final_predictions.parquet` (gzip compressed).

| Model | Output Columns |
|-------|---------------|
| AB | `TCRA_CDR3`, `TCRB_CDR3`, `Epitope`, `Predict`, `Probability (TCRA_Epitope)`, `Probability (TCRB_Epitope)` |
| B | `TCRB_CDR3`, `Epitope`, `Predict`, `Probability (predicted as a positive sample)` |
| A | `TCRA_CDR3`, `Epitope`, `Predict`, `Probability (predicted as a positive sample)` |

`Predict` values: `True TCR-pMHC` (binding) or `False TCR-pMHC` (non-binding).

### Pre-trained Model Files

Model `.h5` files must be placed in `./model/`, PCA dictionary files in `./pca/`:

| File | Chain | Architecture |
|------|-------|-------------|
| `FULL_A_ALL_onehot.h5` | TCRA | FULL (one-hot) |
| `CNN_A_ALL_onehot.h5` | TCRA | CNN (one-hot) |
| `RESNET_A_ALL_pca15.h5` | TCRA | ResNet (PCA-15) |
| `FULL_B_ALL_pca18.h5` | TCRB | FULL (PCA-18) |
| `CNN_B_ALL_pca20.h5` | TCRB | CNN (PCA-20) |
| `RESNET_B_ALL_pca10.h5` | TCRB | ResNet (PCA-10) |

### Ensemble Prediction Logic

- **TCRA / TCRB**: Averages probabilities from FULL, CNN, and RESNET; threshold at 0.5.
- **AB**: A sample is positive only if **both** TCRA and TCRB models independently predict binding.

---

## ERGO-II

### Overview

ERGO-II is a deep learning model for TCR-peptide binding prediction using LSTM-based encoding of TCR sequences. It supports VDJdb and McPAS dataset checkpoints.

### Setup

1. Clone the ERGO-II repository ([GitHub](https://github.com/IdoSpringer/ERGO-II)):

   ```bash
   git clone https://github.com/IdoSpringer/ERGO-II.git
   ```

2. Copy the files from `./inference/ERGO-II_ext/` into the cloned repository:

   ```
   ./ERGO-II/
   ```

### Checkpoints

[Download Checkpoints](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/checkpoint/ERGO-II?csf=1&web=1&e=7zR9pX)

### Scripts

Located in `./inference/ERGO-II_ext/`:

| File | Description |
|------|-------------|
| [`Predict.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/Predict.py) | Main inference script |
| [`get_filenames.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/get_filenames.py) | Utility to extract CSV filenames for batch processing |
| [`run.sh`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/run.sh) | Bash script for parallel multi-GPU inference |
| [`job_inference.slurm`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/job_inference.slurm) | SLURM job script for HPC cluster inference |

### Input Format

Use `data_process.py --target ergo2` to generate the required input. CSV files contain:

| Column | Description |
|--------|-------------|
| `TRB` | TCRβ CDR3 sequence (required) |
| `TRA` | TCRα CDR3 sequence (left empty if not provided) |
| `TRAV` | TCRα V gene (left empty if not provided) |
| `TRAJ` | TCRα J gene (left empty if not provided) |
| `TRBV` | TCRβ V gene (left empty if not provided) |
| `TRBJ` | TCRβ J gene (left empty if not provided) |
| `T-Cell-Type` | T cell type (`UNK` by default; set with `--tcell-type`) |
| `Peptide` | Epitope peptide sequence |
| `MHC` | MHC allele (`UNK` by default; set with `--mhc`) |

### Usage

```bash
python Predict.py \
    --dataset <vdjdb|mcpas> \
    --input_file <path_to_csv> \
    --output_dir <output_directory>
```

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Checkpoint to use: `vdjdb` or `mcpas` |
| `--input_file` | Path to input CSV file |
| `--output_dir` | Directory for output Parquet files |

### Output

Results saved as `<input_name>_predicted.parquet` (gzip compressed). Added column:
- `Score` (float32): ERGO-II binding probability score

### Model File Structure

| Checkpoint | Version |
|------------|---------|
| VDJdb | `1veajht` |
| McPAS | `1meajht` |

Checkpoints must be placed under `./Models/version_<id>/checkpoints/` and hyperparameters at `./Models/version_<id>/meta_tags.csv`. Training pickle files at `./Samples/<dataset>_train_samples.pickle`.

### Batch Inference (Multi-GPU)

**Using [`run.sh`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/run.sh)**:

Edit the `files` array and path variables in [`run.sh`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/run.sh), then run:

```bash
bash run.sh
```

Each file is assigned to a GPU in round-robin order (`gpu_id = i % MAX_GPU`).

**Using SLURM**:

Edit [`job_inference.slurm`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/job_inference.slurm) to set `BASE_PATH`, `OUTPUT_DIR`, and the `files` array, then submit:

```bash
sbatch job_inference.slurm
```

**Extracting filenames**:

```bash
python get_filenames.py \
    --directory <path_to_csv_dir> \
    --output filenames.txt
```

The output can be copy-pasted into the `files=(...)` array in [`run.sh`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/run.sh) or [`job_inference.slurm`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/ERGO-II_ext/job_inference.slurm).

---

## UnifyImmun

### Overview

UnifyImmun is a transformer-based model for TCR-peptide binding prediction using a unified architecture for immunological sequence modeling.

### Setup

1. Clone the UnifyImmun repository ([GitHub](https://github.com/hliulab/UnifyImmun)):

   ```bash
   git clone https://github.com/hliulab/UnifyImmun.git
   ```

2. Copy the files from `./inference/UnifyImmun_ext/` into the cloned repository:

   ```
   ./UnifyImmun/source/
   ```

### Checkpoints

[Download Checkpoints](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/checkpoint/Uniftylmmun?csf=1&web=1&e=qi0SQI)

### Scripts

Located in `./inference/UnifyImmun_ext/`:

| File | Description |
|------|-------------|
| [`TCR_inference.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/UnifyImmun_ext/TCR_inference.py) | Main inference script with `TCRPredictor` class |

### Input Format

Use `data_process.py --target unifyimmun` to generate the required input. CSV files contain:

| Column | Description |
|--------|-------------|
| `tcr` | TCRβ CDR3 sequence |
| `peptide` | Epitope peptide sequence |
| `label` | Binding label (0/1), optional (defaults to 0) |

### Usage

**Single file**:

```bash
python TCR_inference.py \
    --model ../trained_model/TCR_2/model_TCR.pkl \
    --input <path_to_csv> \
    --output <output.parquet> \
    --batch-size 8192
```

**Batch mode (multiple files)**:

```bash
python TCR_inference.py \
    --model ../trained_model/TCR_2/model_TCR.pkl \
    --input file1.csv file2.csv file3.csv \
    --output-dir <output_directory> \
    --batch-size 8192
```

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--model` | `-m` | Path to trained model (`.pkl`) | `../trained_model/TCR_2/model_TCR.pkl` |
| `--input` | `-i` | Input CSV file(s) | `../data/data_TCR/independent_set.csv` |
| `--output` | `-o` | Output Parquet path (single file mode) | auto-generated |
| `--output-dir` | | Output directory (batch mode) | — |
| `--batch-size` | `-b` | Prediction batch size | `8192` |

### Output

Compressed Parquet file (gzip) with original columns plus:
- `score` (float64): softmax binding probability from UnifyImmun

If Parquet writing fails, automatically falls back to CSV.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 66 |
| Max peptide length | 15 |
| Max TCR length | 34 |
| Default batch size | 8192 (max recommended: 65536) |
| Device | CUDA if available, else CPU |

---

## Random Forest

### Overview

A GPU-accelerated Random Forest classifier (cuML) for TCR-peptide binding prediction. Uses Atchley factor encodings with sinusoidal position encoding, with pre-computed sequence embedding caches for fast inference.

Scripts are located in `./inference/Random_Forest/`. No additional repository setup is required.

### Checkpoints

[Download Checkpoints](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/checkpoint/Random_Forest?csf=1&web=1&e=JuVUbx)

### Dependencies

- `cuml` — NVIDIA GPU-accelerated machine learning
- `cudf` — GPU DataFrame library
- `fastparquet` — Fast Parquet I/O
- `joblib` — Model serialization

### Input Format

Use `data_process.py --target rf` to generate the required input. CSV files contain:

| Column | Description |
|--------|-------------|
| `tcr` | TCRβ CDR3 sequence |
| `peptide` | Epitope peptide sequence |
| `label` | Binding label (0/1), optional |

### Usage

```bash
python predict_cuml_rf.py \
    --model <path_to_model.pkl> \
    --data <path_to_csv> \
    --output predictions.parquet \
    --aa_dict <path_to_aa_dict.joblib> \
    --peptide_encoding <path_to_peptide_encoding.npz> \
    --tcr_encoding <path_to_tcr_encoding.npz> \
    --batch_size 50000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to trained cuML RF model (`.pkl`) | required |
| `--data` | Path to input CSV | required |
| `--output` | Output Parquet file path | `predictions.parquet` |
| `--append` | Append to existing Parquet file | `False` |
| `--aa_dict` | Atchley factor dictionary (`.joblib`) | required |
| `--peptide_encoding` | Pre-computed peptide encodings (`.npz`) | required |
| `--tcr_encoding` | Pre-computed TCR encodings (`.npz`) | required |
| `--batch_size` | Samples per inference batch | `50000` |

### Output

Parquet file with original columns plus:
- `score` (float32): binding probability from the Random Forest

### Encoding Strategy

Each TCR-peptide pair is encoded as a concatenated feature vector:

1. **Peptide**: Atchley factors (5 physicochemical properties per amino acid) + sinusoidal position encoding → padded to length 15 → flattened
2. **TCR**: Same encoding → padded to length 25 → flattened
3. **Combined**: `[peptide_embedding | tcr_embedding]` → 1D feature vector

If a sequence is not found in the pre-computed `.npz` cache, the script falls back to real-time encoding. Cache hit statistics are printed at the end of inference.

Supplementary fig 5 [Unitylmmun independent Dataset](https://mailmissouri-my.sharepoint.com/:x:/r/personal/hefe_umsystem_edu/Documents/Panpep reusability report/data/Supplementary_fig_5/Unitylmmun_independent_Data.csv?d=w311745a2b1704f97bd85f46fef405300&csf=1&web=1&e=VbvbSJ)

Supplementary fig 5 [Unifylmmun triple Dataset](https://mailmissouri-my.sharepoint.com/:x:/r/personal/hefe_umsystem_edu/Documents/Panpep reusability report/data/Supplementary_fig_5/Unifylmmun_triple_Data.csv?d=wa0b721b88b37437ea879070454c6a9ef&csf=1&web=1&e=LTYHsz)

---

## UniPMT

### Overview

UniPMT is a unified pre-trained model for TCR-epitope binding prediction.

### Usage

Follow the instructions in the official UniPMT [GitHub](https://github.com/ethanmock/UniPMT) repository directly.

The corresponding directory in this project is `./inference/UniPMT/`.

Supplementary fig 5 [UniPMT Test Data]([UniPMT_data.csv](https://mailmissouri-my.sharepoint.com/:x:/r/personal/hefe_umsystem_edu/Documents/Panpep reusability report/data/Supplementary_fig_5/UniPMT_data.csv?d=wc2b6d66d50d14b1b9f18bc5a0cccf9b5&csf=1&web=1&e=D11YxI))

---

## Metrics Calculation

For the full evaluation pipeline applicable to all methods above, see [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md).
