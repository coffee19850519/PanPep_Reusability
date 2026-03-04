# CASE 5: Training Reproducibility with TCRαβ Extension

## Overview

This case extends PanPep to **paired TCRαβ** (dual-chain) binding recognition. The inference pipeline involves a two-step process: data formatting followed by model inference on both chains.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei2Ef8zmUGBKqh8H0Vhrl9QBOGfwUjV7Oead3UVlV7kVKw?e=NC6JMl)
- **Pre-trained Checkpoints**:
  - [Alpha chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=oXLckH)
  - [Beta chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=86CUNb)
- **Encoding Files** (local):
  - `./PanPep_Weight_Inference/tcr_ab.npz`
  - `./PanPep_Weight_Inference/peptide_ab.npz`

## Inference Scripts

Located in `./inference/PanPep_Weight_Inference/`. The TCRαβ pipeline is a two-step process:

### Step 1 — Data Preparation

Use [`samplingab.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/samplingab.py) to format input data:

```bash
python samplingab.py \
    --positive_data <path_to_positive_data> \
    --tcr_pool <path_to_tcr_pool> \
    --mode <majority|few_shot|zero_shot> \
    --output_dir <output_directory>
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--positive_data` | Positive sample data file | `data/tcrab_majority.csv` |
| `--tcr_pool` | Negative sample pool | `data/pooling_tcrab.csv` |
| `--mode` | Sampling mode: `majority`, `few_shot`, or `zero_shot` | — |
| `--majority_ratio` | Positive ratio for majority mode | `0.8` |
| `--few_shot` | Number of samples for few-shot mode | `2` |
| `--output_dir` | Output directory | — |

### Step 2 — Model Inference

Use [`inferece_ab.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inferece_ab.py) to run prediction:

```bash
python inferece_ab.py \
    --learning_setting <few-shot|zero-shot|Meta-learner|majority> \
    --input_dir <input_directory> \
    --output_dir <output_directory> \
    --fold <fold_number> \
    --peptide_encoding <path_to_peptide_encoding> \
    --tcr_encoding <path_to_tcr_encoding> \
    [--meta_type <zero-shot|majority|few-shot>]   # required when --learning_setting Meta-learner
```

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `--learning_setting` | Learning mode: `few-shot`, `zero-shot`, `Meta-learner`, `majority` |
| `--meta_type` | Meta-learner sub-type: `zero-shot`, `majority`, or `few-shot` (required when `--learning_setting Meta-learner`) |
| `--input_dir` | Directory containing formatted CSV files from Step 1 |
| `--output_dir` | Directory for prediction results |
| `--fold` | Fold number to use (1–10) |
| `--peptide_encoding` | Path to peptide encoding file (default: `./peptide_ab.npz`) |
| `--tcr_encoding` | Path to TCR encoding file (default: `./tcr_ab.npz`) |

**Score weighting by mode**:

| Mode | α weight | β weight |
|------|----------|----------|
| `zero-shot` / `Meta-learner --meta_type zero-shot` | 0.86 | 0.14 |
| `majority` / `Meta-learner --meta_type majority` | 0.59 | 0.41 |
| `few-shot` / `Meta-learner --meta_type few-shot` | 0.00 | 1.00 |

### Example Workflow

```bash
# Step 1: Format data for few-shot learning
python samplingab.py \
    --positive_data data/tcrab_majority.csv \
    --tcr_pool data/pooling_tcrab.csv \
    --mode few_shot \
    --few_shot 5 \
    --output_dir formatted_data/

# Step 2: Run inference (few-shot)
python inferece_ab.py \
    --learning_setting few-shot \
    --input_dir formatted_data/few_shot/ \
    --output_dir results/ \
    --fold 1 \
    --peptide_encoding ./peptide_ab.npz \
    --tcr_encoding ./tcr_ab.npz

# Step 2 (alternative): Meta-learner with majority-style weights
python inferece_ab.py \
    --learning_setting Meta-learner \
    --meta_type majority \
    --input_dir formatted_data/majority/ \
    --output_dir results/ \
    --fold 1 \
    --peptide_encoding ./peptide_ab.npz \
    --tcr_encoding ./tcr_ab.npz
```

## Metrics Calculation

For sample extraction, use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) instead of [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py). See [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md) for the full pipeline.
