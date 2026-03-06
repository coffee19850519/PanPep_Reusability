# Random Forest Training Manual

This document describes the training script in `train/Random_Forest/`, with emphasis on:

- the training entry point,
- the input data format,
- the fixed feature-engineering settings,
- and the external values you need to fill in yourself.

## Overview

The Random Forest baseline is implemented in:

- `train/Random_Forest/train_cuml_rf_simple.py`

This script performs a straightforward pipeline:

1. read the training CSV,
2. load the Atchley factor dictionary,
3. encode each TCR-peptide pair into a fixed-length feature vector,
4. move the feature matrix into `cuDF`,
5. train a GPU `RandomForestClassifier` from cuML,
6. save the trained model.

## Data Requirements

The input CSV must contain the following columns.

| Column | Description |
| --- | --- |
| `tcr` | TCR sequence |
| `peptide` | Peptide sequence |
| `label` | Binary label, typically `1` for binding and `0` for non-binding |

The current script uses the entire CSV for training. It does not split data into train/validation/test internally.

## External Settings to Fill In

The following values depend on your own environment or datasets and are intentionally left blank.

### Data Input

| Parameter | Meaning | Link |
| --- | --- | --- |
| `data_csv` | Training CSV path | `[link](https://mailmissouri-my.sharepoint.com/:x:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/rf_train_data/fold_7_train_all_train_data.csv?d=w21fd0976da9d44aa967f7a3afd11124f&csf=1&web=1&e=wxACKc)` |

### Runtime and Output Settings

| Parameter | Meaning | Placeholder |
| --- | --- | --- |
| `output_model` | Output model file path | `<fill in>` |
| `aa_dict_path` | Atchley factor dictionary path | `<fill in>` |
| `cuda_env` | CUDA / RAPIDS / cuML environment description | `<fill in>` |

## Fixed Feature-Engineering Parameters

These settings are currently hard-coded in the script.

### Sequence Encoding Lengths

| Parameter | Fixed Value | Description |
| --- | --- | --- |
| `peptide_encode_dim` | `15` | Maximum peptide length used for encoding |
| `tcr_encode_dim` | `25` | Maximum TCR length used for encoding |
| `atchley_dim` | `5` | Number of Atchley factors per amino acid |

### Positional Encoding

The script adds sinusoidal positional encoding on top of the Atchley representation:

- total position table length: `40`
- position vector dimension: `5`

### Final Feature Size

Each sample is encoded as follows:

1. peptide -> `15 × 5`
2. TCR -> `25 × 5`
3. concatenate -> `40 × 5`
4. flatten -> `200` features

So the final feature matrix passed to Random Forest has shape:

- `X.shape = [number_of_samples, 200]`

## Current Training Parameters

### Parameters Explicitly Set in Code

| Parameter | Current Value | Description |
| --- | --- | --- |
| model class | `cuml.ensemble.RandomForestClassifier` | GPU Random Forest implementation |
| label dtype | `int32` | Labels are cast to `np.int32` |
| training data usage | full input CSV | No internal validation split |


## Encoding Rules

### Truncation and Padding

- sequences longer than the target length are truncated,
- sequences shorter than the target length are padded with zero vectors.

### Unknown Amino Acids

- any amino acid not found in the Atchley dictionary is replaced with a 5-dimensional zero vector.

### Labels

- the `label` column is read directly from the CSV,
- the script prints the number of positive and negative samples,
- but it does not rebalance the classes or validate label quality.

## Training Output

The script saves one `joblib` file containing a dictionary with the following keys.

| Key | Description |
| --- | --- |
| `model` | Trained cuML Random Forest model |
| `aa_dict_path` | Atchley dictionary path used during training |

The current default output filename is:

- `cuml_rf_model.pkl`

In practice, it is better to replace it with a more specific name that includes the data version, date, or hyperparameter setting.

## General Usage

### Run the Current Script

```bash
python train_cuml_rf_simple.py \
  --data "" \
  --aa_dict "" \
  --output ""
```
