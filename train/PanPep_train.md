# PanPep Training Manual

This document describes how the training pipeline in `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/` works, with emphasis on:

- the training entry points,
- the default training hyperparameters,
- the expected data formats,
- and the external settings you need to fill in yourself.
## Overview

The PanPep training code is a two-stage pipeline:

1. **Meta-learning stage**
2. **Disentanglement distillation stage**

The main files are:

- `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/train/train.py`
- `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/meta_distillation_training.py`
- `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/distillation.py`
- `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml`

`train/train.py` is the example launcher. It currently hard-codes one training file and one output directory, so it should be treated as a template rather than a reusable production entry point.

## Training Pipeline

### Stage 1: Meta-learning

Each peptide is treated as one task.

For each peptide task, the dataset builder creates:

- a **support set** with positive and negative TCRs,
- a **query set** with positive and negative TCRs.

With the current default configuration:

- `support = 2`
- `query = 3`

So each peptide task contains:

- **Support set**: `2` positive + `2` negative samples
- **Query set**: `3` positive + `3` negative samples

Only peptides with at least `support + query` positive TCRs are kept for training. Under the default setting, that means at least `5` positive TCRs per peptide.If you want to use a different configuration, you can change the values of the support set and query set.

### Stage 2: Disentanglement Distillation

After the meta-learning stage finishes, the training code saves intermediate artifacts and uses them to train the memory module.

This stage consumes:

- `models.pkl`
- `prev_loss.pkl`
- `prev_data.pkl`
- `model.pt`

and produces the distilled memory artifacts used later for inference:

- `Content_memory.pkl`
- `Query.pkl`

If you already have the stage-1 artifacts, you can run the distillation stage separately with `distillation.py`.

## Data Requirements

### Main Training CSV

The main training CSV must contain at least these columns:

| Column | Description |
| --- | --- |
| `peptide` | Peptide sequence |
| `binding_TCR` | TCR sequence that binds to the peptide |

The code groups rows by `peptide`, and the `binding_TCR` values under the same peptide are treated as positive samples for that task.

### Negative Sample Inputs

The code supports several training strategies.

| Strategy | Negative-data input | Notes |
| --- | --- | --- |
| `mode2` | TXT file, one TCR per line | Current default in `train/train.py` |
| `mode1` | CSV with at least `peptide`, `binding_TCR` | Peptide-specific negative samples |
| `ranking` | CSV with at least `binding_TCR` | Ranking-oriented negative source |
| `alternating(mixed)` | Two TXT files, one TCR per line | Alternates libraries by odd/even epoch |

### Atchley Factor Dictionary

The sequence encoder uses a 5-dimensional Atchley factor representation for amino acids.

The repository already includes a default dictionary file:

- `train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Requirements/dic_Atchley_factors.pkl`

You can keep that file or replace it with your own path.

## External Settings to Fill In

The following values depend on your own environment or external data and are intentionally left blank.

### Required Data Input

| Parameter | Meaning | Link |
| --- | --- | --- |
| `train_csv` | Main training CSV path | `[link](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/train_data?csf=1&web=1&e=DIGlD5)` |

Note: If you want to reproduce CASE 2, use data other than fold_7_train_Hyperparameter.csv. If you want to reproduce the hyperparameter experiments, use fold_7_train_Hyperparameter.csv.
### Required Runtime Settings

| Parameter | Meaning | Placeholder |
| --- | --- | --- |
| `save_path` | Output directory for checkpoints and logs | `<fill in>` |
| `device` | Training device, e.g. `cuda` or `cpu` | `<fill in>` |

### Strategy-Specific Data Links

#### If `strategy = mode2`

| Parameter | Meaning | Link |
| --- | --- | --- |
| `negative_txt` | Background TCR library or reshuffling_txt  | `[background-draw](https://mailmissouri-my.sharepoint.com/:t:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/PanPep-Provided%20Dataset/Control%20dataset_PanPep.txt?csf=1&web=1&e=M4YPje)  [reshuffling](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/reshuffling.txt) ` |

#### If `strategy = alternating`

| Parameter | Meaning | Link |
| --- | --- | --- |
| `background_draw_txt` | Library used in odd epochs | `[background-draw](https://mailmissouri-my.sharepoint.com/:t:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/PanPep-Provided%20Dataset/Control%20dataset_PanPep.txt?csf=1&web=1&e=M4YPje)` |
| `reshuffling_txt` | Library used in even epochs | `[reshuffling](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/reshuffling.txt)` |

### Optional Settings

| Parameter | Meaning | Placeholder |
| --- | --- | --- |
| `aa_dict_path` | Atchley factor dictionary path | `<fill in>` |
| `seed` | Random seed | `<fill in>` |
| `log_file` | Log file path | `<fill in>` |
| `config_path` | Path to `TrainingConfig.yaml` | `<fill in>` |

## Default Hyperparameters

The values below come from the current code in `Configs/TrainingConfig.yaml` and `train/train.py`.

### Sampling Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_size` | `4096` | Task batch size for `DataLoader` |
| `sample_shuffle` | `True` | Shuffle tasks during training |
| `support` | `2` | Positive/negative samples per class in support set |
| `query` | `3` | Positive/negative samples per class in query set |
| `strategy` | `mode2` | Default negative-sampling strategy |

### Meta-learning Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `meta_lr` | `0.001` | Outer-loop learning rate |
| `inner_loop_lr` | `0.01` | Inner-loop learning rate |
| `inner_update_step` | `3` | Number of inner-loop updates |
| `inner_fine_tuning` | `3` | Fine-tuning steps used in testing/few-shot style adaptation |
| `num_of_index` | `3` | Number of peptide indices `C` |
| `len_of_index` | `3` | Index length `R` |
| `len_of_embedding` | `75` | Peptide embedding length `L` |
| `regular_coefficient` | `0` | Regularization coefficient |
| `epoch` | `500` | Number of meta-learning epochs |

### Distillation Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `distillation_epoch` | `800` | Number of distillation epochs |

### Runtime Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `device` | `cuda` | Default training device |
| `seed` | `44` | Seed hard-coded in `train/train.py` |

## Feature Encoding

The current code uses a fixed sequence encoding scheme.

### Sequence Lengths

| Component | Fixed Length |
| --- | --- |
| Peptide | `15` |
| TCR | `25` |
| Atchley factor dimension | `5` |

### Encoding Rules

- sequences longer than the target length are truncated,
- sequences shorter than the target length are zero-padded,
- unknown amino acids are mapped to zero vectors,
- sinusoidal positional encoding is added after Atchley encoding.

Each sample is represented as:

- peptide embedding: `15 × 5`
- TCR embedding: `25 × 5`
- concatenated sample: `40 × 5`

The peptide-only task embedding used by the memory module is flattened to length `75`.

## Default Model Structure

The default `Model_config` is:

1. `self_attention`
2. `linear(5 -> 5)`
3. `relu`
4. `conv2d(16, 1, kernel=2)`
5. `relu`
6. `batch norm`
7. `max_pool2d(2, 2)`
8. `flatten`
9. `linear(608 -> 2)`

The final output is a 2-class logit vector for peptide-TCR binding classification.

## Sampling Strategy Differences

### `mode2`

- samples negatives from one large background TXT library,
- removes duplicates,
- excludes positive TCRs for the current peptide,
- excludes support negatives from the query negative set.

### `alternating(mixed)`

- odd epochs use `background_draw`,
- even epochs use `reshuffling`.

## Training Outputs

The stage-1 and stage-2 pipeline writes the following files under `save_path`.

| File | Description |
| --- | --- |
| `training.log` | Training log |
| `model.pt` | Meta-learner state dict |
| `models.pkl` | Peptide-specific learners |
| `prev_loss.pkl` | Stored losses for distillation |
| `prev_data.pkl` | Stored task data for distillation |
| `memory_module_init.pt` | Memory module state before distillation |
| `Content_memory.pkl` | Final distilled content memory |
| `Query.pkl` | Final distilled query/read-head parameters |

If `save_train_data=True` and `hook=get_train_data` are used, the code can also save:

- `*_all_train_data.csv`
- `*_distillation_train_data.csv`

## Fill-in Checklist

| Item | Placeholder |
| --- | --- |
| `strategy` | `<fill in>` |
| `train_csv` | `[link](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/train_data?csf=1&web=1&e=DIGlD5)` |
| `save_path` | `<fill in>` |
| `device` | `<fill in>` |
| `seed` | `<fill in>` |
| `negative_txt` / `negative_csv` | `[background-draw](https://mailmissouri-my.sharepoint.com/:t:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/PanPep-Provided%20Dataset/Control%20dataset_PanPep.txt?csf=1&web=1&e=M4YPje)  [reshuffling](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/reshuffling.txt) ` |
| `background_draw_txt` | `[background-draw](https://mailmissouri-my.sharepoint.com/:t:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/PanPep-Provided%20Dataset/Control%20dataset_PanPep.txt?csf=1&web=1&e=M4YPje) ` |
| `reshuffling_txt` | `[reshuffling](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/reshuffling.txt)` |
| `aa_dict_path` | `<fill in>` |

## General Usage

### Call `train_main` directly

```python
from meta_distillation_training import train_main
from utils import MLogger, get_train_data

logger = MLogger("")

train_main(
    train_data=train_data_path,
    save_path=save_path,
    logger_file=logger,
    task_num=unique_peptides_count,
    hook=get_train_data,
    save_train_data=True,
    strategy='mode2'
)
```