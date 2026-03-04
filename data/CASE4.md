# CASE 4: Training Reproducibility with TCRα Extension

## Overview

This case extends PanPep to **TCRα chain** binding recognition by applying the same training pipeline with a TCRα-specific dataset. It demonstrates the framework's extendibility to a new chain type without modifying the source code.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EixVbgsKPD5BuDQP566AvR8BiPXqG85FkCCshSTexHLQgw?e=Jry9LY)
- **Pre-trained Checkpoints**: Download from [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=Iuojsc)
- **Encoding Files**:
  - [tcr_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EfbaOgcR0TVCjZMWigUshYABOa0cqxpJDZaiZBWsm0wMuw?e=9YHM4s)
  - [peptide_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EW2_VBo_t7RAs-ysGbkZMacBX_PTniASmuROxwpVjmt_pw?e=dASilC)

---

## Training

### Scripts

Located in `./train/PanPep_Reproduction_and_Hyperparameter_Sweeps/` — same pipeline as CASE 3:

| File | Description |
|------|-------------|
| [`meta_distillation_training.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/meta_distillation_training.py) | Main training entry point |
| [`Memory_meta.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Memory_meta.py) | Memory-augmented meta-learning module |
| [`learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/learner.py) | Inner-loop learner |
| [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) | All hyperparameter configuration |

### Usage

Edit [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) to point to TCRα data, then run:

```bash
cd train/PanPep_Reproduction_and_Hyperparameter_Sweeps
python meta_distillation_training.py
```
## Inference

After training, use the TCRα checkpoints with `tcr_a.npz` / `peptide_a.npz` encoding files. Scripts are in `./inference/PanPep_Weight_Inference/`.

| Mode | Script |
|------|--------|
| Meta-learner | [`inference_meta_learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_meta_learner.py) |
| Zero-shot | [`inference_zero_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_zero_shot.py) |
| Majority | [`inference_majority.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_majority.py) |
| Few-shot | [`inference_few_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_few_shot.py) |

### Meta-learner

```bash
python inference_meta_learner.py \
    --gpu 0 \
    --distillation 3 \
    --batch_size 10000 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/meta \
    --peptide_encoding ./peptide_a.npz \
    --tcr_encoding ./tcr_a.npz
```

### Zero-shot

```bash
python inference_zero_shot.py \
    --gpu 0 \
    --distillation 50 \
    --batch_size 10000 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/zero \
    --peptide_encoding ./peptide_a.npz \
    --tcr_encoding ./tcr_a.npz \
    --model attention5_conv3_large
```

### Majority

```bash
python inference_majority.py \
    --gpu 0 \
    --distillation 3 \
    --batch_size 10000 \
    --support_rate 0.8 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/majority \
    --support_dir ./support/majority \
    --peptide_encoding ./peptide_a.npz \
    --tcr_encoding ./tcr_a.npz
```

### Few-shot

```bash
python inference_few_shot.py \
    --gpu 0 \
    --mode mixed \
    --distillation 3 \
    --support 4 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --negative_data_background ./data/Control_dataset.txt \
    --negative_data_reshuffling ./data/reshuffling.txt \
    --model_path ./Requirements \
    --result_dir result/few \
    --peptide_encoding ./peptide_a.npz \
    --tcr_encoding ./tcr_a.npz
```

For full parameter descriptions, see [CASE 2](CASE2.md).

---

## Metrics Calculation

For sample extraction, use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) instead of [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py). See [Metrics_Calculation.md](Metrics_Calculation.md) for the full pipeline.
