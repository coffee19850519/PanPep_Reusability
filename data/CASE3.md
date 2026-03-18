# CASE 3: Training Reproducibility with TCRβ Extension

## Overview

This case evaluates PanPep's **training-level reproducibility** by re-training the model using the TCRβ chain dataset and verifying that the reproduced model achieves comparable performance to the originally published results.

## Data Requirements
- **10-fold Split Training Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgDljGvsUg7BQoCfyv6jj0ZlAcmJPmOsdn37KLyJyKzLdWQ?e=tc7ABq)
- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=KVoaZd)
- **Negative TCR Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh)
- **Pre-trained Model Weights**: Download from [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=UYWyXZ)
- **Encoding Files**:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=84cga4)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=CgX01O)

---

## Training

### Scripts

Located in `./train/PanPep_Reproduction_and_Hyperparameter_Sweeps/`:

| File | Description |
|------|-------------|
| [`meta_distillation_training.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/meta_distillation_training.py) | Main training entry point |
| [`Memory_meta.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Memory_meta.py) | Memory-augmented meta-learning module |
| [`learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/learner.py) | Inner-loop learner |
| [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) | All hyperparameter configuration |

### Usage

Training is fully config-driven — edit [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) then run:

```bash
cd train/PanPep_Reproduction_and_Hyperparameter_Sweeps
python meta_distillation_training.py
```

## Inference

After training, use the re-trained checkpoint with any of the four inference modes. Scripts are in `./inference/PanPep_Weight_Inference/`.

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
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
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
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz \
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
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
```

### Few-shot

```bash
python inference_few_shot.py \
    --gpu 0 \
    --mode single \
    --distillation 3 \
    --support 2 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/few \
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
```

For full parameter descriptions, see [CASE 1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE1.md).

---

## Metrics Calculation

For sample extraction, use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) instead of [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py). See [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md) for the full pipeline.


## Sampled Data used in Reusability Report

> The sampled data used for adaptation in both majority and few-shot settings can be downloaded from the following [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgCqmGWuw2JjQ6ZBBdbTIpGCAQwyaVNy5NzcXySe4T7lmKY?e=JDueOv).

> The classification results under Background Drawing in the Reusability Report were generated using the negative sampling data available at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig4.zip?csf=1&web=1&e=zm1RCo).
