# PanPep Hyperparameter Reproduction

## Overview

This case focuses on the training-level reproducibility of PanPep under the hyperparameter setting used in the original study. The model is re-trained on the TCRβ chain dataset, and the reproduced performance is compared against the published results.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=KVoaZd)
- **reshuffling.txt**: Available on [here](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/reshuffling.txt)
- **Control_dataset.txt**: Available on [here](https://mailmissouri-my.sharepoint.com/:t:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/PanPep-Provided%20Dataset/Control%20dataset_PanPep.txt?csf=1&web=1&e=M4YPje)
- Pre-trained Checkpoints: Download from [here](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/checkpoint/hyperparameter_PanPep?csf=1&web=1&e=n9VZ8t)
- **Encoding Files**:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=84cga4)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=CgX01O)

---

## Training

### Scripts

Located in `./train/PanPep_Reproduction_and_Hyperparameter_Sweeps/`:

| File                                                         | Description                           |
| ------------------------------------------------------------ | ------------------------------------- |
| [`meta_distillation_training.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/meta_distillation_training.py) | Main training entry point             |
| [`Memory_meta.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Memory_meta.py) | Memory-augmented meta-learning module |
| [`learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/learner.py) | Inner-loop learner                    |
| [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) | All hyperparameter configuration      |

### Usage

Training is fully config-driven — edit [`Configs/TrainingConfig.yaml`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_Reproduction_and_Hyperparameter_Sweeps/Configs/TrainingConfig.yaml) then run:

```bash
cd train/PanPep_Reproduction_and_Hyperparameter_Sweeps
python meta_distillation_training.py
```

## Inference

After training, use the re-trained checkpoint with any of the four inference modes. Scripts are in `./inference/PanPep_Weight_Inference/`.

| Mode     | Script                                                       |
| -------- | ------------------------------------------------------------ |
| Few-shot | [`inference_few_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_few_shot.py) |



### background-draw

```bash
python inference_few_shot.py \
    --gpu 0 \
    --mode single \
    --distillation 3 \
    --support 1 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/few \
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
```

### reshuffling

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

### mixed

```bash
python inference_few_shot.py \
    --gpu 0 \
    --mode mixed \
    --distillation 3 \
    --support 4 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --negative_data_background ./data/Control_dataset.txt
    --negative_data_reshuffling ./data/reshuffling.txt
    --model_path ./Requirements \
    --result_dir result/few \
    --peptide_encoding ./peptide_b.npz 
    --tcr_encoding ./tcr_b.npz
```

For full parameter descriptions, see [CASE 1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE1.md).

---

## Metrics Calculation

For sample extraction, See [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md) for the full pipeline.