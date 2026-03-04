# CASE 1: Inference Reproducibility with Original Dataset

## Overview

This case evaluates PanPep's inference-level reproducibility using the **original dataset** from the published study.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eoxmw-A2SktDnWKYFDg42TkBmrZP7wOBg1g95kY3nQYPYg?e=vKrGpt)
- **Pre-trained Checkpoints**: Download from the [PanPep's repository](https://github.com/bm2-lab/PanPep/blob/main/Requirements/model.pt)
- **Encoding Files**: Background database and `.npz` encoding files:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=IUYxGG)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=rd0CF9)

## Inference Scripts

Located in `./inference/PanPep_Weight_Inference/`. Four inference modes are available:

| Mode | Script |
|------|--------|
| Meta-learner | [`inference_meta_learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_meta_learner.py) |
| Zero-shot | [`inference_zero_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_zero_shot.py) |
| Majority | [`inference_majority.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_majority.py) |
| Few-shot | [`inference_few_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_few_shot.py) |

## General Usage

```bash
python inference_meta_learner.py \
    --gpu 0,1,2,3,4,5,6,7 \        # Select GPUs
    --distillation 3 \              # Number of distillation steps
    --upper_limit 100000 \          # Batch size
    --k_shot 0 \                    # 0 for meta-learner mode
    --test_data ./few-shot.csv \    # Positive samples
    --negative_data ./Control_dataset.txt \  # Background database
    --model_path ./Requirements \   # Model path
    --result_dir result/few-meta \  # Output directory
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
```

## Mode-Specific Notes

- **Few-shot** and **Majority** modes include an additional `--kshot_dir` parameter for storing selected fine-tuning samples.
- In **Majority** mode, `--k_shot` represents a ratio rather than an absolute count.

## Metrics Calculation

Use [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for sample extraction. See [Metrics_Calculation.md](Metrics_Calculation.md) for the full pipeline.
