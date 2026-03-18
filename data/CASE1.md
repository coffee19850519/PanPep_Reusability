# CASE 1: Inference Reproducibility with Original Dataset

## Overview

This case evaluates PanPep's inference-level reproducibility using the **data**  consisting of updated binders for the peptides from the published study.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eoxmw-A2SktDnWKYFDg42TkBmrZP7wOBg1g95kY3nQYPYg?e=vKrGpt)
- **Pre-trained Model Weights**: Download from the [PanPep's repository](https://github.com/bm2-lab/PanPep/blob/main/Requirements/model.pt)
- **Negative TCR Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh)
- **Encoding Files**: 
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

---
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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpu` | GPU device IDs (comma-separated) | `0,1` |
| `--distillation` | Number of distillation steps | `3` |
| `--batch_size` | Upper limit for batch size | `10000` |
| `--test_data` | Path to test data CSV | `./data/test_data.csv` |
| `--negative_data` | Path to negative TCR data | `./data/Control_dataset.txt` |
| `--model_path` | Path to model checkpoint directory | `./Requirements` |
| `--result_dir` | Directory for output Parquet files | `result/majority_reproduction11111` |
| `--peptide_encoding` | Path to peptide encoding `.npz` file | `./peptide_b.npz` |
| `--tcr_encoding` | Path to TCR encoding `.npz` file | `./tcr_b.npz` |

---

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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpu` | GPU device IDs (comma-separated) | `0` |
| `--distillation` | Number of distillation steps | `50` |
| `--batch_size` | Upper limit for batch size | `10000` |
| `--test_data` | Path to test data CSV | `./data/test_data.csv` |
| `--negative_data` | Path to negative TCR data | `./data/Control_dataset.txt` |
| `--model_path` | Path to model checkpoint directory | `./Requirements` |
| `--result_dir` | Directory for output Parquet files | `result/0000` |
| `--peptide_encoding` | Path to peptide encoding `.npz` file | `./peptide_b.npz` |
| `--tcr_encoding` | Path to TCR encoding `.npz` file | `./tcr_b.npz` |
| `--model` | Model architecture | `attention5_conv3_large` |

---

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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpu` | GPU device IDs (comma-separated) | `0` |
| `--distillation` | Number of distillation steps | `3` |
| `--batch_size` | Upper limit for batch size | `10000` |
| `--support_rate` | Fraction of positive TCRs used as support set | `0.8` |
| `--test_data` | Path to test data CSV | `./data/test_data.csv` |
| `--negative_data` | Path to negative TCR data | `./data/Control_dataset.txt` |
| `--model_path` | Path to model checkpoint directory | `./Requirements` |
| `--result_dir` | Directory for output Parquet files | `result11/majority` |
| `--support_dir` | Directory for pre-saved support data (generated if not provided) | `./support/majority` |
| `--peptide_encoding` | Path to peptide encoding `.npz` file | `./peptide_b.npz` |
| `--tcr_encoding` | Path to TCR encoding `.npz` file | `./tcr_b.npz` |

---

### Few-shot

```bash
python inference_few_shot.py \
    --gpu 0 \
    --mode single \
    --distillation 3 \
    --batch_size 10000 \
    --support 2 \
    --test_data ./data/test_data.csv \
    --negative_data ./data/Control_dataset.txt \
    --model_path ./Requirements \
    --result_dir result/few \
    --peptide_encoding ./peptide_b.npz \
    --tcr_encoding ./tcr_b.npz
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpu` | GPU device IDs (comma-separated) | `0` |
| `--mode` | Inference mode: `single` or `mixed` | `single` |
| `--distillation` | Number of distillation steps | `3` |
| `--batch_size` | Upper limit for batch size | `10000` |
| `--support` | K-shot value (number of support samples) | `2` |
| `--test_data` | Path to test data CSV | `./data/test_data.csv` |
| `--negative_data` | Path to negative TCR data (query negatives) | `./data/Control_dataset.txt` |
| `--negative_data_background` | [mixed only] Background negative library for support set | `./data/Control_dataset.txt` |
| `--negative_data_reshuffling` | [mixed only] Reshuffling negative library for support set | `./data/reshuffling.txt` |
| `--model_path` | Path to model checkpoint directory | `./Requirements` |
| `--result_dir` | Directory for output Parquet files | `result_alternating/few/alternating_s4q6` |
| `--support_dir` | Directory for pre-saved k-shot CSV files (generated if not provided) | `None` |
| `--peptide_encoding` | Path to peptide encoding `.npz` file | `./peptide_b.npz` |
| `--tcr_encoding` | Path to TCR encoding `.npz` file | `./tcr_b.npz` |
| `--update_step_test` | Inner-loop finetuning steps at test time | `3` |

---

## Metrics Calculation

Use [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for sample extraction. See [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md) for the full pipeline.


## Sampled Data used in Reusability Report

> The sampled data used for adaptation in both majority and few-shot settings can be downloaded from the following [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgBQSWL8fo-gS55oFNhTFIZVAZ-VAWHWYzso0a5cvhrkB6M?e=U7B5pp).

> The classification results under Background Drawing in the Reusability Report were generated using the negative sampling data available at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig2.zip?csf=1&web=1&e=jptqb0).
