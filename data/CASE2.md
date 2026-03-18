# CASE 2: Inference Reproducibility with Independent Dataset

## Overview

This case evaluates PanPep's inference-level reproducibility using a **newly curated independent dataset**, assessing generalization beyond the original training distribution. The scripts and workflow are identical to CASE 1 — only the input data differs.

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=moZ7cg)
- **Negative TCR Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh)
- **Pre-trained Model Weights**: Download from the [original PanPep repository](https://github.com/bm2-lab/PanPep/tree/main/Requirements/model.pt)
- **Encoding Files**:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=0QEsas)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=OIM7Jc)

## Inference Scripts

Located in `./inference/PanPep_Weight_Inference/`. Four inference modes are available:

| Mode | Script |
|------|--------|
| Meta-learner | [`inference_meta_learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_meta_learner.py) |
| Zero-shot | [`inference_zero_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_zero_shot.py) |
| Majority | [`inference_majority.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_majority.py) |
| Few-shot | [`inference_few_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_few_shot.py) |

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
    --batch_size 10000 \
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

Use [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for sample extraction. See [Metrics_Calculation.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md) for the full pipeline.


## Unseen Setting

## Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgCKKQwRjGWRRaKiTsPe0x3YAUEmEOb-orOeMjLluv2N8ks?e=tEKna7)
- **Pre-trained Checkpoints**: Download from the [original PanPep repository](https://github.com/bm2-lab/PanPep/tree/main/Requirements/model.pt)
- **Encoding Files**:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=0QEsas)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=OIM7Jc)

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

## Sampled Data used in Reusability Report

> The sampled data used for adaptation in both majority and few-shot settings can be downloaded from the following [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgDhSKn_8b85Sqk-PWoWgmaeATBjASKjN_34jDyz6-n2aiY?e=YCLBJg).

> The classification results under Background Drawing in the Reusability Report were generated using the negative sampling data available at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig3.zip?csf=1&web=1&e=wYVPYW).

> The unseen classification results under Background Drawing in the Reusability Report were generated using the negative sampling data available at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig3_unseen.zip?csf=1&web=1&e=be1H8S).

