# Reusability Report: Meta-Learning for Antigen-Specific T-Cell Receptor Binder Identification 

This epository contains the code associated with our reusability study upon the research "Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition"

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19097872.svg)](https://doi.org/10.5281/zenodo.19097872)

## Overview

![Fig1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/img/Fig1.jpg)

### Experiment Reproduction Table

> **Resources for reproducing the reported results:** The table below lists the datasets, negative TCR controls, model weights, and encoding files used for each experimental case.

| Case | Reproduction experiment | Training data | Test data | Negative TCR data | Model weights | Encoding files |
|---|---|---|---|---|---|---|
| [CASE 1](data/CASE1.md) | Inference reproducibility with the original dataset | — | [Test data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eoxmw-A2SktDnWKYFDg42TkBmrZP7wOBg1g95kY3nQYPYg?e=vKrGpt) | [Negative TCR data](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh) | [Pre-trained model weights](https://github.com/bm2-lab/PanPep/blob/main/Requirements/model.pt) | [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=IUYxGG)<br>[peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=rd0CF9) |
| [CASE 2](data/CASE2.md) | Inference reproducibility with an independent dataset | — | [Independent test data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=moZ7cg)<br>[Unseen-setting test data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgCKKQwRjGWRRaKiTsPe0x3YAUEmEOb-orOeMjLluv2N8ks?e=tEKna7) | [Negative TCR data](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh) | [Pre-trained model weights](https://github.com/bm2-lab/PanPep/tree/main/Requirements/model.pt) | [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=0QEsas)<br>[peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=OIM7Jc) |
| [CASE 3](data/CASE3.md) | Training reproducibility with the TCRβ extension | [10-fold split training data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgDljGvsUg7BQoCfyv6jj0ZlAcmJPmOsdn37KLyJyKzLdWQ?e=tc7ABq) | [Test data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=KVoaZd) | [Negative TCR data](https://mailmissouri-my.sharepoint.com/:t:/g/personal/hefe_umsystem_edu/IQA1Nl3AA947RIzY6aDGDnNzASb1hyvWzldypntzonc-0xY?e=CHIVKh) | [Pre-trained model weights](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=UYWyXZ) | [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=84cga4)<br>[peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=CgX01O) |
| [CASE 4](data/CASE4.md) | Training reproducibility with the TCRα extension | [10-fold cross-validation data split](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgArlOc0U8V5Q7Nw8XLcZ9L7AVWRhrCQLlRJV2b2jWAmZJk) | [10-fold cross-validation data split](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgArlOc0U8V5Q7Nw8XLcZ9L7AVWRhrCQLlRJV2b2jWAmZJk) | [Negative TCR data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EixVbgsKPD5BuDQP566AvR8BiPXqG85FkCCshSTexHLQgw?e=Jry9LY) | [Pre-trained checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=Iuojsc) | [tcr_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EfbaOgcR0TVCjZMWigUshYABOa0cqxpJDZaiZBWsm0wMuw?e=9YHM4s)<br>[peptide_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EW2_VBo_t7RAs-ysGbkZMacBX_PTniASmuROxwpVjmt_pw?e=dASilC) |
| [CASE 5](data/CASE5.md) | Training reproducibility with the paired TCRαβ extension | [TCRβ training data (CASE 3)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgDljGvsUg7BQoCfyv6jj0ZlAcmJPmOsdn37KLyJyKzLdWQ?e=tc7ABq)<br>[TCRα training data (CASE 4)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/IgArlOc0U8V5Q7Nw8XLcZ9L7AVWRhrCQLlRJV2b2jWAmZJk) | [Test data](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei2Ef8zmUGBKqh8H0Vhrl9QBOGfwUjV7Oead3UVlV7kVKw?e=NC6JMl) | — | [Alpha chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=oXLckH)<br>[Beta chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=86CUNb) | [tcr_ab.npz](inference/PanPep_Weight_Inference/tcr_ab.npz)<br>[peptide_ab.npz](inference/PanPep_Weight_Inference/peptide_ab.npz) |

## Documentation

- The following documents provide detailed instructions for different parts of the project:

  * [PanPep Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_train.md): instructions for training PanPep.
  * [Random Forest Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/Random_Forest_train.md): instructions for training the Random Forest baseline.
  * [Data Directory](https://github.com/coffee19850519/PanPep_Reusability/tree/main/data): datasets and related files required for the experiments.
  * [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md): instructions for running baseline methods other than PanPep.
  * [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md): instructions for computing the evaluation metrics.

## Installation
Please refer to PanPep's installation [guide](https://github.com/bm2-lab/PanPep/tree/main?tab=readme-ov-file#requirements).


## Quick Start

## CASE 1: Inference Reproducibility with Original Dataset

This case corresponds to the experiments shown in Fig. 2 and Extended Data Fig. 1.

### PanPep

> [CASE1.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE1.md) \| [tutorial](https://github.com/coffee19850519/PanPep_Reusability/blob/main/tutorials/CASE1.ipynb)

### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 2: Inference Reproducibility with Independent 

This case corresponds to the experiments shown in Fig. 3 and Extended Data Fig. 4.

## PanPep

> [CASE2.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE2.md) \| [tutorial](https://github.com/coffee19850519/PanPep_Reusability/blob/main/tutorials/CASE2.ipynb)

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| UniPMT \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 3: Training Reproducibility with TCRβ Extension

This case corresponds to the experiments shown in Fig. 4 and Extended Data Fig. 5.

## PanPep

> [CASE3.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE3.md) \| [tutorial](https://github.com/coffee19850519/PanPep_Reusability/blob/main/tutorials/CASE3.ipynb)

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 4: Training Reproducibility with TCRα Extension

This case corresponds to the experiments shown in Fig. 5 and Extended Data Fig. 6.

## PanPep

> [CASE4.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE4.md) \| [tutorial](https://github.com/coffee19850519/PanPep_Reusability/blob/main/tutorials/CASE4.ipynb)


### Baseline Methods

> DLpTCR:  [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:see [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 5: Training Reproducibility with TCRαβ Extension

This case corresponds to the experiments shown in Fig. 6 and Extended Data Fig. 7.

## PanPep

> [CASE5.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE5.md)

### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

---

## PanPep Hyperparameter Reproduction

This case corresponds to the experiments shown in Supplementary Figure 8.

Full documentation: [PanPep Hyperparameter Reproduction](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/PanPep_Hyperparameter.md)

## TODO

- [x] Update extracted classification dataset in the paper
- [x] Update training pipeline description and related data
- [x] Update tutorials with Jupyter notebooks

---

## Data and Model Weights
The data and model weights in this study are publicly available on [Zenodo](https://doi.org/10.5281/zenodo.19097872).

## Original Study

Yicheng Gao, Yuli Gao, Qi Liu et al. Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition, Nature Machine Intelligence, 2023.

## Original Source Code

[https://github.com/bm2-lab/PanPep](https://github.com/bm2-lab/PanPep)

## Contact

For any questions or issues, Please contact Fei He via [feihe@usf.edu](feihe@usf.edu)
