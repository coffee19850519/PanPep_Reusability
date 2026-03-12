# Reusability Report: Meta-Learning for Antigen-Specific T-Cell Receptor Binder Identification 

This epository contains the code associated with our reusability study upon the research "Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition"

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18861905.svg)](https://doi.org/10.5281/zenodo.18861905)


## Overview
Our evaluations first examined PanPep’s inference- and training-level reproducibility using both the original dataset and a newly curated independent dataset. We then assessed its extendibility to peptide-TCRα and peptide-TCRαβ binding recognition, applying the same source code to these new task datasets.
![Fig1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/img/Fig1.jpg)

## Documentation

- The following documents provide detailed instructions for different parts of the project:

  \- [PanPep Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_train.md): instructions for training PanPep.
  \- [Random Forest Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/Random_Forest_train.md): instructions for training the Random Forest baseline.
  \- [Data Directory](https://github.com/coffee19850519/PanPep_Reusability/tree/main/data): datasets and related files required for the experiments.
  \- [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md): instructions for running baseline methods other than PanPep.

  \- [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md): instructions for computing the evaluation metrics.

## Installation
Please refer to PanPep's installation [guide](https://github.com/bm2-lab/PanPep/tree/main?tab=readme-ov-file#requirements).


## Quick Start

## CASE 1: Inference Reproducibility with Original Dataset

This case corresponds to the experiments shown in Fig. 2 and Extended Data Fig. 1.

### PanPep

> PanPep: [CASE1.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE1.md)

### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
>
> To reproduce the classification results, please first run inference and then use the classification samples provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig2.zip?csf=1&web=1&e=jptqb0).

## CASE 2: Inference Reproducibility with Independent 

This case corresponds to the experiments shown in Fig. 3 and Extended Data Fig. 4.

## PanPep

> PanPep:[CASE2.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE2.md)

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| UniPMT \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
>
> To reproduce the classification results, please first run inference and then use the classification samples provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig3.zip?csf=1&web=1&e=wYVPYW).
>
> To reproduce the unseen  classification results, please first run inference and then use the classification samples provided at the following  [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig3_unseen.zip?csf=1&web=1&e=be1H8S).

## CASE 3: Training Reproducibility with TCRβ Extension

This case corresponds to the experiments shown in Fig. 4 and Extended Data Fig. 5.

## PanPep

> PanPep:[CASE3.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE3.md)

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
>
> To reproduce the classification results, please first run inference and then use the classification samples provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig4.zip?csf=1&web=1&e=zm1RCo).

## CASE 4: Training Reproducibility with TCRα Extension

This case corresponds to the experiments shown in Fig. 5 and Extended Data Fig. 6.

## PanPep

> PanPep:[CASE4.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE4.md)

### Baseline Methods

> DLpTCR:  [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:see [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
>
> To reproduce the classification results, please first run inference and then use the classification samples provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig5.zip?csf=1&web=1&e=J8eYuP).

## CASE 5: Training Reproducibility with TCRαβ Extension

This case corresponds to the experiments shown in Fig. 6 and Extended Data Fig. 7.

## PanPep

> PanPep:[CASE5.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE5.md)

### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Metrics Calculation:[Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
>
> To reproduce the classification results, please first run inference and then use the classification samples provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/r/personal/hefe_umsystem_edu/Documents/Panpep%20reusability%20report/data/classification/fig6.zip?csf=1&web=1&e=gT9jcX).

---

## PanPep Hyperparameter Reproduction

This case corresponds to the experiments shown in Supplementary Figure 8.

Full documentation: [PanPep Hyperparameter Reproduction](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/PanPep_Hyperparameter.md)

## TODO

- [x] Update extracted classification dataset in the paper
- [x] Update training pipeline description and related data
- [ ] Update Jupyter notebooks

---

## Data and Model Weights
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18913335.svg)](https://doi.org/10.5281/zenodo.18913335)

## Original Study

Yicheng Gao, Yuli Gao, Qi Liu et al. Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition, Nature Machine Intelligence, 2023.

## Original Source Code

[https://github.com/bm2-lab/PanPep](https://github.com/bm2-lab/PanPep)

## Contact

For any questions or issues, Please contact Fei He via [hefe@missouri.edu](hefe@missouri.edu)
