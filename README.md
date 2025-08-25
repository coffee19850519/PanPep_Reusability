# Reusability Report: Meta-Learning for Antigen-Specific T-Cell Receptor Binder Identification 

This epository contains the code associated with our reusability study upon the research "Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition", *Nature Machine Intelligence*.

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

## Overview
Our evaluations first examined PanPep’s inference- and training-level reproducibility using both the original dataset and a newly curated independent dataset. We then assessed its extendibility to peptide-TCRα and peptide-TCRαβ binding recognition, applying the same source code to these new task datasets.
![Fig1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/img/Fig1.jpg)

## Installation
Please refer to PanPep's installation [guide](https://github.com/bm2-lab/PanPep/tree/main?tab=readme-ov-file#requirements).

## Get Started

`meta_distillation_training.py` : Training for few-shot and zero-shot. 

`test_5fold.py` : Using few-shot and zero-shot model test few-shot data. 

`test_zero-shot.py` : Using zero-shot model test zero-shot data. 

`result_metrics.py` : Calculate the RUC-AUC and PR-AUC (Including few-shot results, zero-shot model test few-shot data result, zero-shot result).

`General\general.py`: General training.

`Similarity\get_similarity.py`: Analyze the similarity across peptides and tcrS.

`utils.py`: Common functions and related configurations.

`model_PCA.py`: Do principal component analysis for the model obtained by distillation module.


## Data Download
The data used in this study is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.16943691).

## Original Study

Yicheng Gao, Yuli Gao, Qi Liu et al. Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition, Nature Machine Intelligence, 2023.

## Original Source Code

[https://github.com/bm2-lab/PanPep](https://github.com/bm2-lab/PanPep)

## Contact

For any questions or issues, Please contact Fei He via [hefe@missouri.edu](hefe@missouri.edu)
