# Reusability Report: Meta-Learning for Antigen-Specific T-Cell Receptor Binder Identification 

This epository contains the code associated with our reusability study upon the research "Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition"

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18861905.svg)](https://doi.org/10.5281/zenodo.18861905)


## Overview
Our evaluations first examined PanPep’s inference- and training-level reproducibility using both the original dataset and a newly curated independent dataset. We then assessed its extendibility to peptide-TCRα and peptide-TCRαβ binding recognition, applying the same source code to these new task datasets.
![Fig1](https://github.com/coffee19850519/PanPep_Reusability/blob/main/img/Fig1.jpg)

## Documentation

- [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)
- [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

## Installation
Please refer to PanPep's installation [guide](https://github.com/bm2-lab/PanPep/tree/main?tab=readme-ov-file#requirements).


## Quick Start

### Training Code

For PanPep training, please see the [PanPep Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/PanPep_train.md). For Random Forest baseline training, please see the [Random Forest Training Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/train/Random_Forest_train.md).

## CASE 1: Inference Reproducibility with Original Dataset

> Full documentation: [CASE1.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE1.md)

### Data Requirements
- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Eoxmw-A2SktDnWKYFDg42TkBmrZP7wOBg1g95kY3nQYPYg?e=vKrGpt)
- **Pre-trained Checkpoints**: Download from the [PanPep's repository](https://github.com/bm2-lab/PanPep/blob/main/Requirements/model.pt)
- **Encoding Files**: Background database and `.npz` encoding files available at:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=IUYxGG)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=rd0CF9)

### Inference Modes

Four inference modes are available for different experimental scenarios:

1. **Meta-learner mode** - [`inference_meta_learner.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_meta_learner.py)
2. **Zero-shot mode** - [`inference_zero_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_zero_shot.py)
3. **Majority mode** - [`inference_majority.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_majority.py)
4. **Few-shot mode** - [`inference_few_shot.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inference_few_shot.py)

### **General Usage**

```bash
python inference_meta_learner.py \
    --gpu 0,1,2,3,4,5,6,7 \    # Select GPUs, multiple GPUs can be used
    --distillation 3 \          # Number of distillation steps, usually set to 3
    --upper_limit 100000 \      # Batch size, few-shot typically uses 10GB VRAM with 100,000 batch size
    --k_shot 0 \               # Number of fine-tuning samples, keep at 0 for meta-learner mode
    --test_data ./few-shot.csv \    # Few-shot data (contains only positive samples)
    --negative_data ./Control_dataset.txt \    # Negative sample data (background database)
    --model_path ./Requirements \    # Model path
    --result_dir result/few-meta \    # Output directory for results
    --peptide_encoding ./peptide_b.npz \    # Peptide encoding file
    --tcr_encoding ./tcr_b.npz \    # TCR encoding file
```

To reproduce the classification results, you can first run inference and then extract the values from the files provided at the following [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQAVTro77BJDT4yulW2IClygAWkZfy5BK5i_vpORulhhc9U?e=vl0DDw).

### **Mode-Specific Differences**

**Few-shot** and **Majority** modes include an additional `--kshot_dir` parameter for storing selected fine-tuning samples. Use this parameter to specify a fixed set of fine-tuning samples.

In **Majority** mode, the `--k_shot` parameter represents a ratio rather than an absolute count of samples.
### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Full documentation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

The evaluation metrics pipeline is located in `./metric calculation/` and supports both classification and ranking metrics.

#### Classification Metrics (AUC)

1. **Sample Extraction**: Use the shuffling scripts to extract samples:
   ```bash
   python "./metric calculation/shuffling_index.py"
   python "./metric calculation/random_index.py"
   python "./metric calculation/get_sample_indices_100.py"
   ```

2. **AUC Calculation**: Compute Area Under the Curve:
   ```bash
   python "./metric calculation/AUC.py" --input_file <prediction_results> --output_dir <results_dir>
   ```

#### Ranking Metrics

1. **Sort Predictions**: First sort the prediction results:
   ```bash
   python "./metric calculation/sort.py" --input_file <prediction_file> --output_file <sorted_file>
   ```

2. **Calculate Ranking Metrics**: Compute various ranking-based metrics:

   - **Top Rank Percentile**:
     
     ```bash
     python "./metric calculation/Top_rank_percentile.py" --sorted_file <sorted_file> --output_dir <results_dir>
     ```
     
   - **BEDROC Score**:
     ```bash
     python "./metric calculation/bedroc.py" --sorted_file <sorted_file> --alpha <alpha_value> --output_dir <results_dir>
     ```
   
   - **Success Rate & Hit Rate**:
     ```bash
     python "./metric calculation/success_rate&hit_rate.py" --sorted_file <sorted_file> --threshold <threshold> --output_dir <results_dir>
     ```

## CASE 2: Inference Reproducibility with Independent Dataset

> Full documentation: [CASE2.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE2.md)

### Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=moZ7cg)
- **Pre-trained Checkpoints**: Download from the [original PanPep repository](https://github.com/bm2-lab/PanPep/tree/main/Requirements/model.pt)
- **Encoding Files**: Background database and `.npz` encoding files available at:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=0QEsas)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=OIM7Jc)

### Usage
Use the same inference modes and metrics calculation pipeline as described in [CASE 1](#case-1-inference-reproducibility-with-original-dataset).

To reproduce the classification results, you can first run inference and then extract the values from the files provided at the following  [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQA85ZnPgk7eQ6A6VQOARqoYAYA3CRyPlWlFaNc_X-FUmyM?e=pxXvyA).

To reproduce the unseen classification results, you can first run inference and then extract the values from the files provided at the following link [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQCcTYJOLYSWQYulZ0he9MceAdWm5Nq7LY1wfGH5z8hV7hM?e=Met4Uh).

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| UniPMT \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Full documentation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 3: Training Reproducibility with TCRβ Extension

> Full documentation: [CASE3.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE3.md)

### Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Em3lIjtz-fxOnnz64yzprsoBNVrEvjkbzrBlK4Pa6-FWwg?e=KVoaZd)
- **Pre-trained Checkpoints**: Download from [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=UYWyXZ)
- **Encoding Files**: Background database and `.npz` encoding files available at:
  - [tcr_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/Efq7qjgLxNNKq7QojUzHJZUBOAQA5MZVPwZNtjEVXfo8dQ?e=84cga4)
  - [peptide_b.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EccZz48UFH1AqBnwhLZrCe8BmT9789yEUK7SqF1zlcOv1g?e=CgX01O)

### Usage
For metrics calculation, follow the same pipeline as described in [CASE 1](#metrics-calculation), but use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) instead of [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for sample extraction.

To reproduce the classification results, you can first run inference and then extract the values from the files provided at the following  [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQBNFrh00pltSo1Y1NPUXdznAWczHnrlQ6ue-P8jrRs-nzk?e=J1CYZn).

### Baseline Methods

> DLpTCR \| ERGO-II \| UnifyImmun \| Random Forest: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Full documentation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 4: Training Reproducibility with TCRα Extension

> Full documentation: [CASE4.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE4.md)

### Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EixVbgsKPD5BuDQP566AvR8BiPXqG85FkCCshSTexHLQgw?e=Jry9LY)
- **Pre-trained Checkpoints**: Download from [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=Iuojsc)
- **Encoding Files**: Background database and `.npz` encoding files available at:
  - [tcr_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EfbaOgcR0TVCjZMWigUshYABOa0cqxpJDZaiZBWsm0wMuw?e=9YHM4s)
  - [peptide_a.npz](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/EW2_VBo_t7RAs-ysGbkZMacBX_PTniASmuROxwpVjmt_pw?e=dASilC)

### Usage
For metrics calculation, follow the same pipeline as described in [CASE 1](#metrics-calculation), but use [`get_sample_indices_1.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_1.py) instead of [`get_sample_indices_100.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/get_sample_indices_100.py) for sample extraction.

To reproduce the classification results, you can first run inference and then extract the values from the files provided at the following  [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQAS2W0JXU25R4OQsyiQXabhAdiNIDmtBy1N7Grfgedajyk?e=zBd13f).

### Baseline Methods

> Full documentation: [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> DLpTCR: see [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

## CASE 5: Training Reproducibility with TCRαβ Extension

> Full documentation: [CASE5.md](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/CASE5.md)

### Data Requirements

- **Test Data**: Available on [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ei2Ef8zmUGBKqh8H0Vhrl9QBOGfwUjV7Oead3UVlV7kVKw?e=NC6JMl)
- **Pre-trained Checkpoints**: Download from here:
  - [Alpha chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/Ek35RWplZ-VIkmqhVB6pM_gB8XLzCRfXWGNOlDCIIG5pcA?e=oXLckH)
  - [Beta chain checkpoints](https://mailmissouri-my.sharepoint.com/:f:/g/personal/hefe_umsystem_edu/EsHEqqK-ECBAqQo1_f97IVcBaL1WoQ2euN9Xm497npseOA?e=86CUNb)
- **Encoding Files**: Local encoding files located at:
  - `./PanPep_Weight_Inference/tcr_ab.npz`
  - `./PanPep_Weight_Inference/peptide_ab.npz`
### Inference Modes

For TCRαβ binding recognition, the inference pipeline involves two steps:

1. **Data Preparation**: Use [`samplingab.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/samplingab.py) to format the data for inference:

   ```bash
   python samplingab.py \
       --positive_data <path_to_positive_data> \
       --tcr_pool <path_to_tcr_pool> \
       --mode <majority|few_shot|zero_shot> \
       --output_dir <output_directory>
   ```

   **Parameters**:
   - `--positive_data`: Path to positive sample data file (default: `data/tcrab_majority.csv`)
   - `--tcr_pool`: Path to negative sample data file (default: `data/pooling_tcrab.csv`)
   - `--mode`: Sampling mode - `majority`, `few_shot`, or `zero_shot`
   - `--majority_ratio`: Positive sample ratio for majority mode (default: 0.8)
   - `--few_shot`: Number of samples for few-shot mode (default: 2)
   - `--output_dir`: Output directory path

2. **Model Inference**: Use [`inferece_ab.py`](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/PanPep_Weight_Inference/inferece_ab.py) to perform TCRαβ binding prediction:

   ```bash
   python inferece_ab.py \
       --learning_setting <few-shot|zero-shot|Meta-learner|majority> \
       --input_dir <input_directory> \
       --output_dir <output_directory> \
       --fold <fold_number> \
       --peptide_encoding <path_to_peptide_encoding> \
       --tcr_encoding <path_to_tcr_encoding>
   ```

   **Parameters**:
   - `--learning_setting`: Learning mode (few-shot, zero-shot, Meta-learner, majority)
   - `--input_dir`: Directory containing formatted CSV files from step 1
   - `--output_dir`: Directory to save prediction results
   - `--fold`: Which fold to use (1-10)
   - `--peptide_encoding`: Path to peptide encoding file (default: peptide_ab.npz)
   - `--tcr_encoding`: Path to TCR encoding file (default: tcr_ab.npz)

**Example Workflow**:

```bash
# Step 1: Format data for few-shot learning
python samplingab.py \
    --positive_data data/tcrab_majority.csv \
    --tcr_pool data/pooling_tcrab.csv \
    --mode few_shot \
    --few_shot 5 \
    --output_dir formatted_data/

# Step 2: Run inference
python inferece_ab.py \
    --learning_setting few-shot \
    --input_dir formatted_data/few_shot/ \
    --output_dir results/ \
    --fold 1 \
    --peptide_encoding encoding/peptide_ab.npz \
    --tcr_encoding encoding/tcr_ab.npz
```

To reproduce the classification results, you can first run inference and then extract the values from the files provided at the following  [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/hefe_umsystem_edu/IQBMurKxVCPBQYjtu36vmrK-AVbx31kGFR2udfs1rt4AOyA?e=BG0HJ8).

### Baseline Methods

> DLpTCR \| ERGO-II: see [Baseline Methods Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/inference/README.md)

### Metrics Calculation

> Full documentation: [Metrics Calculation Manual](https://github.com/coffee19850519/PanPep_Reusability/blob/main/metric_calculation/README.md)

---

## PanPep Hyperparameter Reproduction

Full documentation: [PanPep Hyperparameter Reproduction](https://github.com/coffee19850519/PanPep_Reusability/blob/main/data/PanPep_Hyperparameter.md)

## TODO

- [x] Update extracted classification dataset in the paper
- [x] Update training pipeline description and related data
- [ ] Update Jupyter notebooks

---

## Data and Model Weights
The data and model weights in this study are publicly available on [Zenodo](https://doi.org/10.5281/zenodo.18861905).

## Original Study

Yicheng Gao, Yuli Gao, Qi Liu et al. Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition, Nature Machine Intelligence, 2023.

## Original Source Code

[https://github.com/bm2-lab/PanPep](https://github.com/bm2-lab/PanPep)

## Contact

For any questions or issues, Please contact Fei He via [hefe@missouri.edu](hefe@missouri.edu)
