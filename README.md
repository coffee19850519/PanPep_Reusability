`meta_distillation_training.py` : Training for few-shot and zero-shot. 

`test_5fold.py` : Using few-shot and zero-shot model test few-shot data. 

`test_zero-shot.py` : Using zero-shot model test zero-shot data. 

`result_metrics.py` : Calculate the RUC-AUC and PR-AUC (Including few-shot results, zero-shot model test few-shot data result, zero-shot result).

`General\general.py`: General training.

`Similarity\get_similarity.py`: Analyze the similarity across peptides and tcrS.

`utils.py`: Common functions and related configurations.

`model_PCA.py`: Do principal component analysis for the model obtained by distillation module.

control dataset: https://zenodo.org/record/7544387  ([control dataset url](https://zenodo.org/record/7544387/files/Control%20dataset.txt?download=1))(Put it in the root directory of the project.)

