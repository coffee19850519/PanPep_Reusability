***Reproducing the results of PanPep using the same data.*** ```To validate the reproducibility of PanPep, we plan to retrain and evaluate the meta-learner and peptide-specific learners for several rounds using different random seeds while maintaining the same hyperparameter settings and data. We will analyze the performance of PanPep in few-shot, zero-shot, and majority settings by calculating the mean and standard deviation of the performance across these rounds. The variances obtained from these replicate experiments will indicate the robustness of PanPep.``` 
> Additionally, we intend to evaluate the performance of PanPep under different few-shot difficulty levels. One approach we will take is to group the evaluation data into different few-shot groups based on the number of available data for each peptide and the closeness to the most similar training samples based on sequence similarities of the peptide and TCR. This will allow us to determine how PanPep performs with varying degrees of data availability.

***Demonstrating the effectiveness of meta-learning in few-shot and zero-shot tasks.*** We plan to combine all the peptide-TCR binding data and train a general learner using the same architecture and traditional learning modes (e.g., supervised learning and transfer learning), while also implementing popular imbalanced data handling strategies in modeling. This general learner will then be evaluated on the same testing data in both few-shot and zero-shot tasks. Through this head-to-head comparison, we can reveal the benefits of meta-learning and its contribution to these tasks. To further explore the potential of meta-learning, we will experiment with different configurations during meta-training, including variations in the inner loop number, support set/query set split, and convergence requirement of the meta-learner. By summarizing the impact factors in meta-learning, we can better understand how these variables affect the model's performance. Furthermore, we will visualize the distributions of learned representations from the general learner, meta-learner, and peptide-specific learners to compare the effectiveness of meta-learning in few-shot, zero-shot, and majority settings.   

<del>***Analyzing the impact of meta-learning in majority tasks.*** Even though meta-learning was originally proposed to address the few-shot problem, it may be interesting to explore the impact of meta-learning in the majority settings. To this end, we plan to compare the performance of modeling using meta-learning with typical supervised training for the majority tasks. Our goal is to determine whether meta-learning is not conducive to the majority tasks, as suggested by some studies. We will control the similarity of the peptides and TCRs between majority training and majority testing dataset and conduct a rigorous analysis to further demonstrate the impact of meta-learning in the majority task. </del>

***Exploring the properties, mechanism, and potentials of the disentanglement distilled module for zero-shot and few-shot settings.*** PanPep assumes that similar peptides will exhibit similar TCR-binding patterns. It employs a disentanglement distilled module, which reduces all peptide-specific meta-learners to three base learners in the content memory of a Neural Turing Machine (NTM). The resulting distilled base learners can be used to generate new zero-shot meta-learners. To validate this hypothesis, we will analyze the similarity across peptides and the model parameters of meta-learners. Additionally, we will investigate the mechanism of the three distilled base learners by obtaining their principal components. We will further apply the NTM for zero-shot tasks to few-shot tasks to observe its adaptability.   

***Investigating the generalizability of PanPep by extending it to new unseen data.*** PanPep's manuscript only considered the beta chain of HLA-I-TCR binding data. To broaden the scope of this study, we have collected diverse new, unseen data from three accessible databases: IEDB (http://www.iedb.org/), VDJdb58 (https://vdjdb.cdr3.net/), and McPas-TCR60 (http://friedmanlab.weizmann.ac.il/McPAS-TCR/). These databases cover the following: 1) 516 beta-chain of HLA-I-TCR binding data that never appeared in the datasets used in PanPep’s manuscript; 2) 3343 alpha-chain of HLA-I-TCR binding data, 3) 3678 beta-chain of HLA-II-TCR binding data, and 4) 231 alpha-chain of HLA-II-TCR binding data. These unseen datasets will be applied for benchmarking PanPep and other competitive tools to compare their generalizability. For better adapting new data to PanPep in this benchmarking, we will attempt applying PanPep pretrained model and retraining PanPep on the new unseen data. Furthermore, we argue that even in a zero-shot setting, the peptides were unseen for PanPep, but some TCRs were still involved in the meta-training stage. We will intentionally hide some TCRs in a zero-shot setting from meta-training and test the zero-shot learner under such unknown peptides and unknown TCRs condition.



`meta_distillation_training.py` : Training for few-shot and zero-shot. 

`test_5fold.py` : Using few-shot and zero-shot model test few-shot data. 

`test_zero-shot.py` : Using zero-shot model test zero-shot data. 

`result_metrics.py` : Calculate the RUC-AUC and PR-AUC (Including few-shot results, zero-shot model test few-shot data result, zero-shot result).

`General\general.py`: General training.

`Similarity\get_similarity.py`: Analyze the similarity across peptides and tcrS.

`utils.py`: Common functions and related configurations.

`model_PCA.py`: Do principal component analysis for the model obtained by distillation module.

control dataset: https://zenodo.org/record/7544387  ([control dataset url](https://zenodo.org/record/7544387/files/Control%20dataset.txt?download=1))(Put it in the root directory of the project.)

