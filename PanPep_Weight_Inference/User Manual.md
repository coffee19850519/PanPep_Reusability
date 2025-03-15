#                                **User Manual**

Currently, this section contains example use cases. The actual background database and stored encoding `.npz` files can be obtained from the `xxxx` link.

In the example, the background database consists of:

- [pooling_tcrab-b.txt](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/pooling_tcrab-b.txt)
- [peptide_ab.npz](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/peptide_ab.npz)
- [tcr_ab.npz](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/tcr_ab.npz)

These `.npz` files store encoding data. The encoding script can be found at `xxx` (to be added later).

There are four modes available:

1. **Meta-learner mode** ([inference_meta_learner.py](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/inference_meta_learner.py))
2. **Zero-shot mode** ([inference_zero_shot.py](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/inference_zero_shot.py))
3. **Majority mode** ([inference_majority.py](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/inference_majority.py))
4. **Few-shot mode** ([inference_few_shot.py](https://github.com/coffee19850519/PanPep_Reusability/blob/main/PanPep_Weight_Inference/inference_few_shot.py))

### **General Usage**

```bash
nohup python inference_meta_learner.py \
    --gpu 0,1,2,3,4,5,6,7 \    # Select GPUs, multiple GPUs can be used
    --distillation 3 \          # Number of distillation steps, usually set to 3
    --upper_limit 100000 \      # Batch size, few-shot typically uses 10GB VRAM with 100,000 batch size
    --k_shot 0 \               # Number of fine-tuning samples, keep at 0 for meta-learner mode
    --test_data /public/home/wxy/Panpep1/few-shot.csv \    # Few-shot data (contains only positive samples)
    --negative_data /public/home/wxy/Panpep1/Control_dataset.txt \    # Negative sample data (background database)
    --model_path /public/home/wxy/Panpep1/Requirements \    # Model path
    --result_dir result/few-meta \    # Output directory for results
    --peptide_encoding /public/home/wxy/Panpep1/encoding/peptide_b.npz \    # Peptide encoding file
    --tcr_encoding /public/home/wxy/Panpep1/encoding/tcr_b.npz \    # TCR encoding file
    > inference_log.out 2>&1 &
```

### **Mode-Specific Differences**

Other modes have slight variations. **Few-shot** and **Majority** modes include an additional `--kshot_dir` parameter, which stores the selected fine-tuning samples. If you want to specify a fixed set of fine-tuning samples, include this parameter.

In **Majority** mode, the `--k_shot` parameter refers to a ratio rather than a specific count of samples.