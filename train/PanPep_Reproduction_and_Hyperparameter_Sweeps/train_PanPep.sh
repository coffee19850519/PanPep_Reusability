#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=2-00:00:00
#SBATCH --account=bevl-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --job-name=train_PanPep
#SBATCH --output=train_PanPep.%j.out

module load cuda/12.4.0

source /u/coffee19850519/.bashrc

conda activate /work/hdd/bevl/coffee19850519/PanPep

cd /work/nvme/bevl/coffee19850519/code/PanPep_train

python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train.py
python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train1.py
python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train2.py
python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train3.py
python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train4.py
python  -u /work/nvme/bevl/coffee19850519/code/PanPep_train/train5.py


conda deactivate