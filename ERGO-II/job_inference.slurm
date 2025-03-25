#!/bin/bash
#SBATCH --job-name=few_shot        # Job name
#SBATCH -N 1                  # Run all processes on a single node		
#SBATCH --cpus-per-task=40          # Number of CPU cores per task
#SBATCH --gres=gpu:8               # Request 8 GPUs
#SBATCH --output=few_shot111.log  # Standard output and error log
#SBATCH --nodelist=node202
# Load any necessary modules
source /datapool/software/modulefiles.d/mod_env.sh
module load conda/2024.10.1
conda activate ergo


BASE_PATH="/public/home/wxy/Data_processing/ergo-zero"
OUTPUT_DIR="/public/home/wxy/ERGO-II-master/tcr_zero"
NUM_GPUS=8

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

mkdir -p "${OUTPUT_DIR}/logs"

process_file() {
    local filename=$1
    local gpu_id=$2
    local input_file="${BASE_PATH}/${filename}.csv"
    local log_file="${OUTPUT_DIR}/logs/${filename}_output.log"

    echo "Starting process for $filename on GPU $gpu_id" >&2

    CUDA_VISIBLE_DEVICES=$gpu_id python /public/home/wxy/ERGO-II-master/Predict.py \
        --dataset vdjdb \
        --input_file "$input_file" \
        --output_dir "$OUTPUT_DIR" > "$log_file" 2>&1 &

    local pid=$!
    echo "Started process for ${filename} on GPU $gpu_id (PID: $pid)" >&2
    wait $pid
    echo "Completed process for ${filename} on GPU $gpu_id" >&2
}

files=(
    "RLLYPDYQI"
    "RLNQLESKM"
    "RLNQLESKV"
    "RLSFKELLV"
    "RLVDPQIQL"
    "RLYLDAYNM"
)

total_files=${#files[@]}
current_idx=0

while [ $current_idx -lt $total_files ]; do
    echo "Processing files $((current_idx + 1))-$((current_idx + NUM_GPUS)) of $total_files"

    for ((gpu=0; gpu<NUM_GPUS && current_idx<total_files; gpu++)); do
        process_file "${files[$current_idx]}" $gpu &
        current_idx=$((current_idx + 1))
    done

    wait
    echo "Current batch completed"
done

echo "All processes completed"