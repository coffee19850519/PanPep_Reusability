#!/bin/bash

BASE_PATH="/public/home/wxy/generate_combinations/ergo"

process_file() {
    local filename=$1
    local gpu_name=$2
    local output_dir=$3
    local log_name=$(basename "$filename" .csv)
    echo "Processing: $filename"
    CUDA_VISIBLE_DEVICES=$gpu_name nohup python /public/home/wxy/ERGO-II-master/Predict.py --dataset vdjdb --input_file "${filename}" --output_dir "${output_dir}" > "${log_name}_output.log" 2>&1 &
    sleep 2 
}

FILES=(
    "ALNLGETFV"
    "FLQDVMNIL"
)
output_dir="/public/home/wxy/ERGO-II-master"
i=0
MAX_GPU=4
for file in "${FILES[@]}"; do
    gpu_id=$((i % MAX_GPU))
    process_file "${BASE_PATH}/${file}.csv" "$gpu_id" "${output_dir}"
    i=$((i+1))
done

echo "All jobs submitted"