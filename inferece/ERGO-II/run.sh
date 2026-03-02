#!/bin/bash

set -x

BASE_PATH="/public/home/wxy/Data_processing/ergo-few2"
OUTPUT_DIR="/public/home/wxy/ERGO-II-master/all_tcrb1"
MAX_GPU=5

echo "Current process count: $(ps -u $USER | wc -l)"
echo "System limits:"
ulimit -a

mkdir -p "${OUTPUT_DIR}/logs"

process_file() {
    local filename=$1
    local gpu_id=$2
    local input_file="${BASE_PATH}/${filename}.csv"
    local log_file="${OUTPUT_DIR}/logs/${filename}_output.log"

    echo "Starting process for $filename on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python /public/home/wxy/ERGO-II-master/Predict.py \
        --dataset vdjdb \
        --input_file "$input_file" \
        --output_dir "$OUTPUT_DIR" > "$log_file" 2>&1 &

    echo $!
}

files=(
    "LLPPLLEHL"
)

pids=()

i=0
for file in "${files[@]}"; do
    gpu_id=$((i % MAX_GPU))
    pid=$(process_file "$file" "$gpu_id")
    pids+=($pid)
    echo "Started process for $file on GPU $gpu_id with PID $pid"
    
    i=$((i+1))
    sleep 0.5 
done
