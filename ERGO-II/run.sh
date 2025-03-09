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
    # "AIFYLITPV_part1"
    # "AIFYLITPV_part10"
    # "AIFYLITPV_part2"
    # "AIFYLITPV_part3"
    # "AIFYLITPV_part4"
    # "AIFYLITPV_part5"
    # "AIFYLITPV_part6"
    # "AIFYLITPV_part7"
    # "AIFYLITPV_part8"
    # "AIFYLITPV_part9"
    # "ALSKGVHFV_part1"
    # "ALSKGVHFV_part10"
    # "ALSKGVHFV_part2"
    # "ALSKGVHFV_part3"
    # "ALSKGVHFV_part4"
    # "ALSKGVHFV_part5"
    # "ALSKGVHFV_part6"
    # "ALSKGVHFV_part7"
    # "ALSKGVHFV_part8"
    # "ALSKGVHFV_part9"
    # "ALWEIQQVV_part1"
    # "ALWEIQQVV_part10"
    # "ALWEIQQVV_part2"
    # "ALWEIQQVV_part3"
    # "ALWEIQQVV_part4"
    # "ALWEIQQVV_part5"
    # "ALWEIQQVV_part6"
    # "ALWEIQQVV_part7"
    # "ALWEIQQVV_part8"
    # "ALWEIQQVV_part9"
    # "ALWMRLLPL_part1"
    # "ALWMRLLPL_part10"
    # "ALWMRLLPL_part2"
    # "ALWMRLLPL_part3"
    # "ALWMRLLPL_part4"
    # "ALWMRLLPL_part5"
    # "ALWMRLLPL_part6"
    # "ALWMRLLPL_part7"
    # "ALWMRLLPL_part8"
    # "ALWMRLLPL_part9"
    # "AMFWSVPTV_part1"
    # "AMFWSVPTV_part10"
    # "AMFWSVPTV_part2"
    # "AMFWSVPTV_part3"
    # "AMFWSVPTV_part4"
    # "AMFWSVPTV_part5"
    # "AMFWSVPTV_part6"
    # "AMFWSVPTV_part7"
    # "AMFWSVPTV_part8"
    # "AMFWSVPTV_part9"
    # "CLAVHECFV_part1"
    # "CLAVHECFV_part10"
    # "CLAVHECFV_part2"
    # "CLAVHECFV_part3"
    # "CLAVHECFV_part4"
    # "CLAVHECFV_part5"
    # "CLAVHECFV_part6"
    # "CLAVHECFV_part7"
    # "CLAVHECFV_part8"
    # "CLAVHECFV_part9"
    # "CLNEYHLFL_part1"
    # "CLNEYHLFL_part10"
    # "CLNEYHLFL_part2"
    # "CLNEYHLFL_part3"
    # "CLNEYHLFL_part4"
    # "CLNEYHLFL_part5"
    # "CLNEYHLFL_part6"
    # "CLNEYHLFL_part7"
    # "CLNEYHLFL_part8"
    # "CLNEYHLFL_part9"
    # "DTDFVNEFY_part1"
    # "DTDFVNEFY_part10"
    # "DTDFVNEFY_part2"
    # "DTDFVNEFY_part3"
    # "DTDFVNEFY_part4"
    # "DTDFVNEFY_part5"
    # "DTDFVNEFY_part6"
    # "DTDFVNEFY_part7"
    # "DTDFVNEFY_part8"
    # "DTDFVNEFY_part9"
    # "ELAGIGLTV_part1"
    # "ELAGIGLTV_part10"
    # "ELAGIGLTV_part2"
    # "ELAGIGLTV_part3"
    # "ELAGIGLTV_part4"
    # "ELAGIGLTV_part5"
    # "ELAGIGLTV_part6"
    # "ELAGIGLTV_part7"
    # "ELAGIGLTV_part8"
    # "ELAGIGLTV_part9"
    # "FLAHIQWMV_part1"
    # "FLAHIQWMV_part10"
    # "FLAHIQWMV_part2"
    # "FLAHIQWMV_part3"
    # "FLAHIQWMV_part4"
    # "FLAHIQWMV_part5"
    # "FLAHIQWMV_part6"
    # "FLAHIQWMV_part7"
    # "FLAHIQWMV_part8"
    # "FLAHIQWMV_part9"
    # "FLGKIWPSYK_part1"
    # "FLGKIWPSYK_part10"
    # "FLGKIWPSYK_part2"
    # "FLGKIWPSYK_part3"
    # "FLGKIWPSYK_part4"
    # "FLGKIWPSYK_part5"
    # "FLGKIWPSYK_part6"
    # "FLGKIWPSYK_part7"
    # "FLGKIWPSYK_part8"
    # "FLGKIWPSYK_part9"
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