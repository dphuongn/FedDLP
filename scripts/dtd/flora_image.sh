#!/bin/bash
dataset='dtd'
algo='flora'
lr=1e-5
sd=0

# Define the directory where you want to store output and error files
log_dir="./logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

output_file="${log_dir}/${algo}_image_lr${lr}_sd${sd}.out"
error_file="${log_dir}/${algo}_image_lr${lr}_sd${sd}.err"

python system/main.py -data ${dataset} \
    -algo ${algo} \
    -gr 100 \
    -did 0 \
    -nc 10 \
    -lbs 32 \
    -lr ${lr} \
    -wd 0 \
    --lora_rank 2 \
    --lora_alpha 16 \
    --lora_key_vision \
    --lora_query_vision \
    --lora_value_vision \
    --lora_outproj_vision \
    --lora_mlp_vision \
    --lora_head_vision \
    -pfl \
    -sd ${sd} >> $output_file 2>> $error_file
