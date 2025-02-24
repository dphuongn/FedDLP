#!/bin/bash
dataset='dtd'
algo='flora'
lr=1e-5
sd=0

cd system/

# Define the directory where you want to store output and error files
log_dir="../logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

output_file="${log_dir}/${algo}_text_lr${lr}_sd${sd}.out"
error_file="${log_dir}/${algo}_text_lr${lr}_sd${sd}.err"

# Clear previous logs
> $output_file
> $error_file

python main.py -data ${dataset} \
    -algo ${algo} \
    -gr 100 \
    -did 0 \
    -nc 10 \
    -lbs 32 \
    -lr ${lr} \
    -wd 0 \
    --lora_rank 2 \
    --lora_alpha 16 \
    --lora_key_text \
    --lora_query_text \
    --lora_value_text \
    --lora_outproj_text \
    --lora_mlp_text \
    --lora_head_text \
    -pfl \
    -sd ${sd} >> $output_file 2>> $error_file
