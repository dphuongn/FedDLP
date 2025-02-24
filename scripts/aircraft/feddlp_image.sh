#!/bin/bash
dataset='aircraft'
algo='feddlp'
lr=1e-5
sd=0
sparse_lambda=5e-5
gammas_local=0.1

cd system/

# Define the directory where you want to store output and error files
log_dir="../logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

output_file="${log_dir}/${algo}_image_lr${lr}_sd${sd}.out"
error_file="${log_dir}/${algo}_image_lr${lr}_sd${sd}.err"

# Clear previous logs
> $output_file
> $error_file

python main.py -data ${dataset} \
    -algo ${algo} \
    --sparse_lambda ${sparse_lambda} \
    --gamma_local ${gammas_local} \
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
