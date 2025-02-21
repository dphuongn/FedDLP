#!/bin/bash
dataset='pets'
algo='fedclip'
lr=5e-5
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
    --aa_bottleneck_reduction 1 \
    --aa_text \
    -pfl \
    -sd ${sd} >> $output_file 2>> $error_file
