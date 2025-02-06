#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1

attack_method=$2
model_name=$3



for epoch in 29

do
log_file='./results/'$model_name'_'$attack_method'_'$epoch'.log'
log_dir='./results/'$model_name'_'$attack_method'_'$epoch'.csv'
  #echo "freelb_stifiness_adv_${adv} epoch_${total} mag_${mag}: epoch ${epoch}" | tee -a $log_file
  python ./attack_finetune.py \
  --dataset_name imdb \
  --results_file $log_dir \
  --num_examples 800 \
  --attack_method $attack_method \
  --valid 'test' \
  --model_name_or_path ../train/saved_models/$model_name/'epoch'$epoch >> ${log_file}
done
