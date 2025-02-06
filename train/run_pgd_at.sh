#!/bin/bash
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
log_file="pgd_imdb.log"


python run_pgd_at.py \
    --dataset_name imdb \
    --epochs 10 \
    --valid test \
    --adv-steps 5 \
    --adv-lr 0.04 \
    --bsz 16 \
    --adv-init-mag 0.05 \
    --eval_size 32 \
    --num-labels 2 \
    --ckpt-dir ./saved_models/ >> ${log_file}