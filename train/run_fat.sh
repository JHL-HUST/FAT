#!/bin/bash
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
log_file="fat_imdb.log"
valid="test"
adv_lr=0.2


python run_fat.py \
    --dataset_name imdb \
    --epochs 30 \
    --valid $valid \
    --adv-steps 1 \
    --adv-lr 0.2 \
    --bsz 16 \
    --adv-init-mag 0.05 \
    --eval_size 32 \
    --num-labels 2 \
    --nt-at-interval 1 \
    --ckpt-dir ./saved_models >> ${log_file}