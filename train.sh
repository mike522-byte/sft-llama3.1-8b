#!/bin/bash

# =======================
# Script: train.sh
# Description: Launches RAG SFT training with Accelerate + DeepSpeed
# =======================

set -e 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file ./configs/sft.yaml \
    --num_processes 4  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_rag_sft.py \
    --experiment_name RAG-Instruct-training \
    --model_path "/gpfs/share/home/2301110044/pretrained_models/Meta-Llama-3.1-8B-Instruct/" \
    --data_path ./train_data/rag_instruct.json \
    --max_seq_len 2048 \
    --learning_rate 5e-6 \
    --train_bsz_per_gpu 1 \
    --gradient_accumulation_steps 16 \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 3 \
    --gradient_checkpointing