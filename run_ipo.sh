#!/bin/bash
set -e  # stop on error

echo "=========================================="
echo "RUN 1: IPO  from SFT checkpoint (builder preference)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
    --model llama-8b \
    --train_file sft_datasets/train_dpo.jsonl \
    --eval_file sft_datasets/valid_dpo.jsonl \
    --output_dir dpip_ipo_preference_builder_fullrun_hannah \
    --run_name llama8b_ipo_r32_from_sft \
    --sft_checkpoint_path sft_testing/llama/checkpoint-369 \
    --num_epochs 4 \
    --lr 5.0e-6 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --lora_r 32 \
    --lora_alpha 16 \
    --beta 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --loss_type ipo \
    --report_to wandb \
    --use_4bit