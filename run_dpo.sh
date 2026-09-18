#!/bin/bash
set -e  # stop on error

echo "=========================================="
echo "RUN 1: DPO from SFT checkpoint (builder preference)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
    --model qwen-7b \
    --train_file sft_datasets/train_dpo.jsonl \
    --eval_file sft_datasets/valid_dpo.jsonl \
    --output_dir dpip_dpo_preference_builder_fullrun \
    --run_name qwen7b_dpo_r32_from_sft \
    --sft_checkpoint_path sft_testing/qwen7b_r32_baseline/checkpoint-369 \
    --num_epochs 4 \
    --lr 5.0e-6 \
    --batch_size 6 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --lora_r 32 \
    --lora_alpha 16 \
    --beta 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --report_to wandb \
    --use_4bit

echo "=========================================="
echo "RUN 1 complete. Starting RUN 2: Temporal DPO"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
    --model qwen-7b \
    --train_file dpip_director_dpo_temporal/train_dpo.jsonl \
    --eval_file dpip_director_dpo_temporal/valid_dpo.jsonl \
    --output_dir dpip_dpo_temporal_preference_builder \
    --run_name qwen7b_dpo_r32_from_sftpref_temporal \
    --sft_checkpoint_path sft_testing/qwen7b_r32_baseline/checkpoint-369 \
    --num_epochs 4 \
    --lr 5.0e-6 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1600 \
    --lora_r 32 \
    --lora_alpha 16 \
    --beta 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --report_to wandb \
    --use_4bit

echo "=========================================="
echo "Both runs complete."
echo "=========================================="