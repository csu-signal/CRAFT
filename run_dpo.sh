#!/bin/bash
set -e  # stop on error

echo "=========================================="
echo "RUN 1: DPO from SFT checkpoint (builder preference)"
echo "=========================================="

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
    --model qwen-3.5-9b \
    --train_file /data/dpip_agent_weights/hannah_test/sft_datasets/train_dpo.jsonl \
    --eval_file /data/dpip_agent_weights/hannah_test/sft_datasets/valid_dpo.jsonl \
    --output_dir dpip_dpo_preference_builder_fullrun_hannah \
    --run_name qwen-3.5-9b_dpo_r32_from_sft \
    --sft_checkpoint_path /data/dpip_agent_weights/hannah_test/sft_testing/qwen-3.5-9b_r32_baseline/checkpoint-369 \
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
    --report_to wandb \
    --use_4bit

echo "=========================================="
echo "RUN 1 complete."
echo "=========================================="