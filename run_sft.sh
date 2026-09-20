#qwen-3.5-9b baseline — default LoRA r=32, no quantization, 3 epochs
CUDA_VISIBLE_DEVICES=1 python train_sft.py \
    --model qwen-3.5-9b \
    --train_file sft_datasets/train_sft_preference.jsonl \
    --eval_file sft_datasets/valid_sft_preference.jsonl \
    --output_dir sft_testing \
    --run_name qwen-3.5-9b_r32_baseline \
    --num_epochs 3 \
    --lr 2e-5 \
    --batch_size 12 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --eval_strategy steps \
    --eval_steps 60 \
    --save_steps 60 \
    --logging_steps 5 \
    --report_to wandb