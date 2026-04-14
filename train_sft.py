"""
train_sft_craft.py
-----------------
SFT training script for CRAFT director models using TRL + PEFT LoRA.
Targets Qwen-7B-Instruct or Llama-3-8B-Instruct from local paths.

Usage:
    python train_sft_craft.py --model qwen-7b
    python train_sft_craft.py --model llama-8b --lora_r 64
"""

import argparse
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.data_utils import is_conversational
import torch

# ── Local model paths ─────────────────────────────────────────────────────────

LOCAL_MODELS = {
    "qwen-7b":  "/data/open-weight-llms/models/qwen-7b",
    "llama-8b": "/data/open-weight-llms/models/llama-8b",
}

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model",          type=str,   default="qwen-7b",
                    choices=list(LOCAL_MODELS.keys()))
parser.add_argument("--train_file",     type=str,   default="train_sft.jsonl")
parser.add_argument("--eval_file",      type=str,   default="valid_sft.jsonl")
parser.add_argument("--output_dir",     type=str,   default="craft_sft_output")
parser.add_argument("--run_name",       type=str,   default=None)

# Training hyperparams
parser.add_argument("--num_epochs",     type=int,   default=3)
parser.add_argument("--lr",             type=float, default=2e-5)
parser.add_argument("--batch_size",     type=int,   default=2)
parser.add_argument("--grad_accum",     type=int,   default=8)
parser.add_argument("--max_seq_length", type=int,   default=1024)
parser.add_argument("--warmup_ratio",   type=float, default=0.05)
parser.add_argument("--weight_decay",   type=float, default=0.01)

# Eval
parser.add_argument("--eval_strategy", type=str,   default="steps",
                    choices=["steps", "epoch", "no"])
parser.add_argument("--eval_steps",    type=int,   default=100)
parser.add_argument("--save_steps",    type=int,   default=100)
parser.add_argument("--logging_steps", type=int,   default=10)

# LoRA
parser.add_argument("--lora_r",        type=int,   default=32)
parser.add_argument("--lora_alpha",    type=int,   default=16)
parser.add_argument("--lora_dropout",  type=float, default=0.05)
parser.add_argument("--assistant_only_loss", action="store_true", default=False)
parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
# Quantization
parser.add_argument("--use_4bit",      action="store_true",
                    help="Load model in 4-bit for QLoRA")

args = parser.parse_args()

# ── Derived config ────────────────────────────────────────────────────────────

model_path = LOCAL_MODELS[args.model]
run_name   = args.run_name or f"craft_sft_{args.model}_r{args.lora_r}"
output_dir = os.path.join(args.output_dir, run_name)

print(f"\n{'='*60}")
print(f"  CRAFT SFT Training")
print(f"  model:      {args.model} ({model_path})")
print(f"  output_dir: {output_dir}")
print(f"  epochs:     {args.num_epochs}  lr: {args.lr}")
print(f"  lora_r:     {args.lora_r}  lora_alpha: {args.lora_alpha}")
print(f"  4bit:       {args.use_4bit}")
print(f"{'='*60}\n")

def patch_qwen_chat_template(tokenizer):
    original = tokenizer.chat_template

    old = (
        "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first)"
        " or (message.role == \"assistant\" and not message.tool_calls) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    )

    new = (
        "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
        "\n    {%- elif message.role == \"assistant\" and not message.tool_calls %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' }}"
        "{% generation %}{{- message.content }}{% endgeneration %}"
        "{{- '<|im_end|>\\n' }}"
    )

    if old not in original:
        print("  ✗ Pattern not found — template may have changed")
        print("  Falling back to assistant_only_loss=False")
        return False

    patched = original.replace(old, new)
    tokenizer.chat_template = patched
    print("  ✓ Qwen chat template patched with {% generation %}")
    return True

def check_assistant_mask_support(tokenizer, sample_messages):
    """
    Check if the tokenizer's chat template supports assistant masking
    (required for assistant_only_loss=True).
    Also checks that apply_chat_template produces non-zero assistant masks.
    """
    print("\n--- Assistant mask support check ---")

    # Check 1: does the template contain {% generation %}
    template = tokenizer.chat_template or ""
    has_generation_keyword = "{% generation %}" in template
    print(f"  chat_template has {'{%'} generation {'%}'}: {has_generation_keyword}")

    if not has_generation_keyword:
        print("  ✗ Template missing {% generation %} — assistant_only_loss=True will fail")
        print("  → Fix: patch the tokenizer template or use assistant_only_loss=False")
        return False

    # Check 2: does it actually produce non-zero masks
    try:
        result = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        masks = result.get("assistant_masks", [])
        n_assistant = sum(masks)
        print(f"  assistant_masks total tokens: {len(masks)}")
        print(f"  assistant tokens masked:      {n_assistant}")
        if n_assistant == 0:
            print("  ✗ All assistant masks are 0 — template not generating masks correctly")
            return False
        print("  ✓ assistant_only_loss=True is supported")
        return True
    except Exception as e:
        print(f"  ✗ apply_chat_template failed: {e}")
        return False

# ── Load tokenizer ────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

# Qwen uses <|im_end|>, Llama uses <|eot_id|> — set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
if args.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

model.config.use_cache = False   # required for gradient checkpointing

# ── LoRA config ───────────────────────────────────────────────────────────────

# Target modules differ between Qwen and Llama
# Qwen2: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# Llama3: same names, same pattern
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading SFT dataset...")
dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

print(f"  train: {len(dataset['train'])}  eval: {len(dataset['eval'])}")


# drop extra columns — only messages needed for conversational SFT
keep_cols = ["messages"]
for split in ["train", "eval"]:
    extra = [c for c in dataset[split].column_names if c not in keep_cols]
    if extra:
        dataset[split] = dataset[split].remove_columns(extra)
        print(f"  dropped from {split}: {extra}")

print("columns after cleanup:", dataset["train"].column_names)
# verify conversational detection
from trl.data_utils import is_conversational
print("is_conversational:", is_conversational(dataset["train"][0]))

patch_qwen_chat_template(tokenizer)
mask_supported = check_assistant_mask_support(tokenizer, dataset["train"][0]["messages"])
use_assistant_only_loss = args.assistant_only_loss and mask_supported

# # set assistant_only_loss based on check result
# use_assistant_only_loss = args.assistant_only_loss and mask_supported
# if args.assistant_only_loss and not mask_supported:
#     print("  WARNING: falling back to assistant_only_loss=False")
# Verify the messages field is present
# assert "messages" in dataset["train"].column_names, \
#     "Dataset must have a 'messages' column with [system, user, assistant] format"

# ── SFT config ────────────────────────────────────────────────────────────────

# EOS token for Qwen vs Llama
eos_token = "<|im_end|>" if "qwen" in args.model else "<|eot_id|>"
print("args report to",args.report_to )
sft_config = SFTConfig(
    # Output
    output_dir=output_dir,
    run_name=run_name,

    # Training
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
 
    # Sequence
    max_length=args.max_seq_length, #Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right. Ref:https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
    packing=False,   # off — our samples vary in length and packing can mix game contexts

    # Data — tell SFTTrainer to use the messages column with chat template
    # dataset_text_field=None,
    # dataset_kwargs={"skip_prepare_dataset": False},

    dataset_text_field="messages",      # ← tells TRL this is conversational
    assistant_only_loss=use_assistant_only_loss,           # ← loss only on assistant tokens

    # Eval
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
    per_device_eval_batch_size=args.batch_size,

    # Logging + saving
    logging_steps=args.logging_steps,
    save_strategy=args.eval_strategy,
    save_steps=args.save_steps if args.eval_strategy == "steps" else None,
    # save_total_limit=2,         # keep only last 2 checkpoints
    load_best_model_at_end=True if args.eval_strategy != "no" else False,

    # Misc
    report_to=args.report_to,           # swap to "wandb" when ready
    eos_token=eos_token,
    seed=42,
)

# ── Trainer ───────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"] if args.eval_strategy != "no" else None,
    processing_class=tokenizer,
)

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nStarting training...")
trainer.train()

print("\nTraining complete. Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training config for reproducibility
config_path = os.path.join(output_dir, "training_config.json")
with open(config_path, "w") as f:
    json.dump(vars(args), f, indent=2)
print(f"Config saved to {config_path}")

print(f"\nDone. Model saved to {output_dir}")


# # Qwen with custom eval cadence
# python train_sft.py --model qwen-7b --eval_steps 50 --save_steps 50 --logging_steps 5 --output_dir "sft_testing"

# Qwen-7b baseline — default LoRA r=32, no quantization, 3 epochs
# CUDA_VISIBLE_DEVICES=1 python train_sft.py \
#     --model qwen-7b \
#     --train_file sft_datasets/train_sft_preference.jsonl \
#     --eval_file sft_datasets/valid_sft_preference.jsonl \
#     --output_dir sft_testing \
#     --run_name qwen7b_r32_baseline \
#     --num_epochs 3 \
#     --lr 2e-5 \
#     --batch_size 12 \
#     --grad_accum 4 \
#     --max_seq_length 1024 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --eval_strategy steps \
#     --eval_steps 60 \
#     --save_steps 60 \
#     --logging_steps 5 \
#     --report_to wandb

# llama 8b baseline — default LoRA r=32, no quantization, 3 epochs
# CUDA_VISIBLE_DEVICES=1 python train_sft.py \
#     --model llama-8b \
#     --train_file sft_datasets/train_sft_preference.jsonl \
#     --eval_file sft_datasets/valid_sft_preference.jsonl \
#     --output_dir sft_testing_llama-8b_r32_baseline \
#     --run_name llama-8b_r32_baseline \
#     --num_epochs 3 \
#     --lr 2e-5 \
#     --batch_size 12 \
#     --grad_accum 4 \
#     --max_seq_length 1024 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --eval_strategy steps \
#     --eval_steps 60 \
#     --save_steps 60 \
#     --logging_steps 5 \
#     --report_to wandb

# # Qwen-7b with higher rank LoRA r=64 — more expressive, slightly more memory
# python train_sft.py \
#     --model qwen-7b \
#     --train_file train_sft.jsonl \
#     --eval_file valid_sft.jsonl \
#     --output_dir sft_testing \
#     --run_name qwen7b_r64 \
#     --num_epochs 3 \
#     --lr 2e-4 \
#     --batch_size 2 \
#     --grad_accum 8 \
#     --max_seq_length 2048 \
#     --lora_r 64 \
#     --lora_alpha 32 \
#     --lora_dropout 0.05 \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --save_steps 50 \
#     --logging_steps 5

# # Llama-8b with QLoRA 4bit — fits on single GPU, lower memory footprint
# python train_sft.py \
#     --model llama-8b \
#     --train_file train_sft.jsonl \
#     --eval_file valid_sft.jsonl \
#     --output_dir sft_testing \
#     --run_name llama8b_qlora_r32 \
#     --num_epochs 3 \
#     --lr 2e-4 \
#     --batch_size 2 \
#     --grad_accum 8 \
#     --max_seq_length 2048 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --use_4bit \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --save_steps 50 \
#     --logging_steps 5

# # Qwen-7b quick smoke test — 1 epoch, eval every 20 steps, small batch
# # use this first to verify the pipeline runs end to end before full training
# python train_sft.py \
#     --model qwen-7b \
#     --train_file train_sft.jsonl \
#     --eval_file valid_sft.jsonl \
#     --output_dir sft_testing \
#     --run_name qwen7b_smoketest \
#     --num_epochs 1 \
#     --lr 2e-4 \
#     --batch_size 1 \
#     --grad_accum 4 \
#     --max_seq_length 2048 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --eval_strategy steps \
#     --eval_steps 20 \
#     --save_steps 20 \
#     --logging_steps 5

# # Qwen-7b epoch-level eval — cleaner for longer runs, saves more frequently
# python train_sft_craft.py \
#     --model qwen-7b \
#     --train_file train_sft.jsonl \
#     --eval_file valid_sft.jsonl \
#     --output_dir sft_testing \
#     --run_name qwen7b_epocheval \
#     --num_epochs 5 \
#     --lr 1e-4 \
#     --batch_size 2 \
#     --grad_accum 8 \
#     --max_seq_length 2048 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.01 \
#     --eval_strategy epoch \
#     --logging_steps 10