"""
train_dpo.py
-----------------
DPO training script for CRAFT director models using TRL + PEFT LoRA.
Targets Qwen-7B-Instruct from local paths.

Data format expected (from prep_dpo_data.py):
    chosen:   [system, user, chosen_assistant]
    rejected: [system, user, rejected_assistant]

Usage:
    python train_dpo_craft.py --model qwen-7b

    # With custom dataset
    python train_dpo_craft.py \
        --model qwen-7b \
        --train_file sft_datasets/train_dpo.jsonl \
        --eval_file  sft_datasets/valid_dpo.jsonl

    # Full run example
    CUDA_VISIBLE_DEVICES=1 python train_dpo_craft.py \
        --model qwen-7b \
        --train_file sft_datasets/train_dpo.jsonl \
        --eval_file  sft_datasets/valid_dpo.jsonl \
        --output_dir craft_dpo_output \
        --run_name qwen7b_dpo_r32_baseline \
        --num_epochs 3 \
        --lr 5e-6 \
        --batch_size 2 \
        --grad_accum 8 \
        --max_seq_length 1024 \
        --lora_r 32 \
        --lora_alpha 16 \
        --beta 0.1 \
        --eval_steps 100 \
        --save_steps 100 \
        --report_to wandb
"""

import argparse
import os
import json
import torch
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,PeftModel
from trl import DPOTrainer, DPOConfig
from trl import get_dataset, get_kbit_device_map, get_peft_config, get_quantization_config
 
# ── Local model paths ─────────────────────────────────────────────────────────

LOCAL_MODELS = {
    # ── Local paths ───────────────────────────────────────────
    "qwen-7b":  "/data/open-weight-llms/models/qwen-7b",
    "qwen-14b": "/data/open-weight-llms/models/qwen-14b",
    "llama-8b": "/data/open-weight-llms/models/llama-8b",

    # ── HuggingFace paths ─────────────────────────────────────
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b-v3": "Qwen/Qwen3-4B-Instruct-2507",
}

# Qwen/Qwen2.5-1.5B-Instruct
#Qwen/Qwen3-4B-Instruct-2507
#Qwen/Qwen2.5-3B-Instruct
#Qwen/Qwen0.5-3B-Instruct


# ── Argument parsing ──────────────────────────────────────────────────────────
 
parser = argparse.ArgumentParser()
parser.add_argument("--model",          type=str,   default="qwen-7b",
                    choices=list(LOCAL_MODELS.keys()))
parser.add_argument("--sft_checkpoint_path",          type=str,   default="sft_testing/qwen7b_r32_baseline/checkpoint-369")
          
parser.add_argument("--train_file",     type=str,   default="sft_datasets/train_dpo.jsonl")
parser.add_argument("--eval_file",      type=str,   default="sft_datasets/valid_dpo.jsonl")
parser.add_argument("--output_dir",     type=str,   default="craft_dpo_output")
parser.add_argument("--run_name",       type=str,   default=None)

# Training hyperparams
parser.add_argument("--num_epochs",     type=int,   default=3)
parser.add_argument("--lr",             type=float, default=5e-6)
parser.add_argument("--batch_size",     type=int,   default=2)
parser.add_argument("--grad_accum",     type=int,   default=8)
parser.add_argument("--max_seq_length", type=int,   default=1024)
parser.add_argument("--warmup_ratio",   type=float, default=0.05)
parser.add_argument("--weight_decay",   type=float, default=0.01)

# DPO-specific
parser.add_argument("--beta",           type=float, default=0.1,
                    help="DPO regularization parameter (KL penalty weight)")
parser.add_argument("--loss_type",       type=str,   default='sigmoid')

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

# Misc
parser.add_argument("--report_to",     type=str,   default="none",
                    choices=["none", "wandb", "tensorboard"])
parser.add_argument("--use_4bit",      action="store_true",
                    help="Load model in 4-bit for QLoRA")

args = parser.parse_args()

# ── Derived config ────────────────────────────────────────────────────────────

model_path = LOCAL_MODELS[args.model]
run_name   = args.run_name or f"craft_dpo_{args.model}_r{args.lora_r}"
output_dir = os.path.join(args.output_dir, run_name)

print(f"\n{'='*60}")
print(f"  CRAFT DPO Training")
print(f"  model:      {args.model} ({model_path})")
print(f"  output_dir: {output_dir}")
print(f"  epochs:     {args.num_epochs}  lr: {args.lr}  beta: {args.beta}")
print(f"  lora_r:     {args.lora_r}  lora_alpha: {args.lora_alpha}")
print(f"  4bit:       {args.use_4bit}")
print(f"{'='*60}\n")



def setup_model_and_tokenizer(self, config, checkpoint_path=None):
    model_name = config["model_name"]
    print(f"\nLoading model: {model_name}")
    
    model_size_gb = self.get_model_size_gb(model_name)
    torch_dtype = torch.bfloat16 if model_size_gb <= 14 else torch.float16
    
    if checkpoint_path:
        if checkpoint_path.startswith("s3://"):
            # Download checkpoint first
            local_checkpoint_path = "./test_sft_checkpoint/"
            print(f"Downloading SFT checkpoint from S3...")
            subprocess.run([
                "aws", "s3", "cp", checkpoint_path, local_checkpoint_path, "--recursive"
            ], check=True)
            checkpoint_path = local_checkpoint_path  # Use local path
        
            
        print(f"Loading LoRA checkpoint from: {checkpoint_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)  # Now uses local path
        ref_model = None
    else:
        # Load fresh models
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

# ── Load tokenizer ────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load model ────────────────────────────────────────────────────────────────

# ── BitsAndBytes config ───────────────────────────────────────
bnb_config = None
if args.use_4bit:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

if args.sft_checkpoint_path:
    print(f"Loading SFT LoRA checkpoint from: {args.sft_checkpoint_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        quantization_config=bnb_config,  # None is silently ignored
    )
    model = PeftModel.from_pretrained(base_model, args.sft_checkpoint_path)
    model.enable_input_require_grads()
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    apply_lora_via_trainer = False

else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        quantization_config=bnb_config,  # None is silently ignored
    )
    apply_lora_via_trainer = True

model.config.use_cache = False



# print("Loading model...")
 
# if args.sft_checkpoint_path:
#     print(f"Loading SFT LoRA checkpoint from: {args.sft_checkpoint_path}")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         use_cache=False,
#     )
#     model = PeftModel.from_pretrained(base_model, args.sft_checkpoint_path)
#     model.enable_input_require_grads()

#     model.train()  # set training mode

#     # Force LoRA params trainable
#     for name, param in model.named_parameters():
#         if "lora_" in name:
#             param.requires_grad_(True)

#     print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
#     # Should be >0, expect ~80M for r=32 on Qwen-7B

#     apply_lora_via_trainer = False  # DPOTrainer gets PeftModel, no peft_config
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         use_cache=False,
#     )
     
#     apply_lora_via_trainer = True  # DPOTrainer applies LoRA via peft_config

# model.config.use_cache = False

 

# ── LoRA config ───────────────────────────────────────────────────────────────

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

# model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading DPO dataset...")
dataset = load_dataset("json", data_files={
    "train": args.train_file,
    "eval":  args.eval_file,
})

print(f"  train: {len(dataset['train'])}  eval: {len(dataset['eval'])}")

# Keep only chosen/rejected — drop debug fields from prep_dpo_data.py
keep_cols = ["chosen", "rejected", "prompt"]   
for split in ["train", "eval"]:
    extra = [c for c in dataset[split].column_names if c not in keep_cols]
    if extra:
        dataset[split] = dataset[split].remove_columns(extra)
        print(f"  dropped from {split}: {extra}")

print("Columns after cleanup:", dataset["train"].column_names)

# Sanity check: confirm chosen != rejected
n_identical = sum(
    1 for s in dataset["train"]
    if s["chosen"][-1]["content"].strip() == s["rejected"][-1]["content"].strip()
)
print(f"  Identical chosen/rejected pairs: {n_identical} "
      f"({100*n_identical/len(dataset['train']):.1f}%) ← should be 0")

dummy_train_dataset = Dataset.from_dict({

    'prompt': dataset['train']['prompt'][:100],
    'chosen':   dataset['train']['chosen'][:100],
    'rejected': dataset['train']['rejected'][:100],
})
dummy_eval_dataset = Dataset.from_dict({
    'prompt': dataset['eval']['prompt'][:50],
    'chosen':   dataset['eval']['chosen'][:50],
    'rejected': dataset['eval']['rejected'][:50],
})
print("size of train/eval dataset:", dataset['train'], dataset['eval'])
print("size of dummy_train/eval dataset:", dummy_train_dataset, dummy_eval_dataset)

for split in ["train", "eval"]:
    sample = dataset[split][0]
    print(f"\n{'='*60}")
    print(f"  SAMPLE FROM: {split}")
    print(f"{'='*60}")
    # print("PROMPT")
    print("\n--- CHOSEN ---")
    for turn in sample["chosen"]:
        print(f"[{turn['role'].upper()}]\n{turn['content']}\n")
    print("\n--- REJECTED ---")
    for turn in sample["rejected"]:
        print(f"[{turn['role'].upper()}]\n{turn['content']}\n")
    print(f"{'='*60}\n")


# ── DPO config ────────────────────────────────────────────────────────────────

dpo_config = DPOConfig(
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
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},   
    max_grad_norm=0.3,

    # Sequence lengths
    max_length=args.max_seq_length,
    max_prompt_length=args.max_seq_length // 2,

    # DPO-specific
    beta=args.beta,

    # Eval
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
    per_device_eval_batch_size=args.batch_size,

    # Logging + saving
    logging_steps=args.logging_steps,
    save_strategy=args.eval_strategy,
    save_steps=args.save_steps if args.eval_strategy == "steps" else None,
    load_best_model_at_end=True if args.eval_strategy != "no" else False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    loss_type = args.loss_type, #change this to 'ipo' to get the IPO loss, sigmoid represents DPO 
    # Misc
    report_to=args.report_to,
    seed=42,
)

# ── Trainer ───────────────────────────────────────────────────────────────────

# ref_model=None: with PEFT/LoRA, TRL handles the reference model implicitly
# by running the base (non-adapted) forward pass internally
#set loss type for dpo or ipo
dpo_config.loss_type = args.loss_type
print("loss type", dpo_config.loss_type)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset['train'],
    eval_dataset= dataset['eval'],
    processing_class=tokenizer,
    
    peft_config=lora_config if apply_lora_via_trainer else None,
)

print("Adapter training mode:", model.training)
print("Trainable params during eval setup:")
print([(n, p.requires_grad) for n, p in model.named_parameters() if "lora_" in n][:3])

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nStarting DPO training...")
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




# Qwen-7b DPO baseline — r=32, no quantization, 3 epochs
# CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
#     --model qwen-7b \
#     --train_file sft_datasets/train_dpo.jsonl \
#     --eval_file sft_datasets/valid_dpo.jsonl \
#     --output_dir craft_dpo_output \
#     --run_name qwen7b_dpo_r32_baseline \
#     --num_epochs 3 \
#     --lr 5e-6 \
#     --batch_size 2 \
#     --grad_accum 8 \
#     --max_seq_length 1024 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --beta 0.1 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --logging_steps 10 \
#     --report_to wandb

# # Qwen-7b DPO from SFT checkpoint — warm init from SFT LoRA weights
# CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
#     --model qwen-7b \
#     --train_file sft_datasets/train_dpo.jsonl \
#     --eval_file sft_datasets/valid_dpo.jsonl \
#     --output_dir craft_dpo_preference_builder_fullrun \
#     --run_name qwen7b_dpo_r32_from_sft \
#     --sft_checkpoint_path sft_testing/qwen7b_r32_baseline/checkpoint-369 \
#     --num_epochs 4 \
#     --lr 5.0e-6 \
#     --batch_size 6 \
#     --grad_accum 4 \
#     --max_seq_length 1024 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --beta 0.1 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --logging_steps 5 \
#     --report_to wandb \
#     --use_4bit


# dpip_director_dpo_temporal/train_dpo.jsonl

# CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
#     --model qwen-7b \
#     --train_file dpip_director_dpo_temporal/train_dpo.jsonl \
#     --eval_file dpip_director_dpo_temporal/valid_dpo.jsonl \
#     --output_dir craft_dpo_temporal_preference_builder \
#     --run_name qwen7b_dpo_r32_from_sftpref_temporal \
#     --sft_checkpoint_path sft_testing/qwen7b_r32_baseline/checkpoint-369 \
#     --num_epochs 4 \
#     --lr 5.0e-6 \
#     --batch_size 2 \
#     --grad_accum 4 \
#     --max_seq_length 1600 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --beta 0.1 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --logging_steps 1 \
#     --report_to wandb \
#     --use_4bit


# dpip_director_dpo_temporal/valid_dpo.jsonl
#tuning commands


# CUDA_VISIBLE_DEVICES=1 python train_dpo.py \
#     --model qwen-7b \
#     --train_file sft_datasets/train_dpo.jsonl \
#     --eval_file sft_datasets/valid_dpo.jsonl \
#     --output_dir craft_dpo_output_qwen \
#     --run_name qwen7b_dpo_r32_from_sft \
#     --sft_checkpoint_path sft_testing/qwen7b_r32_baseline/checkpoint-369 \
#     --num_epochs 20 \
#     --lr 5.0e-6 \
#     --batch_size 6 \
#     --grad_accum 4 \
#     --max_seq_length 1024 \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --beta 0.1 \
#     --eval_strategy steps \
#     --eval_steps 5 \
#     --save_steps 20 \
#     --logging_steps 1 \
#     --report_to wandb