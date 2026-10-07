#!/bin/bash
#SBATCH --job-name=craft-echo
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=240:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=begin        	# send email when job begins
#SBATCH --mail-type=end          	# send email when job ends
#SBATCH --mail-user=sifatul.anindho@colostate.edu

srun python train.py \
    --mode rloo_per_turn \
    --log_dir /s/babbage/h/nobackup/nblancha/public-datasets/sifat/craft_echo_runs \
    --steps 350 \
    --max_turns 20 \
    --report_to wandb \
    --resume_from_checkpoint /s/babbage/h/nobackup/nblancha/public-datasets/sifat/craft_echo_runs/craft_episode_return_Qwen2.5-7B-Instruct_seed42_20260921_1939/checkpoint-200
