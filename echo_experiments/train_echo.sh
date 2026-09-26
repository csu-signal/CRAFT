#!/bin/bash
#SBATCH --job-name=craft-echo
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --time=240:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=begin        	# send email when job begins
#SBATCH --mail-type=end          	# send email when job ends
#SBATCH --mail-user=sifatul.anindho@colostate.edu

set -euo pipefail

module purge
module load python/anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate craft

cd "$(dirname "$(realpath "$0")")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

srun --unbuffered python train.py \
    --mode echo \
    --steps 500 \
    --max_turns 20 \
    --report_to wandb