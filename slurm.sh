#!/bin/bash
#SBATCH --job-name=swissper2
#SBATCH --output=swissper_acc_100.txt
#SBATCH --error=swissper-%j_100.txt
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=6              # Increase CPUs for dataset preprocessing
#SBATCH --mem=150GB                    # Ensure enough RAM for dataset loading
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:2           # Request 2 GPUs
#SBATCH --mail-user=sukanya.nath@unibe.ch
#SBATCH --mail-type=begin,end,fail

# Load necessary modules
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/11.8.0
module load FFmpeg/6.0-GCCcore-12.3.0

# Activate virtual environment
source /storage/homefs/sn23a250/swissper/swissper_venv/bin/activate

# Ensure accelerate is configured correctly
accelerate config

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# Run training with multi-GPU setup
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 finetune_ubelix_accelerate_handle_oom.py

~                                                                                                                 