#!/bin/bash
#SBATCH --job-name=swissper
#SBATCH --output=swissper_logs.txt
#SBATCH --error=swissper-%j.txt
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=14GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mail-user=sukanya.nath@unibe.ch
#SBATCH --mail-type=begin,end,fail


# Load the CUDA module
module load Python/3.10.8-GCCcore-12.2.0 
module load CUDA/11.8.0
module load FFmpeg/6.0-GCCcore-12.3.0
python -m pip install --upgrade pip
#python -m venv swissper_venv
source swissper_venv/bin/activate
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install transformers accelerate evaluate lxml datasets[audio] jupyter jiwer tensorboard chardet tabulate
pip install ffmpeg ffmpeg-python

# Load the Python environment
# Set the working directory to the project
cd /storage/homefs/sn23a250/swissper

# Activate Python environment
source /storage/homefs/sn23a250/swissper/swissper_venv/bin/activate

# Run the script
python finetune_ubelix.py
