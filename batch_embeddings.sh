#!/bin/bash
#SBATCH --time=00:00:20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_example
#SBATCH --mem=1024
conda activate nlp
module load PyTorch
python pytorch_gpu_example.py
