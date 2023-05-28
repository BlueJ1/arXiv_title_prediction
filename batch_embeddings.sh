#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_example
#SBATCH --mem=8192
module load PyTorch
conda activate nlp
python embeddings_from_filtered_raw_data.py
