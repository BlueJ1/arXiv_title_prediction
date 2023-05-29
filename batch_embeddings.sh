#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=embeddings
#SBATCH --mem=8192

module load PyTorch
source $HOME/.envs/nlp/bin/activate

python embeddings_from_filtered_raw_data.py
