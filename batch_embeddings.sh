#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --job-name=embeddings_2gpu
#SBATCH --mem=12000

module load PyTorch
source $HOME/.envs/nlp/bin/activate

python embeddings_from_filtered_raw_data.py
