#!/bin/bash
#SBATCH --time=00:01:20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=python_example
#SBATCH --mem=12000
module load PyTorch
source $HOME/.envs/nlp/bin/activate
python embeddings_from_filtered_raw_data.py
