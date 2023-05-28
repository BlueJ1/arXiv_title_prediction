#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=3000
#SBATCH --job-name=generate_data_batches
module load PyTorch
source $HOME/.envs/nlp/bin/activate
python generate_data_batches.py
