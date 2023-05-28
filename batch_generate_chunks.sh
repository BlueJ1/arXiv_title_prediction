#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --job-name=generate_data_batches
conda activate nlp
module load PyTorch
python generate_data_batches.py
