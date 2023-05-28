#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=python_example
#SBATCH --mem=800
Python/3.10.8-GCCcore-12.2.0
python python_example.py
