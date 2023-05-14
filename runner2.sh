#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=70GB
#SBATCH --job-name=try
#SBATCH --output=/scratch/vb2184/attempt2.txt
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
module purge
module load intel/19.1.2
module load anaconda3/2020.07
module load python3/intel/3.7.3

python3 try_python_attempt2.py
