#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00

# $1 = Python script

# Set dynamic output and error files using job name and script name
#SBATCH --output=~/logs/logs/%x_%j.out
#SBATCH --error=~/logs/%x_%j.err
#SBATCH --chdir=~/MoltenSaltCalc

# Stop on first error
set -e

# Load modules
module purge
module load 2025

# Activate Python environment
source ~/PythonVenvs/MSPrediction/bin/activate
pip install -e .  # Make sure the latest version of the moltensaltcalc package is installed

# Show GPU info
nvidia-smi

# Run Python script with all arguments
python -u "$@"
