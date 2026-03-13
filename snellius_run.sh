#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00

# $1 = Python script
# $2 = Job name (optional, default: M_NaF_DI)
JOB_NAME=${2:-ms_prediction}

# Get script base name without extension
SCRIPT_NAME=$(basename "$1" .py)

# Set dynamic output and error files using job name and script name
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=~/logs/${JOB_NAME}_${SCRIPT_NAME}_%j.out
#SBATCH --error=~/logs/${JOB_NAME}_${SCRIPT_NAME}_%j.err
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
