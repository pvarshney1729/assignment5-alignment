#!/bin/bash
#SBATCH --job-name=sft_math
#SBATCH --partition=a4-batch
#SBATCH --qos=a4-batch-qos
#SBATCH --gpus=2
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=sft_job_%j.out
#SBATCH --error=sft_job_%j.err

# Run the experiment
# uv run python cs336_alignment/sft_experiment.py --dataset_sizes 128 256 512 1024 --run_full --run_filtered

uv run python cs336_alignment/sft_experiment.py --run_filtered

# uv run python cs336_alignment/sft_experiment.py --learning_rate 1e-4 --batch_size 4 --num_epochs 5