#!/bin/bash
#SBATCH --job-name=expert_iteration
#SBATCH --partition=a4-batch
#SBATCH --qos=a4-batch-qos
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=ei_job_%j.out
#SBATCH --error=ei_job_%j.err

# Run expert iteration experiment
uv run python cs336_alignment/expert_iteration.py --batch_size_rollouts 2048 --rollout_configs 4 8 --epoch_configs 1 2