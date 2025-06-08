#!/bin/bash
#SBATCH --job-name=expert_iteration
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=ei_batch_2048_rollout_4_8_epoch_1_2_job_%j.out
#SBATCH --error=ei_batch_2048_rollout_4_8_epoch_1_2_job_%j.err

# Run expert iteration experiment
uv run python cs336_alignment/expert_iteration.py --batch_size_rollouts 2048 --rollout_configs 4 8 --epoch_configs 1 2