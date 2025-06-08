#!/bin/bash
#SBATCH --job-name=grpo_training
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=grpo_question_only_job_%j.out
#SBATCH --error=grpo_question_only_job_%j.err

# Run GRPO training with different configurations
uv run python cs336_alignment/grpo_train_loop_deterministic_off_policy.py --output-dir /data/c-vprateek/grpo_question_only

# # Test different loss types
# echo "Testing reinforce_with_baseline..."
# uv run python cs336_alignment/grpo_train_loop.py \
#     --output-dir outputs/grpo_reinforce_baseline \
#     --config-path configs/grpo_reinforce.json

# echo "Testing grpo_clip..."
# uv run python cs336_alignment/grpo_train_loop.py \
#     --output-dir outputs/grpo_clip \
#     --config-path configs/grpo_clip.json

echo "GRPO training completed!"