#!/bin/bash

# echo "Starting Broad Hyperparameter Sweep..."

# # --- Sweep Configuration ---
# # Note: rollout_batch_size is fixed at 256 in the python script's default config.
# # Note: loss_type is set to grpo_clip, which is suitable for off-policy learning.
# # TRAIN_BATCH_SIZE_SWEEP=(64 128 256)
# # EPOCHS_PER_ROLLOUT_SWEEP=(1 2 4 8)
# TRAIN_BATCH_SIZE_SWEEP=(64 128 256)
# EPOCHS_PER_ROLLOUT_SWEEP=(2 4 8)

BASE_OUTPUT_DIR="/data/c-vprateek/grpo_off_policy_sweep"
PROJECT_NAME="cs336_alignment_off_policy_sweep"

# # Loop over all hyperparameter combinations
# for epochs in "${EPOCHS_PER_ROLLOUT_SWEEP[@]}"; do
#     for tbs in "${TRAIN_BATCH_SIZE_SWEEP[@]}"; do
#         # Per hint, keep micro-batch size constant (at 2) for consistent memory usage.
#         grad_accum=$((tbs / 2))

#         # --- Create unique names and directories for this run ---
#         run_name="epochs_${epochs}_tbs_${tbs}"
#         output_dir="${BASE_OUTPUT_DIR}/broad/${run_name}"
#         mkdir -p "$output_dir"

#         # --- Generate a temporary config file for this run ---
#         config_file="${output_dir}/config.json"
#         cat > "$config_file" <<EOL
# {
#     "n_grpo_steps": 50,
#     "epochs_per_rollout_batch": ${epochs},
#     "train_batch_size": ${tbs},
#     "gradient_accumulation_steps": ${grad_accum},
#     "loss_type": "grpo_clip",
#     "eval_every": 10,
#     "log_every": 1,
#     "cliprange": 0.2,
#     "use_std_normalization": false,
#     "seed": 42,
#     "learning_rate": 3e-5,
#     "advantage_eps": 1e-6,
#     "rollout_batch_size": 256,
#     "group_size": 8,
#     "sampling_temperature": 1.0,
#     "sampling_min_tokens": 4,
#     "sampling_max_tokens": 1024,
#     "gpu_memory_utilization": 0.2,
#     "grad_clip_value": 1.0,
#     "log_every": 2,
#     "eval_every": 2,
#     "eval_batch_size": 512,
#     "model_path": "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
#     "dataset_path": "/data/a5-alignment/MATH/train.jsonl",
#     "val_dataset_path": "/data/a5-alignment/MATH/validation.jsonl",
#     "weight_decay": 0.0,
#     "betas": [0.9, 0.95],
#     "seed": 42,
#     "length_normalization_strategy": "mean"
# }
# EOL

#         # --- Generate a temporary sbatch script for this run ---
#         sbatch_file="${output_dir}/run.sbatch"
#         cat > "$sbatch_file" <<EOL
# #!/bin/bash
# #SBATCH --job-name=grpo_sweep_${run_name}
# #SBATCH --partition=a5-batch
# #SBATCH --qos=a5-batch-qos
# #SBATCH --gpus=1
# #SBATCH --mem=100G
# #SBATCH --time=2:00:00
# #SBATCH --output=${output_dir}/slurm_%j.out
# #SBATCH --error=${output_dir}/slurm_%j.err

# echo "===== RUNNING GRPO OFF-POLICY SWEEP CONFIG ====="
# echo "Job Name: grpo_sweep_${run_name}"
# echo "Epochs per rollout: \${epochs}"
# echo "Train batch size: \${tbs}"
# echo "Grad accum steps: \${grad_accum}"
# echo "Output dir: \${output_dir}"
# echo "Config file: \${config_file}"
# echo "================================================"

# uv run python cs336_alignment/grpo_train_loop_deterministic_off_policy.py \\
#     --config-path "${config_file}" \\
#     --output-dir "${output_dir}"

# echo "Job finished."
# EOL

#         # --- Submit the job to the Slurm scheduler ---
#         echo "Submitting job for ${run_name}"
#         sbatch "$sbatch_file"
#     done
# done

# echo "All broad sweep jobs have been submitted."
# echo "Monitor their progress and check for results in ${BASE_OUTPUT_DIR}/broad/"

# ======================================================================================
# Phase 2: Focused Hyperparameter Sweep (Manual Steps)
# ======================================================================================
echo "PHASE 2: Starting Focused Hyperparameter Sweep..."

# --- Focused Sweep Configuration ---
FOCUSED_EPOCHS_SWEEP=(2) 
FOCUSED_TBS_SWEEP=(64)   

for epochs in "${FOCUSED_EPOCHS_SWEEP[@]}"; do
    for tbs in "${FOCUSED_TBS_SWEEP[@]}"; do
        grad_accum=$((tbs / 2))

        run_name="epochs_${epochs}_tbs_${tbs}_steps_200"
        output_dir="${BASE_OUTPUT_DIR}/focused/${run_name}"
        mkdir -p "$output_dir"

        config_file="${output_dir}/config.json"
        cat > "$config_file" <<EOL
{
    "n_grpo_steps": 200,
    "epochs_per_rollout_batch": ${epochs},
    "train_batch_size": ${tbs},
    "gradient_accumulation_steps": ${grad_accum},
    "loss_type": "grpo_clip",
    "eval_every": 10,
    "log_every": 1,
    "cliprange": 0.2,
    "use_std_normalization": false,
    "seed": 42,
    "learning_rate": 1e-5,
    "advantage_eps": 1e-6,
    "rollout_batch_size": 256,
    "group_size": 8,
    "sampling_temperature": 1.0,
    "sampling_min_tokens": 4,
    "sampling_max_tokens": 1024,
    "gpu_memory_utilization": 0.2,
    "grad_clip_value": 1.0,
    "log_every": 2,
    "eval_every": 2,
    "eval_batch_size": 512,
    "model_path": "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    "dataset_path": "/data/a5-alignment/MATH/train.jsonl",
    "val_dataset_path": "/data/a5-alignment/MATH/validation.jsonl",
    "weight_decay": 0.0,
    "betas": [0.9, 0.95],
    "seed": 42,
    "length_normalization_strategy": "mean"
}
EOL
        sbatch_file="${output_dir}/run.sbatch"
        cat > "$sbatch_file" <<EOL
#!/bin/bash
#SBATCH --job-name=grpo_focused_${run_name}
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=${output_dir}/slurm_%j.out
#SBATCH --error=${output_dir}/slurm_%j.err

echo "===== RUNNING GRPO FOCUSED SWEEP CONFIG ====="
echo "Job Name: grpo_focused_${run_name}"
echo "Config file: ${config_file}"
echo "================================================"

uv run python cs336_alignment/grpo_train_loop_deterministic_off_policy.py \\
    --config-path "${config_file}" \\
    --output-dir "${output_dir}"

echo "Job finished."
EOL
        echo "Submitting focused job for ${run_name}"
        sbatch "$sbatch_file"
    done
done

echo "All focused sweep jobs have been submitted."
echo "Monitor their progress and check for results in ${BASE_OUTPUT_DIR}/focused/"
