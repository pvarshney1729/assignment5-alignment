import typer
import orjson
import json
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import load_dataset
from vllm_utils import evaluate_model, init_vllm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tqdm import tqdm

from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.masked_mean import masked_mean

from cs336_alignment.data_utils import load_math_data
from cs336_alignment.vllm_utils import load_policy_into_vllm_instance

from vllm import SamplingParams

import wandb

with open("cs336_alignment/prompts/r1_zero.prompt", 'r') as f:
    prompt_template = f.read()

@dataclass
class GRPOConfig:
    n_grpo_steps: int = 200
    learning_rate: float = 3e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True

    cliprange: float = 0.2
    grad_clip_value: float = 1.0

    log_every: int = 1
    eval_every: int = 3
    # eval_batch_size: int = 1024
    eval_batch_size: int = 512

    model_path: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    dataset_path: str = "/data/a5-alignment/MATH/train.jsonl"
    val_dataset_path: str = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir: str = "/data/a5-alignment/outputs/grpo"

    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)

    seed: int = 42

    length_normalization_strategy: Literal["mean", "normalize"] = "normalize"


def make_json_serializable(obj):
    """Convert tensors and numpy types to Python native types."""
    if torch.is_tensor(obj):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj
    
def generate_grpo_rollouts(
    vllm_model,
    question_batch: list,
    prompt_template: str,
    group_size: int,
    config: GRPOConfig,
    step: int
) -> tuple[list[str], list[str], list[str]]:
    """Generate rollouts for GRPO training, returning flattened lists."""
    
    # Extract questions and create prompts
    questions = [ex['problem'] for ex in question_batch]
    prompts = [prompt_template.format(question=q) for q in questions]
    
    # Generate rollouts using vLLM
    sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        max_tokens=config.sampling_max_tokens,
        min_tokens=config.sampling_min_tokens,
        n=group_size,  # Generate group_size responses per prompt
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=config.seed + step,
    )
    
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # Flatten results and create repeated structures
    rollout_responses = []
    repeated_prompts = []
    repeated_ground_truths = []
    
    for i, output in enumerate(outputs):
        for completion in output.outputs:
            rollout_responses.append(completion.text)
            repeated_prompts.append(prompts[i])
            repeated_ground_truths.append(question_batch[i]['answer'])
    
    return rollout_responses, repeated_prompts, repeated_ground_truths

def grpo_train_loop(config: GRPOConfig):
    assert config.train_batch_size % config.gradient_accumulation_steps == 0
    assert config.rollout_batch_size % config.group_size == 0
    assert config.train_batch_size >= config.group_size
    
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    num_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size
    num_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size

    logger.info("Micro Batch Size: %d", micro_train_batch_size)
    logger.info("Prompts per Rollout Batch: %d", num_prompts_per_rollout_batch)
    logger.info("Microbatches per Rollout Batch: %d", num_microbatches_per_rollout_batch)

    train_dataset = load_math_data(config.dataset_path)
    val_dataset = load_math_data(config.val_dataset_path)
    logger.info("Loaded %d training examples, %d validation examples", len(train_dataset), len(val_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    device_policy = "cuda:0" 
    device_vllm = "cuda:1" if num_gpus >= 2 else device_policy

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
    )
    policy_model = policy_model.to(device_policy)
    policy_model.train()

    model_engine = init_vllm(
        model_id=config.model_path,
        device=device_vllm,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization
    )
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas)

    metrics_history = []

    logger.info(f"Policy model device: {next(policy_model.parameters()).device}")
    logger.info(f"vLLM target device: {device_vllm}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


    experiment_name = f"use_std_normalization_false_grpo_{config.loss_type}_batch_{config.rollout_batch_size}_group_{config.group_size}_lr_{config.learning_rate}"
    wandb.init(
        project="cs336_alignment",
        name=experiment_name,
        config={
            "n_grpo_steps": config.n_grpo_steps,
            "learning_rate": config.learning_rate,
            "rollout_batch_size": config.rollout_batch_size,
            "group_size": config.group_size,
            "loss_type": config.loss_type,
            "use_std_normalization": config.use_std_normalization,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "cliprange": config.cliprange,
            "seed": config.seed,
            "log_every": config.log_every,
            "eval_every": config.eval_every,
            "eval_batch_size": config.eval_batch_size,
        }
    )
    
    wandb.define_metric("grpo_step")
    wandb.define_metric("grpo/*", step_metric="grpo_step")

    for step in tqdm(range(config.n_grpo_steps), desc="GRPO Steps"):
        torch.cuda.empty_cache()

        question_batch = np.random.choice(train_dataset, num_prompts_per_rollout_batch, replace=False)

        load_policy_into_vllm_instance(policy_model, model_engine)

        torch.cuda.empty_cache()

        rollout_responses, repeated_prompts, repeated_ground_truths = generate_grpo_rollouts(
            vllm_model=model_engine,
            question_batch=question_batch,
            prompt_template=prompt_template,
            group_size=config.group_size,
            config=config,
            step=step
        )

        torch.cuda.empty_cache()

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        advantages = advantages.to(device_policy)
        raw_rewards = raw_rewards.to(device_policy)

        policy_model.train()

        if config.loss_type == "grpo_clip":
            with torch.no_grad():
                old_policy_model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                ).to(device_policy) 
                old_policy_model.load_state_dict(policy_model.state_dict())
                old_policy_model.eval()

        for epoch in range(config.epochs_per_rollout_batch):

            full_responses = [prompt + response for prompt, response in zip(repeated_prompts, rollout_responses)]
            tokenized = tokenize_prompt_and_output(
                prompt_strs=repeated_prompts,
                output_strs=rollout_responses,
                tokenizer=tokenizer,
            )

            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)
            max_seq_len = response_mask.shape[1]

            # current_policy_outputs = get_response_log_probs(
            #     model=policy_model,
            #     input_ids=input_ids,
            #     labels=labels,
            #     return_token_entropy=True,
            # )

            # current_policy_log_probs = current_policy_outputs["log_probs"]
            # current_policy_token_entropy = current_policy_outputs["token_entropy"]

            # old_policy_log_probs = None
            # if config.loss_type == "grpo_clip":
            #     old_policy_outputs = get_response_log_probs(
            #         model=old_policy_model,
            #         input_ids=input_ids,
            #         labels=labels,
            #         return_token_entropy=False,
            #     )

            #     old_policy_log_probs = old_policy_outputs["log_probs"]

            old_policy_log_probs_all = None
            if config.loss_type == "grpo_clip":
                # Process old policy in chunks too
                old_policy_log_probs_all = []
                for microbatch_idx in range(num_microbatches_per_rollout_batch):
                    start_idx = microbatch_idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size
                    
                    with torch.no_grad():
                        old_outputs = get_response_log_probs(
                            model=old_policy_model,
                            input_ids=input_ids[start_idx:end_idx],
                            labels=labels[start_idx:end_idx],
                            return_token_entropy=False,
                        )
                        old_policy_log_probs_all.append(old_outputs["log_probs"])
                
                old_policy_log_probs_all = torch.cat(old_policy_log_probs_all, dim=0)

            total_loss = 0.0
            all_metadata = {}

            optimizer.zero_grad()

            all_token_entropies = []
            all_response_masks = []

            for microbatch_idx in range(num_microbatches_per_rollout_batch):
                
                start_idx = microbatch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size

                current_policy_outputs = get_response_log_probs(
                    model=policy_model,
                    input_ids=input_ids[start_idx:end_idx],    # Only 2 sequences
                    labels=labels[start_idx:end_idx],          # Only 2 sequences  
                    return_token_entropy=False,
                )

                microbatch_policy_log_probs = current_policy_outputs["log_probs"]
                # microbatch_token_entropy = current_policy_outputs["token_entropy"]
                microbatch_response_mask = response_mask[start_idx:end_idx]
                microbatch_advantages = advantages[start_idx:end_idx]
                microbatch_raw_rewards = raw_rewards[start_idx:end_idx]
                microbatch_old_log_probs = old_policy_log_probs_all[start_idx:end_idx] if old_policy_log_probs_all is not None else None

                # all_token_entropies.append(microbatch_token_entropy)
                all_response_masks.append(microbatch_response_mask)
    

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=microbatch_policy_log_probs,
                    response_mask=microbatch_response_mask,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    loss_type=config.loss_type,
                    raw_rewards=microbatch_raw_rewards.unsqueeze(1),
                    advantages=microbatch_advantages.unsqueeze(1),
                    old_log_probs=microbatch_old_log_probs,
                    cliprange=config.cliprange,
                    length_normalization_strategy=config.length_normalization_strategy,
                    # normalization_constant=max_seq_len,
                    normalization_constant=config.sampling_max_tokens,
                )

                total_loss += loss

                for key, value in metadata.items():
                    if key not in all_metadata:
                        all_metadata[key] = []
                    all_metadata[key].append(value.item() if torch.is_tensor(value) else value)

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip_value)
            optimizer.step()
                        
            # if all_token_entropies:
            #     combined_token_entropy = torch.cat(all_token_entropies, dim=0)
            #     combined_response_mask = torch.cat(all_response_masks, dim=0)
            #     average_token_entropy = masked_mean(combined_token_entropy, combined_response_mask, dim=None).item()
            # else:
            #     average_token_entropy = 0.0

            average_token_entropy = 0.0   

            if step % config.log_every == 0:
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "loss": total_loss / num_microbatches_per_rollout_batch,
                    "grad_norm": grad_norm.item(),
                    "token_entropy": average_token_entropy,
                    "train_reward_mean": reward_metadata["mean_raw_reward"],
                    "train_reward_std": reward_metadata["std_raw_reward"],
                    **{f"train_{k}": float(np.mean(v)) for k, v in all_metadata.items()},
                }
                logger.info(f"Step {step}: {metrics}")
                wandb.log({
                    "grpo/loss": metrics["loss"],
                    "grpo/grad_norm": metrics["grad_norm"],
                    "grpo/token_entropy": metrics["token_entropy"],
                    "grpo/train_reward_mean": metrics["train_reward_mean"],
                    "grpo/train_reward_std": metrics["train_reward_std"],
                    **{f"grpo/{k}": v for k, v in metrics.items() if k.startswith("train_")},
                    "grpo_step": step
                })
                            
                metrics_history.append(metrics)

        if step % config.eval_every == 0:
            logger.info("Running validation...")
            val_metrics = evaluate_model(
                policy_model,
                model_engine,
                val_dataset,
                tokenizer,
                device_policy,
                max_eval_examples=config.eval_batch_size,
            )
            logger.info(f"Validation: {val_metrics}")

            wandb.log({
                "grpo/val_accuracy": val_metrics["accuracy_statistics"]["accuracy"],
                "grpo/val_format_accuracy": val_metrics["accuracy_statistics"]["format_accuracy"],
                "grpo/val_avg_entropy": val_metrics["entropy_statistics"].get("mean_entropy", 0),
                "grpo_step": step
            })
            
            # Add validation metrics to last training metrics
            if metrics_history:
                metrics_history[-1]["validation_metrics"] = val_metrics
    
    
    logger.info("\nFinal comprehensive evaluation...")
    policy_model.eval()
    with torch.no_grad():
        final_eval = evaluate_model(
            policy_model=policy_model,
            vllm_model=model_engine,
            validation_data=val_dataset,
            tokenizer=tokenizer,
            device=device_policy,
            # No max_eval_examples = evaluate on full validation set
        )
    
    final_accuracy = final_eval["accuracy_statistics"]["accuracy"]
    logger.info(f"Final validation accuracy: {final_accuracy:.3f}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_accuracy": final_accuracy,
        "final_format_accuracy": final_eval["accuracy_statistics"]["format_accuracy"],
        "final_avg_entropy": final_eval["entropy_statistics"].get("mean_entropy", 0),
        "final_avg_response_length": final_eval["length_statistics"].get("avg_length_all", 0),
    })
    
    wandb.finish()

    return policy_model, metrics_history



def main(
    config_path: Optional[str] = None,
    output_dir: str = "outputs/grpo",
):
    if config_path:
        with open(config_path, "r") as f:
            config_dict = orjson.load(f)
        config = GRPOConfig(**config_dict)
    else:
        config = GRPOConfig()

    config.output_dir = output_dir

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    policy, metric_history = grpo_train_loop(config)

    torch.save(policy.state_dict(), f"{output_dir}/final_policy.pt")
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(make_json_serializable(metric_history), f, indent=2)    
    
    logger.info(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    typer.run(main)
