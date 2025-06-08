import typer
import orjson
import json
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from pathlib import Path
from dataclasses import asdict

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
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
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
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 2 # On-policy
    train_batch_size: int = 64 # On-policy
    gradient_accumulation_steps: int = 32 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.2
    loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    "GRPO-No-Clip",
    ] = "grpo_clip"
    use_std_normalization: bool = False

    cliprange: float = 0.2
    grad_clip_value: float = 1.0

    log_every: int = 5
    eval_every: int = 5
    eval_batch_size: int = 512
    # eval_batch_size: int = None

    model_path: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    dataset_path: str = "/data/a5-alignment/MATH/train.jsonl"
    val_dataset_path: str = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir: str = "/data/a5-alignment/outputs/grpo"

    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)

    seed: int = 42

    length_normalization_strategy: Literal["mean", "normalize"] = "mean"
    # prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    # reward_fn_name: Literal["r1_zero_reward_fn", "question_only_reward_fn"] = "r1_zero_reward_fn"
    prompt_path: str = "cs336_alignment/prompts/question_only.prompt"
    reward_fn_name: Literal["r1_zero_reward_fn", "question_only_reward_fn"] = "question_only_reward_fn"


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

    with open(config.prompt_path, 'r') as f:
        prompt_template = f.read()

    if config.reward_fn_name == "r1_zero_reward_fn":
        reward_fn = r1_zero_reward_fn
    elif config.reward_fn_name == "question_only_reward_fn":
        reward_fn = question_only_reward_fn
    else:
        raise ValueError(f"Unknown reward_fn_name: {config.reward_fn_name}")

    print("Inside GRPO train loop")
    print(json.dumps(asdict(config), indent=2))
    
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
    # device_vllm = "cuda:1" if num_gpus >= 2 else device_policy
    device_vllm = "cuda:0" 

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


    experiment_name = f"train_batch_size_{config.train_batch_size}_epochs_per_rollout_batch_{config.epochs_per_rollout_batch}_lr_{config.learning_rate}_grpo"
    wandb.init(
        project="cs336_alignment",
        name=experiment_name,
        config=asdict(config)
    )
    
    wandb.define_metric("grpo_step")
    wandb.define_metric("grpo/*", step_metric="grpo_step")

    step = 0
    epoch_num = 0
    dataset_indices = np.arange(len(train_dataset))

    pbar = tqdm(total=config.n_grpo_steps, desc="GRPO Steps")

    while step < config.n_grpo_steps:
        logger.info(f"Starting Epoch {epoch_num+1}...")
        np.random.shuffle(dataset_indices)

        for i in range(0, len(dataset_indices), num_prompts_per_rollout_batch):
            if step >= config.n_grpo_steps:
                break 

            batch_indices = dataset_indices[i : i + num_prompts_per_rollout_batch]

            # Drop the last batch if it's smaller than the configured size
            if len(batch_indices) < num_prompts_per_rollout_batch:
                logger.info(f"Skipping incomplete batch of size {len(batch_indices)}.")
                continue

            question_batch = [train_dataset[idx] for idx in batch_indices]

            torch.cuda.empty_cache()

            load_policy_into_vllm_instance(policy_model, model_engine)

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
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=config.group_size,
                advantage_eps=config.advantage_eps,
                normalize_by_std=config.use_std_normalization,
            )

            advantages = advantages.to(device_policy)
            raw_rewards = raw_rewards.to(device_policy)

            policy_model.train()

            # if config.loss_type == "grpo_clip":
            #     with torch.no_grad():
            #         old_policy_model = AutoModelForCausalLM.from_pretrained(
            #             config.model_path,
            #             torch_dtype=torch.bfloat16,
            #             attn_implementation="flash_attention_2"
            #         ).to(device_policy) 
            #         old_policy_model.load_state_dict(policy_model.state_dict())
            #         old_policy_model.eval()

            full_responses = [prompt + response for prompt, response in zip(repeated_prompts, rollout_responses)]
            tokenized = tokenize_prompt_and_output(
                prompt_strs=repeated_prompts,
                output_strs=rollout_responses,
                tokenizer=tokenizer,
            )

            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)

            old_policy_log_probs_all = None
            if config.loss_type in ["grpo_clip", "GRPO-No-Clip"]:
                logger.info(f"Computing old log probabilities for {config.loss_type}...")
                old_log_probs_list = []
                policy_model.eval()  # Set to eval mode for inference
                with torch.inference_mode():
                    for microbatch_idx in range(num_microbatches_per_rollout_batch):
                        start_idx = microbatch_idx * micro_train_batch_size
                        end_idx = start_idx + micro_train_batch_size
                        
                        old_outputs = get_response_log_probs(
                            model=policy_model,
                            input_ids=input_ids[start_idx:end_idx],
                            labels=labels[start_idx:end_idx],
                            return_token_entropy=False,
                        )
                        old_log_probs_list.append(old_outputs["log_probs"])
                
                old_policy_log_probs_all = torch.cat(old_log_probs_list, dim=0)
                policy_model.train()  # Set back to train mode for updates
                logger.info("Finished computing old log probabilities.")



            for epoch in range(config.epochs_per_rollout_batch):

                total_loss = 0.0
                all_metadata = {}

                optimizer.zero_grad()

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
                        if torch.is_tensor(value):
                            # For logging, we want a single scalar. If the metadata is a
                            # tensor (e.g., per-token values), log its mean.
                            if value.dtype == torch.bool:
                                all_metadata[key].append(value.float().mean().item())
                            else:
                                all_metadata[key].append(value.mean().item())
                        else:
                            all_metadata[key].append(value)

                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip_value)
                optimizer.step()

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
                    prompt_template,
                    reward_fn,
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

            step += 1
            pbar.update(1)

        epoch_num += 1

    pbar.close()
        
    
    logger.info("\nFinal comprehensive evaluation...")
    policy_model.eval()
    with torch.no_grad():
        final_eval = evaluate_model(
            policy_model=policy_model,
            vllm_model=model_engine,
            validation_data=val_dataset,
            tokenizer=tokenizer,
            device=device_policy,
            prompt_template=prompt_template,
            reward_fn=reward_fn,
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
            config_dict = json.load(f)
        config = GRPOConfig(**config_dict)
    else:
        config = GRPOConfig()

    print(json.dumps(asdict(config), indent=2))

    config.output_dir = output_dir

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    policy, metric_history = grpo_train_loop(config)

    torch.save(policy.state_dict(), f"{output_dir}/final_policy.pt")
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(make_json_serializable(metric_history), f, indent=2)    
    
    logger.info(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    typer.run(main)
