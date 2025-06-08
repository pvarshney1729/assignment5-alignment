import argparse
import torch
import numpy as np
import random
import wandb
from cs336_alignment.data_utils import load_math_data
from vllm import LLM, SamplingParams
from typing import List, Dict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from cs336_alignment.data_utils import load_sft_data, load_math_data, MATHSFTDataset
from tqdm import tqdm

from cs336_alignment.vllm_utils import init_vllm, load_policy_into_vllm_instance, evaluate_model

from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_experiment import sft_microbatch_train_step

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_utils import init_vllm, load_policy_into_vllm_instance, evaluate_model, generate_rollouts


def filter_correct_rollouts(
    questions: List[str],
    rollouts: List[List[str]],
    ground_truth_answers: List[str],
    reward_function,
    prompt_template: str,
) -> List[Dict[str, str]]:
    correct_examples = []

    for question, question_rollouts, ground_truth_answer in zip(questions, rollouts, ground_truth_answers):
        
        formatted_prompt = prompt_template.format(question=question)
        
        for rollout in question_rollouts:

            reward = reward_function(rollout, ground_truth_answer)
            if reward["reward"] > 0:
                correct_examples.append({
                    "prompt": formatted_prompt,
                    "response": rollout,
                    "reward": reward,
                })

                # print(f"correct_examples:")
                # import json
                # print(json.dumps(correct_examples, indent=4))
                # exit()
    
    return correct_examples


def train_sft_step(
    model,
    sft_data: List[Dict[str, str]],
    tokenizer,
    device: str,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Run SFT on filtered data for one or more epochs."""
    
    if len(sft_data) == 0:
        print("Warning: No correct examples found for SFT step!")
        return {"average_loss": 0.0}
    
    print(f"Running SFT step on {len(sft_data)} correct examples for {num_epochs} epochs...")
    
    # Create dataset and dataloader
    dataset = MATHSFTDataset(sft_data)
    
    def sft_collate_fn(batch):
        prompt_strs = [item["prompt"] for item in batch]
        output_strs = [item["response"] for item in batch]
        
        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=prompt_strs,
            output_strs=output_strs,
            tokenizer=tokenizer
        )
        
        return {
            "input_ids": tokenized_batch["input_ids"],
            "labels": tokenized_batch["labels"],
            "response_mask": tokenized_batch["response_mask"],
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"SFT Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            response_mask = batch['response_mask'].to(device)
            
            log_probs_results = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            
            log_probs = log_probs_results["log_probs"]
            
            loss, meta_data = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
    
    return {"average_loss": total_loss / num_batches if num_batches > 0 else 0.0}

    
def run_expert_iteration_experiment(
    num_rollouts: int = 8,
    num_epochs_per_step: int = 1,
    num_expiter_steps: int = 5,
    batch_size_rollouts: int = 512,
    learning_rate: float = 5e-5,
    batch_size_sft: int = 8,
    gradient_accumulation_steps: int = 4,
    device_policy: str = "cuda:0",
    device_vllm: str = "cuda:0",
    seed: int = 42,
    save_dir: str = "/data/c-vprateek/expert_iteration_models",
):

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available. Setting vLLM device to {device_policy}")
        device_vllm = device_policy

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    experiment_name = f"expert_iteration_batch_size_{batch_size_rollouts}_rollouts_{num_rollouts}_epochs_{num_epochs_per_step}"
    wandb.init(
        project="cs336_alignment",
        name=experiment_name,
        config={
            "num_rollouts": num_rollouts,
            "num_epochs_per_step": num_epochs_per_step,
            "num_expiter_steps": num_expiter_steps,
            "batch_size_rollouts": batch_size_rollouts,
            "learning_rate": learning_rate,
            "batch_size_sft": batch_size_sft,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "seed": seed,
        }
    )

    wandb.define_metric("expert_iteration_step")
    wandb.define_metric("expert_iteration/*", step_metric="expert_iteration_step")

    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    print("Loading initial model and tokenizer...")

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_policy
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading training and validation data...")
    train_data = load_math_data("/data/a5-alignment/MATH/train.jsonl")
    validation_data = load_math_data("/data/a5-alignment/MATH/validation.jsonl")

    print(f"Training dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(validation_data)}")
    
    with open("cs336_alignment/prompts/r1_zero.prompt", 'r') as f:
        prompt_template = f.read()

    print("Initializing vLLM...")
    vllm_model = init_vllm(model_path, device_vllm, seed, gpu_memory_utilization=0.2)

    print("Initial evaluation...")
    policy_model.eval()
    with torch.no_grad():
        initial_eval = evaluate_model(
            policy_model=policy_model,
            vllm_model=vllm_model,
            validation_data=validation_data,
            tokenizer=tokenizer,
            device=device_policy,
            max_eval_examples=512
        )
    
    print(f"Initial validation accuracy: {initial_eval['accuracy_statistics']['accuracy']:.3f}")

    wandb.log({
        "expert_iteration/accuracy": initial_eval["accuracy_statistics"]["accuracy"],
        "expert_iteration/format_accuracy": initial_eval["accuracy_statistics"]["format_accuracy"],
        "expert_iteration/avg_entropy": initial_eval["entropy_statistics"].get("mean_entropy", 0),
        "expert_iteration/num_correct_examples": 0,  # No rollouts yet
        "expert_iteration_step": 0
    })

    for expiter_step in range(1, num_expiter_steps + 1):
        print(f"\n{'='*80}")
        print(f"Expert Iteration Step {expiter_step}/{num_expiter_steps}")
        print('='*80)

        torch.cuda.empty_cache()
        

        rollout_questions = random.sample(train_data, min(batch_size_rollouts, len(train_data)))
        questions = [example['problem'] for example in rollout_questions]
        ground_truth_answers = [example['answer'] for example in rollout_questions]

        print("Loading current policy into vLLM...")
        load_policy_into_vllm_instance(policy_model, vllm_model)

        print(f"Generating {num_rollouts} rollouts per question...")
        rollouts = generate_rollouts(
            vllm_model=vllm_model,
            questions=questions,
            prompt_template=prompt_template,
            num_rollouts=num_rollouts,
            seed=seed+expiter_step,
        )

        print("Releasing VLLM to free memory for SFT...")
        del vllm_model
        torch.cuda.empty_cache()

        print("Filtering correct rollouts...")
        correct_examples = filter_correct_rollouts(
            questions=questions,
            rollouts=rollouts,
            ground_truth_answers=ground_truth_answers,
            reward_function=r1_zero_reward_fn,
            prompt_template=prompt_template,
        )

        print(f"Found {len(correct_examples)} correct rollouts")

        if len(correct_examples) == 0:
            print("Warning: No correct examples found! Skipping SFT step.")
            continue

        print("Running SFT...")
        sft_stats = train_sft_step(
            model=policy_model,
            sft_data=correct_examples,
            tokenizer=tokenizer,
            device=device_policy,
            num_epochs=num_epochs_per_step,
            learning_rate=learning_rate,
            batch_size=batch_size_sft,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        torch.cuda.empty_cache()

        print("Re-initializing VLLM for evaluation...")
        vllm_model = init_vllm(model_path, device_vllm, seed, gpu_memory_utilization=0.7)

        torch.cuda.empty_cache()

        print("Evaluating updated policy...")
        policy_model.eval()
        with torch.no_grad():
            eval_stats = evaluate_model(
                policy_model=policy_model,
                vllm_model=vllm_model,
                validation_data=validation_data,
                tokenizer=tokenizer,
                device=device_policy,
                max_eval_examples=512
            )
        
        current_accuracy = eval_stats["accuracy_statistics"]["accuracy"]
        print(f"Validation accuracy after EI step {expiter_step}: {current_accuracy:.3f}")
        
        wandb.log({
            "expert_iteration/accuracy": current_accuracy,
            "expert_iteration/format_accuracy": eval_stats["accuracy_statistics"]["format_accuracy"],
            "expert_iteration/avg_entropy": eval_stats["entropy_statistics"].get("mean_entropy", 0),
            "expert_iteration/sft_loss": sft_stats["average_loss"],
            "expert_iteration/num_correct_examples": len(correct_examples),
            "expert_iteration/rollout_success_rate": len(correct_examples) / (len(questions) * num_rollouts),
            "expert_iteration_step": expiter_step
        })
        
        checkpoint_dir = os.path.join(save_dir, experiment_name, f"step_{expiter_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        policy_model.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

    print("\nFinal comprehensive evaluation...")
    policy_model.eval()
    with torch.no_grad():
        final_eval = evaluate_model(
            policy_model=policy_model,
            vllm_model=vllm_model,
            validation_data=validation_data,
            tokenizer=tokenizer,
            device=device_policy,
            max_eval_examples=None
        )

    final_accuracy = final_eval["accuracy_statistics"]["accuracy"]
    print(f"Final validation accuracy: {final_accuracy:.3f}")
    
    wandb.log({
        "final_accuracy": final_accuracy,
        "final_format_accuracy": final_eval["accuracy_statistics"]["format_accuracy"],
        "final_avg_entropy": final_eval["entropy_statistics"].get("mean_entropy", 0)
    })
    
    # Save final model
    final_dir = os.path.join(save_dir, experiment_name, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")
    
    wandb.finish()
    return final_accuracy


def main():
    parser = argparse.ArgumentParser(description="Run Expert Iteration experiment on MATH dataset")
    parser.add_argument("--rollout_configs", nargs="+", type=int, default=[4, 8], 
                       help="Different rollout counts to try")
    parser.add_argument("--epoch_configs", nargs="+", type=int, default=[1, 2], 
                       help="Different epoch counts per Expert Iteration step to try")
    parser.add_argument("--num_expiter_steps", type=int, default=5, help="Number of Expert Iteration steps")
    parser.add_argument("--batch_size_rollouts", nargs="+", type=int, default=[512], help="Batch size for rollouts")  # Fix: Default to 512
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for SFT steps")
    parser.add_argument("--batch_size_sft", type=int, default=8, help="Batch size for SFT")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_dir", type=str, default="/data/c-vprateek/ei_models", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = {}

    for batch_size in args.batch_size_rollouts:
        for rollouts in args.rollout_configs:
            for epochs in args.epoch_configs:
                config_name = f"batch_size_{batch_size}_rollouts_{rollouts}_epochs_{epochs}"
                print(f"\n{'='*100}")
                print(f"Running Expert Iteration: batch_size {batch_size}, {rollouts} rollouts, {epochs} epochs per step")
                print('='*80)

                save_directory = args.save_dir + f"/batch_size_{batch_size}_rollouts_{rollouts}_epochs_{epochs}"

                accuracy = run_expert_iteration_experiment(
                    num_rollouts=rollouts,
                    num_epochs_per_step=epochs,
                    num_expiter_steps=args.num_expiter_steps,
                    batch_size_rollouts = batch_size,
                    learning_rate = args.learning_rate,
                    batch_size_sft = args.batch_size_sft,
                    gradient_accumulation_steps = args.gradient_accumulation_steps,
                    save_dir = save_directory,
                    seed = args.seed,
                )

                results[config_name] = accuracy

    print(f"\n{'='*80}")
    print("EXPERT ITERATION EXPERIMENT SUMMARY")
    print('='*80)
    for config_name, accuracy in results.items():
        print(f"{config_name}: {accuracy:.3f}")
        if accuracy >= 0.15:
            print(f"Achieved target accuracy of ≥15%")


if __name__ == "__main__":
    main()