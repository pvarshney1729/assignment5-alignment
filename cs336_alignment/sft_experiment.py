import torch
from typing import Dict, Optional
from tqdm import tqdm
import wandb
import random
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
import argparse
import os   

from cs336_alignment.data_utils import load_sft_data, load_math_data, MATHSFTDataset
from cs336_alignment.vllm_utils import init_vllm, load_policy_into_vllm_instance, evaluate_model


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
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

            if (batch_idx + 1) % (log_interval * gradient_accumulation_steps) == 0:
                avg_loss = total_loss / num_batches
                print(f"Batch {batch_idx + 1}, Average Loss: {avg_loss:.4f}")
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train_step": batch_idx + 1
                })

    return {"average_loss": total_loss / num_batches if num_batches > 0 else 0.0}


def run_sft_experiment(
    dataset_size: Optional[int] = None,
    filter_correct: bool = False,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    eval_interval: int = 500,
    save_dir: str = "/data/yourusername/sft_models",
    device_policy: str = "cuda:0",
    device_vllm: str = "cuda:1",
    seed: int = 42
):
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available. Setting vLLM device to {device_policy}")
        device_vllm = device_policy

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    experiment_name = f"sft_size_{dataset_size}_filter_{filter_correct}"
    wandb.init(
        project="cs336_alignment",
        name=experiment_name,
        config={
            "dataset_size": dataset_size,
            "filter_correct": filter_correct,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "seed": seed
        }
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step") 
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    print(f"Running SFT experiment: dataset_size={dataset_size}, filter_correct={filter_correct}")

    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    print("Loading model and tokenizer...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_policy 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("Loading training data...")
    sft_data = load_sft_data(
        "/data/a5-alignment/MATH/sft.jsonl",
        max_examples=dataset_size,
        filter_correct=filter_correct
    )
    print(f"Training dataset size: {len(sft_data)}")
    
    print("Loading validation data...")
    validation_data = load_math_data("/data/a5-alignment/MATH/validation.jsonl")
    print(f"Validation dataset size: {len(validation_data)}")
    
    # Create dataset and dataloader
    train_dataset = MATHSFTDataset(sft_data)

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
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    
    # Initialize vLLM for evaluation
    print("Initializing vLLM...")
    vllm_model = init_vllm(model_path, device_vllm, seed)
    
    # Training loop
    print("Starting training...")
    eval_step = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        epoch_stats = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device_policy,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Evaluate at intervals
        if (epoch + 1) % max(1, num_epochs // 3) == 0:
            print("\nRunning evaluation...")
            model.eval()
            with torch.no_grad():
                eval_stats = evaluate_model(
                    policy_model=model,
                    vllm_model=vllm_model,
                    validation_data=validation_data,
                    tokenizer=tokenizer,
                    device=device_policy,
                    max_eval_examples=100
                )
            
            # Log evaluation metrics
            wandb.log({
                "eval/accuracy": eval_stats["accuracy_statistics"]["accuracy"],
                "eval/format_accuracy": eval_stats["accuracy_statistics"]["format_accuracy"],
                "eval/avg_response_length": eval_stats["length_statistics"].get("avg_length_all", 0),
                "eval/avg_entropy": eval_stats["entropy_statistics"].get("mean_entropy", 0),
                "eval_step": eval_step
            })
            eval_step += 1
            
            print(f"Validation Accuracy: {eval_stats['accuracy_statistics']['accuracy']:.3f}")
            print(f"Format Accuracy: {eval_stats['accuracy_statistics']['format_accuracy']:.3f}")
    
    # Save final model
    output_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    with torch.no_grad():
        final_eval_stats = evaluate_model(
            policy_model=model,
            vllm_model=vllm_model,
            validation_data=validation_data,
            tokenizer=tokenizer,
            device=device_policy,
            max_eval_examples=200
        )
    
    final_accuracy = final_eval_stats["accuracy_statistics"]["accuracy"]
    print(f"Final Validation Accuracy: {final_accuracy:.3f}")
    
    wandb.log({
        "final_accuracy": final_accuracy,
        "final_format_accuracy": final_eval_stats["accuracy_statistics"]["format_accuracy"]
    })
    
    wandb.finish()
    return final_accuracy



def main():
    parser = argparse.ArgumentParser(description="Run SFT experiment on MATH dataset")
    parser.add_argument("--dataset_sizes", nargs="+", type=int, default=None, 
                       help="Dataset sizes to experiment with")
    parser.add_argument("--run_full", action="store_true", help="Also run with full dataset")
    parser.add_argument("--run_filtered", action="store_true", help="Run with filtered correct examples")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="/data/yourusername/sft_models", help="Save directory")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.dataset_sizes is not None:
        # Run experiments with different dataset sizes
        for size in args.dataset_sizes:
            print(f"\n{'='*60}")
            print(f"Running experiment with dataset size: {size}")
            print('='*60)
            
            accuracy = run_sft_experiment(
                dataset_size=size,
                filter_correct=False,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=args.num_epochs,
                save_dir=args.save_dir
            )
            results[f"size_{size}"] = accuracy
        
    # Run with full dataset
    if args.run_full:
        print(f"\n{'='*60}")
        print("Running experiment with full dataset")
        print('='*60)
        
        accuracy = run_sft_experiment(
            dataset_size=None,
            filter_correct=False,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir
        )
        results["full_dataset"] = accuracy
    
    # Run with filtered dataset
    if args.run_filtered:
        print(f"\n{'='*60}")
        print("Running experiment with filtered correct examples")
        print('='*60)
        
        accuracy = run_sft_experiment(
            dataset_size=None,
            filter_correct=True,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir
        )
        results["filtered_correct"] = accuracy
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print('='*60)
    for exp_name, accuracy in results.items():
        print(f"{exp_name}: {accuracy:.3f}")


if __name__ == "__main__":
    main()