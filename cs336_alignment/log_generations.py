import torch
from typing import List, Dict, Any, Callable, Optional
from vllm import LLM, SamplingParams
import json
from collections import defaultdict

from cs336_alignment.compute_entropy import compute_entropy
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

def log_generations(
    model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    reward_function: Callable[[str, str], Dict[str, float]],
    tokenizer,
    sampling_params: Optional[SamplingParams] = None,
    max_examples: int = 10,
    log_file: Optional[str] = None,
    policy_model = None,
) -> Dict[str, Any]:
    
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

    num_examples = min(len(prompts), len(ground_truths), max_examples)
    selected_prompts = prompts[:num_examples]
    selected_ground_truths = ground_truths[:num_examples]

    print(f"Generating responses for {num_examples} examples...")

    outputs = model.generate(
        selected_prompts,
        sampling_params=sampling_params,
    )

    generated_responses = [output.outputs[0].text for output in outputs]

    detailed_logs = []
    reward_stats = defaultdict(list)
    length_stats = defaultdict(list)
    entropy_stats = []

    print("Evaluating responses and computing statistics...")

    for i, (prompt, response, ground_truth) in enumerate(zip(selected_prompts, generated_responses, selected_ground_truths)):
        rewards = reward_function(response, ground_truth)

        try:
            if policy_model is not None: # Check if policy_model is provided
                tokenized_result = tokenize_prompt_and_output([prompt], [response], tokenizer)
                # Move tensors to the device of the policy_model
                input_ids = tokenized_result["input_ids"].to(policy_model.device)
                labels = tokenized_result["labels"].to(policy_model.device)
                response_mask = tokenized_result["response_mask"].to(policy_model.device)

                with torch.no_grad():
                    model_outputs = get_response_log_probs(
                        model=policy_model,  # Use the passed HuggingFace policy_model
                        input_ids=input_ids, # Now on policy_model.device
                        labels=labels,       # Now on policy_model.device
                        return_token_entropy=True,
                    )

                token_entropy = model_outputs["token_entropy"]
                # Only compute entropy for response tokens
                response_entropies = token_entropy * response_mask
                response_entropy_sum = response_entropies.sum()
                response_token_count = response_mask.sum()
                
                if response_token_count > 0:
                    average_entropy = (response_entropy_sum / response_token_count).item()
                else:
                    average_entropy = 0.0
            else: # If policy_model is not provided, we cannot compute entropy this way
                average_entropy = 0.0
                # Optionally, print a warning
                #  if policy_model is expected but not provided
                print(f"Warning: policy_model not provided for entropy calculation for example {i}.")

        except Exception as e:
            print(f"Warning: Could not compute entropy for example {i}: {e}")
            average_entropy = 0.0

        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        response_length = len(response_tokens)

        log_entry = {
            "example_id": i,
            "prompt": prompt,
            "generated_response": response,
            "ground_truth": ground_truth,
            "rewards": rewards,
            "response_length": response_length,
            "average_token_entropy": average_entropy,
            "is_correct": rewards.get("answer_reward", 0.0) > 0.5,
            "is_properly_formatted": rewards.get("format_reward", 0.0) > 0.5,
        }

        detailed_logs.append(log_entry)

        for reward_type, reward_value in rewards.items():
            reward_stats[reward_type].append(reward_value)

        length_stats["all"].append(response_length)
        
        if log_entry["is_correct"]:
            length_stats["correct"].append(response_length)
        else:
            length_stats["incorrect"].append(response_length)
            
        if log_entry["is_properly_formatted"]:
            length_stats["properly_formatted"].append(response_length)
        else:
            length_stats["unformatted"].append(response_length)

        entropy_stats.append(average_entropy)

    aggregate_stats = {
        "num_examples": num_examples,
        "reward_statistics": {},
        "length_statistics": {},
        "entropy_statistics": {},
        "accuracy_statistics": {},
    }

    # Reward statistics
    for reward_type, values in reward_stats.items():
        aggregate_stats["reward_statistics"][reward_type] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    
    # Length statistics
    for length_type, lengths in length_stats.items():
        if lengths:
            aggregate_stats["length_statistics"][f"avg_length_{length_type}"] = sum(lengths) / len(lengths)
            aggregate_stats["length_statistics"][f"min_length_{length_type}"] = min(lengths)
            aggregate_stats["length_statistics"][f"max_length_{length_type}"] = max(lengths)
        else:
            aggregate_stats["length_statistics"][f"avg_length_{length_type}"] = 0.0
    
    # Entropy statistics
    if entropy_stats:
        aggregate_stats["entropy_statistics"] = {
            "mean_entropy": sum(entropy_stats) / len(entropy_stats),
            "min_entropy": min(entropy_stats),
            "max_entropy": max(entropy_stats),
        }
    
    # Accuracy statistics
    correct_count = sum(1 for log in detailed_logs if log["is_correct"])
    formatted_count = sum(1 for log in detailed_logs if log["is_properly_formatted"])
    
    aggregate_stats["accuracy_statistics"] = {
        "accuracy": correct_count / num_examples if num_examples > 0 else 0.0,
        "format_accuracy": formatted_count / num_examples if num_examples > 0 else 0.0,
        "correct_count": correct_count,
        "formatted_count": formatted_count,
    }
    
    # Print summary statistics
    print("\n" + "="*60)
    print("GENERATION STATISTICS SUMMARY")
    print("="*60)
    print(f"Examples processed: {num_examples}")
    print(f"Accuracy: {aggregate_stats['accuracy_statistics']['accuracy']:.3f}")
    print(f"Format accuracy: {aggregate_stats['accuracy_statistics']['format_accuracy']:.3f}")
    print(f"Average response length: {aggregate_stats['length_statistics'].get('avg_length_all', 0):.1f} tokens")
    print(f"Average entropy: {aggregate_stats['entropy_statistics'].get('mean_entropy', 0):.3f}")
    
    # Show some example generations
    print("\nSample generations:")
    for i, log in enumerate(detailed_logs[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {log['prompt']}...")
        print(f"Response: {log['generated_response']}...")
        print(f"Ground truth: {log['ground_truth']}")
        print(f"Correct: {log['is_correct']}, Formatted: {log['is_properly_formatted']}")
        print(f"Rewards: {log['rewards']}")
    
    # Save detailed logs if requested
    if log_file:
        with open(log_file, 'w') as f:
            json.dump({
                "aggregate_statistics": aggregate_stats,
                "detailed_logs": detailed_logs
            }, f, indent=2)
        print(f"\nDetailed logs saved to: {log_file}")
    
    return {
        "aggregate_statistics": aggregate_stats,
        "detailed_logs": detailed_logs
    }

