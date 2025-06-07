import json
import os
from pathlib import Path
from typing import Callable, List, Dict, Any
from collections import Counter

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_math_validation_data(dataset_path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def format_prompts(examples: List[Dict[str, Any]], prompt_template: str) -> List[str]:
    prompts = []
    for example in examples:
        formatted_prompt = prompt_template.format(
            question=example['problem'],
        )
        prompts.append(formatted_prompt)
    return prompts

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str = "math_baseline_results.json",
) -> Dict[str, Any]:
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)

    results = []
    rewards_Counter = Counter()
    
    for i, (prompt, response, ground_truth) in enumerate(zip(prompts, generated_texts, ground_truths)):
        scores = reward_fn(response, ground_truth)

        result = {
            "index": i,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "scores": scores,
        }
        results.append(result)
        
        format_reward = scores['format_reward']
        answer_reward = scores['answer_reward']

        if format_reward == 1 and answer_reward == 1:
            rewards_Counter['both_correct'] += 1
        elif format_reward == 1 and answer_reward == 0:
            rewards_Counter['format_only_correct'] += 1
        elif format_reward == 0 and answer_reward == 1:
            rewards_Counter['answer_only_correct'] += 1
        else:
            rewards_Counter['neither_correct'] += 1

    total_examples = len(results)
    avg_reward = sum(result['scores']['reward'] for result in results) / total_examples
    avg_format_reward = sum(result['scores']['format_reward'] for result in results) / total_examples
    avg_answer_reward = sum(result['scores']['answer_reward'] for result in results) / total_examples

    evaluation_results = {
        "total_examples": total_examples,
        "avg_reward": avg_reward,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
        "reward_categories": dict(rewards_Counter),
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    return evaluation_results

def print_evaluation_summary(evaluation_results: Dict[str, Any]):
    total = evaluation_results['total_examples']
    categories = evaluation_results['reward_categories']

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total examples: {total}")
    print(f"Average reward: {evaluation_results['avg_reward']:.4f}")
    print(f"Average format reward: {evaluation_results['avg_format_reward']:.4f}")
    print(f"Average answer reward: {evaluation_results['avg_answer_reward']:.4f}")
    print()
    print("Reward Categories:")
    print(f"  (1) Correct format AND answer (reward=1): {categories.get('correct_both', 0)}")
    print(f"  (2) Correct format, wrong answer (format=1, answer=0): {categories.get('format_only', 0)}")
    print(f"  (3) Wrong format AND answer (format=0, answer=0): {categories.get('neither', 0)}")
    print()

    results = evaluation_results["results"]
    
    # Find examples for each category
    correct_both_examples = [r for r in results if r["scores"]["format_reward"] == 1.0 and r["scores"]["answer_reward"] == 1.0]
    format_only_examples = [r for r in results if r["scores"]["format_reward"] == 1.0 and r["scores"]["answer_reward"] == 0.0]
    neither_examples = [r for r in results if r["scores"]["format_reward"] == 0.0 and r["scores"]["answer_reward"] == 0.0]
    
    def show_examples(examples, category_name, max_examples=3):
        if examples:
            print(f"\nExample(s) from {category_name} (showing up to {max_examples}):")
            for i, ex in enumerate(examples[:max_examples]):
                print(f"\n--- Example {i+1} ---")
                print(f"Ground truth: {ex['ground_truth']}")
                print(f"Response: {ex['response'][:200]}{'...' if len(ex['response']) > 200 else ''}")
                print(f"Scores: {ex['scores']}")
    
    show_examples(correct_both_examples, "Correct Format AND Answer")
    show_examples(format_only_examples, "Correct Format, Wrong Answer")
    show_examples(neither_examples, "Wrong Format AND Answer")

def main():
    MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    DATASET_PATH = "/data/a5-alignment/MATH/validation.jsonl"
    PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
    OUTPUT_FILE = "math_baseline_results.json"

    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset file not found at", DATASET_PATH)
        return
    
    if not os.path.exists(PROMPT_PATH):
        print("Error: Prompt file not found at", PROMPT_PATH)
        return
    
    print("Loading MATH validation dataset...")
    examples = load_math_validation_data(DATASET_PATH)
    print(f"Loaded {len(examples)} examples")

    print(f"Sample example:\n{examples[0]}")
    # exit()

    print("Loading r1_zero prompt template...")
    with open(PROMPT_PATH, 'r') as f:
        prompt_template = f.read()

    print("Formatting prompts...")
    prompts = format_prompts(examples, prompt_template)

    ground_truths = [example['answer'] for example in examples]

    print("Initializing VLLM model...")
    vllm_model = LLM(model=MODEL_PATH, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print("Evaluating VLLM model...")
    evaluation_results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, ground_truths, sampling_params, OUTPUT_FILE)

    print_evaluation_summary(evaluation_results)

if __name__ == "__main__":
    main()