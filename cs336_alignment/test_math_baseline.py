#!/usr/bin/env python3
"""
Test script for math baseline evaluation with a small subset of data.
"""

import json
import tempfile
from cs336_alignment.math_baseline import load_math_validation_data, format_prompts, evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


def create_sample_math_data():
    """Create a small sample MATH dataset for testing."""
    sample_data = [
        {
            "question": "What is 2 + 2?",
            "solution": "4"
        },
        {
            "question": "Find the value of x if 3x = 12.",
            "solution": "4"
        },
        {
            "question": "What is the square root of 16?",
            "solution": "4"
        }
    ]
    return sample_data


def test_with_sample_data():
    """Test the evaluation pipeline with sample data."""
    print("Testing math baseline evaluation with sample data...")
    
    # Create sample data
    sample_examples = create_sample_math_data()
    
    # Load prompt template
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()
    
    prompts = format_prompts(sample_examples, prompt_template)
    
    ground_truths = [example['solution'] for example in sample_examples]
    
    print(f"Sample prompt:\n{prompts[0]}")
    print(f"\nGround truth: {ground_truths[0]}")
    
    # Test with a small model path (you might need to adjust this)
    MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    
    try:
        print(f"Initializing vLLM with model: {MODEL_PATH}")
        llm = LLM(model=MODEL_PATH)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
        
        # Run evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            evaluation_results = evaluate_vllm(
                vllm_model=llm,
                reward_fn=r1_zero_reward_fn,
                prompts=prompts,
                ground_truths=ground_truths,
                eval_sampling_params=sampling_params,
                output_file=tmp_file.name
            )
        
        print("Test completed successfully!")
        print(f"Results: {json.dumps(evaluation_results, indent=4)}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        print("This might be expected if running outside the cluster environment.")


if __name__ == "__main__":
    test_with_sample_data()