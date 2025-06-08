from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import torch
from vllm import LLM
from typing import List, Dict,Any
from vllm import SamplingParams
from cs336_alignment.log_generations import log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Optional

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching = True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_model(
    policy_model,
    vllm_model,
    validation_data: List[Dict[str, Any]],
    tokenizer,
    device: str,
    prompt_template: str,
    reward_function,
    max_eval_examples: Optional[int] = None
) -> Dict[str, float]:
    
    num_examples_to_eval = len(validation_data)
    if max_eval_examples is not None:
        num_examples_to_eval = min(len(validation_data), max_eval_examples)

    print(f"Evaluating on {num_examples_to_eval} examples...")

    load_policy_into_vllm_instance(policy_model, vllm_model)

    eval_examples = validation_data[:num_examples_to_eval]
    prompts = [prompt_template.format(question=ex['problem']) for ex in eval_examples]
    ground_truths = [example['answer'] for example in eval_examples]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    results = log_generations(
        model=vllm_model,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_function=reward_function,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_examples=num_examples_to_eval,
        policy_model=policy_model,
    )

    return results["aggregate_statistics"]


def generate_rollouts(
    vllm_model: LLM,
    questions: List[str],
    prompt_template: str,
    num_rollouts: int,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 1024,
    seed: int = 42
) -> List[List[str]]:
    
    prompts = [prompt_template.format(question=question) for question in questions]

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=4,
        n=num_rollouts,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    outputs = vllm_model.generate(prompts, sampling_params)

    rollouts = []
    for output in outputs:
        question_rollouts = [completion.text for completion in output.outputs]
        rollouts.append(question_rollouts)

    return rollouts