from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import orjson

from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


class MATHSFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        return {
                "prompt": example["prompt"],
                "response": example["response"],
            }
    

def load_sft_data(filepath: str, max_examples: Optional[int] = None, filter_correct: bool = False) -> List[Dict[str, str]]:
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(orjson.loads(line.strip()))

    original_size = len(data)

    if filter_correct:
        print("Filtering for correct examples...")
        filtered_data = []

        for example in tqdm(data, desc="Filtering examples"):
            response = example["response"]
            ground_truth = example["ground_truth"]
            
            # Use the reward function to check if the response is correct
            reward_result = r1_zero_reward_fn(response, ground_truth)
            
            # Only keep examples where the answer is correct (reward > 0)
            if reward_result["reward"] > 0:
                filtered_data.append(example)

        data = filtered_data
        print(f"Filtered dataset: {len(data)} correct examples out of {original_size} total")


    if max_examples is not None:
        data = data[:max_examples]

    return data


def load_math_data(filepath: str) -> List[Dict[str, str]]:
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            examples.append(orjson.loads(line.strip()))

    return examples

def load_dataset(path: str, max_examples: Optional[int] = None):
    """Load JSONL dataset."""
    examples = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            examples.append(json.loads(line))
    return examples