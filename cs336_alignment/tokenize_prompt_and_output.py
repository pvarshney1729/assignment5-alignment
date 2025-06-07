from typing import List, Dict
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Tensor]:
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size

    all_full_sequences = []
    all_prompt_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)

        full_sequence = prompt_tokens + output_tokens

        all_full_sequences.append(full_sequence)
        all_prompt_lens.append(len(prompt_tokens))

        
    max_length = max(len(sequence) for sequence in all_full_sequences)
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_input_ids = []
    padded_labels = []
    padded_response_masks = []

    for full_sequence, prompt_len in zip(all_full_sequences, all_prompt_lens):
        padded_full_sequence = full_sequence + [pad_token] * (max_length - len(full_sequence))

        input_ids = padded_full_sequence[:-1]
        labels = padded_full_sequence[1:]

        response_mask = [0] * (prompt_len - 1) + [1] * (len(full_sequence) - prompt_len) + [0] * (max_length - len(full_sequence))
        response_mask = response_mask[:len(labels)]

        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_response_masks.append(response_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "labels": torch.tensor(padded_labels),
        "response_mask": torch.tensor(padded_response_masks),
    }