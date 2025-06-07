import torch
from typing import Dict, Tuple
from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    negative_log_likelihood = -policy_log_probs

    normalized_loss = masked_normalize(
        tensor=negative_log_likelihood,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )

    scaled_loss = normalized_loss / (gradient_accumulation_steps * len(policy_log_probs))

    scaled_loss.backward()

    meta_data = {
        "total_response_tokens": response_mask.sum(),
        "raw_loss": (negative_log_likelihood * response_mask).sum(),
    }

    return scaled_loss, meta_data