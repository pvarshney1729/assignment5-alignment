import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Dict

from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        logs_prob_all = F.log_softmax(logits, dim=-1)
        
        labels_expanded = labels.unsqueeze(-1)

        log_probs = torch.gather(logs_prob_all, dim=-1, index=labels_expanded)

        log_probs = log_probs.squeeze(-1)

        result = {
            "log_probs": log_probs
        }

        if return_token_entropy:
            token_entropy = compute_entropy(logits)
            result["token_entropy"] = token_entropy

        return result