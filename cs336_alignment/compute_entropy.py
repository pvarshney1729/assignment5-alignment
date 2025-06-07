import torch
import torch.nn.functional as F

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = - (probs * log_probs).sum(dim=-1)
    return entropy