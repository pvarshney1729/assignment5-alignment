import torch

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    term_1 = ratio * advantages
    term_2 = clipped_ratio * advantages
    
    loss = -torch.min(term_1, term_2)

    is_clipped = term_2 < term_1
    
    metadata = {
        "is_clipped": is_clipped,                           # Per-token clipping indicator
        "clip_fraction": torch.mean(is_clipped.float()),    # Fraction of tokens clipped
        "mean_ratio": torch.mean(ratio),                    # Average policy ratio
        "mean_clipped_ratio": torch.mean(clipped_ratio),    # Average clipped ratio
    }
    
    return loss, metadata