import torch
from typing import Literal
from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss
from cs336_alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "GRPO-No-Clip", None],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided when loss_type='reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided when loss_type='grpo_clip'"
        assert old_log_probs is not None, "old_log_probs must be provided when loss_type='grpo_clip'"
        assert cliprange is not None, "cliprange must be provided when loss_type='grpo_clip'"
        
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )

    elif loss_type == "GRPO-No-Clip":
        assert advantages is not None, "advantages must be provided when loss_type='GRPO-No-Clip'"
        assert old_log_probs is not None, "old_log_probs must be provided when loss_type='GRPO-No-Clip'"

        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        loss = -ratio * advantages

        metadata = {"ratio": ratio}

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of ['no_baseline', 'reinforce_with_baseline', 'grpo_clip']")
    
    return loss, metadata