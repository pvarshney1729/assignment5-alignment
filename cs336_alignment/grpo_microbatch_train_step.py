import torch
from typing import Literal
from cs336_alignment.run_compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.masked_mean import masked_mean
from cs336_alignment.masked_normalize import masked_normalize

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "GRPO-No-Clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization_strategy: str = "mean",
    normalization_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    
    if length_normalization_strategy == "mean":
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    elif length_normalization_strategy == "normalize":
        per_example_loss = masked_normalize(
            per_token_loss, 
            response_mask, 
            dim=1,
            normalize_constant=normalization_constant
        )
    else:
        raise ValueError(f"Unknown length normalization strategy: {length_normalization_strategy}")

    batch_loss = torch.mean(per_example_loss) / gradient_accumulation_steps

    batch_loss.backward()

    meta_data = {
        "per_example_loss_mean": torch.mean(per_example_loss),
        "per_example_loss_std": torch.std(per_example_loss),
        "batch_loss": batch_loss,
        **loss_metadata,
    }

    return batch_loss, meta_data