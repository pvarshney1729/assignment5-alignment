import torch
import numpy as np
from typing import Callable, List, Dict, Tuple, Optional

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
    
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    
    n_groups = len(rollout_responses) // group_size
    advantages = torch.zeros_like(raw_rewards)
    
    group_means = []
    group_stds = []
    
    for group_idx in range(n_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        
        group_rewards = raw_rewards[start_idx:end_idx]
        
        group_mean = torch.mean(group_rewards)
        group_std = torch.std(group_rewards)  # Use population std
        
        group_means.append(group_mean.item())
        group_stds.append(group_std.item())
        
        if normalize_by_std:
            group_advantages = (group_rewards - group_mean) / (group_std + advantage_eps)
        else:
            group_advantages = group_rewards - group_mean
            
        advantages[start_idx:end_idx] = group_advantages
    
    metadata = {
        "mean_raw_reward": torch.mean(raw_rewards).item(),
        "std_raw_reward": torch.std(raw_rewards).item(),
        "max_raw_reward": torch.max(raw_rewards).item(),
        "min_raw_reward": torch.min(raw_rewards).item(),
        "mean_advantage": torch.mean(advantages).item(),
        "std_advantage": torch.std(advantages).item(),
        "mean_group_mean": np.mean(group_means),
        "mean_group_std": np.mean(group_stds),
    }
    
    return advantages, raw_rewards, metadata