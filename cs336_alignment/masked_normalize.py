import torch
from typing import Optional

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: Optional[int] = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask

    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    normalized_tensor = summed / normalize_constant

    return normalized_tensor