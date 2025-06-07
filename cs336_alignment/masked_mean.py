import torch

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    
    assert tensor.shape == mask.shape, "tensor and mask must have the same shape"

    mask_float = mask.float()
    masked_tensor = tensor * mask_float

    if dim is None:
        masked_sum = torch.sum(masked_tensor)
        mask_count = torch.sum(mask_float)
        
        # Avoid division by zero
        if mask_count == 0:
            return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        
        return masked_sum / mask_count
    else:
        # Compute mean along the specified dimension
        masked_sum = torch.sum(masked_tensor, dim=dim)
        mask_count = torch.sum(mask_float, dim=dim)
        
        # Avoid division by zero: where mask_count is 0, set result to 0
        # Use torch.where to handle division by zero safely
        result = torch.where(
            mask_count > 0,
            masked_sum / mask_count,
            torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        )
        
        return result