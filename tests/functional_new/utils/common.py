import numpy as np
import torch

from .types import TensorLike


def to_numpy(value: TensorLike) -> np.ndarray:
    """Convert various types to numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        return np.array(value)
    elif isinstance(value, (int, float)):
        return np.array([value])
    elif isinstance(value, np.ndarray):
        return value
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
