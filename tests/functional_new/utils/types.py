from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor, List, float, int]


@dataclass
class StatResult:
    """Output of a reducer (pure statistics; no pass/fail)."""

    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionResult:
    """Judgment after applying a Criterion to a StatResult."""

    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
