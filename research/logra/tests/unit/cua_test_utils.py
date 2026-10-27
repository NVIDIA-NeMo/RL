"""Small shared optimizer fixture for CUA numerical tests."""

import torch
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn


def create_optimizer(
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Sequential, LoGRAOptimizer]:
    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(2, 3, bias=False, dtype=dtype))
    optimizer = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0),
        LoGRAConfig(
            rank=2, optimizer="sgd", target_modules=["0"], freeze_non_target=True
        ),
    )
    return net, optimizer
