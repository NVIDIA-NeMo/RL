import torch
from logra import optimizer as logra_optimizer
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn


def test_sketches_are_reduced_with_a_single_collective(monkeypatch):
    """Two target layers must cost one all_reduce, not one per layer."""
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3))
    opt = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01),
        LoGRAConfig(rank=3, target_modules=["0", "2"], optimizer="sgd"),
    )
    reduced = []
    monkeypatch.setattr(logra_optimizer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(logra_optimizer.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        logra_optimizer.dist,
        "all_reduce",
        lambda tensor, op=None, group=None: reduced.append(tuple(tensor.shape)),
    )
    net(torch.randn(4, 7)).square().sum().backward()
    opt.synchronize_and_clip(None, 1.0)
    # Two one-time projection fingerprint checks, one for every sketch at once,
    # and one for the scalar squared norm.
    assert len(reduced) == 4, reduced
    total_rows = net[0].out_features + net[2].out_features
    assert reduced.count((total_rows, 3)) == 1
