import copy

import pytest
import torch
from logra.compression import install_sketches
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn
from torch.utils.checkpoint import checkpoint


def model():
    return nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_direct_sketch_and_input_gradient(recompute, distribution):
    torch.manual_seed(5)
    dense = model()
    compressed = copy.deepcopy(dense)
    states = install_sketches(
        compressed, LoGRAConfig(rank=3, target_modules=["0"], distribution=distribution)
    )
    x = torch.randn(6, 7, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    dense(x).square().sum().backward()
    for chunk in z.chunk(2):
        output = (
            checkpoint(compressed, chunk, use_reentrant=False)
            if recompute
            else compressed(chunk)
        )
        output.square().sum().backward()
    torch.testing.assert_close(
        states[0].sketch, dense[0].weight.grad @ states[0].projection.T
    )
    torch.testing.assert_close(x.grad, z.grad)
    torch.testing.assert_close(dense[2].weight.grad, compressed[2].weight.grad)
    assert compressed[0].weight.grad is None


@pytest.mark.parametrize("rule", ["sgd", "row_adam"])
def test_resume_and_non_target_update(rule):
    torch.manual_seed(4)
    net = model()
    config = LoGRAConfig(rank=3, target_modules=["0"], optimizer=rule)
    opt = LoGRAOptimizer(
        net, torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01), config
    )
    x = torch.randn(4, 7, requires_grad=True)

    def step(m, o):
        o.zero_grad()
        m(x).square().sum().backward()
        o.synchronize_and_clip(None, 1.0)
        o.step()

    before = net[2].weight.detach().clone()
    step(net, opt)
    assert not torch.equal(before, net[2].weight)
    state = copy.deepcopy(opt.state_dict())
    clone = model()
    clone.load_state_dict(net.state_dict())
    other = LoGRAOptimizer(
        clone,
        torch.optim.AdamW(clone.parameters(), lr=0.003, weight_decay=0.01),
        config,
    )
    other.load_state_dict(state)
    step(net, opt)
    step(clone, other)
    for p, q in zip(net.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q)
    assert all(p.grad is None for p in [net[0].weight, clone[0].weight])


def test_sgd_matches_explicit_reconstruction_and_clipping():
    torch.manual_seed(9)
    net = model()
    reference = copy.deepcopy(net)
    config = LoGRAConfig(rank=3, target_modules=["0"], optimizer="sgd")
    opt = LoGRAOptimizer(
        net, torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0), config
    )
    x = torch.randn(4, 7, requires_grad=True)
    reference(x).square().sum().backward()
    net(x).square().sum().backward()
    a = opt.layers[0].projection
    expected = reference[0].weight.grad @ a.T @ a
    norm2 = expected.square().sum() + sum(
        p.grad.square().sum()
        for n, p in reference.named_parameters()
        if n != "0.weight"
    )
    norm = norm2.sqrt()
    actual_norm = opt.synchronize_and_clip(None, 0.1)
    assert actual_norm == pytest.approx(norm.item(), rel=1e-5)
    before = net[0].weight.detach().clone()
    opt.step()
    torch.testing.assert_close(
        net[0].weight, before - 0.01 * expected * min(1.0, 0.1 / (norm.item() + 1e-6))
    )


def test_backward_leaves_no_cyclic_garbage():
    """Sketch accumulation must be freed by reference counting alone.

    Objects that only the cyclic collector can reclaim accumulate across steps
    and trigger full collections that stall training for hundreds of ms.
    """
    import gc

    net = model()
    install_sketches(net, LoGRAConfig(rank=3, target_modules=["0", "2"]))
    x = torch.randn(4, 7, requires_grad=True)
    net(x).square().sum().backward()  # warm up any one-time caches
    gc.collect()
    gc.disable()
    try:
        for _ in range(3):
            net(x).square().sum().backward()
        unreachable = gc.collect()
    finally:
        gc.enable()
    assert unreachable == 0
