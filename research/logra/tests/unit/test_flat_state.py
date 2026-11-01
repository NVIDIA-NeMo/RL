import pytest
import torch
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn


def build(dtype, optimizer="row_adam", weight_decay=0.1):
    torch.manual_seed(3)
    net = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).to(dtype)
    opt = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=weight_decay),
        LoGRAConfig(rank=3, target_modules=["0", "2"], optimizer=optimizer),
    )
    return net, opt


def backward(net, dtype):
    x = torch.randn(4, 7, dtype=dtype, requires_grad=True)
    net(x).float().square().sum().backward()


def shares_flat_moments(opt):
    base = opt.row_second_moment.untyped_storage().data_ptr()
    return all(
        opt.state[layer.weight]["row_second_moment"].untyped_storage().data_ptr()
        == base
        for layer in opt.layers
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_step_matches_per_layer_reference_with_masters_and_scale(dtype):
    net, opt = build(dtype)
    backward(net, dtype)
    opt.synchronize_and_clip(None, 1.0)
    group = opt.param_groups[0]
    expected = []
    for layer in opt.layers:
        sketch = layer.sketch.clone()
        moment = (sketch.square().mean(dim=1, keepdim=True)) * (1 - opt.config.beta2)
        direction = sketch / (
            (moment / (1 - opt.config.beta2)).sqrt() + opt.config.epsilon
        )
        base = opt.state[layer.weight].get("master_weight", layer.weight.detach())
        decayed = base.float() * (1 - group["lr"] * group["weight_decay"])
        expected.append(decayed - 0.5 * group["lr"] * direction @ layer.projection)
    opt.step(scale=0.5)
    for layer, reference in zip(opt.layers, expected, strict=True):
        master = opt.state[layer.weight].get("master_weight")
        if dtype == torch.float32:
            assert master is None
            torch.testing.assert_close(layer.weight.detach(), reference)
        else:
            assert master.dtype == torch.float32
            torch.testing.assert_close(master, reference)
            assert torch.equal(layer.weight.detach(), master.to(dtype))


def test_resume_relinks_flat_moments_in_fp32():
    net, opt = build(torch.bfloat16)
    for _ in range(2):
        opt.zero_grad()
        backward(net, torch.bfloat16)
        opt.synchronize_and_clip(None, 1.0)
        opt.step()
    saved = opt.state_dict()
    expected = opt.row_second_moment.clone()

    net2, opt2 = build(torch.bfloat16)
    net2.load_state_dict(net.state_dict())
    opt2.load_state_dict(saved)
    assert torch.equal(opt2.row_second_moment, expected)
    assert shares_flat_moments(opt2)
    for layer in opt2.layers:
        assert layer.sketch.dtype == torch.float32
        for value in opt2.state[layer.weight].values():
            assert value.dtype == torch.float32

    # The resumed optimizer updates the moments it exposes for checkpoints.
    opt2.zero_grad()
    backward(net2, torch.bfloat16)
    opt2.synchronize_and_clip(None, 1.0)
    opt2.step()
    assert not torch.equal(opt2.row_second_moment, expected)
    assert shares_flat_moments(opt2)


def test_generic_offload_then_move_row_moments_restores_linkage():
    net, opt = build(torch.float32)
    backward(net, torch.float32)
    opt.synchronize_and_clip(None, 1.0)
    opt.step()
    # The native worker replaces every state tensor with an independent copy.
    for state in opt.state.values():
        for key, value in state.items():
            state[key] = value.to("cpu").clone()
    assert not shares_flat_moments(opt)
    opt.move_row_moments("cpu")
    assert shares_flat_moments(opt)


def test_trial_step_restores_weights_and_masters():
    net, opt = build(torch.bfloat16, optimizer="sgd", weight_decay=0.0)
    for parameter in net.parameters():
        if parameter not in {layer.weight for layer in opt.layers}:
            parameter.requires_grad_(False)
    opt.native.param_groups[0]["params"] = []
    backward(net, torch.bfloat16)
    opt.synchronize_and_clip(None, None)
    before = [layer.weight.detach().clone() for layer in opt.layers]
    masters = [opt.state[layer.weight]["master_weight"].clone() for layer in opt.layers]
    with opt.trial_step(scale=2.0):
        for layer, master in zip(opt.layers, masters, strict=True):
            proposal = torch.addmm(
                master, layer.sketch, layer.projection, alpha=-0.02
            ).to(torch.bfloat16)
            assert torch.equal(layer.weight.detach(), proposal)
    for layer, weight, master in zip(opt.layers, before, masters, strict=True):
        assert torch.equal(layer.weight.detach(), weight)
        assert torch.equal(opt.state[layer.weight]["master_weight"], master)
