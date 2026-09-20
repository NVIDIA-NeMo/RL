# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import pytest
import torch
from nemo_rl.optimizers.bf16_cpu_adamw import BF16CPUAdamW

ARGS = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


@pytest.mark.parametrize("chunk", [1, 17, 4096])
def test_matches_fp32_adam_with_rounded_state(chunk):
    torch.manual_seed(17)
    p = torch.nn.Parameter(torch.randn(73))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = BF16CPUAdamW([p], **ARGS, chunk_numel=chunk)
    baseline = torch.optim.AdamW([ref], **ARGS, foreach=False)
    for step in range(30):
        gradient = torch.randn_like(p) * (0.001 if step % 2 else 1)
        p.grad, ref.grad = gradient.clone(), gradient.clone()
        torch.nn.utils.clip_grad_norm_([p], 0.5)
        torch.nn.utils.clip_grad_norm_([ref], 0.5)
        optimizer.step()
        baseline.step()
        torch.testing.assert_close(p, ref, rtol=0, atol=1e-7)
        for key in ("exp_avg", "exp_avg_sq"):
            expected = baseline.state[ref][key].bfloat16()
            assert torch.equal(optimizer.state[p][key], expected)
            baseline.state[ref][key].copy_(expected)
        optimizer.zero_grad(set_to_none=True)
        baseline.zero_grad(set_to_none=True)


def test_none_zero_gradient_and_state_restore():
    p = torch.nn.Parameter(torch.ones(31))
    untouched = torch.nn.Parameter(torch.ones(4))
    opt = BF16CPUAdamW([p, untouched], **ARGS, chunk_numel=7)
    p.grad = torch.arange(31).float()
    opt.step()
    assert untouched not in opt.state
    saved = copy.deepcopy(opt.state_dict())
    q, other = (
        torch.nn.Parameter(p.detach().clone()),
        torch.nn.Parameter(untouched.detach().clone()),
    )
    loaded = BF16CPUAdamW([q, other], **ARGS, chunk_numel=11)
    loaded.load_state_dict(saved)
    for _ in range(3):
        p.grad, q.grad = torch.zeros_like(p), torch.zeros_like(q)
        opt.step()
        loaded.step()
        assert torch.equal(p, q)
        for key in ("exp_avg", "exp_avg_sq"):
            assert loaded.state[q][key].dtype == torch.bfloat16
            assert torch.equal(opt.state[p][key], loaded.state[q][key])
    saved["state"][0]["exp_avg"] = saved["state"][0]["exp_avg"].float()
    with pytest.raises(ValueError, match="Moment dtype"):
        loaded.load_state_dict(saved)


def test_bounded_casts(monkeypatch):
    p = torch.nn.Parameter(torch.ones(1003))
    p.grad = torch.ones_like(p)
    original = torch.Tensor.float
    sizes = []

    def checked(tensor, *args, **kwargs):
        sizes.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", checked)
    BF16CPUAdamW([p], **ARGS, chunk_numel=29).step()
    assert sizes and max(sizes) <= 29


@pytest.mark.parametrize(
    "override",
    [dict(lr=-1), dict(eps=float("nan")), dict(betas=(1.0, 0.95)), dict(chunk_numel=0)],
)
def test_bad_options_fail(override):
    with pytest.raises(ValueError):
        BF16CPUAdamW(
            [torch.nn.Parameter(torch.ones(2))],
            **{**ARGS, "chunk_numel": 8, **override},
        )


def test_noncontiguous_parameter_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        BF16CPUAdamW([torch.nn.Parameter(torch.ones(2, 3).T)], **ARGS, chunk_numel=8)


def _dtensor_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        # Rank 1 has an empty shard, as can happen for tiny FSDP parameters.
        value = torch.ones(1 if rank == 0 else 0)
        make = lambda x: DTensor.from_local(
            x, mesh, [Shard(0)], shape=torch.Size([1]), stride=(1,)
        )
        p = torch.nn.Parameter(make(value.clone()))
        opt = BF16CPUAdamW([p], **ARGS, chunk_numel=1)
        p.grad = make(value.clone())
        opt.step()
        saved = copy.deepcopy(opt.state_dict())
        q = torch.nn.Parameter(make(p.to_local().detach().clone()))
        restored = BF16CPUAdamW([q], **ARGS, chunk_numel=1)
        restored.load_state_dict(saved)
        p.grad, q.grad = make(value.clone()), make(value.clone())
        opt.step()
        restored.step()
        assert torch.equal(p.to_local(), q.to_local())
        assert restored.state[q]["exp_avg"].dtype == torch.bfloat16
        assert restored.state[q]["exp_avg"].placements == (Shard(0),)
        assert restored.state[q]["step"] == 2
    finally:
        dist.destroy_process_group()


def test_dtensor_restore_and_empty_local_shard(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_dtensor_worker, args=(str(tmp_path / "mesh"),), nprocs=2, join=True)


def test_streaming_rejects_foreign_or_cpu_gradients_without_update():
    p = torch.nn.Parameter(torch.ones(7))
    opt = BF16CPUAdamW([p], **ARGS, chunk_numel=3)
    with pytest.raises(ValueError, match="not owned"):
        opt.step(gradient_shards={id(p) + 1: torch.ones(7)})
    with pytest.raises(ValueError, match="CUDA"):
        opt.step(gradient_shards={id(p): torch.ones(7)})
    assert torch.equal(p, torch.ones(7))
    assert not opt.state
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="mix"):
        opt.step(gradient_shards={id(p): torch.ones(7)})
    assert not opt.state
