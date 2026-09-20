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


def test_dcp_roundtrip_preserves_bf16_moments_and_next_update(tmp_path):
    """Exercise the flattened DCP path used by AutoModel, not native state_dict."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    torch.manual_seed(41)
    model = torch.nn.Linear(5, 3)
    opt = BF16CPUAdamW(model.parameters(), **ARGS, chunk_numel=7)
    for _ in range(8):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    saved = get_optimizer_state_dict(model, opt, options=options)
    assert not any('bf16_cpu_adamw_version' in key for key in saved)
    dcp.save({'optim': saved}, checkpoint_id=tmp_path / 'optim')
    resumed = copy.deepcopy(model)
    loaded = BF16CPUAdamW(resumed.parameters(), **ARGS, chunk_numel=7)
    skeleton = {'optim': get_optimizer_state_dict(resumed, loaded, options=options)}
    dcp.load(skeleton, checkpoint_id=tmp_path / 'optim')
    set_optimizer_state_dict(resumed, loaded, skeleton['optim'], options=options)
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert loaded.state[q]['step'] == 8
        for key in ('exp_avg', 'exp_avg_sq'):
            assert loaded.state[q][key].dtype == torch.bfloat16
            assert torch.equal(opt.state[p][key], loaded.state[q][key])
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    opt.step()
    loaded.step()
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q)
        assert loaded.state[q]['step'] == 9
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.equal(opt.state[p][key], loaded.state[q][key])


def test_unknown_checkpoint_version_rejected():
    p = torch.nn.Parameter(torch.ones(3))
    opt = BF16CPUAdamW([p], **ARGS, chunk_numel=2)
    saved = opt.state_dict()
    saved['bf16_cpu_adamw_version'] = 2
    with pytest.raises(ValueError, match='version 1'):
        opt.load_state_dict(saved)


def _dcp_dtensor_worker(rank, checkpoint_dir):
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    dist.init_process_group(
        'gloo', init_method=f'file://{checkpoint_dir}/rendezvous', rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh('cpu', (2,))
        model = torch.nn.Module()
        # Exercise both regular and empty local shards.
        for name, size in [('weight', 5), ('tiny', 1)]:
            model.register_parameter(name, torch.nn.Parameter(
                distribute_tensor(torch.arange(size).float(), mesh, [Shard(0)])
            ))
        opt = BF16CPUAdamW(model.parameters(), **ARGS, chunk_numel=2)
        for _ in range(8):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            opt.step()
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        saved = {'optim': get_optimizer_state_dict(model, opt, options=options)}
        dcp.save(saved, checkpoint_id=f'{checkpoint_dir}/optim')
        resumed = copy.deepcopy(model)
        loaded = BF16CPUAdamW(resumed.parameters(), **ARGS, chunk_numel=2)
        # AutoModel materializes Adam state for every parameter before DCP,
        # including empty local shards that DCP's generic initializer skips.
        for p in resumed.parameters():
            p.grad = torch.zeros_like(p)
        loaded.step()
        loaded.zero_grad(set_to_none=True)
        resumed.load_state_dict(model.state_dict())
        skeleton = {'optim': get_optimizer_state_dict(resumed, loaded, options=options)}
        dcp.load(skeleton, checkpoint_id=f'{checkpoint_dir}/optim')
        set_optimizer_state_dict(resumed, loaded, skeleton['optim'], options=options)
        for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
            assert loaded.state[q]['step'] == 8
            for key in ('exp_avg', 'exp_avg_sq'):
                state = loaded.state[q][key]
                assert state.dtype == torch.bfloat16
                assert state.placements == p.placements
                assert torch.equal(state.to_local(), opt.state[p][key].to_local())
            p.grad = torch.full_like(p, 0.25)
            q.grad = torch.full_like(q, 0.25)
        opt.step()
        loaded.step()
        for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
            assert loaded.state[q]['step'] == 9
            assert torch.equal(p.to_local(), q.to_local())
            for key in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(opt.state[p][key].to_local(), loaded.state[q][key].to_local())
    finally:
        dist.destroy_process_group()


def test_dcp_dtensor_roundtrip_and_empty_shard(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_dcp_dtensor_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_markerless_dcp_state_still_rejects_fp32_moments():
    p = torch.nn.Parameter(torch.ones(3))
    opt = BF16CPUAdamW([p], **ARGS, chunk_numel=2)
    p.grad = torch.ones_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    del saved['bf16_cpu_adamw_version']
    saved['state'][0]['exp_avg'] = saved['state'][0]['exp_avg'].float()
    with pytest.raises(ValueError, match='Moment dtype'):
        opt.load_state_dict(saved)
