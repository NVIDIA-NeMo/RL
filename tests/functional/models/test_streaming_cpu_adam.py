# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank GPU parity: singleton and sharded FSDP, accumulation, two updates."""

from contextlib import nullcontext
from functools import partial
from datetime import timedelta
import gc
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from nemo_rl.models.deferred_grad import (
    backward_scope,
    _pending,
    scale_grads_and_clip_grad_norm as streaming_scale_and_clip,
    optimizer_step,
)
from nemo_rl.optimizers.bf16_cpu_adamw import BF16CPUAdamW


class TinyEPExperts(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        from torch.distributed.tensor import distribute_tensor, Shard

        self.weight = torch.nn.Parameter(
            distribute_tensor(torch.ones(4, 4, 4, device="cuda"), mesh, [Shard(0)])
        )

    def forward(self, x):
        return x @ self.weight.to_local()[0]


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def run(mesh, enabled, rank, max_norm, mixed=False):
    torch.manual_seed(123)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)
    ).cuda()
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    for layer in (model[0], model[2]):
        fully_shard(
            layer,
            mesh=mesh,
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        offload_policy=CPUOffloadPolicy(pin_memory=False),
        mp_policy=policy,
    )
    moe_mesh = None
    input_dim = 128
    if mixed:
        moe_mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("ep_shard", "ep"))
        expert = TinyEPExperts(moe_mesh["ep"])
        from torch.distributed.tensor import Shard

        fully_shard(
            expert,
            mesh=moe_mesh["ep_shard"],
            shard_placement_fn=lambda p: Shard(1),
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            mp_policy=policy,
        )
        dense = torch.nn.Linear(4, 4).cuda()
        fully_shard(
            dense,
            mesh=mesh,
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            mp_policy=policy,
        )
        model = torch.nn.Sequential(dense, expert)
        fully_shard(
            model,
            mesh=mesh,
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            mp_policy=policy,
        )
        input_dim = 4
    model.cpu()
    opt = BF16CPUAdamW(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        chunk_numel=8192,
    )
    results = []
    memory = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with backward_scope(opt) if enabled else nullcontext():
            for micro in range(2):
                torch.manual_seed(200 + 10 * step + micro + rank)
                x = torch.randn(8, input_dim, device="cuda", dtype=torch.bfloat16)
                model(x).float().square().mean().backward()
                if enabled:
                    assert _pending, "No GPU gradients were captured"
                    assert all(v.gradient.is_cuda for v in _pending.values())
                else:
                    assert not _pending
        from nemo_automodel.components.training.utils import (
            scale_grads_and_clip_grad_norm,
        )

        clipper = (
            partial(streaming_scale_and_clip, optimizer=opt)
            if enabled
            else scale_grads_and_clip_grad_norm
        )
        norm = clipper(
            max_norm,
            [model],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=mesh,
            moe_mesh=moe_mesh,
            ep_axis_name="ep" if mixed else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=1,
            dp_group_size=2,
        )
        norm = torch.as_tensor(norm).detach().cpu()
        assert torch.isfinite(norm) and norm > 0
        if enabled:
            assert _pending and all(p.grad is None for p in model.parameters())
            by_param = {
                id(e.param.sharded_param): e.gradient for e in _pending.values()
            }
            # Small-test-only snapshots; production never materializes full CPU grads.
            grads = [by_param[id(p)].cpu().clone() for p in model.parameters()]
            del by_param
        else:
            assert not _pending
            grads = [local(p.grad).clone() for p in model.parameters()]
        assert any(g.count_nonzero() for g in grads)
        assert all(g.device.type == "cpu" for g in grads)
        if enabled:
            optimizer_step(opt)
            assert opt.last_streamed_chunk_bytes == 8192 * 4
            assert not _pending and all(p.grad is None for p in model.parameters())
        else:
            opt.step()
        params = [local(p).detach().clone() for p in model.parameters()]
        moments = [
            local(v).clone()
            for s in opt.state.values()
            for k, v in s.items()
            if k in ("exp_avg", "exp_avg_sq")
        ]
        results.append([norm] + grads + params + moments)
        opt.zero_grad(set_to_none=True)
        del x
        gc.collect()
        torch.cuda.synchronize()
        memory.append(torch.cuda.memory_allocated())
    assert memory[1] <= memory[0] + 1024 * 1024, memory
    del opt, model
    gc.collect()
    torch.cuda.synchronize()
    return results, memory


def worker(rank, rendezvous):
    from nemo_automodel.shared.torch_patches import apply_torch_patches

    apply_torch_patches()
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    singleton = init_device_mesh(
        "cuda", (2, 1), mesh_dim_names=("replica", "singleton")
    )["singleton"]
    report = []
    for name, m, mixed in [
        ("sharded", mesh, False),
        ("singleton", singleton, False),
        ("mixed_ep", mesh, True),
    ]:
        for max_norm in (0.01, 1e6):
            baseline, base_memory = run(m, False, rank, max_norm, mixed)
            candidate, deferred_memory = run(m, True, rank, max_norm, mixed)
            max_error = 0.0
            for a, b in zip(baseline, candidate):
                for x, y in zip(a, b):
                    # CPU/GPU reduction order can differ; BF16 state adds rounding.
                    tol = 0.01 if x.dtype == torch.bfloat16 else 2e-6
                    torch.testing.assert_close(x, y, rtol=tol, atol=1e-8)
                    max_error = max(
                        max_error, float((x.float() - y.float()).abs().max())
                    )
            report.append(
                dict(
                    mode=name,
                    max_norm=max_norm,
                    max_abs_error=max_error,
                    baseline_memory=base_memory,
                    deferred_memory=deferred_memory,
                )
            )
    out = Path(rendezvous).parent
    (out / f"parity-rank{rank}.json").write_text(json.dumps(report, indent=2))
    print("STREAM_ADAM_PARITY_PASSED", rank, report, flush=True)
    dist.destroy_process_group()


def test_streaming_cpu_adam_parity():
    if torch.cuda.device_count() < 2:
        raise unittest.SkipTest("Requires two CUDA GPUs")
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(worker, args=(d + "/rdzv",), nprocs=2, join=True)


if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs are required")
    test_streaming_cpu_adam_parity()
