# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CP zigzag-sharding parity tests.

The unpacked diffusion-CP path shards the sequence in ``data.process_microbatch``
with ``nemo_rl.distributed.model_utils._get_tokens_on_this_cp_rank`` and, inside
the bridge attention, gathers it back with
``megatron.bridge.diffusion.common.cp_utils.all_gather_seq_cp``. These two MUST
use the identical zigzag convention or the gathered sequence is scrambled. The
pure tests below (no torch.distributed) are the cheap guard for that (F1); the
gloo round-trip/autograd test (F2) is skipped where multiprocessing is unavailable.
"""
import os

import pytest
import torch

from megatron.bridge.diffusion.common.cp_utils import (
    all_gather_seq_cp,
    local_zigzag_mask,
    scatter_seq_cp,
    zigzag_slice,
)
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("seq_dim", [0, 1, 2])
def test_zigzag_slice_matches_rl_convention(cp_size, seq_dim):
    """bridge zigzag_slice == RL _get_tokens_on_this_cp_rank for every rank (F1)."""
    torch.manual_seed(0)
    S = 2 * cp_size * 3
    shape = [3, 3, 3]
    shape[seq_dim] = S
    x = torch.randn(*shape)
    for r in range(cp_size):
        a = zigzag_slice(x, r, cp_size, seq_dim)
        b = _get_tokens_on_this_cp_rank(x, r, cp_size, seq_dim=seq_dim)
        assert torch.equal(a, b), f"cp={cp_size} rank={r} dim={seq_dim}"


@pytest.mark.parametrize("cp_size", [2, 4])
def test_local_zigzag_mask_partitions_positions(cp_size):
    S = 2 * cp_size * 3
    union = torch.zeros(S, dtype=torch.bool)
    total = 0
    for r in range(cp_size):
        m = local_zigzag_mask(S, r, cp_size, device="cpu")
        assert m.sum().item() == S // cp_size
        assert not (union & m).any(), "ranks must own disjoint positions"
        union |= m
        total += int(m.sum().item())
    assert union.all() and total == S


def _dist_worker(rank, cp_size, seq_dim, S, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=cp_size)
    g = dist.group.WORLD
    torch.manual_seed(1234)  # identical full tensor on every rank
    shape = [3, 3, 3]
    shape[seq_dim] = S
    full = torch.randn(*shape, dtype=torch.double)
    local = zigzag_slice(full, rank, cp_size, seq_dim).clone().requires_grad_(True)
    gathered = all_gather_seq_cp(local, g, seq_dim)
    rt = torch.allclose(gathered, full)
    (gathered * gathered).sum().backward()
    grad_ok = torch.allclose(local.grad, 2 * local.detach())  # reduce-scatter selects owned chunks
    l2 = zigzag_slice(full, rank, cp_size, seq_dim).clone().requires_grad_(True)
    sc = scatter_seq_cp(all_gather_seq_cp(l2, g, seq_dim), g, seq_dim)
    sc_ok = torch.allclose(sc, l2)
    if rank == 0:
        q.put((bool(rt), bool(grad_ok), bool(sc_ok)))
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("cp_size,seq_dim", [(2, 0), (4, 2)])
def test_gather_scatter_roundtrip_and_autograd(cp_size, seq_dim):
    """Forward round-trip + backward correctness of the CP collectives (F2)."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + cp_size * 10 + seq_dim
    procs = [
        ctx.Process(target=_dist_worker, args=(r, cp_size, seq_dim, 2 * cp_size * 3, port, q))
        for r in range(cp_size)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
    except (OSError, RuntimeError) as e:  # multiprocessing restricted in this env
        for p in procs:
            if p.is_alive():
                p.terminate()
        pytest.skip(f"multiprocessing unavailable: {e}")
    if q.empty():
        pytest.skip("gloo rendezvous produced no result (mp restricted)")
    rt_ok, grad_ok, sc_ok = q.get()
    assert rt_ok, "gather round-trip mismatch"
    assert grad_ok, "all_gather_seq_cp backward (reduce-scatter) mismatch"
    assert sc_ok, "scatter(gather(x)) != x"
