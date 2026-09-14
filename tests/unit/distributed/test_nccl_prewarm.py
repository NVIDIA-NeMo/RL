# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.distributed.nccl_prewarm import (
    prewarm_checkpoint_gather,
    prewarm_enabled,
    prewarm_expert_alltoall,
)


def test_prewarm_enabled_default():
    """Pre-warming is on unless explicitly disabled."""
    os.environ.pop("NRL_NCCL_PREWARM", None)
    assert prewarm_enabled()


@pytest.mark.parametrize(
    "value,expected", [("0", False), ("1", True), ("", True), ("true", True)]
)
def test_prewarm_enabled_env(monkeypatch, value, expected):
    """Only the exact string "0" disables it; anything else leaves it on."""
    monkeypatch.setenv("NRL_NCCL_PREWARM", value)
    assert prewarm_enabled() is expected


def test_disabled_is_a_noop_without_a_process_group(monkeypatch):
    """When disabled, neither entry point touches torch.distributed.

    This matters because the kill switch must be usable on a rank that never
    joined a process group -- if the disabled path still asserted on
    dist.is_initialized() it would crash instead of being inert.
    """
    monkeypatch.setenv("NRL_NCCL_PREWARM", "0")
    assert not dist.is_initialized()
    prewarm_checkpoint_gather()
    prewarm_expert_alltoall(None)  # group is never touched when disabled


def _gather_worker(rank, world_size, tmp_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Must be collective-safe and callable more than once: NCCL caches the
        # connections, so a second call is a no-op, and the driver may end up
        # invoking it again after a worker restart.
        prewarm_checkpoint_gather()
        prewarm_checkpoint_gather()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_prewarm_checkpoint_gather_runs_collectively(tmp_path):
    """The gather completes on every rank and is safe to repeat.

    Uses gloo so this runs in CI without GPUs. The NCCL-specific behaviour
    being worked around (lazy point-to-point transport setup) cannot be
    observed here; this only pins the collective's shape and idempotence.
    """
    world_size = 2
    tmp_file = tmp_path / "rendezvous"
    mp.spawn(
        _gather_worker,
        args=(world_size, str(tmp_file)),
        nprocs=world_size,
        join=True,
    )


def _alltoall_worker(rank, world_size, tmp_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{tmp_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        prewarm_expert_alltoall(dist.group.WORLD)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs for an all_to_all"
)
def test_prewarm_expert_alltoall_runs(tmp_path):
    """The expert all_to_all pre-warm completes over a real NCCL group."""
    world_size = 2
    tmp_file = tmp_path / "rendezvous_ep"
    mp.spawn(
        _alltoall_worker,
        args=(world_size, str(tmp_file)),
        nprocs=world_size,
        join=True,
    )


def test_prewarm_expert_alltoall_skips_single_rank_group(monkeypatch):
    """A group of one has no peers to connect, so it must return immediately.

    Guards the EP=1 case: reaching the collective would need a CUDA device and
    would be pointless work on a group with no remote peer.
    """
    monkeypatch.setenv("NRL_NCCL_PREWARM", "1")

    class _SingleRankGroup:
        pass

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    prewarm_expert_alltoall(_SingleRankGroup())
