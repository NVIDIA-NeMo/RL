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

"""Unit tests for the broadcast-based golden reshard (``xferdtensor_golden``).

Two layers:
- Pure ``_compute_shard_slices`` math (which global-tensor slice a rank owns) —
  no process group, runs anywhere.
- End-to-end golden reshards over a real 2-rank **gloo (CPU)** group spawned
  locally (no GPU, no NCCL): a tiny fake process_group forwards
  ``.broadcast`` to ``torch.distributed.broadcast``. Covers shard->replicate
  (gather) and replicate->shard (scatter).

``xferdtensor.py`` imports the real nccl.xfer op under try/except, so it loads
on CPU with no extras.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_rl.distributed.nccl_xfer_utils import MeshInfo
from nemo_rl.distributed.xferdtensor import (
    DTensorRef,
    _compute_shard_slices,
    xferdtensor_golden,
)


# --------------------------------------------------------------------------
# Pure: _compute_shard_slices
# --------------------------------------------------------------------------
def test_compute_shard_slices_replicate():
    # No Shard placement -> whole extent on every dim.
    sl = _compute_shard_slices((4, 3), [2], [0], [Replicate()])
    assert sl == [slice(None), slice(None)]


@pytest.mark.parametrize("coord,expected", [(0, slice(0, 2)), (1, slice(2, 4))])
def test_compute_shard_slices_shard_dim0(coord, expected):
    # TP=2 mesh sharding tensor dim 0 of a (4,3) tensor -> 2 row-chunks.
    sl = _compute_shard_slices((4, 3), [2], [coord], [Shard(0)])
    assert sl == [expected, slice(None)]


@pytest.mark.parametrize("coord,expected", [(0, slice(0, 3)), (1, slice(3, 6))])
def test_compute_shard_slices_shard_dim1(coord, expected):
    # Row-parallel: shard tensor dim 1 of a (4,6) tensor.
    sl = _compute_shard_slices((4, 6), [2], [coord], [Shard(1)])
    assert sl == [slice(None), expected]


def test_compute_shard_slices_two_axes_same_dim():
    # Two mesh axes (each size 2) both shard tensor dim 0 -> 4 chunks of 2.
    # coords [1, 0] -> row-major chunk index 1*2 + 0 = 2 -> rows [4:6].
    sl = _compute_shard_slices((8,), [2, 2], [1, 0], [Shard(0), Shard(0)])
    assert sl == [slice(4, 6)]


# --------------------------------------------------------------------------
# End-to-end golden reshard over a 2-rank gloo (CPU) group
# --------------------------------------------------------------------------
class _FakePG:
    """Minimal process_group duck-type: ``.rank`` + ``.broadcast(t, src)``."""

    def __init__(self, rank):
        self.rank = rank

    def broadcast(self, tensor, src):
        dist.broadcast(tensor, src=src)


def _golden_worker(rank, world_size, port, case):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # Known global tensor; CPU float32 so .view(torch.uint8) is valid.
        g = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        pg = _FakePG(rank)
        mesh = MeshInfo(torch.tensor([0, 1]))  # 1-D TP2 mesh over global ranks 0,1

        if case == "shard_to_replicate":
            # src: each rank owns its Shard(0) row-block; dst: fully replicated.
            src = DTensorRef(g[2 * rank : 2 * rank + 2].clone(), (4, 3))
            dst_buf = torch.zeros(4, 3, dtype=torch.float32)
            xferdtensor_golden(
                src,
                mesh,
                [Shard(0)],
                DTensorRef(dst_buf, (4, 3)),
                mesh,
                [Replicate()],
                pg,
            )
            assert torch.equal(dst_buf, g), f"rank {rank}: {dst_buf} != {g}"

        elif case == "replicate_to_shard":
            # src: full tensor replicated; dst: each rank gets its Shard(0) block.
            src = DTensorRef(g.clone(), (4, 3))
            dst_buf = torch.zeros(2, 3, dtype=torch.float32)
            xferdtensor_golden(
                src,
                mesh,
                [Replicate()],
                DTensorRef(dst_buf, (4, 3)),
                mesh,
                [Shard(0)],
                pg,
            )
            expected = g[2 * rank : 2 * rank + 2]
            assert torch.equal(dst_buf, expected), (
                f"rank {rank}: {dst_buf} != {expected}"
            )
        else:
            raise AssertionError(f"unknown case {case}")
    finally:
        dist.destroy_process_group()


def _run_gloo(case, world_size=2):
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    # mp.spawn re-raises child exceptions (incl. assertion failures) in the parent.
    mp.spawn(
        _golden_worker, args=(world_size, port, case), nprocs=world_size, join=True
    )


@pytest.mark.parametrize("case", ["shard_to_replicate", "replicate_to_shard"])
def test_xferdtensor_golden_reshard_cpu(case):
    _run_gloo(case, world_size=2)
