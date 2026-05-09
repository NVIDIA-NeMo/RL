# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit test for ``_broadcast_batched_data_dict`` on a 2-rank gloo group.

Exercises the helper that backs ``_fetch(fetch_policy="leader_broadcast")``.
Runs on CPU (gloo) so it stays in the no-GPU Tier 1 lane.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data_plane.worker_mixin import _broadcast_batched_data_dict


def _worker(rank: int, world_size: int, tmp_init_file: str, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        if rank == 0:
            data = BatchedDataDict(
                {
                    "input_ids": torch.arange(12, dtype=torch.long).reshape(3, 4),
                    "input_lengths": torch.tensor([4, 3, 2], dtype=torch.int32),
                    "scalar_meta": "step_42",
                }
            )
        else:
            data = None

        out = _broadcast_batched_data_dict(data, src=0, group=dist.group.WORLD)

        assert torch.equal(
            out["input_ids"], torch.arange(12, dtype=torch.long).reshape(3, 4)
        )
        assert torch.equal(
            out["input_lengths"], torch.tensor([4, 3, 2], dtype=torch.int32)
        )
        assert out["scalar_meta"] == "step_42"
        q.put((rank, "ok"))
    except Exception as e:  # pragma: no cover — surface failures to parent
        q.put((rank, f"err: {type(e).__name__}: {e}"))
    finally:
        dist.destroy_process_group()


def test_leader_broadcast_round_trip(tmp_path):
    init_file = str(tmp_path / "init")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, 2, init_file, q)) for rank in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    results = sorted([q.get() for _ in range(2)])
    assert results == [(0, "ok"), (1, "ok")], results


def test_get_replica_group_default_is_none():
    """TQWorkerMixin._get_replica_group must default to None.

    The base default lets ``_fetch(fetch_policy="leader_broadcast")``
    fall back to the independent path when no backend override exists
    (Phase 1 / FSDP2 with TP=CP=PP=1).
    """
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class _Stub(TQWorkerMixin):
        pass

    assert _Stub()._get_replica_group() is None
