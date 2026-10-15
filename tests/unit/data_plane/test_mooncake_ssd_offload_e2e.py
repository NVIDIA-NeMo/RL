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
"""End-to-end Mooncake SSD offload through a real TransferQueue.

One storage unit owns all Mooncake memory; the test writes more than that
memory holds, so the run only passes if the master offloads objects to the
unit's SSD and every value still reads back unchanged. Like
``test_ssd_offload_e2e.py`` it boots its own TQ and Ray in a spawned child:
TQ keeps one named controller per Ray cluster, and mooncake's engine reads
its environment once per process.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from ._rollout_shapes import mooncake_available, rdma_available

MiB = 1 << 20


def _spill_and_read_back(ssd_root: str) -> None:  # pragma: no cover
    import os

    import ray
    import torch
    from tensordict import TensorDict

    unit_bytes = 256 * MiB
    cfg = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "mooncake_cpu",
        "claim_meta_poll_interval_s": 0.5,
        "mooncake_cpu": {
            "local_buffer_size": 128 * MiB,
            "staging_buffer_size": 32 * MiB,
            "storage_unit_segment_size": unit_bytes,
            "ssd_offload": {
                "enabled": True,
                "path": ssd_root,
                "offload_threshold": 0.5,
            },
        },
    }
    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.data_plane.mooncake_storage_unit import start_storage_units

    os.environ.pop("RAY_ADDRESS", None)
    ray.init(num_cpus=4, object_store_memory=2 << 30, include_dashboard=False)
    try:
        node_ids = [ray.get_runtime_context().get_node_id()]
        client = build_data_plane_client(cfg, storage_unit_node_ids=node_ids)
        units = start_storage_units(cfg, node_ids)

        # 24 x 16 MiB = 384 MiB: 1.5x what the unit's memory can hold.
        rows, row_floats = 4, MiB  # 4 MiB per row
        batches = {}
        for p in range(24):
            batches[p] = torch.full((rows, row_floats), float(p))
            client.register_partition(
                partition_id=f"p{p}",
                fields=["big"],
                num_samples=rows,
                consumer_tasks=["read"],
            )
            client.put_samples(
                sample_ids=[f"p{p}-{i}" for i in range(rows)],
                partition_id=f"p{p}",
                fields=TensorDict({"big": batches[p]}, batch_size=[rows]),
            )
            # Objects are evictable only once their lease (10 s) expires.
            time.sleep(2.0)

        spilled = [f for f in Path(ssd_root).rglob("*") if f.is_file()]
        assert spilled, f"no file was offloaded under {ssd_root}"

        for p, expected in batches.items():
            meta = client.claim_meta(
                partition_id=f"p{p}",
                task_name="read",
                required_fields=["big"],
                batch_size=rows,
                timeout_s=60.0,
            )
            data = client.get_data(meta)
            assert torch.equal(data["big"], expected), f"partition p{p} changed"

        client.close()
        for unit in units:
            ray.kill(unit)
    finally:
        ray.shutdown()


@pytest.mark.skipif(not mooncake_available(), reason="mooncake not installed")
@pytest.mark.skipif(not rdma_available(), reason="mooncake_cpu requires RDMA")
def test_mooncake_ssd_offload_spills_beyond_unit_memory(tmp_path) -> None:
    proc = mp.get_context("spawn").Process(
        target=_spill_and_read_back, args=(str(tmp_path / "ssd"),)
    )
    proc.start()
    proc.join(timeout=600)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("Mooncake SSD offload child timed out after 600 s")
    assert proc.exitcode == 0, f"child exited with {proc.exitcode}"
