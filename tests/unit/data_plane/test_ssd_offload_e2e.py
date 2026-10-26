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
"""End-to-end SimpleStorage SSD offload through a real TransferQueue.

The shared session client (see ``conftest.py``) is initialized once per pytest
worker, and TQ keeps one named controller per Ray cluster, so a second
``tq.init`` here would silently reuse the first config and never see
``ssd_offload``. This test therefore boots its own TQ in a spawned child with
its own local Ray cluster, using a tiny ``threshold_bytes`` so that the large
field spills to SSD while the small one stays in host memory. The child's
assertions fail it with a nonzero exit code and a traceback on stderr.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest


def _round_trip_with_ssd_offload(ssd_root: str) -> None:  # pragma: no cover
    import os

    import ray
    import torch
    from tensordict import TensorDict

    from nemo_rl.data_plane import build_data_plane_client

    n = 4
    threshold_bytes = 1024
    # 4 B x 512 = 2 KiB per row: at or above the threshold, so it goes to SSD.
    big = torch.arange(n * 512, dtype=torch.float32).reshape(n, 512)
    # 8 B per row: far below the threshold, so it stays in host memory.
    small = torch.arange(n, dtype=torch.int64)

    # A fresh local cluster: attaching to the parent's would reuse its
    # already-initialized TQ controller and ignore this config.
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(num_cpus=4, include_dashboard=False)
    try:
        client = build_data_plane_client(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "backend": "simple",
                "claim_meta_poll_interval_s": 0.5,
                "simple": {
                    "storage_capacity": 64,
                    "num_storage_units": 1,
                    "ssd_offload": {
                        "enabled": True,
                        "path": ssd_root,
                        "threshold_bytes": threshold_bytes,
                    },
                },
            }
        )
        keys = [f"k{i}" for i in range(n)]
        client.register_partition(
            partition_id="ssd-e2e",
            fields=["big", "small"],
            num_samples=n,
            consumer_tasks=["read"],
        )
        client.put_samples(
            sample_ids=keys,
            partition_id="ssd-e2e",
            fields=TensorDict({"big": big, "small": small}, batch_size=[n]),
        )

        # Only the large field crosses the threshold, one file per sample.
        spilled = sorted(Path(ssd_root).rglob("*.bin"))
        assert len(spilled) >= n, f"expected >= {n} SSD files, found {spilled}"

        meta = client.claim_meta(
            partition_id="ssd-e2e",
            task_name="read",
            required_fields=["big", "small"],
            batch_size=n,
            timeout_s=30.0,
        )
        data = client.get_data(meta)
        order = [keys.index(k) for k in meta.sample_ids]
        assert torch.equal(data["big"], big[order]), "SSD-backed field changed"
        assert torch.equal(data["small"], small[order]), "in-memory field changed"

        client.close()
        left = sorted(Path(ssd_root).rglob("*.bin"))
        assert not left, f"close() left SSD files behind: {left}"
    finally:
        ray.shutdown()


def test_ssd_offload_round_trip_spills_and_cleans_up(tmp_path) -> None:
    proc = mp.get_context("spawn").Process(
        target=_round_trip_with_ssd_offload, args=(str(tmp_path / "ssd"),)
    )
    proc.start()
    proc.join(timeout=180)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("SSD offload child timed out after 180 s")
    assert proc.exitcode == 0, f"child exited with {proc.exitcode}"
