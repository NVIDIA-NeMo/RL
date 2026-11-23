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
"""SSD offload and data-plane checkpoints together, on every backend.

Each backend gets less memory than the data, so values are on SSD when the
checkpoint is saved; a fresh system with SSD on (another path) restores it and
every value reads back unchanged. Each half runs in a spawned child with its
own Ray and TQ: TQ keeps one named controller per Ray cluster, and mooncake's
engine reads its environment once per process.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from ._rollout_shapes import mooncake_available, rdma_available

MiB = 1 << 20
ROWS, PARTITIONS = 4, 24  # 24 x 4 rows x 1 MiB = 96 MiB


def _cfg(backend: str, ssd_root: str, node_id: str) -> dict:
    if backend == "simple":
        block = {
            "storage_capacity": 256,
            "num_storage_units": 1,
            # 1 KiB: every 1 MiB value goes to SSD.
            "ssd_offload": {"enabled": True, "path": ssd_root, "threshold_bytes": 1024},
        }
    elif backend == "mooncake_cpu":
        block = {
            "local_buffer_size": 128 * MiB,
            "staging_buffer_size": 32 * MiB,
            "global_segment_size": 64 * MiB,
            "storage_unit_segment_size": 64 * MiB,
            "ssd_offload": {
                "enabled": True,
                "path": ssd_root,
                "offload_threshold": 0.5,
            },
        }
    else:  # nixl: 24 MiB put area + 8 MiB IO area
        block = {
            "num_storage_units": 1,
            "node_ids": [node_id],
            "unit_slab_bytes": 32 * MiB,
            "op_timeout_s": 60.0,
            "ssd": {"enabled": True, "path": ssd_root, "staging_bytes": 8 * MiB},
        }
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": backend,
        "claim_meta_poll_interval_s": 0.5,
        backend: block,
    }


def _start(backend: str, ssd_root: str):  # pragma: no cover
    import os

    import ray

    from nemo_rl.data_plane import build_data_plane_client

    os.environ.pop("RAY_ADDRESS", None)
    ray.init(num_cpus=4, object_store_memory=2 << 30, include_dashboard=False)
    node_id = ray.get_runtime_context().get_node_id()
    cfg = _cfg(backend, ssd_root, node_id)
    if backend != "mooncake_cpu":
        return build_data_plane_client(cfg, checkpointing=True)

    from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
        configure_checkpoint_workers,
    )
    from nemo_rl.data_plane.mooncake_storage_unit import start_storage_units

    client = build_data_plane_client(
        cfg, checkpointing=True, storage_unit_node_ids=[node_id]
    )
    configure_checkpoint_workers(list(start_storage_units(cfg, [node_id])))
    return client


def _expected(p: int):  # pragma: no cover
    import torch

    return torch.full((ROWS, MiB // 4), float(p))  # 1 MiB per row


def _on_ssd(backend: str, ssd_root: str) -> int:  # pragma: no cover
    """How much is on SSD now (files, or blobs for nixl)."""
    if backend != "nixl":  # not mooncake's always-present .lock
        return sum(
            f.is_file() and not f.name.startswith(".lock")
            for f in Path(ssd_root).rglob("*")
        )
    import ray

    from nemo_rl.data_plane.adapters.tq_nixl import DEFAULT_NAMESPACE

    # A nixl unit unlinks each spill file once open, so count its disk blobs.
    unit = ray.get_actor("NixlStorageUnit#0", namespace=DEFAULT_NAMESPACE)
    return ray.get(unit.stats.remote())["disk_only_blobs"]


def _spill_and_save(
    backend: str, ssd_root: str, checkpoint_dir: str
) -> None:  # pragma: no cover
    import ray
    from tensordict import TensorDict

    client = _start(backend, ssd_root)
    try:
        for p in range(PARTITIONS):
            client.register_partition(
                partition_id=f"p{p}",
                fields=["big"],
                num_samples=ROWS,
                consumer_tasks=["read"],
            )
            client.put_samples(
                sample_ids=[f"p{p}-{i}" for i in range(ROWS)],
                partition_id=f"p{p}",
                fields=TensorDict({"big": _expected(p)}, batch_size=[ROWS]),
            )
            if backend == "mooncake_cpu":
                time.sleep(1.5)  # objects are evictable once their 1 s lease ends
        assert _on_ssd(backend, ssd_root), f"{backend}: nothing on SSD before the save"
        client.save_checkpoint(checkpoint_dir)
        client.close()
    finally:
        ray.shutdown()


def _restore_and_read_back(
    backend: str, ssd_root: str, checkpoint_dir: str
) -> None:  # pragma: no cover
    import ray
    import torch

    client = _start(backend, ssd_root)
    try:
        # 96 MiB into less memory: the restore itself must use the SSD.
        client.load_checkpoint(checkpoint_dir)
        for p in range(PARTITIONS):
            meta = client.claim_meta(
                partition_id=f"p{p}",
                task_name="read",
                required_fields=["big"],
                batch_size=ROWS,
                timeout_s=60.0,
            )
            data = client.get_data(meta)
            assert torch.equal(data["big"], _expected(p)), f"{backend}: p{p} changed"
        client.close()
    finally:
        ray.shutdown()


def _run_child(monkeypatch, backend: str, target, *args) -> None:
    if backend == "mooncake_cpu":
        from nemo_rl.data_plane.adapters.transfer_queue_env import _wanted_engine_env

        # Set on the driver before Ray starts in a run (configure_engine_env).
        for name, value in _wanted_engine_env().items():
            monkeypatch.setenv(name, value)
    proc = mp.get_context("spawn").Process(target=target, args=(backend, *args))
    proc.start()
    proc.join(timeout=600)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(f"{backend} {target.__name__} child timed out after 600 s")
    assert proc.exitcode == 0, (
        f"{backend} {target.__name__} child exited with {proc.exitcode}"
    )


@pytest.mark.parametrize(
    "backend",
    [
        "simple",
        pytest.param(
            "mooncake_cpu",
            marks=[
                pytest.mark.skipif(
                    not mooncake_available(), reason="mooncake not installed"
                ),
                pytest.mark.skipif(
                    not rdma_available(), reason="mooncake_cpu requires RDMA"
                ),
            ],
        ),
        pytest.param("nixl", marks=pytest.mark.nixl),
    ],
)
def test_checkpoint_round_trips_with_ssd_offload(
    monkeypatch, tmp_path, backend
) -> None:
    """Save with values on SSD; restore with SSD on at another path."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _run_child(
        monkeypatch,
        backend,
        _spill_and_save,
        str(tmp_path / "ssd-save"),
        str(checkpoint_dir),
    )
    _run_child(
        monkeypatch,
        backend,
        _restore_and_read_back,
        str(tmp_path / "ssd-load"),
        str(checkpoint_dir),
    )
