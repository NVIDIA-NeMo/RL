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

Either one storage unit or the driver's own client owns all Mooncake
memory; the test writes more than that memory holds, so the run only passes
if the master offloads objects to the owner's SSD and every value still
reads back unchanged. Like
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

pytestmark = [
    pytest.mark.skipif(not mooncake_available(), reason="mooncake not installed"),
    pytest.mark.skipif(not rdma_available(), reason="mooncake_cpu requires RDMA"),
]

MiB = 1 << 20


ROWS, PARTITIONS = 4, 24  # 24 x 4 rows x 4 MiB = 384 MiB


def _cfg(ssd_root: str, storage_units: bool, segment_bytes: int = 256 * MiB) -> dict:
    # 256 MiB by default: 1.5x smaller than what _write puts.
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "mooncake_cpu",
        "claim_meta_poll_interval_s": 0.5,
        "mooncake_cpu": {
            "local_buffer_size": 128 * MiB,
            "staging_buffer_size": 32 * MiB,
            "global_segment_size": segment_bytes,
            "storage_unit_segment_size": segment_bytes if storage_units else 0,
            "ssd_offload": {
                "enabled": True,
                "path": ssd_root,
                "offload_threshold": 0.5,
            },
        },
    }


def _start(
    cfg: dict, storage_units: bool, checkpointing: bool = False
):  # pragma: no cover
    import os

    import ray

    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.data_plane.mooncake_storage_unit import start_storage_units

    os.environ.pop("RAY_ADDRESS", None)
    ray.init(num_cpus=4, object_store_memory=2 << 30, include_dashboard=False)
    node_ids = [ray.get_runtime_context().get_node_id()] if storage_units else None
    client = build_data_plane_client(
        cfg, checkpointing=checkpointing, storage_unit_node_ids=node_ids
    )
    return client, start_storage_units(cfg, node_ids)


def _expected(p: int):  # pragma: no cover
    import torch

    return torch.full((ROWS, MiB), float(p))  # 4 MiB per row


def _write(client) -> None:  # pragma: no cover
    from tensordict import TensorDict

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
        # Objects are evictable only once their lease (1 s) expires.
        time.sleep(2.0)


def _read_back(client) -> None:  # pragma: no cover
    import torch

    for p in range(PARTITIONS):
        meta = client.claim_meta(
            partition_id=f"p{p}",
            task_name="read",
            required_fields=["big"],
            batch_size=ROWS,
            timeout_s=60.0,
        )
        data = client.get_data(meta)
        assert torch.equal(data["big"], _expected(p)), f"partition p{p} changed"


def _spill_and_read_back(
    ssd_root: str, storage_units: bool
) -> None:  # pragma: no cover
    import ray

    client, units = _start(_cfg(ssd_root, storage_units), storage_units)
    try:
        _write(client)
        spilled = [f for f in Path(ssd_root).rglob("*") if f.is_file()]
        assert spilled, f"no file was offloaded under {ssd_root}"
        _read_back(client)

        # As the SingleController does at shutdown: Ray's SIGTERM skips the
        # units' own exit cleanup, so they are asked to remove their dirs.
        ray.get([unit.close.remote() for unit in units])
        if storage_units:
            leftover = list(Path(ssd_root, "nemo_rl_mooncake_offload").iterdir())
            assert not leftover, f"offload directories left: {leftover}"
        client.close()
    finally:
        ray.shutdown()


def _disk_only_keys() -> list[str]:  # pragma: no cover
    """Data-plane keys whose only copy is on SSD (offloaded, evicted)."""
    import transfer_queue as tq

    store = tq.get_client().storage_manager.storage_client._store
    # TQ keys are "<global index>@<field>"; probe well past the 96 written.
    replicas = store.batch_get_replica_desc([f"{i}@big" for i in range(1024)])
    present = {key: values for key, values in replicas.items() if values}
    assert len(present) == PARTITIONS * ROWS, sorted(present)
    return [
        key
        for key, values in present.items()
        if not any(value.is_memory_replica() for value in values)
        and any(value.is_local_disk_replica() for value in values)
    ]


def _spill_and_save(ssd_root: str, checkpoint_dir: str) -> None:  # pragma: no cover
    import ray

    from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
        configure_checkpoint_workers,
    )

    client, units = _start(_cfg(ssd_root, True), True, checkpointing=True)
    try:
        configure_checkpoint_workers(list(units))
        _write(client)
        disk_only = _disk_only_keys()
        assert disk_only, "nothing was evicted to SSD before the save"
        print(
            f"saving with {len(disk_only)} of {PARTITIONS * ROWS} objects on SSD only"
        )
        client.save_checkpoint(checkpoint_dir)
        client.close()
    finally:
        ray.shutdown()


def _restore_and_read_back(
    ssd_root: str, checkpoint_dir: str
) -> None:  # pragma: no cover
    import ray

    from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
        configure_checkpoint_workers,
    )

    client, units = _start(_cfg(ssd_root, True), True, checkpointing=True)
    try:
        configure_checkpoint_workers(list(units))
        # 384 MiB into a 256 MiB unit: restore must wait for eviction to SSD.
        client.load_checkpoint(checkpoint_dir)
        _read_back(client)
        client.close()
    finally:
        ray.shutdown()


def _run_child(monkeypatch, target, *args) -> None:
    from nemo_rl.data_plane.adapters.transfer_queue_env import _wanted_engine_env

    # The child inherits this env. A run sets it on the driver before Ray
    # starts (configure_engine_env); without MC_ENABLE_DEST_DEVICE_AFFINITY,
    # mooncake can pair rails across subnets on RoCE and transfers fail.
    for name, value in _wanted_engine_env().items():
        monkeypatch.setenv(name, value)
    proc = mp.get_context("spawn").Process(target=target, args=args)
    proc.start()
    proc.join(timeout=600)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(f"{target.__name__} child timed out after 600 s")
    assert proc.exitcode == 0, f"{target.__name__} child exited with {proc.exitcode}"


@pytest.mark.parametrize("storage_units", [True, False], ids=["units", "clients"])
def test_mooncake_ssd_offload_spills_beyond_owner_memory(
    monkeypatch, tmp_path, storage_units
) -> None:
    _run_child(monkeypatch, _spill_and_read_back, str(tmp_path / "ssd"), storage_units)


def test_mooncake_checkpoint_round_trips_with_ssd_offload(
    monkeypatch, tmp_path
) -> None:
    """Save while objects sit only on SSD; restore into a fresh, smaller-than-data
    system; every value reads back unchanged."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _run_child(
        monkeypatch, _spill_and_save, str(tmp_path / "ssd-save"), str(checkpoint_dir)
    )
    _run_child(
        monkeypatch,
        _restore_and_read_back,
        str(tmp_path / "ssd-load"),
        str(checkpoint_dir),
    )


def _fill_small_disk(ssd_root: str) -> None:  # pragma: no cover
    """Offload more than the disk's quota: the run must stop."""
    import ray
    from tensordict import TensorDict

    # 1 GiB unit: ample room for what is still under its 1 s lease at ~20 MiB/s.
    client, _units = _start(_cfg(ssd_root, True, segment_bytes=1 << 30), True)
    try:
        # ~4 GiB in 16 MiB partitions, paced at ~20 MiB/s so eviction keeps up
        # and more than the 3 GiB disk's 90% quota is offloaded. The watcher
        # (polling every 30 s) interrupts this main thread once Mooncake has
        # deleted SSD files for space.
        for p in range(256):
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
            time.sleep(0.8)
        time.sleep(90)
        import shutil

        usage = shutil.disk_usage(ssd_root)
        raise AssertionError(
            "the run was not stopped after SSD evictions "
            f"(disk used {usage.used >> 20} of {usage.total >> 20} MiB)"
        )
    except KeyboardInterrupt:
        print("run stopped by the SSD eviction watcher")
    finally:
        ray.shutdown()


def test_mooncake_ssd_eviction_stops_the_run(monkeypatch, tmp_path) -> None:
    """Mooncake deletes SSD files once one owner's files pass 90% of the disk.

    Manual only: mounting the small tmpfs needs CAP_SYS_ADMIN, which CI's
    container lacks, so CI skips it. The CPU watcher tests cover the rest.

    (Below the free-space floor of one object + 256 MiB it fails the write
    instead, which ends the run through failed puts, so the disk must be big
    enough that the 90% quota is reached first: 10% > 256 MiB, i.e. > 2.6 GiB.)
    """
    import subprocess

    small_disk = tmp_path / "small-ssd"
    small_disk.mkdir()
    mounted = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", "size=3g", "tmpfs", str(small_disk)],
        capture_output=True,
    )
    if mounted.returncode != 0:
        pytest.skip(f"cannot mount a small tmpfs: {mounted.stderr.decode().strip()}")
    try:
        _run_child(monkeypatch, _fill_small_disk, str(small_disk))
    finally:
        subprocess.run(["umount", str(small_disk)], check=False)
