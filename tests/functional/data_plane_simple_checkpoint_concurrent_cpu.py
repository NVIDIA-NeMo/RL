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
"""CPU-only test: concurrent writers versus a SimpleStorage capture.

Capture is not atomic across the controller and the storage shards. It is N+1
separate Ray RPCs. The cut is consistent only while the caller keeps every
canonical write out. The single-controller does that with the real
`DataPlaneCheckpointBarrier`: capture runs inside `barrier.checkpoint()` and
every writer runs inside `barrier.mutation(kind)`. This test uses that real
barrier with real Ray/ZMQ actors and no GPU.

  unfenced : a clear lands between the controller RPC and the shard RPCs,
             with no barrier. Asserted: the restore has a dangling row. This
             is the contract the barrier exists to protect.
  drained  : a writer is inside `mutation()` when `checkpoint()` is asked for.
             The checkpoint must wait for it, so the cut contains the clear.
  fenced   : a writer asks for `mutation()` while `checkpoint()` is held across
             capture. It must wait, so the restored state equals the cut.
  stress   : an async writer mutates continuously through `mutation()` while
             captures run through `checkpoint()`. Every restored checkpoint
             must equal an oracle copied inside the checkpoint section.
             Persistence runs after the checkpoint section ends.

Writers that skip the barrier are not covered here: this test cannot prove
that every production caller takes it.

Run in a process with no existing Ray cluster or TQ client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue.storage.bootstrap import simple_storage_bootstrap as bootstrap

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.data_plane.adapters.tq_controller_patch import configure_controller_patch
from nemo_rl.data_plane.adapters.tq_simple_checkpoint import (
    capture_simple_checkpoint,
    configure_simple_checkpoint,
)

PARTITION = "concurrent-test"
NUM_UNITS = 2
SEED = {f"k{i}": i for i in range(8)}
CLEAR_KIND = "sample_clears"
PUT_KIND = "group_commits"


def wait_for(predicate: Any, what: str, timeout_s: float = 30.0) -> None:
    """Poll a condition; fail loudly instead of hanging."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {what}")
        time.sleep(0.01)


async def await_for(predicate: Any, what: str, timeout_s: float = 30.0) -> None:
    """Same as `wait_for`, but keeps the event loop running."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {what}")
        await asyncio.sleep(0.01)


def configure_gated_units() -> None:
    """Keep the production storage actor; add only a gate before shard capture."""
    actor_class = bootstrap.SimpleStorageUnit
    base = actor_class.__ray_metadata__.modified_class

    class GatedUnit(base):
        def test_configure(self, gate: str | None) -> None:
            self.test_gate = gate

        def nrl_capture_snapshot(self, token: str) -> str:
            gate = getattr(self, "test_gate", None)
            if gate is not None:
                (Path(gate) / f"entered-{self.storage_unit_id}").touch()
                wait_for(lambda: (Path(gate) / "release").exists(), "gate release")
            return super().nrl_capture_snapshot(token)

    bootstrap.SimpleStorageUnit = ray.remote(**actor_class._default_options)(GatedUnit)


def put(keys: list[str], values: list[int]) -> None:
    tq.kv_batch_put(
        keys=keys,
        partition_id=PARTITION,
        fields=TensorDict(
            {"values": torch.tensor(values)[:, None]}, batch_size=[len(keys)]
        ),
        tags=[{"value": value} for value in values],
    )


def restart(config: Any) -> None:
    """Replace the controller and the shards with fresh actors."""
    tq.close()

    def gone() -> bool:
        try:
            ray.get_actor("TransferQueueController", namespace="transfer_queue")
        except ValueError:
            return True
        return False

    wait_for(gone, "old controller to terminate", 20.0)
    tq.init(config)


def read_state() -> tuple[dict[str, int], list[str]]:
    """Return {key: value} for readable rows, and keys the controller lists
    but the shards cannot serve (dangling)."""
    listing = tq.kv_list(partition_id=PARTITION).get(PARTITION, {})
    state: dict[str, int] = {}
    dangling: list[str] = []
    for key in sorted(listing):
        try:
            row = tq.kv_batch_get(keys=[key], partition_id=PARTITION)
            value = int(row["values"].unbind()[0].item())
        except Exception:  # noqa: BLE001 - any failure means "cannot serve"
            dangling.append(key)
            continue
        if value != listing[key]["value"]:
            dangling.append(key)
        else:
            state[key] = value
    return state, dangling


def restore_and_read(config: Any, path: Path) -> tuple[dict[str, int], list[str]]:
    restart(config)
    tq.load_checkpoint(path)
    return read_state()


def shard_actors() -> list[Any]:
    return [ray.get_actor(f"TransferQueueStorageUnit#{rank}") for rank in range(NUM_UNITS)]


def capture(path: Path) -> Any:
    return capture_simple_checkpoint(
        path, num_storage_units=NUM_UNITS, timeout_s=60.0, metadata=None
    )


async def gated_capture(directory: Path, name: str, between: Any) -> tuple[Path, Any]:
    """Capture with every shard gated. `await between()` runs after the
    controller has captured and every shard is waiting, then the gate opens."""
    gate = directory / f"{name}-gate"
    gate.mkdir()
    units = shard_actors()
    ray.get([unit.test_configure.remote(str(gate)) for unit in units])
    path = directory / name
    capturing = asyncio.ensure_future(asyncio.to_thread(capture, path))
    await await_for(
        lambda: len(list(gate.glob("entered-*"))) == NUM_UNITS,
        "all shards to reach the capture gate "
        "(is the SimpleStorageUnit test patch applied?)",
    )
    await between()
    (gate / "release").touch()
    captured = await asyncio.wait_for(capturing, timeout=30)
    ray.get([unit.test_configure.remote(None) for unit in units])
    return path, captured


def finish(captured: Any) -> None:
    try:
        captured.persist()
    finally:
        captured.release()


async def mutate_clear(barrier: DataPlaneCheckpointBarrier, key: str) -> None:
    """A canonical clear the way production does it: inside `mutation()`."""
    async with barrier.mutation(CLEAR_KIND):
        await asyncio.to_thread(tq.kv_clear, keys=key, partition_id=PARTITION)


async def scenario_unfenced(directory: Path, config: Any) -> dict[str, Any]:
    # `exercise` just ran tq.init, so the actors are already fresh.
    put(list(SEED), list(SEED.values()))

    async def clear_without_barrier() -> None:
        # No barrier: this clear lands after the controller RPC, before the shards.
        await asyncio.to_thread(tq.kv_clear, keys="k0", partition_id=PARTITION)

    path, captured = await gated_capture(directory, "unfenced", clear_without_barrier)
    finish(captured)
    state, dangling = restore_and_read(config, path)
    # The gate makes the race deterministic. If no row dangles, the fenced
    # scenario below proves nothing, so fail loudly.
    assert dangling == ["k0"], f"expected k0 to dangle after the clear, got {dangling}"
    return {"restored_keys": sorted(state), "dangling_keys": dangling}


async def scenario_drained(directory: Path, config: Any) -> dict[str, Any]:
    restart(config)
    put(list(SEED), list(SEED.values()))
    barrier = DataPlaneCheckpointBarrier()
    inside = asyncio.Event()
    leave = asyncio.Event()

    async def slow_writer() -> None:
        async with barrier.mutation(CLEAR_KIND):
            await asyncio.to_thread(tq.kv_clear, keys="k0", partition_id=PARTITION)
            inside.set()
            await leave.wait()

    writing = asyncio.ensure_future(slow_writer())
    await asyncio.wait_for(inside.wait(), timeout=30)

    async def checkpoint_after_drain() -> tuple[Path, Any]:
        async with barrier.checkpoint():
            async def nothing() -> None:
                return None

            return await gated_capture(directory, "drained", nothing)

    checkpointing = asyncio.ensure_future(checkpoint_after_drain())
    await asyncio.sleep(0.5)
    assert not checkpointing.done(), "checkpoint ran before the writer left"
    assert not list(directory.glob("drained-gate")), "capture started before the drain"
    leave.set()
    await writing
    path, captured = await asyncio.wait_for(checkpointing, timeout=60)
    finish(captured)
    state, dangling = restore_and_read(config, path)
    assert not dangling, f"drained cut has dangling rows: {dangling}"
    expected = {key: value for key, value in SEED.items() if key != "k0"}
    assert state == expected, f"drained cut differs from the oracle: {state}"
    return {"restored_keys": sorted(state)}


async def scenario_fenced(directory: Path, config: Any) -> dict[str, Any]:
    restart(config)
    put(list(SEED), list(SEED.values()))
    barrier = DataPlaneCheckpointBarrier()
    writers: list[asyncio.Future[None]] = []

    async def between() -> None:
        writers.append(asyncio.ensure_future(mutate_clear(barrier, "k0")))
        await asyncio.sleep(0.5)
        assert not writers[0].done(), "writer ran inside the capture cut"

    async with barrier.checkpoint():
        path, captured = await gated_capture(directory, "fenced", between)
    await asyncio.wait_for(writers[0], timeout=30)
    live, live_dangling = read_state()
    assert "k0" not in live and not live_dangling, (live, live_dangling)
    finish(captured)
    state, dangling = restore_and_read(config, path)
    assert not dangling, f"fenced cut has dangling rows: {dangling}"
    assert state == SEED, f"fenced cut differs from the oracle: {state}"
    return {"restored_keys": sorted(state)}


async def scenario_stress(directory: Path, config: Any) -> dict[str, Any]:
    restart(config)
    barrier = DataPlaneCheckpointBarrier()
    oracle: dict[str, int] = {}
    stop = asyncio.Event()
    mutations = 0
    rng = random.Random(0)

    async def writer() -> None:
        nonlocal mutations
        counter = 0
        while not stop.is_set():
            key = f"k{rng.randrange(16)}"
            if key in oracle and rng.random() < 0.4:
                async with barrier.mutation(CLEAR_KIND):
                    await asyncio.to_thread(
                        tq.kv_clear, keys=key, partition_id=PARTITION
                    )
                    del oracle[key]
            else:
                counter += 1
                async with barrier.mutation(PUT_KIND):
                    await asyncio.to_thread(put, [key], [counter])
                    oracle[key] = counter
            mutations += 1

    checkpoints: list[tuple[Path, dict[str, int]]] = []
    writing = asyncio.ensure_future(writer())
    try:
        for index in range(3):
            # Let the writer make progress between captures.
            start = mutations
            await await_for(lambda: mutations >= start + 3, "writer progress")
            path = directory / f"stress-{index}"
            async with barrier.checkpoint():  # capture and oracle see one cut
                captured = await asyncio.to_thread(capture, path)
                cut = dict(oracle)
            # Persistence runs after the checkpoint section, with the writer live.
            await asyncio.to_thread(finish, captured)
            checkpoints.append((path, cut))
    finally:
        stop.set()
        await asyncio.wait_for(writing, timeout=30)
    for path, cut in checkpoints:
        state, dangling = restore_and_read(config, path)
        assert not dangling, f"{path.name}: dangling rows {dangling}"
        assert state == cut, f"{path.name}: restored {state} != cut {cut}"
    return {
        "checkpoints": len(checkpoints),
        "total_mutations": mutations,
        "cut_sizes": [len(cut) for _, cut in checkpoints],
    }


async def run_scenarios(directory: Path, config: Any) -> dict[str, Any]:
    return {
        "unfenced": await scenario_unfenced(directory, config),
        "drained": await scenario_drained(directory, config),
        "fenced": await scenario_fenced(directory, config),
        "stress": await scenario_stress(directory, config),
        "passed": True,
    }


def exercise(directory: Path) -> dict[str, Any]:
    configure_controller_patch()
    configure_simple_checkpoint()
    configure_gated_units()
    config = OmegaConf.create(
        {
            "metrics": {"enabled": False},
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": NUM_UNITS,
                    "total_storage_size": 1024,
                    "ssd_offload": {"enabled": False},
                },
            },
        }
    )
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        namespace="transfer_queue",
        object_store_memory=128 * 1024**2,
        _node_ip_address="127.0.0.1",
    )
    try:
        tq.init(config)
        return asyncio.run(run_scenarios(directory, config))
    finally:
        tq.close()
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="nrl-tq-concurrent-cpu-") as temporary:
        result = exercise(Path(temporary))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
