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
"""CPU-only real Ray/ZMQ capture, concurrent mutation, failure and restore test.

Run in a process with no existing Ray cluster or TQ client. A test-only actor
subclass gates persistence so real client requests must finish before the file
write is released. The checkpoint mutation cut is enforced by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ray
import torch
import transfer_queue as tq
import transfer_queue.interface as interface
from omegaconf import OmegaConf
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.tq_controller_patch import configure_controller_patch
from nemo_rl.data_plane.adapters.tq_simple_checkpoint import (
    capture_simple_checkpoint,
    configure_simple_checkpoint,
)


def wait_for_file(path: Path) -> None:
    """Wait for the actor gate without hanging a failed test indefinitely."""
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Actor did not reach persistence gate: {path}")
        time.sleep(0.01)


def configure_test_controller() -> None:
    """Preserve the production actor, adding only a persistence gate/fault."""
    original = interface.TransferQueueController
    base = original.__ray_metadata__.modified_class

    class TestController(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.test_gate: str | None = None
            self.test_fail_capture = False
            torch.set_num_threads(1)

        def test_configure(self, gate: str | None, fail_capture: bool = False) -> None:
            self.test_gate = gate
            self.test_fail_capture = fail_capture

        def test_retained(self) -> int:
            return len(self._controller_snapshot_views)

        def nrl_capture_snapshot(self, token: str) -> str:
            if self.test_fail_capture:
                raise ValueError("injected controller capture failure")
            return super().nrl_capture_snapshot(token)

        def nrl_persist_snapshot(self, token: str, path: str) -> int:
            if self.test_gate is not None:
                gate = Path(self.test_gate)
                (gate / "entered").touch()
                wait_for_file(gate / "release")
            return super().nrl_persist_snapshot(token, path)

    interface.TransferQueueController = ray.remote(**original._default_options)(
        TestController
    )


def put(keys: list[str], values: list[int]) -> None:
    """Exercise the native ZMQ client with real tensor payloads and tags."""
    tq.kv_batch_put(
        keys=keys,
        partition_id="capture-test",
        fields=TensorDict(
            {"values": torch.tensor(values)[:, None]}, batch_size=[len(keys)]
        ),
        tags=[{"value": value, "nested": {"items": [value]}} for value in values],
    )


def checkpoint_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in path.rglob("*")
        if item.is_file()
    }


def exercise(directory: Path) -> dict[str, Any]:
    """Use actual actors and sockets; no model, Gym server or GPU is needed."""
    configure_controller_patch()
    configure_simple_checkpoint()
    configure_test_controller()
    config = OmegaConf.create(
        {
            "metrics": {"enabled": False},
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 2,
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
        controller = ray.get_actor(
            "TransferQueueController", namespace="transfer_queue"
        )
        put(["a", "b", "c"], [1, 2, 3])
        path = directory / "committed"
        captured = capture_simple_checkpoint(
            path, num_storage_units=2, metadata={"step": 7}
        )
        assert not (path / "controller_state.pkl").exists()
        gate = directory / "gate"
        gate.mkdir()
        ray.get(controller.test_configure.remote(str(gate)))
        mutations = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            writing = pool.submit(captured.persist)
            try:
                wait_for_file(gate / "entered")
                # Persistence is blocked in the actor; ZMQ mutations must finish.
                for index in range(20):
                    put(["a", f"new-{index}"], [100 + index, 200 + index])
                    tq.kv_clear(keys=f"new-{index}", partition_id="capture-test")
                    mutations += 2
                tq.kv_clear(keys="b", partition_id="capture-test")
                mutations += 1
                assert not writing.done()
            finally:
                (gate / "release").touch()
                writing.result(timeout=30)
                captured.release()
        ray.get(controller.test_configure.remote(None))
        assert ray.get(controller.test_retained.remote()) == 0
        committed_hashes = checkpoint_hashes(path)

        # A partial persistence failure must not publish the completion marker.
        partial = directory / "failed-persistence"
        failed = capture_simple_checkpoint(partial, num_storage_units=2, metadata=None)
        (partial / "controller_state.pkl").touch()
        try:
            try:
                failed.persist()
            except ray.exceptions.RayTaskError as error:
                assert "FileExistsError" in str(error)
            else:
                raise AssertionError("Expected exclusive-create persistence failure")
        finally:
            failed.release()
        assert not (partial / "metadata.json").exists()
        assert ray.get(controller.test_retained.remote()) == 0
        assert checkpoint_hashes(path) == committed_hashes

        ray.get(controller.test_configure.remote(None, True))
        failed_path = directory / "failed-capture"
        try:
            capture_simple_checkpoint(failed_path, num_storage_units=2, metadata=None)
        except ray.exceptions.RayTaskError as error:
            assert "injected controller capture failure" in str(error)
        else:
            raise AssertionError("Expected capture failure")
        assert not (failed_path / "metadata.json").exists()
        assert ray.get(controller.test_retained.remote()) == 0
        ray.get(controller.test_configure.remote(None))
        put(["after-failure"], [999])

        tq.close()
        deadline = time.monotonic() + 20
        while True:
            try:
                ray.get_actor("TransferQueueController", namespace="transfer_queue")
            except ValueError:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Old controller did not terminate")
            time.sleep(0.05)
        tq.init(config)
        tq.load_checkpoint(path)
        listing = tq.kv_list(partition_id="capture-test")["capture-test"]
        assert set(listing) == {"a", "b", "c"}
        assert [listing[key]["value"] for key in ("a", "b", "c")] == [1, 2, 3]
        actual = tq.kv_batch_get(keys=["a", "b", "c"], partition_id="capture-test")
        assert [row.item() for row in actual["values"].unbind()] == [1, 2, 3]
        assert checkpoint_hashes(path) == committed_hashes
        return {
            "passed": True,
            "mutations_during_persistence": mutations,
            "fresh_controller_restore": True,
            "fresh_storage_restore": True,
            "capture_failure_checked": True,
            "persistence_failure_checked": True,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "tq": tq.__version__,
            "test_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    finally:
        tq.close()
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="nrl-tq-controller-cpu-") as temporary:
        result = exercise(Path(temporary))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
