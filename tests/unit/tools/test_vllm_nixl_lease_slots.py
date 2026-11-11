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

"""Exercise the bundled verifier against asynchronous and failed NIXL reads."""

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def lease_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Load the exact newly added module from the distributable unified patch."""
    bundle = (
        Path(__file__).resolve().parents[3]
        / "tools/vllm_patches/nixl-lease-v0.30/lease-verification.patch"
    )
    text = bundle.read_text()
    section = next(
        part
        for part in text.split("diff --git ")
        if part.startswith(
            "a/vllm/distributed/kv_transfer/kv_connector/v1/nixl/lease_slots.py "
        )
    )
    source = "".join(
        line[1:]
        for line in section.splitlines(keepends=True)
        if line.startswith("+") and not line.startswith("+++")
    )
    path = tmp_path_factory.mktemp("bundled-lease") / "lease_slots.py"
    path.write_text(source)
    dependencies = {
        name: ModuleType(name)
        for name in [
            "torch",
            "vllm.distributed.device_communicators.shm_broadcast",
            "vllm.distributed.kv_transfer.kv_connector.utils",
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata",
            "vllm.logger",
        ]
    }

    class Tensor:
        def __init__(self, count: int) -> None:
            self.values = np.zeros(count, dtype=np.int64)

        def numpy(self) -> np.ndarray:
            return self.values

    dependencies["torch"].Tensor = Tensor
    dependencies["torch"].int64 = np.int64
    dependencies["torch"].zeros = lambda count, dtype: Tensor(count)
    dependencies[
        "vllm.distributed.device_communicators.shm_broadcast"
    ].memory_fence = Mock()
    dependencies["vllm.distributed.kv_transfer.kv_connector.utils"].EngineId = str
    dependencies[
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata"
    ].ReqId = str
    dependencies["vllm.logger"].init_logger = logging.getLogger
    spec = importlib.util.spec_from_file_location("bundled_nixl_lease", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**dependencies, spec.name: module}):
        spec.loader.exec_module(module)
    return module


def verifier(
    lease_module: ModuleType,
    *,
    cells: int = 1,
    actual_nonce: int = 71,
    state: str = "DONE",
) -> tuple:
    """Emulate NIXL posting, delayed completion, and writes into landing cells."""
    wire = Mock()
    subject = lease_module.LeaseVerifier(wire, cells)
    subject.local_handle = 10
    subject.add_remote("P", 0, 20)
    subject.add_remote("P", 1, 21)
    requests = {}
    wire.state = state
    wire.post_state = "PROC"

    def prepare(operation, local, local_cells, remote, remote_slots):
        handle = len(requests) + 100
        requests[handle] = (local_cells, remote_slots)
        return handle

    def transfer(handle):
        for cell in requests[handle][0]:
            subject.tensor.numpy()[cell] = actual_nonce
        return wire.post_state

    wire.make_prepped_xfer.side_effect = prepare
    wire.transfer.side_effect = transfer
    wire.check_xfer_state.side_effect = lambda handle: wire.state
    return subject, wire


@pytest.mark.parametrize("nonce", [0, 72])
def test_nonce_mismatch_rejects_request(lease_module: ModuleType, nonce: int) -> None:
    subject, _ = verifier(lease_module, actual_nonce=nonce)
    subject.add_check("request", "P", 0, 0, 71)
    accepted, failed = subject.step(["request"])
    assert not accepted and "released the KV blocks" in failed["request"]


def test_matching_nonce_on_all_ranks(lease_module: ModuleType) -> None:
    subject, wire = verifier(lease_module, cells=2)
    subject.add_check("request", "P", 0, 0, 71)
    subject.add_check("request", "P", 1, 0, 71)
    assert subject.step(["request"]) == ({"request"}, {})
    assert wire.make_prepped_xfer.call_count == 2
    assert len(subject._free_cells) == 2


def test_failed_read_rejects_request(lease_module: ModuleType) -> None:
    subject, _ = verifier(lease_module, state="ERR")
    subject.add_check("request", "P", 0, 0, 71)
    accepted, failed = subject.step(["request"])
    assert not accepted and "could not be read" in failed["request"]
    assert subject._free_cells == [0]


def test_temporary_contention_waits_then_reuses_released_cell(
    lease_module: ModuleType,
) -> None:
    subject, wire = verifier(lease_module, state="PROC")
    subject.add_check("first", "P", 0, 0, 71)
    assert subject.step(["first"]) == (set(), {})
    subject.add_check("second", "P", 0, 0, 71)
    assert subject.step(["second"]) == (set(), {})
    assert wire.make_prepped_xfer.call_count == 1
    wire.state = "DONE"
    assert subject.step([]) == ({"first"}, {})
    assert subject.step([]) == ({"second"}, {})
    assert wire.make_prepped_xfer.call_count == 2


@pytest.mark.parametrize("queued_together", [False, True])
def test_release_failure_quarantines_cell_and_fails_later_request(
    lease_module: ModuleType, queued_together: bool
) -> None:
    subject, wire = verifier(lease_module)
    wire.release_xfer_handle.side_effect = RuntimeError("release failed")
    subject.add_check("first", "P", 0, 0, 71)
    if queued_together:
        subject.add_check("second", "P", 0, 0, 71)
    assert subject.step(["first", "second"] if queued_together else ["first"]) == (
        {"first"},
        {},
    )
    if not queued_together:
        subject.add_check("second", "P", 0, 0, 71)
    accepted, failed = subject.step([] if queued_together else ["second"])
    assert not accepted and "not enough reusable" in failed["second"]
    assert not subject._queued and not subject._reads and not subject._free_cells
    assert wire.make_prepped_xfer.call_count == 1


def test_request_exceeding_total_capacity_fails_without_waiting(
    lease_module: ModuleType,
) -> None:
    subject, wire = verifier(lease_module)
    subject.add_check("too-large", "P", 0, 0, 71)
    subject.add_check("too-large", "P", 1, 0, 71)
    accepted, failed = subject.step(["too-large"])
    assert not accepted and "not enough reusable" in failed["too-large"]
    wire.make_prepped_xfer.assert_not_called()


def test_post_failure_with_unreleased_handle_does_not_reuse_cell(
    lease_module: ModuleType,
) -> None:
    subject, wire = verifier(lease_module)
    wire.post_state = "ERR"
    wire.release_xfer_handle.side_effect = RuntimeError("release failed")
    subject.add_check("first", "P", 0, 0, 71)
    accepted, failed = subject.step(["first"])
    assert not accepted and "could not be read" in failed["first"]
    subject.add_check("second", "P", 0, 0, 71)
    accepted, failed = subject.step(["second"])
    assert not accepted and "not enough reusable" in failed["second"]
    assert wire.make_prepped_xfer.call_count == 1 and not subject._free_cells


def test_bundled_new_module_matches_manifest(lease_module: ModuleType) -> None:
    bundle = Path(__file__).resolve().parents[3] / "tools/vllm_patches/nixl-lease-v0.30"
    manifest = json.loads((bundle / "manifest.json").read_text())
    record = next(
        row for row in manifest["files"] if row["path"].endswith("/lease_slots.py")
    )
    assert (
        hashlib.sha256(Path(lease_module.__file__).read_bytes()).hexdigest()
        == record["patched_sha256"]
    )
