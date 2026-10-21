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
"""TQ must count the host's physical cores once per process, not per put/get.

Upstream ``limit_pytorch_auto_parallel_threads`` wraps every client put and
get and calls ``psutil.cpu_count(logical=False)`` each time. That globs
``/sys`` and opens a file per CPU, releasing the GIL on every access, so in a
GIL-saturated process (a vLLM worker) the count dominated each TQ write.

The tests drive TQ's own context manager, so they fail if a TQ revision
stops looking the count up through the module's ``psutil`` — the case where
the patch would silently stop doing anything.
"""

from __future__ import annotations

import subprocess
import sys

import psutil
import pytest
import torch

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

tq_common = pytest.importorskip(
    "transfer_queue.utils.common",
    reason="transfer_queue not installed",
)


class _CountingPsutil:
    """Stands in for ``psutil``: records each count and reports fixed core counts."""

    def __init__(self, physical: int, logical: int) -> None:
        self.physical = physical
        self.logical = logical
        self.calls: list[bool] = []

    def cpu_count(self, logical: bool = True) -> int:
        self.calls.append(logical)
        return self.logical if logical else self.physical


def test_tq_counts_physical_cores_once_across_puts_and_gets(monkeypatch) -> None:
    """Every put and get enters this context; only the first may count."""
    counting = _CountingPsutil(physical=4, logical=8)
    monkeypatch.setattr(
        tq_common, "psutil", tq_adapter._PhysicalCoreCountCache(counting)
    )

    for _ in range(3):
        with tq_common.limit_pytorch_auto_parallel_threads(info="test"):
            # TQ caps torch's threads at the physical core count it was given,
            # so the cached count is the one TQ actually used.
            assert torch.get_num_threads() == 4

    assert counting.calls == [False]


def test_logical_counts_and_other_psutil_calls_pass_through() -> None:
    counting = _CountingPsutil(physical=4, logical=8)
    cache = tq_adapter._PhysicalCoreCountCache(counting)

    assert cache.cpu_count() == 8
    assert cache.cpu_count(logical=True) == 8
    assert counting.calls == [True, True]
    # TQ reaches psutil for more than the core count elsewhere; the proxy must
    # keep every other attribute.
    assert tq_adapter._PhysicalCoreCountCache(psutil).Process is psutil.Process


def test_an_unknown_physical_count_is_also_counted_once() -> None:
    """psutil returns None when it cannot tell; that answer is cached too."""
    counting = _CountingPsutil(physical=None, logical=8)  # type: ignore[arg-type]
    cache = tq_adapter._PhysicalCoreCountCache(counting)

    assert cache.cpu_count(logical=False) is None
    assert cache.cpu_count(logical=False) is None
    assert counting.calls == [False]


def test_installed_proxy_reports_the_real_physical_count() -> None:
    assert isinstance(tq_common.psutil, tq_adapter._PhysicalCoreCountCache)
    assert tq_common.psutil.cpu_count(logical=False) == psutil.cpu_count(logical=False)


def test_patch_is_idempotent() -> None:
    """A second call must not wrap the proxy in another proxy."""
    first = tq_common.psutil
    tq_adapter._patch_physical_core_count()
    assert tq_common.psutil is first


def test_missing_psutil_reference_raises_drift_error(monkeypatch) -> None:
    """If TQ stops counting through this module, say so instead of patching
    nothing."""
    monkeypatch.setattr(tq_common, "_nrl_core_count_cached", False)
    monkeypatch.delattr(tq_common, "psutil")

    with pytest.raises(RuntimeError, match="pin in pyproject.toml has moved"):
        tq_adapter._patch_physical_core_count()


def test_patch_is_installed_by_import_alone() -> None:
    """Importing the adapter must arm the process — no client required.

    ``SingleControllerActor`` receives its client by cloudpickle, so
    ``TQDataPlaneClient.__init__`` never runs there; a fresh interpreter that
    only imports is the one thing that detects the patch moving out of import.
    """
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import nemo_rl.data_plane.adapters.transfer_queue;"
            "from transfer_queue.utils import common;"
            "assert common._nrl_core_count_cached",
        ]
    )
