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

These tests are the patch's drift guard: they drive TQ's own context manager,
so they fail if a TQ pin stops looking the count up through the module's
``psutil`` — the case where the patch would silently stop doing anything.
"""

from __future__ import annotations

import subprocess
import sys
import types

import psutil
import pytest
import torch

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

tq_common = pytest.importorskip(
    "transfer_queue.utils.common",
    reason="transfer_queue not installed",
)


class _CountingPsutil(types.SimpleNamespace):
    """Stands in for ``psutil``: records each count and reports fixed core counts."""

    def __init__(self, *, physical_cores: int, logical_cores: int) -> None:
        calls: list[bool] = []

        def cpu_count(logical: bool = True) -> int:
            calls.append(logical)
            return logical_cores if logical else physical_cores

        super().__init__(cpu_count=cpu_count, calls=calls, Process=psutil.Process)


def test_tq_counts_physical_cores_once_across_puts_and_gets(monkeypatch) -> None:
    """Every put and get enters this context; only the first may count."""
    counting = _CountingPsutil(physical_cores=4, logical_cores=8)
    monkeypatch.setattr(
        tq_common, "psutil", tq_adapter._with_cached_cpu_count(counting)
    )

    for _ in range(3):
        with tq_common.limit_pytorch_auto_parallel_threads(info="test"):
            # TQ caps torch's threads at the physical core count it was given,
            # so the cached count is the one TQ actually used.
            assert torch.get_num_threads() == 4

    assert counting.calls == [False]


def test_other_psutil_attributes_pass_through() -> None:
    """A later TQ that uses more of ``psutil`` in this module must keep working."""
    cached = tq_adapter._with_cached_cpu_count(psutil)

    assert cached.Process is psutil.Process
    assert cached.virtual_memory is psutil.virtual_memory
    assert cached.cpu_count(logical=True) == psutil.cpu_count(logical=True)


def test_installed_namespace_reports_the_real_physical_count() -> None:
    assert tq_common._nrl_core_count_cached
    assert tq_common.psutil.cpu_count(logical=False) == psutil.cpu_count(logical=False)


def test_patch_is_idempotent() -> None:
    """A second call must not cache the cached namespace again."""
    first = tq_common.psutil
    tq_adapter._patch_physical_core_count()
    assert tq_common.psutil is first


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
