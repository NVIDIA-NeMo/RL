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

"""Regression tests for the RayExecutorV2 TCPStore port patch (RL-1104).

vLLM's ``RayExecutorV2`` allocates the torch.distributed TCPStore port with a
bind-probe that immediately releases the socket, then builds the broadcast
``MessageQueue``, which allocates from the same ``VLLM_PORT`` base and *binds
and holds* what it gets. For an engine that spans nodes the queue needs a real
TCP socket, so it deterministically takes the port the probe just released and
the rank-0 worker dies with ``EADDRINUSE``.

That failure needs a >= 2-node engine to show up at runtime, and no nightly test
has one. These tests pin the port arithmetic instead, which needs no GPU and no
second node, so the whole class of failure is covered by the unit suite.
"""

import ast
import logging
import socket
from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm import patches
from tests.unit.models.generation.vllm_patch_source_utils import (
    write_unpatched_copy,
)

pytestmark = pytest.mark.vllm

_VLLM_EXECUTOR_SOURCE = "v1/executor/ray_executor_v2.py"
_WINDOW = 32
# What ``ParallelConfig`` hands a plain non-DP engine: it takes the offline-SPMD
# path and copies VLLM_DP_RANK_LOCAL / VLLM_DP_MASTER_PORT, which default to 0.
# So ``local_dp_rank`` is 0 rather than None -- the reason a fix placed inside
# the ``local_dp_rank is None`` branch is dead code.
_NON_DP_LOCAL_RANK = 0
_NON_DP_MASTER_PORT = 0


def _find_free_port_band(width: int) -> int:
    """Return a base port with `width` consecutive bindable ports above it."""
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("", 0))
            base = probe.getsockname()[1]
        held = []
        try:
            for offset in range(width):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", base + offset))
                held.append(sock)
        except OSError:
            continue
        finally:
            for sock in held:
                sock.close()
        return base
    pytest.skip("could not find a free port band for the test")


def _load_select_tcpstore_port(source_path: Path):
    """Return ``_select_tcpstore_port`` as defined in `source_path`.

    The function is exec'd on its own rather than importing the module so the
    test reads the file the worker actually patches without pulling Ray in.
    """
    import vllm.envs as envs
    from vllm.utils.network_utils import _get_open_port, get_open_port

    tree = ast.parse(source_path.read_text())
    func = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "RayExecutorV2"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_tcpstore_port"
    )
    func.decorator_list = []  # drop @staticmethod so the result is callable
    namespace: dict = {
        "envs": envs,
        "_get_open_port": _get_open_port,
        "get_open_port": get_open_port,
    }
    module = ast.Module(body=[func], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace
    )
    return namespace["_select_tcpstore_port"]


@pytest.fixture
def pristine_source(tmp_path) -> Path:
    """A copy of the installed vLLM ray_executor_v2.py, unpatched.

    The installed copy may already carry the patch: the vLLM lane rewrites
    site-packages in place as soon as any earlier test builds a generation
    worker. Reversing it keeps this fixture honest whatever the test order.
    """
    return write_unpatched_copy(
        _VLLM_EXECUTOR_SOURCE,
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        tmp_path / "ray_executor_v2.py",
    )


@pytest.fixture
def patched_source(pristine_source, monkeypatch) -> Path:
    """The same copy after the NeMo-RL TCPStore port patch is applied to it."""
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(pristine_source)
    )
    patches._patch_vllm_ray_executor_v2_tcpstore_port(logging.getLogger(__name__))
    return pristine_source


def test_patch_anchor_still_matches_installed_vllm(patched_source):
    """The patch is a source edit, so an upstream rename makes it a silent no-op."""
    assert "start_port=envs.VLLM_PORT + 32" in patched_source.read_text(), (
        "the TCPStore port patch did not apply to the installed vLLM; its anchor "
        "snippet has probably changed upstream"
    )
    ast.parse(patched_source.read_text())  # the edit must leave valid Python


def test_unpatched_vllm_picks_an_unusable_tcpstore_port(pristine_source, monkeypatch):
    """Documents the bug, and proves the patch is load-bearing.

    Unpatched, ``local_dp_rank`` is 0 rather than None, so vLLM takes its DP
    branch and searches from ``master_port + 100 + local_dp_rank * 32`` = 100.
    What happens next depends on whether the process may bind privileged ports,
    and *both* outcomes are broken:

    * cannot bind < 1024 (the CI cluster) -- all 32 attempts fail and it falls
      through to ``get_open_port()``, i.e. straight back to ``VLLM_PORT``. That
      is the port the MessageQueue binds and holds, so rank 0 dies with
      EADDRINUSE. This is what DeepSeek-V3 hit on port 7000.
    * can bind < 1024 (this unit-test container runs as root) -- it returns 100,
      outside the engine's reserved band and identical for *every* engine on the
      node, since each computes the same ``0 + 100 + 0 * 32``.

    Asserting either specific port would pin the test to one environment, so
    assert the property the patch actually restores: a port inside this engine's
    reserved band that the MessageQueue will not also take.
    """
    from nemo_rl.distributed.virtual_cluster import DEFAULT_VLLM_PORTS_PER_ENGINE

    base = _find_free_port_band(_WINDOW * 2)
    monkeypatch.setenv("VLLM_PORT", str(base))

    select = _load_select_tcpstore_port(pristine_source)
    from vllm.utils.network_utils import get_open_port

    tcpstore_port = select(_NON_DP_LOCAL_RANK, _NON_DP_MASTER_PORT)
    # get_open_port() is what MessageQueue calls for its remote subscribe socket.
    in_band = base <= tcpstore_port < base + DEFAULT_VLLM_PORTS_PER_ENGINE
    disjoint = tcpstore_port != get_open_port()

    assert not (in_band and disjoint), (
        f"unpatched vLLM returned TCPStore port {tcpstore_port} for VLLM_PORT="
        f"{base}, which is both in-band and disjoint from the MessageQueue's "
        "port -- the bug this patch exists for would not reproduce, so the "
        "patch may no longer be needed"
    )


def test_patched_tcpstore_port_avoids_the_messagequeue_scan(
    patched_source, monkeypatch
):
    """The fix: the TCPStore lands a window above what the MessageQueue takes."""
    base = _find_free_port_band(_WINDOW * 2)
    monkeypatch.setenv("VLLM_PORT", str(base))

    select = _load_select_tcpstore_port(patched_source)
    from vllm.utils.network_utils import get_open_port

    tcpstore_port = select(_NON_DP_LOCAL_RANK, _NON_DP_MASTER_PORT)
    assert tcpstore_port == base + _WINDOW
    # get_open_port() is what MessageQueue calls for its remote subscribe socket.
    assert tcpstore_port != get_open_port()


def test_patched_tcpstore_port_stays_in_the_engines_reserved_band(
    patched_source, monkeypatch
):
    """Ports must stay below the OS ephemeral floor (as low as 9000 on GB200)."""
    from nemo_rl.distributed.virtual_cluster import DEFAULT_VLLM_PORTS_PER_ENGINE

    base = _find_free_port_band(_WINDOW * 2)
    monkeypatch.setenv("VLLM_PORT", str(base))

    select = _load_select_tcpstore_port(patched_source)
    tcpstore_port = select(_NON_DP_LOCAL_RANK, _NON_DP_MASTER_PORT)
    assert base <= tcpstore_port < base + DEFAULT_VLLM_PORTS_PER_ENGINE


def test_patched_select_falls_back_when_vllm_port_is_unset(patched_source, monkeypatch):
    """Without VLLM_PORT there is no reserved band; keep vLLM's own behaviour."""
    monkeypatch.delenv("VLLM_PORT", raising=False)

    select = _load_select_tcpstore_port(patched_source)
    assert isinstance(select(_NON_DP_LOCAL_RANK, _NON_DP_MASTER_PORT), int)


def test_patch_is_idempotent(patched_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_source)
    )
    patches._patch_vllm_ray_executor_v2_tcpstore_port(logging.getLogger(__name__))
    assert patched_source.read_text() == before
