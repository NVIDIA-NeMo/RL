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

"""Regression tests for the MessageQueue remote-socket bind patch (RL-1111).

``MessageQueue.__init__`` picks the port for its remote (TCP) socket with
``get_open_port()``, which binds a probe socket, *releases it*, and returns the
number; ZMQ only binds the port for real several statements later. Every
``RayWorkerProc`` on a non-driver node takes ``n_local_reader=0`` and so needs
one of these TCP ports, and they all scan from the same ``VLLM_PORT``. Because
``_init_message_queues`` runs immediately after ``init_device()`` -- whose
process-group setup is a collective barrier -- the workers on a node arrive at
the probe together, all see the same port free, and all but one die with
``zmq.error.ZMQError: Address already in use``.

Workers on the *driver* node use an ``ipc://`` socket instead and never take a
TCP port, so only an engine spanning >= 2 nodes can hit this -- and no nightly
test runs one (none has ``tensor_parallel_size * pipeline_parallel_size >
cluster.gpus_per_node``). These tests reproduce the race with plain processes,
so it is covered without a GPU or a second node.
"""

import ast
import logging
import multiprocessing as mp
import socket
from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm import patches
from tests.unit.models.generation.vllm_patch_source_utils import (
    write_unpatched_copy,
)

pytestmark = pytest.mark.vllm

_VLLM_MQ_SOURCE = "distributed/device_communicators/shm_broadcast.py"
_MARKER = "_nrl_bind_attempts"
# Enough concurrent binders that a lost race is essentially certain: unpatched,
# this leaves 6 of 8 workers dead.
_RACERS = 8
_LOOPBACK = "127.0.0.1"


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


def _load_message_queue(source_path: Path):
    """Import `source_path` as a standalone module and return its MessageQueue.

    Loaded from the file rather than from ``vllm`` so the test exercises the
    exact bytes the worker patches on disk. Every import inside the file is
    absolute, so it loads cleanly under a private module name.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"nrl_test_shm_broadcast_{source_path.parent.name}", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MessageQueue


def _bind_racer(message_queue_cls, ready, hold, results, index: int) -> None:
    """One worker: build a remote-reader MessageQueue, then hold its port."""
    ready.wait(timeout=120)  # stands in for the init_device() collective
    try:
        queue = message_queue_cls(n_reader=1, n_local_reader=0, connect_ip=_LOOPBACK)
        results.put((index, queue.handle.remote_subscribe_addr))
    except Exception as error:  # noqa: BLE001 - the failure under test
        results.put((index, f"ERROR {type(error).__name__}: {error}"))
        queue = None
    # A real RayWorkerProc keeps its response queue for the engine's lifetime.
    # Returning here would drop the socket and free the port mid-race, letting a
    # straggler legitimately reuse it and masking a genuine collision.
    try:
        hold.wait(timeout=120)
    except Exception:  # noqa: BLE001 - a crashed peer breaks the barrier
        pass
    del queue


def _race(message_queue_cls, racers: int = _RACERS) -> list[str]:
    """Build `racers` MessageQueues at once; return each one's result string."""
    # fork so the children inherit the already-loaded module object under test;
    # spawn would re-import vLLM per child and cannot carry it across.
    ctx = mp.get_context("fork")
    ready, hold = ctx.Barrier(racers), ctx.Barrier(racers)
    results = ctx.Queue()

    procs = [
        ctx.Process(
            target=_bind_racer, args=(message_queue_cls, ready, hold, results, index)
        )
        for index in range(racers)
    ]
    for proc in procs:
        proc.start()
    collected = dict(results.get(timeout=180) for _ in range(racers))
    for proc in procs:
        proc.join(timeout=180)
    return [collected[index] for index in range(racers)]


@pytest.fixture
def pristine_source(tmp_path) -> Path:
    """A copy of the installed vLLM shm_broadcast.py, unpatched.

    The installed copy may already carry the patch: the vLLM lane rewrites
    site-packages in place as soon as any earlier test builds a generation
    worker. Reversing it keeps this fixture honest whatever the test order --
    without that, the unpatched racer never loses and its negative-control
    test skips itself as "the race did not materialize".
    """
    return write_unpatched_copy(
        _VLLM_MQ_SOURCE,
        "_patch_vllm_shm_broadcast_bind_retry",
        tmp_path / "pristine" / "shm_broadcast.py",
    )


@pytest.fixture
def patched_source(tmp_path, monkeypatch) -> Path:
    """The same copy after the NeMo-RL MessageQueue bind patch is applied."""
    copied = write_unpatched_copy(
        _VLLM_MQ_SOURCE,
        "_patch_vllm_shm_broadcast_bind_retry",
        tmp_path / "patched" / "shm_broadcast.py",
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_shm_broadcast_bind_retry(logging.getLogger(__name__))
    return copied


@pytest.fixture
def reserved_band(monkeypatch) -> int:
    """Point VLLM_PORT at a free band, as configure_worker does for an engine."""
    import vllm.envs as envs

    # Another test in this process may have frozen envs after a fake init.
    if hasattr(envs, "disable_envs_cache"):
        envs.disable_envs_cache()
    base = _find_free_port_band(_RACERS * 2)
    monkeypatch.setenv("VLLM_PORT", str(base))
    return base


def test_patch_anchor_still_matches_installed_vllm(patched_source):
    """The patch is a source edit, so an upstream rename makes it a silent no-op."""
    assert _MARKER in patched_source.read_text(), (
        "the MessageQueue bind-retry patch did not apply to the installed vLLM; "
        "its anchor snippet has probably changed upstream"
    )
    ast.parse(patched_source.read_text())  # the edit must leave valid Python


def test_patch_is_idempotent(patched_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_source)
    )
    patches._patch_vllm_shm_broadcast_bind_retry(logging.getLogger(__name__))
    assert patched_source.read_text() == before


def test_probe_releases_the_port_it_returns(reserved_band):
    """The root cause, without a race: the probe does not hold what it hands out."""
    from vllm.utils.network_utils import get_open_port

    # Two independent callers -- i.e. two workers on the same node -- are handed
    # the same port, because nothing is holding it between the calls.
    assert get_open_port() == get_open_port() == reserved_band


def test_unpatched_message_queue_loses_the_port_race(pristine_source, reserved_band):
    """Documents the bug: concurrent workers collide on one port."""
    outcomes = _race(_load_message_queue(pristine_source))
    failures = [outcome for outcome in outcomes if outcome.startswith("ERROR")]
    if not failures:
        pytest.skip("the port race did not materialize on this machine")
    assert any("Address already in use" in failure for failure in failures)


def test_patched_message_queue_survives_the_port_race(patched_source, reserved_band):
    """The fix: every worker binds, on its own port, inside the reserved band."""
    from nemo_rl.distributed.virtual_cluster import DEFAULT_VLLM_PORTS_PER_ENGINE

    outcomes = _race(_load_message_queue(patched_source))

    assert not [outcome for outcome in outcomes if outcome.startswith("ERROR")], (
        f"workers failed to bind their response queue: {outcomes}"
    )
    ports = [int(outcome.rsplit(":", 1)[1]) for outcome in outcomes]
    assert len(set(ports)) == _RACERS, f"workers share a port: {sorted(ports)}"
    # Ports must stay below the OS ephemeral floor (as low as 9000 on GB200);
    # falling back to kernel-assigned ports is the contention #2380 fixed.
    assert all(
        reserved_band <= port < reserved_band + DEFAULT_VLLM_PORTS_PER_ENGINE
        for port in ports
    ), f"ports escaped the engine's reserved band: {sorted(ports)}"


def test_patched_message_queue_still_works_without_vllm_port(
    patched_source, monkeypatch
):
    """Without VLLM_PORT there is no reserved band; keep vLLM's own behaviour."""
    import vllm.envs as envs

    if hasattr(envs, "disable_envs_cache"):
        envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_PORT", raising=False)

    message_queue_cls = _load_message_queue(patched_source)
    queue = message_queue_cls(n_reader=1, n_local_reader=0, connect_ip=_LOOPBACK)
    assert queue.handle.remote_subscribe_addr.startswith(f"tcp://{_LOOPBACK}:")
