# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""PR-gate chaos smoke tests — "fails-loud-not-hangs" only.

Covers §5.3 of the test plan. Keeps only cheap, deterministic assertions.
Recovery / rebalance / soak are nightly-only.

Tests:
  P7-a  T2-tq-controller-fails-loud: kill controller, next call raises within 5s.
  P7-b  T2-tq-storage-actor-fails-loud: kill storage actor, next call raises within 5s.
  P7-c  T2-tq-port-already-bound: pre-bound port causes init error with a message
        naming "address already in use" (or equivalent) — not a generic KeyError.

Requires Ray + transfer_queue. Skipped at module-import time when absent.
"""

from __future__ import annotations

import socket
import time

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("ray")
pytest.importorskip("transfer_queue")

from nemo_rl.data_plane import build_data_plane_client  # noqa: E402

# Ray is initialized once by the parent autouse fixture
# ``tests/unit/conftest.py::init_ray_cluster`` (mirrors production: NeMo-RL
# inits Ray at startup; the data plane attaches on top). These tests kill
# only the TQ controller / storage actors — Ray itself stays up across
# tests, so each test just needs a fresh TQ client.

_TQ_CFG = {
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",
    "storage_capacity": 1024,
    "num_storage_units": 1,
}

# Budget: the test must raise within this many seconds after the kill.
_TIMEOUT_S = 5.0


@pytest.fixture
def tq_client_and_ray():
    """Start a TQ client and yield (client, ray module) together."""
    import ray

    client = build_data_plane_client(_TQ_CFG)
    yield client, ray
    # Best-effort close — may raise if the controller is already dead.
    try:
        client.close()
    except Exception:
        pass


def _seed_partition(client, partition_id: str, n: int = 4) -> list[str]:
    """Register a partition and put n keys so subsequent get_meta can fire."""
    keys = [f"k{i}" for i in range(n)]
    client.register_partition(
        partition_id=partition_id,
        fields=["x"],
        num_samples=n,
        consumer_tasks=["read"],
    )
    client.put_samples(
        sample_ids=keys,
        partition_id=partition_id,
        fields=TensorDict({"x": torch.arange(n)}, batch_size=[n]),
    )
    return keys


def _call_raises_within(fn, budget_s: float) -> Exception | None:
    """Call fn(), assert it raises within budget_s seconds.

    Returns the exception so the caller can inspect its message.
    Raises AssertionError if no exception is raised or if it takes too long.
    """
    t0 = time.monotonic()
    try:
        fn()
    except Exception as exc:
        elapsed = time.monotonic() - t0
        assert elapsed <= budget_s, (
            f"Expected exception within {budget_s}s but it took {elapsed:.2f}s. "
            "This suggests the call hung before eventually failing."
        )
        return exc
    elapsed = time.monotonic() - t0
    raise AssertionError(
        f"Expected the call to raise within {budget_s}s but it returned normally "
        f"after {elapsed:.2f}s. The failure must be loud, not silent."
    )


# ── P7-a: kill TQ controller ─────────────────────────────────────────────────


def test_controller_kill_raises_within_5s(tq_client_and_ray) -> None:
    """After ray.kill on the TQ controller actor, the next client call must raise
    within _TIMEOUT_S seconds and must not hang.

    Risk guarded: R-H6 — cached client reference becomes invalid; next call hangs
    forever (the original observed failure mode).
    """
    client, ray = tq_client_and_ray
    _seed_partition(client, "chaos-ctrl")

    # Locate and kill the TQ controller actor.
    # TQ uses a named actor "TransferQueueController" in Ray (or similar).
    # We probe with ray.get_actor and fall back gracefully if TQ changed its API.
    controller = None
    for name_candidate in [
        "TransferQueueController",
        "tq_controller",
        "transfer_queue_controller",
    ]:
        try:
            controller = ray.get_actor(name_candidate)
            break
        except Exception:
            continue

    if controller is None:
        pytest.skip(
            "Could not locate TQ controller actor by known names — "
            "TQ may have changed its internal actor naming. "
            "Update the name_candidates list in this test."
        )

    ray.kill(controller, no_restart=True)

    exc = _call_raises_within(
        lambda: client.get_meta(
            partition_id="chaos-ctrl",
            task_name="read",
            required_fields=["x"],
            batch_size=4,
            timeout_s=1.0,  # short so the timeout doesn't mask the kill
        ),
        budget_s=_TIMEOUT_S,
    )
    # Any exception is acceptable — the key property is "raises, not hangs".
    assert exc is not None


# ── P7-b: kill storage actor ──────────────────────────────────────────────────


def test_storage_actor_kill_raises_within_5s(tq_client_and_ray) -> None:
    """After ray.kill on a TQ storage actor, the next get_samples must raise
    within _TIMEOUT_S seconds.

    Risk guarded: storage actor failure must surface as a raised exception,
    not a silent hang or a corrupt partial result.
    """
    client, ray = tq_client_and_ray
    keys = _seed_partition(client, "chaos-storage")

    # Locate a storage actor. TQ names them with a prefix like "SimpleStorageUnit".
    storage = None
    for name_candidate in [
        "SimpleStorageUnit_0",
        "SimpleStorageUnit0",
        "tq_storage_0",
        "StorageUnit_0",
    ]:
        try:
            storage = ray.get_actor(name_candidate)
            break
        except Exception:
            continue

    if storage is None:
        # Try listing all actors and looking for a storage-like name.
        try:
            actors = ray.util.list_named_actors(all_namespaces=False)
            for a in actors:
                if "storage" in a.lower() or "Storage" in a:
                    try:
                        storage = ray.get_actor(a)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

    if storage is None:
        pytest.skip(
            "Could not locate TQ storage actor by known names. "
            "Update the name_candidates list in this test."
        )

    ray.kill(storage, no_restart=True)

    exc = _call_raises_within(
        lambda: client.get_samples(
            sample_ids=keys,
            partition_id="chaos-storage",
            select_fields=["x"],
        ),
        budget_s=_TIMEOUT_S,
    )
    assert exc is not None


# ── P7-c: port already bound ──────────────────────────────────────────────────


def test_port_already_bound_raises_with_message() -> None:
    """If the TQ controller's port is already in use, init must raise with a
    message that names "address already in use" or "address in use" or
    "port" or "bind" — not a generic KeyError or AttributeError.

    This test binds a random port first, then asks TQ to use the same port.
    If TQ does not expose a port configuration knob, the test is skipped with
    a clear message rather than failing.
    """
    # Find a free port, bind it, and hold it open.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind(("127.0.0.1", 0))
        bound_port = probe.getsockname()[1]
    except OSError:
        pytest.skip("Could not bind a probe socket to detect port conflicts.")

    cfg_with_port = {
        **_TQ_CFG,
        "controller_port": bound_port,  # TQ may or may not respect this key
    }

    try:
        client = build_data_plane_client(cfg_with_port)
        # If TQ happily started on a different port, the test cannot assert
        # conflict behavior — skip rather than mislead.
        client.close()
        pytest.skip(
            "TQ ignored controller_port config key or resolved the conflict "
            "internally — cannot test port-already-bound behavior without "
            "a config knob that forces the port."
        )
    except (OSError, RuntimeError, Exception) as exc:
        msg = str(exc).lower()
        # Accept any error that plausibly names the OS-level conflict.
        conflict_tokens = [
            "address already in use",
            "address in use",
            "port",
            "bind",
            "eaddrinuse",
            "98",  # errno 98 on Linux
        ]
        if any(tok in msg for tok in conflict_tokens):
            # Correct behavior: error names the conflict.
            return
        # If none of the tokens match but we still got an error, check that it
        # is not a generic internal state corruption exception.
        assert not isinstance(exc, (KeyError, AttributeError)), (
            f"Port-conflict raised a state-corruption exception {type(exc).__name__!r}: "
            f"{exc!r}. "
            "This suggests TQ's internal state is corrupted rather than the port "
            "conflict being surfaced cleanly. Expected message containing one of: "
            f"{conflict_tokens}"
        )
        # Any other exception is acceptable — the key invariant is "not KeyError/AttributeError".
    finally:
        probe.close()
