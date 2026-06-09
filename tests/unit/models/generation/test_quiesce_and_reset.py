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
"""Unit tests for the RL-412 §2 non-fatal-rendezvous helpers.

Pure/lightweight: no Ray cluster, no HTTP. Covers (a) the actor-death vs
peer-side-timeout classifier that decides reuse-vs-respawn, and (b) the
quiesce-wait that paces retries to a settled gen world.
"""

from __future__ import annotations

import nemo_rl.models.generation.remote_generation as rg
from nemo_rl.models.generation.remote_generation import (
    RemoteGeneration,
    _should_respawn_refit_worker,
)


# =====================================================================
# _should_respawn_refit_worker: reuse on clean pre-NCCL timeout, respawn on
# NCCL poison / actor death / anything else.
# =====================================================================


def test_reuse_on_clean_rendezvous_timeout():
    import torch

    # A pure TCPStore rendezvous timeout (peer didn't check in; no NCCL comm
    # formed) → reuse the RefitWorker (do NOT respawn).
    assert _should_respawn_refit_worker(
        torch.distributed.DistStoreError("7/9 clients joined")
    ) is False
    # DistNetworkError is also pre-NCCL → reuse.
    if hasattr(torch.distributed, "DistNetworkError"):
        assert _should_respawn_refit_worker(
            torch.distributed.DistNetworkError("conn reset")
        ) is False


def test_respawn_on_nccl_poison_actor_death_and_unknown():
    from ray.exceptions import ActorDiedError, RayActorError

    # NCCLError-class failures partially ran the NCCL bootstrap → poisoned
    # context → MUST respawn (reusing it fails forever with NCCLError).
    assert _should_respawn_refit_worker(RuntimeError("NCCL error: unhandled cuda error")) is True
    # Actor death → respawn.
    assert _should_respawn_refit_worker(RayActorError()) is True
    assert _should_respawn_refit_worker(ActorDiedError.__new__(ActorDiedError)) is True
    # Unknown/other → conservative respawn (clears any poison).
    assert _should_respawn_refit_worker(ValueError("???")) is True


# =====================================================================
# _quiesce_wait: pace retries until the gen world settles
# =====================================================================


class _FakeClock:
    """Deterministic stand-in for the module ``time`` so the wait loop runs
    without real sleeping. ``sleep`` advances the monotonic clock."""

    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _bare_remote_generation(world_seq):
    """A RemoteGeneration with __init__ bypassed; `_fetch_gen_world` returns the
    next tuple from ``world_seq`` (repeating the last forever)."""
    rgen = object.__new__(RemoteGeneration)
    calls = {"n": 0}

    def _fake_fetch():
        i = min(calls["n"], len(world_seq) - 1)
        calls["n"] += 1
        return world_seq[i]

    rgen._fetch_gen_world = _fake_fetch  # type: ignore[attr-defined]
    rgen._quiesce_calls = calls  # type: ignore[attr-defined]
    return rgen


def test_quiesce_wait_returns_when_settled(monkeypatch):
    monkeypatch.setattr(rg, "time", _FakeClock())
    # (alive, joinable, stable_s, epoch, reinit): stable past QUIESCE_S(10),
    # not re-initing, world steady at 8 — settled once seen stable across two reads.
    rgen = _bare_remote_generation([(8, 8, 50.0, 0, False)])
    rgen._quiesce_wait()
    # First read sets prev_alive; second confirms stability → ~2 reads.
    assert rgen._quiesce_calls["n"] >= 2


def test_quiesce_wait_bounded_when_never_settles(monkeypatch):
    monkeypatch.setattr(rg, "time", _FakeClock())
    # Always re-initing → never settled → must still return (bounded by max wait).
    rgen = _bare_remote_generation([(8, 4, 0.0, 0, True)])
    rgen._quiesce_wait()  # returns (does not hang)
    assert rgen._quiesce_calls["n"] >= 2


def test_quiesce_wait_requires_stable_world_across_two_reads(monkeypatch):
    monkeypatch.setattr(rg, "time", _FakeClock())
    # World churns 8→5 then holds at 5: settled only once two consecutive reads
    # agree (and stable_s/reinit are fine throughout).
    rgen = _bare_remote_generation([
        (8, 8, 50.0, 0, False),
        (5, 5, 50.0, 0, False),
        (5, 5, 50.0, 0, False),
    ])
    rgen._quiesce_wait()
    assert rgen._quiesce_calls["n"] >= 3


def test_quiesce_wait_returns_immediately_on_older_server(monkeypatch):
    monkeypatch.setattr(rg, "time", _FakeClock())
    # _fetch_gen_world None (older gen server / transport error) → return at once.
    rgen = _bare_remote_generation([None])
    rgen._quiesce_wait()
    assert rgen._quiesce_calls["n"] == 1
