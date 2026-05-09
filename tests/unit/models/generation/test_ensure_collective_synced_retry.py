# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke tests for ensure_collective_synced — straggler exclusion + retry.

These tests pin down the recovery behaviour without needing a Ray cluster
or real NCCL: ray.wait/get/cancel are mocked, the router HTTP is mocked,
and the policy worker's init/abort_collective return fake "futures" that
are designated ready or pending.

The path being smoke-tested:
  1. ensure_collective_synced reads /current_gen_world_size.
  2. Dispatches init_collective on train + gen via the policy and self.
  3. ray.wait(timeout=...) returns ``ready`` and ``pending`` sets.
  4. If pending is non-empty, identifies which gen-side futures are
     stragglers, calls /admin/remove_shard for each, aborts the train
     comm, sleeps briefly, and retries at the new (smaller) world size.
  5. Eventual success → _last_synced_world_size updated.

This is the load-bearing fix for the post-recover NCCL rendezvous wedge
that took down both 4B and 30B during continuous fault-injection
stress tests.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy optional deps that aren't needed for this test (decord is
# imported by nemo_rl.data.multimodal_utils but only functions on Linux).
sys.modules.setdefault("decord", types.ModuleType("decord"))

from nemo_rl.models.generation.remote_generation import RemoteGeneration


def _make_rg(server_url: str = "http://router:8089") -> RemoteGeneration:
    """Build a RemoteGeneration without hitting a real server.

    Same pattern as test_remote_generation_http.py — short-circuit the
    HTTP round-trips in the constructor so the instance is usable.
    """
    fake_cfg = {
        "vllm_cfg": {"max_model_len": 16},
        "max_new_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "model_name": "test-model",
    }
    with (
        patch.object(RemoteGeneration, "_fetch_remote_config", return_value=fake_cfg),
        patch.object(
            RemoteGeneration,
            "_fetch_shard_urls",
            return_value=[
                "http://shard-0:8000/v1",
                "http://shard-1:8001/v1",
                "http://shard-2:8002/v1",
                "http://shard-3:8003/v1",
            ],
        ),
    ):
        return RemoteGeneration(generation=None, server_url=server_url, config={})


def _build_policy_mock(train_ws: int = 4):
    """Mock policy with the fields ensure_collective_synced touches."""
    policy = MagicMock()
    policy.worker_group.cluster.world_size.return_value = train_ws
    policy.worker_group.cluster.get_master_address_and_port.return_value = (
        "127.0.0.1",
        12345,
    )
    return policy


def _patch_router_and_ray(
    *,
    initial_gen_ws: int,
    after_eviction_gen_ws: int,
    initial_shards: list[dict],
    train_futures_per_attempt: list[list[object]],
    gen_futures_per_attempt: list[list[object]],
    pending_per_attempt: list[set[object]],
    remove_shard_calls: list,
):
    """Yield (start, stop) helpers that wire requests.get/post + ray.

    The test passes lists of fake futures and pending sets indexed by
    attempt (0 = first attempt). On each attempt, ray.wait returns
    (ready, pending) per the lists; ray.get on ready futures succeeds.

    Captures /admin/remove_shard calls into ``remove_shard_calls`` for
    assertions, and updates the gen world size from initial → after on
    the second attempt (post-eviction).
    """
    # Make `requests.get/post` return router-shaped responses.
    state = {"gen_ws": initial_gen_ws, "shards": list(initial_shards)}

    def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/refit_ready" in url:
            resp.json.return_value = {"ready": True, "reason": "ok"}
        elif "/current_gen_world_size" in url:
            resp.json.return_value = {"world_size": state["gen_ws"]}
        elif "/shards" in url:
            resp.json.return_value = list(state["shards"])
        else:
            raise AssertionError(f"unexpected GET {url}")
        return resp

    def fake_post(url, *args, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/admin/remove_shard" in url:
            sid = (json or {}).get("shard_id")
            remove_shard_calls.append(sid)
            # Simulate router behaviour: drop the shard, world size
            # shrinks proportionally.
            state["shards"] = [s for s in state["shards"] if s["shard_id"] != sid]
            state["gen_ws"] = after_eviction_gen_ws
            resp.json.return_value = {"shard_id": sid, "removed": True}
        else:
            raise AssertionError(f"unexpected POST {url}")
        return resp

    # Stateful mocks for ray.wait — return per-attempt (ready, pending).
    attempt_idx = {"i": 0}

    def fake_ray_wait(futures, *, num_returns=None, timeout=None):
        i = attempt_idx["i"]
        # Each call to ray.wait that includes init_collective futures
        # advances the attempt counter. The abort_collective ray.wait
        # also passes through here but uses a tiny timeout (15s) on a
        # single-element list — distinguish by length and timeout.
        if timeout == 15.0:
            # abort_collective's ray.wait — pretend it completed.
            return list(futures), []
        ready = [f for f in futures if f not in pending_per_attempt[i]]
        pending = [f for f in futures if f in pending_per_attempt[i]]
        attempt_idx["i"] += 1
        return ready, pending

    def fake_ray_get(refs):
        # ray.get on the resolved futures — assume success.
        return [None] * (len(refs) if hasattr(refs, "__len__") else 1)

    def fake_ray_cancel(ref, force=False):
        return None

    # Wire policy.init_collective to return per-attempt train futures.
    train_attempt = {"i": 0}

    def policy_init_collective(*args, **kwargs):
        out = train_futures_per_attempt[train_attempt["i"]]
        train_attempt["i"] += 1
        return out

    # Stub abort_collective to return an empty list of futures (the
    # ray.wait above special-cases timeout=15.0 to mean "abort path").
    def policy_abort_collective():
        return []

    # Wire RemoteGeneration.init_collective for the gen side.
    gen_attempt = {"i": 0}

    def rg_init_collective(self, ip, port, world_size, *, train_world_size):
        out = gen_futures_per_attempt[gen_attempt["i"]]
        gen_attempt["i"] += 1
        # init_collective is supposed to set _last_synced_world_size as
        # a side-effect — match the real behaviour.
        self._last_synced_world_size = world_size
        return out

    return {
        "fake_get": fake_get,
        "fake_post": fake_post,
        "fake_ray_wait": fake_ray_wait,
        "fake_ray_get": fake_ray_get,
        "fake_ray_cancel": fake_ray_cancel,
        "policy_init_collective": policy_init_collective,
        "policy_abort_collective": policy_abort_collective,
        "rg_init_collective": rg_init_collective,
    }


def test_ensure_collective_synced_happy_path():
    """No stragglers → first attempt succeeds, no eviction calls."""
    rg = _make_rg()
    rg._last_synced_world_size = None
    policy = _build_policy_mock(train_ws=4)

    train_futures = [object() for _ in range(4)]
    gen_futures = [object() for _ in range(4)]

    remove_shard_calls: list = []
    mocks = _patch_router_and_ray(
        initial_gen_ws=4,
        after_eviction_gen_ws=4,
        initial_shards=[
            {"shard_id": f"dp-{i}", "status": "ready"} for i in range(4)
        ],
        train_futures_per_attempt=[train_futures],
        gen_futures_per_attempt=[gen_futures],
        pending_per_attempt=[set()],  # nothing pending
        remove_shard_calls=remove_shard_calls,
    )

    policy.init_collective.side_effect = mocks["policy_init_collective"]
    policy.abort_collective.side_effect = mocks["policy_abort_collective"]

    with (
        patch("nemo_rl.models.generation.remote_generation.requests.get", side_effect=mocks["fake_get"]),
        patch("nemo_rl.models.generation.remote_generation.requests.post", side_effect=mocks["fake_post"]),
        patch("nemo_rl.models.generation.remote_generation.ray.wait", side_effect=mocks["fake_ray_wait"]),
        patch("nemo_rl.models.generation.remote_generation.ray.get", side_effect=mocks["fake_ray_get"]),
        patch("nemo_rl.models.generation.remote_generation.ray.cancel", side_effect=mocks["fake_ray_cancel"]),
        patch.object(RemoteGeneration, "init_collective", new=mocks["rg_init_collective"]),
        patch.object(RemoteGeneration, "_wait_until_refit_ready", new=lambda self, **k: None),
    ):
        rg.ensure_collective_synced(policy, rendezvous_timeout_s=10.0, max_attempts=4)

    assert rg._last_synced_world_size == 8, "world size = train(4) + gen(4)"
    assert remove_shard_calls == [], "no eviction expected on happy path"


def test_ensure_collective_synced_evicts_straggler_then_succeeds():
    """First attempt: 1 gen future pending → eviction fires for that
    shard → second attempt at smaller world succeeds."""
    rg = _make_rg()
    rg._last_synced_world_size = None
    policy = _build_policy_mock(train_ws=4)

    # New retry policy: first attempt(s) retry at FULL world; eviction
    # only kicks in at attempt evict_threshold = max(2, max_attempts//2).
    # With max_attempts=4: evict_threshold=2, so attempt 1 retries
    # full-world, attempt 2 evicts dp-2 → attempt 3 succeeds.
    train_futs_a = [object() for _ in range(4)]
    gen_futs_a = [object() for _ in range(4)]
    # Attempt 1: full-world retry, same world.
    train_futs_b = [object() for _ in range(4)]
    gen_futs_b = [object() for _ in range(4)]
    # Attempt 2 evicts; attempt 3 fresh futures at smaller world.
    train_futs_c = [object() for _ in range(4)]
    gen_futs_c = [object() for _ in range(3)]

    pending_attempt_1 = {gen_futs_a[2]}  # dp-2 stalls
    pending_attempt_2 = {gen_futs_b[2]}  # still stalled, triggers eviction
    pending_attempt_3 = set()             # clean rendezvous at WS=7

    remove_shard_calls: list = []
    mocks = _patch_router_and_ray(
        initial_gen_ws=4,
        after_eviction_gen_ws=3,
        initial_shards=[
            {"shard_id": f"dp-{i}", "status": "ready"} for i in range(4)
        ],
        train_futures_per_attempt=[train_futs_a, train_futs_b, train_futs_c],
        gen_futures_per_attempt=[gen_futs_a, gen_futs_b, gen_futs_c],
        pending_per_attempt=[pending_attempt_1, pending_attempt_2, pending_attempt_3],
        remove_shard_calls=remove_shard_calls,
    )

    policy.init_collective.side_effect = mocks["policy_init_collective"]
    policy.abort_collective.side_effect = mocks["policy_abort_collective"]

    with (
        patch("nemo_rl.models.generation.remote_generation.requests.get", side_effect=mocks["fake_get"]),
        patch("nemo_rl.models.generation.remote_generation.requests.post", side_effect=mocks["fake_post"]),
        patch("nemo_rl.models.generation.remote_generation.ray.wait", side_effect=mocks["fake_ray_wait"]),
        patch("nemo_rl.models.generation.remote_generation.ray.get", side_effect=mocks["fake_ray_get"]),
        patch("nemo_rl.models.generation.remote_generation.ray.cancel", side_effect=mocks["fake_ray_cancel"]),
        patch("nemo_rl.models.generation.remote_generation.time.sleep", new=lambda *a, **k: None),
        patch.object(RemoteGeneration, "init_collective", new=mocks["rg_init_collective"]),
        patch.object(RemoteGeneration, "_wait_until_refit_ready", new=lambda self, **k: None),
    ):
        rg.ensure_collective_synced(policy, rendezvous_timeout_s=0.01, max_attempts=4)

    assert remove_shard_calls == ["dp-2"], (
        f"expected dp-2 evicted on attempt 2 (after attempt 1 full-world retry); got {remove_shard_calls}"
    )
    assert rg._last_synced_world_size == 7, (
        "after eviction at attempt 2 + success at attempt 3: train(4) + gen(3) = 7"
    )


def test_ensure_collective_synced_evicts_then_smaller_world_succeeds_at_attempt_3():
    """Multi-attempt recovery: two evictions before a successful
    rendezvous. Tests that the retry loop keeps shrinking the world
    until rendezvous completes, rather than giving up after one failure.
    """
    rg = _make_rg()
    rg._last_synced_world_size = None
    policy = _build_policy_mock(train_ws=4)

    # With max_attempts=4 the eviction kicks in at attempt 2; so to
    # exercise multiple sequential evictions we need max_attempts=6.
    # Attempt 1: full-world retry (dp-3 stalls).
    # Attempt 2: evict dp-3 (still pending) → world shrinks to 3.
    # Attempt 3: full-world (now WS=3) retry (dp-2 stalls — idx 2).
    # Attempt 4 (>= evict_threshold=3 with max_attempts=6): evict dp-2.
    # Attempt 5: clean at WS=2.
    train_futs_a = [object() for _ in range(4)]
    gen_futs_a = [object() for _ in range(4)]
    train_futs_b = [object() for _ in range(4)]
    gen_futs_b = [object() for _ in range(4)]  # still WS=4 at attempt 2 (eviction during)
    train_futs_c = [object() for _ in range(4)]
    gen_futs_c = [object() for _ in range(3)]
    train_futs_d = [object() for _ in range(4)]
    gen_futs_d = [object() for _ in range(3)]
    train_futs_e = [object() for _ in range(4)]
    gen_futs_e = [object() for _ in range(2)]

    pending_attempt_1 = {gen_futs_a[3]}  # full-world retry
    pending_attempt_2 = {gen_futs_b[3]}  # eviction triggers
    pending_attempt_3 = {gen_futs_c[2]}  # WS=3, dp-2 stalls (full-world retry)
    pending_attempt_4 = {gen_futs_d[2]}  # eviction triggers again
    pending_attempt_5 = set()             # clean

    # State driven by remove_shard mutations rather than the attempt
    # counter — this is how the real router behaves and avoids
    # ordering bugs where /shards GET during eviction sees the
    # post-eviction state instead of pre-eviction.
    shard_state = {
        "shards": [{"shard_id": f"dp-{i}", "status": "ready"} for i in range(4)],
        "ws": 4,
    }

    remove_shard_calls: list = []
    state = {"attempt": 0}

    def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/refit_ready" in url:
            resp.json.return_value = {"ready": True, "reason": "ok"}
        elif "/current_gen_world_size" in url:
            resp.json.return_value = {"world_size": shard_state["ws"]}
        elif "/shards" in url:
            resp.json.return_value = list(shard_state["shards"])
        else:
            raise AssertionError(f"unexpected GET {url}")
        return resp

    def fake_post(url, *args, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/admin/remove_shard" in url:
            sid = (json or {}).get("shard_id")
            remove_shard_calls.append(sid)
            shard_state["shards"] = [
                s for s in shard_state["shards"] if s["shard_id"] != sid
            ]
            shard_state["ws"] = len(shard_state["shards"])
            resp.json.return_value = {"shard_id": sid, "removed": True}
        else:
            raise AssertionError(f"unexpected POST {url}")
        return resp

    pendings = [pending_attempt_1, pending_attempt_2, pending_attempt_3, pending_attempt_4, pending_attempt_5]
    train_futs_per = [train_futs_a, train_futs_b, train_futs_c, train_futs_d, train_futs_e]
    gen_futs_per = [gen_futs_a, gen_futs_b, gen_futs_c, gen_futs_d, gen_futs_e]

    def fake_ray_wait(futures, *, num_returns=None, timeout=None):
        if timeout == 15.0:
            return list(futures), []
        i = state["attempt"]
        ready = [f for f in futures if f not in pendings[i]]
        pending = [f for f in futures if f in pendings[i]]
        state["attempt"] += 1
        return ready, pending

    train_attempt = {"i": 0}

    def policy_init_collective(*args, **kwargs):
        out = train_futs_per[train_attempt["i"]]
        train_attempt["i"] += 1
        return out

    gen_attempt = {"i": 0}

    def rg_init_collective(self, ip, port, world_size, *, train_world_size):
        out = gen_futs_per[gen_attempt["i"]]
        gen_attempt["i"] += 1
        self._last_synced_world_size = world_size
        return out

    policy.init_collective.side_effect = policy_init_collective
    policy.abort_collective.side_effect = lambda: []

    with (
        patch("nemo_rl.models.generation.remote_generation.requests.get", side_effect=fake_get),
        patch("nemo_rl.models.generation.remote_generation.requests.post", side_effect=fake_post),
        patch("nemo_rl.models.generation.remote_generation.ray.wait", side_effect=fake_ray_wait),
        patch("nemo_rl.models.generation.remote_generation.ray.get", side_effect=lambda r: [None]),
        patch("nemo_rl.models.generation.remote_generation.ray.cancel", side_effect=lambda r, **k: None),
        patch("nemo_rl.models.generation.remote_generation.time.sleep", new=lambda *a, **k: None),
        patch.object(RemoteGeneration, "init_collective", new=rg_init_collective),
        patch.object(RemoteGeneration, "_wait_until_refit_ready", new=lambda self, **k: None),
    ):
        rg.ensure_collective_synced(policy, rendezvous_timeout_s=0.01, max_attempts=6)

    assert remove_shard_calls == ["dp-3", "dp-2"], (
        f"expected dp-3 then dp-2 evicted across two attempt-2/4 timeouts; got {remove_shard_calls}"
    )
    assert rg._last_synced_world_size == 6, (
        "after two evictions: train(4) + gen(2) = 6"
    )
