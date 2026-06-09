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
"""Unit tests for GenerationRouter (step 1: data plane + cordon + health)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from nemo_rl.models.generation.generation_router import GenerationRouter, ShardEntry


# =====================================================================
# Fake shard server (aiohttp) — mimics vLLM's /health + /v1/completions.
# =====================================================================


class FakeShard:
    """Minimal aiohttp server that mimics one vLLM DP shard.

    Records every /v1/completions hit and lets the test toggle health and
    completions behavior at runtime.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.healthy = True
        self.completions_status = 200
        self.completions_body = {"choices": [{"text": "ok", "finish_reason": "stop"}]}
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.url: str = ""

    async def _health(self, request: web.Request) -> web.Response:
        if self.healthy:
            return web.json_response({"status": "ok"})
        return web.Response(status=503, text="unhealthy")

    async def _openapi(self, request: web.Request) -> web.Response:
        # The router's health probe hits /openapi.json (the only stable
        # always-present endpoint on a real vLLM OpenAI server), NOT /health.
        # Mirror vLLM: serve it, gated on the same `healthy` toggle so tests
        # can simulate an unresponsive engine.
        if self.healthy:
            return web.json_response({"openapi": "3.0.0"})
        return web.Response(status=503, text="unhealthy")

    async def _completions(self, request: web.Request) -> web.Response:
        body = await request.read()
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"_raw": body.decode("utf-8", "replace")}
        payload["_headers"] = dict(request.headers)
        self.calls.append(payload)
        if self.completions_status >= 500:
            return web.Response(status=self.completions_status, text="boom")
        return web.json_response(self.completions_body, status=self.completions_status)

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/openapi.json", self._openapi)
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._completions)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}/v1"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


@pytest.fixture
async def two_shards():
    a, b = FakeShard(), FakeShard()
    await a.start()
    await b.start()
    yield a, b
    await a.stop()
    await b.stop()


@pytest.fixture
async def router_and_client(two_shards: tuple[FakeShard, FakeShard]):
    """Build a router + an httpx AsyncClient bound to its FastAPI app.

    httpx.ASGITransport doesn't fire FastAPI's lifespan by default, so we
    bring up the router's http session and health task by hand on the
    test loop. This keeps the request handler, the proxy session, and the
    health poll loop all sharing one event loop. (FastAPI's TestClient
    spawns a thread with its own loop, which leaves the aiohttp
    ClientSession bound to a stale loop and every proxy call times out.)
    """
    import httpx

    a, b = two_shards
    r = GenerationRouter(
        port=0,
        health_poll_interval_s=0.1,
        health_timeout_s=0.2,
        failure_threshold=2,
        join_success_threshold=1,
        proxy_timeout_s=2.0,
    )
    r.register_shards(
        [("dp-a", a.url), ("dp-b", b.url)], per_shard_world_size=2
    )
    r._http_session = aiohttp.ClientSession()
    health_task = asyncio.create_task(r._health_poll_loop())
    transport = httpx.ASGITransport(app=r.get_app())
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://router.test"
        ) as client:
            yield r, client
    finally:
        health_task.cancel()
        try:
            await health_task
        except (asyncio.CancelledError, Exception):
            pass
        await r._http_session.close()


@pytest.fixture
async def router(router_and_client):
    yield router_and_client[0]


@pytest.fixture
async def aclient(router_and_client):
    yield router_and_client[1]


# =====================================================================
# Tests
# =====================================================================


@pytest.mark.asyncio
async def test_register_shards_initial_state(router: GenerationRouter):
    """All registered shards start `ready` so an existing-VllmGeneration
    bring-up flow doesn't have to wait two health poll intervals before
    accepting traffic."""
    statuses = {s.shard_id: s.status for s in router._shards.values()}
    assert statuses == {"dp-a": "ready", "dp-b": "ready"}
    assert router.current_gen_world_size() == 4  # 2 shards * per_shard_world_size=2
    ready, reason = router.refit_ready_state()
    assert ready, reason


@pytest.mark.asyncio
async def test_proxy_round_robin(router: GenerationRouter, aclient, two_shards):
    """Proxy fans out across ready shards round-robin when no sticky key."""
    a, b = two_shards
    for _ in range(4):
        resp = await aclient.post("/v1/completions", json={"prompt": "hi"})
        assert resp.status_code == 200, resp.text
    # 4 requests across 2 shards → each should see at least 1.
    assert len(a.calls) >= 1
    assert len(b.calls) >= 1
    assert len(a.calls) + len(b.calls) == 4


@pytest.mark.asyncio
async def test_sticky_session_pins_to_one_shard(
    router: GenerationRouter, aclient, two_shards
):
    """X-NRL-Session-Id pins all requests for a session to the same shard."""
    a, b = two_shards
    for i in range(6):
        resp = await aclient.post(
            "/v1/completions",
            json={"prompt": f"q-{i}"},
            headers={"X-NRL-Session-Id": "stick-42"},
        )
        assert resp.status_code == 200
    sticky_count = max(len(a.calls), len(b.calls))
    other_count = min(len(a.calls), len(b.calls))
    assert sticky_count == 6 and other_count == 0


@pytest.mark.asyncio
async def test_5xx_cordons_and_replays(
    router: GenerationRouter, aclient, two_shards
):
    """A 5xx from one shard cordons it and the request is replayed elsewhere."""
    a, b = two_shards
    a.completions_status = 503  # a will reject
    # Force round-robin to land on dp-a first (initial _rr_index=0 would
    # increment to 1 → dp-b; -1 → dp-a). We want to exercise the cordon
    # path, which only runs if a 5xx response was actually received.
    router._rr_index = -1
    resp = await aclient.post("/v1/completions", json={"prompt": "hi"})
    assert resp.status_code == 200, resp.text  # replay landed on b
    assert len(a.calls) == 1  # a got the failed attempt
    assert len(b.calls) == 1  # b got the replay
    statuses = {s.shard_id: s.status for s in router._shards.values()}
    assert statuses["dp-a"] == "cordoned"
    assert statuses["dp-b"] == "ready"


@pytest.mark.asyncio
async def test_no_ready_shards_returns_5xx(
    router: GenerationRouter, aclient, two_shards
):
    """When every shard is cordoned the proxy returns a structured error,
    not a hang."""
    a, b = two_shards
    a.completions_status = 502
    b.completions_status = 502
    resp = await aclient.post("/v1/completions", json={"prompt": "hi"})
    assert resp.status_code >= 500
    statuses = {s.shard_id: s.status for s in router._shards.values()}
    assert all(v == "cordoned" for v in statuses.values())


@pytest.mark.asyncio
async def test_health_poll_cordons_and_does_not_auto_recover(
    router: GenerationRouter, two_shards
):
    """Two consecutive failures cordon a ready shard; once probes succeed
    again the health poller does NOT auto-promote it back. Cordon is a
    deliberate decision — the router observed bad behavior — and the
    k8s-native answer is to remove and replace via the autoscaler, not
    to second-guess the cordon when /health flickers green again.

    A shard that flapped once is more likely to keep flapping than to
    have silently healed; auto-recovery would also race the on_cordon
    -> remove_shard hook in the FT path.
    """
    a, b = two_shards
    a.healthy = False
    await asyncio.sleep(0.5)
    assert router._shards["dp-a"].status == "cordoned"

    # Probes go green again — but the shard MUST stay cordoned. The
    # router won't bring it back without explicit operator action.
    a.healthy = True
    await asyncio.sleep(0.5)
    assert router._shards["dp-a"].status == "cordoned"


@pytest.mark.asyncio
async def test_uncordon_flips_to_joining_not_ready(
    router: GenerationRouter, two_shards
):
    """Admin /admin/uncordon flips cordoned -> joining, NOT directly to
    ready. We can't prove a cordoned shard's weights are current (cordon
    may have fired because the actor was hung mid-broadcast), so the
    safe path is to gate the data plane until the next refit broadcasts
    weights into it and promote_all_joining moves it to ready.
    """
    a, _ = two_shards
    a.healthy = False
    await asyncio.sleep(0.5)
    assert router._shards["dp-a"].status == "cordoned"

    a.healthy = True
    await router.uncordon("dp-a")
    assert router._shards["dp-a"].status == "joining"

    # And the health poller must NOT auto-promote it from joining.
    await asyncio.sleep(0.5)
    assert router._shards["dp-a"].status == "joining"


@pytest.mark.asyncio
async def test_joining_shard_not_auto_promoted_by_health_poll(
    router: GenerationRouter, two_shards
):
    """Regression test for the stale-weight corruption bug: a `joining`
    shard MUST NOT be auto-promoted to `ready` by the health poller, even
    if vLLM `/health` answers OK. Joining means "freshly added, weights
    are stale" — promotion only happens via `promote_all_joining()` after
    a successful weight broadcast.

    Without this gate, the data plane (which only routes to `ready`
    shards) would start picking up the new shard as soon as vLLM's
    HTTP server came up, well before the next refit pushed fresh
    weights — corrupting rollouts for the entire add-shard-to-next-
    refit window.
    """
    a, _ = two_shards
    # Drive dp-a into `joining` (simulating a freshly add_shard'd entry).
    router._shards["dp-a"].status = "joining"
    router._shards["dp-a"].consecutive_successes = 0

    # Health probe is healthy: the fake shard's /health returns 200.
    # Wait several poll intervals — enough that consecutive_successes
    # would have crossed any reasonable join_success_threshold.
    a.healthy = True
    await asyncio.sleep(0.5)

    # Status MUST still be `joining`. Promotion has to come from
    # `promote_all_joining()` after a successful refit broadcast.
    assert router._shards["dp-a"].status == "joining"

    # Promote explicitly (simulating successful refit).
    promoted = router.promote_all_joining()
    assert "dp-a" in promoted
    assert router._shards["dp-a"].status == "ready"


@pytest.mark.asyncio
async def test_data_plane_skips_joining_shards(
    router: GenerationRouter, aclient, two_shards
):
    """The actual safety net for the corruption bug: even if everything
    else fails, _pick_shard only picks `ready` shards. A `joining` shard
    (stale weights) never receives data-plane traffic.
    """
    a, b = two_shards
    # Park dp-a in joining — its FakeShard server is up, but the router
    # MUST not route to it because weights are unproven.
    router._shards["dp-a"].status = "joining"

    for _ in range(6):
        resp = await aclient.post("/v1/completions", json={"prompt": "hi"})
        assert resp.status_code == 200, resp.text

    # All requests landed on dp-b, none on dp-a.
    assert len(b.calls) == 6
    assert len(a.calls) == 0


@pytest.mark.asyncio
async def test_admin_cordon_uncordon_endpoints(router: GenerationRouter, aclient):
    """POST /admin/cordon and /admin/uncordon flip status without health
    intervention. Useful for the --fault-mode http-error path. Uncordon
    flips cordoned -> ready directly: cordoned shards have fresh weights
    (cordon never kills the actor or skips broadcasts)."""
    resp = await aclient.post(
        "/admin/cordon", json={"shard_id": "dp-a", "reason": "test"}
    )
    assert resp.status_code == 200
    assert router._shards["dp-a"].status == "cordoned"
    resp = await aclient.post("/admin/uncordon", json={"shard_id": "dp-a"})
    assert resp.status_code == 200
    assert router._shards["dp-a"].status == "ready"


@pytest.mark.asyncio
async def test_on_cordon_hook_fires(router: GenerationRouter, two_shards):
    """on_cordon hook is called exactly once per cordon transition. Step 2
    will use this to trigger remove_shard from health-poll cordoning."""
    a, _ = two_shards
    fired: list[tuple[str, str]] = []

    async def _hook(shard_id: str, reason: str) -> None:
        fired.append((shard_id, reason))

    router.on_cordon = _hook
    a.healthy = False
    await asyncio.sleep(0.5)
    assert len(fired) == 1
    assert fired[0][0] == "dp-a"


@pytest.mark.asyncio
async def test_shard_entry_health_url_normalization():
    """ShardEntry strips a trailing /v1 to derive _health_url so vLLM's
    /health (mounted at the root, not /v1/health) is hit correctly."""
    e = ShardEntry(shard_id="x", url="http://10.0.0.1:8000/v1")
    assert e._health_url == "http://10.0.0.1:8000/health"
    e2 = ShardEntry(shard_id="x", url="http://10.0.0.1:8000")
    assert e2._health_url == "http://10.0.0.1:8000/health"
    e3 = ShardEntry(shard_id="x", url="http://10.0.0.1:8000/")
    assert e3._health_url == "http://10.0.0.1:8000/health"


# =====================================================================
# Lifecycle (remove_shard) — uses fakes for ray actors / placement group.
# =====================================================================


class _FakeActor:
    def __init__(self, name: str):
        self.name = name
        self.killed = False


class _FakePG:
    def __init__(self, name: str):
        self.name = name
        self.removed = False


class _FakeGeneration:
    """Stand-in for VllmGeneration. The router calls reset_collective() on
    the surviving workers after a remove_shard."""

    def __init__(self):
        self.reset_calls = 0

    def reset_collective(self):
        self.reset_calls += 1
        return []  # Empty futures list — `ray.get([])` is a no-op.


@pytest.mark.asyncio
async def test_remove_shard_kills_actors_frees_pg_and_resets(monkeypatch):
    """remove_shard kills every actor handle, removes the PG, and calls
    reset_collective() on the surviving generation. After it returns, the
    shard is gone from /shards and current_gen_world_size shrinks."""
    import nemo_rl.models.generation.generation_router as gr

    # Patch ray.kill / remove_placement_group with fakes since pytest doesn't
    # bring up a real Ray cluster.
    killed_actors: list[str] = []
    removed_pgs: list[str] = []

    fake_ray_kill = lambda actor, no_restart=True: (  # noqa: E731
        killed_actors.append(actor.name) or setattr(actor, "killed", True)
    )

    def fake_remove_pg(pg):
        removed_pgs.append(pg.name)
        pg.removed = True

    class _StubRay:
        def kill(self, actor, no_restart=True):
            fake_ray_kill(actor, no_restart=no_restart)

        def get(self, futures):
            return []

    # Inside the executor _kill_and_free does `import ray` and
    # `from ray.util.placement_group import remove_placement_group`. Patch
    # both via sys.modules so the import resolves to our stubs.
    import sys
    import types

    stub_ray = types.SimpleNamespace(kill=fake_ray_kill, get=lambda fs: [])
    stub_pg_mod = types.SimpleNamespace(remove_placement_group=fake_remove_pg)
    monkeypatch.setitem(sys.modules, "ray", stub_ray)
    # ray.util.placement_group is imported via `from ray.util.placement_group
    # import remove_placement_group`; the import system needs the parent
    # package present too. The router only references the function, so a
    # minimal stub works.
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", stub_pg_mod)

    fake_gen = _FakeGeneration()
    actor_a = _FakeActor("a-leader")
    actor_b1 = _FakeActor("b-leader")
    actor_b2 = _FakeActor("b-helper")
    pg_a = _FakePG("pg-a")
    pg_b = _FakePG("pg-b")

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://shard-a:8000/v1"), ("dp-b", "http://shard-b:8000/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={"dp-a": [actor_a], "dp-b": [actor_b1, actor_b2]},
        pg_by_shard={"dp-a": pg_a, "dp-b": pg_b},
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 4

    result = await r.remove_shard("dp-a", reason="test")

    assert result["removed"] is True
    assert result["world_size"] == 2  # one shard left, per_shard_ws=2
    assert "dp-a" not in r._shards
    assert "dp-b" in r._shards
    assert actor_a.killed is True
    assert pg_a.removed is True
    # dp-b is untouched.
    assert actor_b1.killed is False
    assert pg_b.removed is False
    # reset_collective dispatched once on the surviving generation.
    assert fake_gen.reset_calls == 1
    # Refit gate is open again (router cleared the flag).
    ready, _ = r.refit_ready_state()
    assert ready


@pytest.mark.asyncio
async def test_remove_shard_unknown_id_is_noop(monkeypatch):
    """Removing an unknown shard is a no-op and reports removed=False."""
    import nemo_rl.models.generation.generation_router as gr

    r = gr.GenerationRouter(port=0)
    r.register_shards([("dp-a", "http://shard-a/v1")], per_shard_world_size=1)
    result = await r.remove_shard("does-not-exist", reason="test")
    assert result == {
        "shard_id": "does-not-exist",
        "removed": False,
        "reason": "not found",
    }


# =====================================================================
# Cascade-protection: _filter_targets_by_liveness + _evict_failed_workers_inline
#
# At TP>1, a burst kill of M shards leaves SURVIVORS' init_collective
# futures failing alongside the dead shards' futures. Without a
# liveness check, _evict_failed_workers_inline would mass-evict all
# of them and cascade the world to zero. The fix: probe each
# candidate's leader actor with is_alive; evict only confirmed-dead
# shards.
# =====================================================================


class _LiveActor:
    """Stand-in actor whose ``is_alive.remote()`` returns a future that ray.get
    resolves to True (mocked via the monkeypatched ``ray.get``)."""

    def __init__(self, name: str):
        self.name = name
        self.killed = False

    @property
    def is_alive(self) -> "_LiveAliveStub":
        return _LiveAliveStub(self.name, alive=True)


class _DeadActor:
    """Stand-in actor whose ``is_alive.remote()`` returns a future that ray.get
    raises ``RayActorError`` for (mocked via the monkeypatched ``ray.get``)."""

    def __init__(self, name: str):
        self.name = name
        self.killed = False

    @property
    def is_alive(self) -> "_LiveAliveStub":
        return _LiveAliveStub(self.name, alive=False)


class _LiveAliveStub:
    """Returned by ``actor.is_alive`` so ``actor.is_alive.remote()`` works.

    The remote() call returns a sentinel future tagged with the actor name
    and its alive-ness; the patched ``ray.get`` resolves that sentinel.
    """

    def __init__(self, name: str, alive: bool):
        self.name = name
        self.alive = alive

    def remote(self):
        return _IsAliveFuture(self.name, self.alive)


class _IsAliveFuture:
    def __init__(self, name: str, alive: bool):
        self.name = name
        self.alive = alive


def _stub_ray_for_eviction(monkeypatch):
    """Wire stub ray.* into sys.modules + the gr.ray reference.

    ``_filter_targets_by_liveness`` does ``import ray`` then
    ``ray.get(fut, timeout=...)`` to resolve is_alive futures, and
    ``remove_shard`` does the same for actor kill / pg removal. We
    return a single ``ray`` stub that handles both code paths.
    """
    import sys
    import types

    killed_actors: list[str] = []
    removed_pgs: list[str] = []

    def fake_kill(actor, no_restart=True):  # noqa: ARG001
        killed_actors.append(getattr(actor, "name", repr(actor)))
        try:
            setattr(actor, "killed", True)
        except Exception:  # noqa: BLE001
            pass

    def fake_get(fut, timeout=None):  # noqa: ARG001
        # is_alive future: resolve based on the alive flag carried on it.
        if isinstance(fut, _IsAliveFuture):
            if fut.alive:
                return True
            # Mimic ray's RayActorError on a dead actor probe.
            raise RuntimeError(f"RayActorError: actor {fut.name} dead")
        return None  # generic ray.get on lists / placeholders

    def fake_remove_pg(pg):
        removed_pgs.append(getattr(pg, "name", repr(pg)))
        try:
            setattr(pg, "removed", True)
        except Exception:  # noqa: BLE001
            pass

    stub_ray = types.SimpleNamespace(kill=fake_kill, get=fake_get)
    stub_pg_mod = types.SimpleNamespace(remove_placement_group=fake_remove_pg)
    monkeypatch.setitem(sys.modules, "ray", stub_ray)
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", stub_pg_mod)
    return killed_actors, removed_pgs


@pytest.mark.asyncio
async def test_filter_targets_by_liveness_splits_dead_and_alive(monkeypatch):
    """_filter_targets_by_liveness pings each candidate's leader and
    classifies based on the response. Live survivors are kept; truly-dead
    shards are flagged for eviction."""
    import nemo_rl.models.generation.generation_router as gr

    _stub_ray_for_eviction(monkeypatch)

    # Two shards: dp-a is dead (FI killed it), dp-b is alive (survivor
    # whose init_collective future timed out on the dead peer's TCPStore).
    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-a": [_DeadActor("a-leader"), _DeadActor("a-helper")],
            "dp-b": [_LiveActor("b-leader"), _LiveActor("b-helper")],
        },
    )
    dead, alive = await r._filter_targets_by_liveness(["dp-a", "dp-b"])
    assert dead == ["dp-a"]
    assert alive == ["dp-b"]


@pytest.mark.asyncio
async def test_evict_failed_workers_inline_skips_alive_survivors(monkeypatch):
    """The TP=2 cascade scenario: 2 shards killed in a burst, survivors'
    init_collective futures fail alongside the dead shards' futures.
    Eviction must skip alive survivors and only remove dead shards.

    Without the liveness gate this would mass-evict ALL 4 shards and
    cascade the world to zero (the production bug)."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, removed_pgs = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGeneration()
    # 4 DP shards × TP=2 actors. dp-0/dp-1 = killed by FI (DeadActor —
    # is_alive raises RayActorError). dp-2/dp-3 = alive but their
    # init_collective futures failed because the dead shards never joined
    # the TCPStore (so the gen workers raised DistNetworkError up the
    # stack).
    pg_0 = _FakePG("pg-0")
    pg_1 = _FakePG("pg-1")
    pg_2 = _FakePG("pg-2")
    pg_3 = _FakePG("pg-3")

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [
            ("dp-0", "http://a/v1"),
            ("dp-1", "http://b/v1"),
            ("dp-2", "http://c/v1"),
            ("dp-3", "http://d/v1"),
        ],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-0": [_DeadActor("dp0-leader"), _DeadActor("dp0-helper")],
            "dp-1": [_DeadActor("dp1-leader"), _DeadActor("dp1-helper")],
            "dp-2": [_LiveActor("dp2-leader"), _LiveActor("dp2-helper")],
            "dp-3": [_LiveActor("dp3-leader"), _LiveActor("dp3-helper")],
        },
        pg_by_shard={"dp-0": pg_0, "dp-1": pg_1, "dp-2": pg_2, "dp-3": pg_3},
        generation=fake_gen,
    )
    # World size = 4 shards × per_shard_ws=2 = 8 (gen ranks).
    assert r.current_gen_world_size() == 8

    # All 4 leader futures failed (the dispatch path doesn't know which
    # are dead vs blocked). Without the liveness gate we'd evict all 4.
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=[0, 1, 2, 3],
        reason="init_collective: per-worker failure",
    )
    # Only the 2 dead shards are evicted.
    assert num_evicted == 2
    assert set(evicted_ids) == {"dp-0", "dp-1"}
    # The 2 alive survivors stayed in the table.
    assert "dp-0" not in r._shards
    assert "dp-1" not in r._shards
    assert "dp-2" in r._shards
    assert "dp-3" in r._shards
    # World size is now 2 shards × 2 = 4 (alive survivors).
    assert r.current_gen_world_size() == 4
    # The dead shards' actors got ray.kill'd; survivors' actors are
    # untouched.
    assert "dp0-leader" in killed_actors
    assert "dp1-leader" in killed_actors
    assert "dp2-leader" not in killed_actors
    assert "dp3-leader" not in killed_actors


@pytest.mark.asyncio
async def test_evict_failed_workers_inline_dead_shard_evicts_normally(monkeypatch):
    """Sanity check: when a single shard is genuinely dead (TP=1 path),
    eviction proceeds as before. The liveness gate must not over-correct
    and skip eviction in the case it was originally designed for."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, _ = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGeneration()
    pg_a = _FakePG("pg-a")
    pg_b = _FakePG("pg-b")

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1")],
        per_shard_world_size=1,
        actor_handles_by_shard={
            "dp-a": [_DeadActor("a-leader")],
            "dp-b": [_LiveActor("b-leader")],
        },
        pg_by_shard={"dp-a": pg_a, "dp-b": pg_b},
        generation=fake_gen,
    )
    # Only dp-a's future failed (dp-b succeeded). failed_idxs=[0].
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=[0],
        reason="init_collective: per-worker failure",
    )
    assert num_evicted == 1
    assert evicted_ids == ["dp-a"]
    assert "dp-a" not in r._shards
    assert "dp-b" in r._shards
    assert "a-leader" in killed_actors


@pytest.mark.asyncio
async def test_admin_remove_shard_endpoint(router_and_client, monkeypatch):
    """POST /admin/remove_shard is wired to remove_shard. Integration with
    the FastAPI app (the lifecycle path test above hits the method directly)."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "ray",
                        types.SimpleNamespace(kill=lambda *a, **k: None,
                                              get=lambda fs: []))
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ray.util.placement_group",
                        types.SimpleNamespace(remove_placement_group=lambda pg: None))

    router, aclient = router_and_client
    resp = await aclient.post(
        "/admin/remove_shard", json={"shard_id": "dp-a", "reason": "test"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shard_id"] == "dp-a"
    assert body["removed"] is True
    # World size shrinks (per_shard_ws=2, was 2 shards → 1 shard remains).
    assert body["world_size"] == 2
    assert "dp-a" not in router._shards


# =====================================================================
# Fault-tolerance metrics snapshot
# =====================================================================


@pytest.mark.asyncio
async def test_metrics_snapshot_initial_state():
    """Bootstrap fleet count is captured; counters start at zero."""
    import nemo_rl.models.generation.generation_router as gr

    r = gr.GenerationRouter(port=0)
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1"), ("dp-c", "http://c/v1")],
        per_shard_world_size=2,
    )
    snap = r.metrics_snapshot()
    assert snap["num_ready_shards"] == 3
    assert snap["num_total_shards"] == 3
    assert snap["total_shards_at_bootstrap"] == 3
    assert snap["cumulative_shards_removed"] == 0
    assert snap["cumulative_shards_added"] == 0
    assert snap["per_shard_world_size"] == 2
    assert snap["current_gen_world_size"] == 6
    assert snap["last_fault_event"] is None
    assert snap["nccl_reinit_in_progress"] is False
    # Per-shard breakdown carries every shard with non-None health age
    # (register_shards sets last_health_ok_at = now).
    assert {s["shard_id"] for s in snap["per_shard"]} == {"dp-a", "dp-b", "dp-c"}
    for s in snap["per_shard"]:
        assert s["status"] == "ready"
        assert s["last_health_ok_age_s"] is not None
        assert s["last_health_ok_age_s"] >= 0


@pytest.mark.asyncio
async def test_metrics_snapshot_cordon_records_fault_event():
    """A cordon transition records the fault event and bumps consec failures."""
    import nemo_rl.models.generation.generation_router as gr

    r = gr.GenerationRouter(port=0)
    r.register_shards([("dp-a", "http://a/v1")], per_shard_world_size=1)
    await r.cordon("dp-a", reason="manual cordon for test")
    snap = r.metrics_snapshot()
    assert snap["num_ready_shards"] == 0
    assert snap["num_cordoned_shards"] == 1
    fault = snap["last_fault_event"]
    assert fault is not None
    assert fault["kind"] == "cordon"
    assert fault["shard_id"] == "dp-a"
    assert fault["reason"] == "manual cordon for test"
    assert "monotonic_ts" in fault


@pytest.mark.asyncio
async def test_metrics_snapshot_remove_then_recover_counters(monkeypatch):
    """remove_shard bumps cumulative_shards_removed and stamps a fault event."""
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "ray",
        types.SimpleNamespace(kill=lambda *a, **k: None, get=lambda fs: []),
    )
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "ray.util.placement_group",
        types.SimpleNamespace(remove_placement_group=lambda pg: None),
    )

    import nemo_rl.models.generation.generation_router as gr

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1")],
        per_shard_world_size=1,
    )
    await r.remove_shard("dp-a", reason="injected fault")
    snap = r.metrics_snapshot()
    assert snap["cumulative_shards_removed"] == 1
    assert snap["cumulative_shards_added"] == 0
    assert snap["num_total_shards"] == 1
    assert snap["total_shards_at_bootstrap"] == 2
    fault = snap["last_fault_event"]
    assert fault is not None
    assert fault["kind"] == "remove"
    assert fault["shard_id"] == "dp-a"
    assert fault["reason"] == "injected fault"

    # Manually bump _cumulative_shards_added the way a future add_shard
    # path would. This locks in the counter shape used by the snapshot
    # contract — when add_shard lands it just has to bump this field.
    r._cumulative_shards_added += 1
    snap2 = r.metrics_snapshot()
    assert snap2["cumulative_shards_added"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_snapshot(router_and_client):
    """GET /metrics on the router returns the metrics_snapshot dict."""
    router, aclient = router_and_client
    resp = await aclient.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    # Smoke-check the shape; the exhaustive checks are above.
    assert body["num_total_shards"] == 2
    assert body["total_shards_at_bootstrap"] == 2
    assert "per_shard" in body
    assert "cumulative_shards_removed" in body


# =====================================================================
# TP>1 add_shard worker_indices remap
#
# When a shard with TP*PP>1 is added after a remove, spawn_workers_for_shard
# compacts dead indices first; the router then has to re-derive each
# surviving shard's worker_indices so a future remove_shard targets the
# right actors. For TP=N, each shard owns N consecutive indices starting
# at its leader's post-compaction position.
# =====================================================================


class _FakeWorkerGroup:
    """Just enough of a RayWorkerGroup to back the router's add_shard remap.

    The router reads ``dp_leader_worker_indices`` and pairs leaders with
    shard-table entries positionally. After compaction inside
    ``spawn_workers_for_shard``, surviving leaders sit at indices
    ``[0, N, 2N, ...]`` for ``per_shard_world_size = N``.
    """

    def __init__(self, leader_indices):
        self.dp_leader_worker_indices = list(leader_indices)
        self.dead_indices: set = set()

    @property
    def dp_size(self) -> int:
        return len(self.dp_leader_worker_indices)


class _FakeGenerationWithWG:
    """Stand-in for VllmGeneration that exposes ``worker_group`` and
    ``add_dp_worker``. The real method runs Ray; here we simulate the
    post-compaction state and return shaped data."""

    def __init__(self, per_shard_ws, n_existing):
        self.per_shard_ws = per_shard_ws
        # Initial leaders post-bring-up: [0, N, 2N, ...]
        self.worker_group = _FakeWorkerGroup(
            [i * per_shard_ws for i in range(n_existing)]
        )
        self.last_pre_append_hook = None

    def add_dp_worker(self, pre_append_hook=None):
        # Simulate the Phase B append: leader of new shard goes to the
        # next consecutive slot in dp_leader_worker_indices, and the shard
        # owns N actor handles + N worker indices.
        self.last_pre_append_hook = pre_append_hook
        if pre_append_hook is not None:
            pre_append_hook()
        new_leader_idx = (
            len(self.worker_group.dp_leader_worker_indices) * self.per_shard_ws
        )
        self.worker_group.dp_leader_worker_indices.append(new_leader_idx)
        actors = [_FakeActor(f"new-{i}") for i in range(self.per_shard_ws)]
        worker_indices = list(
            range(new_leader_idx, new_leader_idx + self.per_shard_ws)
        )
        return actors, _FakePG("pg-new"), worker_indices, "http://new/v1"

    # Stubs the router calls under add_shard's lock.
    def reset_collective(self):
        return []


@pytest.mark.asyncio
async def test_add_shard_tp2_remaps_survivor_worker_indices(monkeypatch):
    """After compacting dead indices on add_shard, the router must
    re-derive each surviving TP>1 shard's worker_indices as N consecutive
    slots starting at its leader's NEW (post-compaction) index.

    Pre: 4 shards × TP=2. dp-1 was killed → its old worker indices [2,3]
    were dropped by mark_workers_dead. Compaction (inside
    spawn_workers_for_shard) shifts dp-2's leader from 4 to 2 and dp-3's
    leader from 6 to 4. Add a new shard → new leader at 6. Each surviving
    shard's worker_indices must reflect the post-compaction reality so
    the next remove_shard kills the right actors.
    """
    import nemo_rl.models.generation.generation_router as gr

    fake_gen = _FakeGenerationWithWG(per_shard_ws=2, n_existing=3)

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    # Survivors after dp-1 was removed. The router still holds the OLD
    # (pre-compaction) worker_indices for each survivor.
    r.register_shards(
        [
            ("dp-0", "http://a/v1"),
            ("dp-2", "http://c/v1"),
            ("dp-3", "http://d/v1"),
        ],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-0": [_FakeActor("dp0-l"), _FakeActor("dp0-h")],
            "dp-2": [_FakeActor("dp2-l"), _FakeActor("dp2-h")],
            "dp-3": [_FakeActor("dp3-l"), _FakeActor("dp3-h")],
        },
        # Stale (pre-compaction) indices: dp-2 used to be at [4,5], dp-3
        # at [6,7]. After compaction these will shift to [2,3] and [4,5].
        worker_indices_by_shard={
            "dp-0": [0, 1],
            "dp-2": [4, 5],
            "dp-3": [6, 7],
        },
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 6

    result = await r.add_shard(reason="test")
    assert result["added"] is True
    # World size grew by per_shard_ws=2 (new shard added).
    assert result["world_size"] == 8

    # Survivors' worker_indices were remapped to post-compaction positions.
    assert r._shards["dp-0"].worker_indices == [0, 1]
    assert r._shards["dp-2"].worker_indices == [2, 3]
    assert r._shards["dp-3"].worker_indices == [4, 5]
    # New shard owns the next 2 indices.
    new_id = result["shard_id"]
    assert r._shards[new_id].worker_indices == [6, 7]
    # And carries N actor handles, not 1.
    assert len(r._shards[new_id].actor_handles) == 2


@pytest.mark.asyncio
async def test_add_shard_tp2_then_remove_kills_all_followers(monkeypatch):
    """End-to-end TP>1 lifecycle: add a fresh shard with N=2 actors,
    then remove it. mark_workers_dead must receive ALL N indices, and
    every actor handle must be ray.kill'd."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, removed_pgs = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGenerationWithWG(per_shard_ws=2, n_existing=2)
    # Bolt on the bookkeeping the router expects post-mark-dead.
    fake_gen.dp_size = 2
    fake_gen.worker_group.dead_indices = set()

    def _mark_dead(indices):
        fake_gen.worker_group.dead_indices.update(indices)
        fake_gen.worker_group.dp_leader_worker_indices = [
            i
            for i in fake_gen.worker_group.dp_leader_worker_indices
            if i not in fake_gen.worker_group.dead_indices
        ]

    fake_gen.worker_group.mark_workers_dead = _mark_dead

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-0", "http://a/v1"), ("dp-1", "http://b/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-0": [_FakeActor("dp0-l"), _FakeActor("dp0-h")],
            "dp-1": [_FakeActor("dp1-l"), _FakeActor("dp1-h")],
        },
        worker_indices_by_shard={"dp-0": [0, 1], "dp-1": [2, 3]},
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 4

    add_result = await r.add_shard(reason="test")
    assert add_result["added"] is True
    new_id = add_result["shard_id"]
    # New shard's actors are tracked; 2 of them.
    new_actor_names = {a.name for a in r._shards[new_id].actor_handles}
    assert len(new_actor_names) == 2

    # Remove the freshly-added shard. ray.kill must hit BOTH followers.
    rm_result = await r.remove_shard(new_id, reason="test")
    assert rm_result["removed"] is True
    # Both new actors were killed.
    for a in new_actor_names:
        assert a in killed_actors
    # mark_workers_dead got the full 2-index range, not just the leader.
    assert {4, 5}.issubset(fake_gen.worker_group.dead_indices)


# =====================================================================
# TP>1 ActorDiedError mid-init_collective: the WHOLE shard (every actor
# in its TP group) must be ray.kill'd and its worker_indices marked dead.
# Live run on the 30B TP=2 cluster hit this path: dp-4 (freshly-added)
# died with ActorDiedError during init_collective. Without this fix,
# _filter_targets_by_liveness (which probes only the leader) could
# mis-classify a half-dead TP group as alive and skip eviction —
# leading to worker_indices stale + the next refit dispatching onto a
# zombie actor.
# =====================================================================


@pytest.mark.asyncio
async def test_evict_failed_workers_inline_actor_died_force_evicts_tp2(
    monkeypatch,
):
    """When init_collective surfaces RayActorError/ActorDiedError for a
    TP>1 shard, _evict_failed_workers_inline must force-evict that shard
    REGARDLESS of what is_alive returns on the leader. The leader's
    actor wrapper can still respond to is_alive even after its
    engine_core process died (e.g. vLLM forks a subprocess, the parent
    actor outlives the engine), or a TP follower can die while the
    leader is up — Ray surfaces the death via ActorDiedError on the
    future, and that's authoritative."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, _ = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGeneration()
    pg_4 = _FakePG("pg-4")
    pg_5 = _FakePG("pg-5")
    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    # Both leaders return alive=True from is_alive (the misleading case
    # we hit on the live run). Without force-eviction on
    # ActorDiedError, the eviction path would skip both as "alive
    # survivors" and leave the dead one in the table.
    r.register_shards(
        [("dp-4", "http://x/v1"), ("dp-5", "http://y/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-4": [_LiveActor("dp4-l"), _LiveActor("dp4-h")],
            "dp-5": [_LiveActor("dp5-l"), _LiveActor("dp5-h")],
        },
        pg_by_shard={"dp-4": pg_4, "dp-5": pg_5},
        generation=fake_gen,
    )
    # Two failed futures: dp-4 raised ActorDiedError (Ray confirmed
    # actor death), dp-5 raised DistNetworkError (peer-loss, alive
    # survivor).
    failed_idxs = [0, 1]
    exc_types = ["ActorDiedError", "DistNetworkError"]
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=failed_idxs,
        reason="init_collective: per-worker failure",
        exception_types=exc_types,
    )
    # dp-4 force-evicted via ActorDiedError; dp-5 retained as alive
    # survivor (its DistNetworkError is the rendezvous-peer-loss
    # signature and is_alive returned True).
    assert num_evicted == 1
    assert evicted_ids == ["dp-4"]
    assert "dp-4" not in r._shards
    assert "dp-5" in r._shards
    # Both actors of the dead shard's TP group were ray.kill'd, not just
    # the leader.
    assert "dp4-l" in killed_actors
    assert "dp4-h" in killed_actors
    # Survivor's actors untouched.
    assert "dp5-l" not in killed_actors
    assert "dp5-h" not in killed_actors


@pytest.mark.asyncio
async def test_evict_failed_workers_inline_ray_actor_error_force_evicts(
    monkeypatch,
):
    """Same shape as above, but for RayActorError exception name (older
    Ray versions surface this name; newer versions surface
    ActorDiedError). Both must force-evict bypassing the liveness probe."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, _ = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGeneration()
    pg_a = _FakePG("pg-a")
    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-a": [_LiveActor("dpa-l"), _LiveActor("dpa-h")],
        },
        pg_by_shard={"dp-a": pg_a},
        generation=fake_gen,
    )
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=[0],
        reason="init_collective: per-worker failure",
        exception_types=["RayActorError"],
    )
    assert num_evicted == 1
    assert evicted_ids == ["dp-a"]
    # Both actors in the TP group killed.
    assert "dpa-l" in killed_actors
    assert "dpa-h" in killed_actors


@pytest.mark.asyncio
async def test_evict_failed_workers_inline_fast_fail_force_evicts_poisoned(
    monkeypatch,
):
    """Over-eviction fix: a POISONED worker (EngineCore dead, or NCCL state
    poisoned by a prior in-band comm_abort so its next comm_init raises in
    <1s) is still POD-ALIVE — is_alive returns True — so the liveness probe
    would wrongly KEEP it. _per_worker_results flags it as a fast-failer
    (raised inside the fast-fail window) via force_evict_idxs, and eviction
    must force-remove it. Meanwhile a healthy survivor that merely TIMED OUT
    (~30s NCCL bootstrap) waiting on the rendezvous for the dead peer — same
    exception type, but SLOW — must be retained.

    Timing, not exception type, is the discriminator. Pre-fix, both shards
    (exception NCCLError ∉ the keep set) would have been force-evicted,
    cascading the gen world toward zero on every fault."""
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, _ = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGeneration()
    pg_a = _FakePG("pg-a")
    pg_b = _FakePG("pg-b")
    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    # Both shards report is_alive=True. dp-a is POISONED (fast-fail); dp-b is
    # a healthy survivor that timed out on the rendezvous (slow-fail).
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1")],
        per_shard_world_size=1,
        actor_handles_by_shard={
            "dp-a": [_LiveActor("a-leader")],
            "dp-b": [_LiveActor("b-leader")],
        },
        pg_by_shard={"dp-a": pg_a, "dp-b": pg_b},
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 2

    # Both futures "failed" with the same exception name, but only dp-a
    # (idx 0) raised inside the fast-fail window. dp-b raised at the ~30s
    # bootstrap timeout (slow → healthy survivor blocked on the dead peer).
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=[0, 1],
        reason="update_weights_from_collective: per-worker error",
        exception_types=["NCCLError", "NCCLError"],
        force_evict_idxs=[0],
    )
    # Only the poisoned fast-failer is evicted; the healthy slow-failer is
    # kept (is_alive=True). It rejoins on train's next retry.
    assert num_evicted == 1
    assert evicted_ids == ["dp-a"]
    assert "dp-a" not in r._shards
    assert "dp-b" in r._shards
    assert "a-leader" in killed_actors
    assert "b-leader" not in killed_actors
    assert r.current_gen_world_size() == 1


@pytest.mark.asyncio
async def test_add_shard_tp2_actor_died_during_init_collective_cleans_up_full_shard(
    monkeypatch,
):
    """End-to-end recreation of the live-run failure on dp-4: a freshly-
    added TP=2 shard's leader dies during init_collective with
    ActorDiedError. The router must remove the WHOLE shard (every actor
    + its worker_indices marked dead), not orphan the follower.

    Verifies: after eviction, (1) all actors in the dead shard are
    ray.kill'd, (2) mark_workers_dead got every index in the shard's TP
    group, (3) the surviving shard is intact, (4) world size shrinks
    from 4 to 2.
    """
    import nemo_rl.models.generation.generation_router as gr

    killed_actors, _ = _stub_ray_for_eviction(monkeypatch)

    fake_gen = _FakeGenerationWithWG(per_shard_ws=2, n_existing=2)
    fake_gen.dp_size = 2
    fake_gen.worker_group.dead_indices = set()
    marked_dead: list[int] = []

    def _mark_dead(indices):
        marked_dead.extend(indices)
        fake_gen.worker_group.dead_indices.update(indices)
        fake_gen.worker_group.dp_leader_worker_indices = [
            i
            for i in fake_gen.worker_group.dp_leader_worker_indices
            if i not in fake_gen.worker_group.dead_indices
        ]

    fake_gen.worker_group.mark_workers_dead = _mark_dead

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    # 2 shards × TP=2; dp-0 alive, dp-1 will simulate the just-added
    # shard whose leader died during init_collective.
    pg_0 = _FakePG("pg-0")
    pg_1 = _FakePG("pg-1")
    r.register_shards(
        [("dp-0", "http://a/v1"), ("dp-1", "http://b/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-0": [_LiveActor("dp0-l"), _LiveActor("dp0-h")],
            "dp-1": [_LiveActor("dp1-l"), _LiveActor("dp1-h")],
        },
        pg_by_shard={"dp-0": pg_0, "dp-1": pg_1},
        worker_indices_by_shard={"dp-0": [0, 1], "dp-1": [2, 3]},
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 4

    # Future for dp-1's leader raised ActorDiedError (Ray confirmed the
    # actor died — this is the live-run signature). dp-0's future
    # succeeded so failed_idxs=[1] only.
    num_evicted, evicted_ids = await r._evict_failed_workers_inline(
        failed_idxs=[1],
        reason="init_collective: per-worker failure",
        exception_types=[None, "ActorDiedError"],
    )
    assert num_evicted == 1
    assert evicted_ids == ["dp-1"]
    # All actors in dp-1's TP group killed (NOT just the leader).
    assert "dp1-l" in killed_actors
    assert "dp1-h" in killed_actors
    # Survivor untouched.
    assert "dp0-l" not in killed_actors
    assert "dp0-h" not in killed_actors
    # mark_workers_dead got the full 2-index range for dp-1.
    assert {2, 3}.issubset(set(marked_dead))
    # World size shrank to 1 shard × 2 = 2.
    assert r.current_gen_world_size() == 2
    assert "dp-1" not in r._shards
    assert "dp-0" in r._shards


@pytest.mark.asyncio
async def test_add_shard_tp2_partial_init_failure_does_not_orphan_followers(
    monkeypatch,
):
    """If add_dp_worker raises mid-spawn (e.g. prepare_refit_info on the
    leader times out / crashes after followers are already alive on the
    PG), the cleanup path must ray.kill ALL spawned actors (leader +
    every follower) AND release the PG. Otherwise the followers leak
    onto the PG and the next add_shard gets a partial slot.

    This exercises the cleanup-on-failure path of add_dp_worker — we
    simulate it via a fake generation whose add_dp_worker raises after
    mock-spawning all N actors and signals back via a side-channel
    which actors got killed."""
    import nemo_rl.models.generation.generation_router as gr

    class _FakeAddDpFailingGen:
        """Fake VllmGeneration whose add_dp_worker simulates a Phase A
        failure: all N actors are spawned (on the simulated PG), but
        prepare_refit_info raises before append. The cleanup path is
        expected to ray.kill all N actors (recorded here) and release
        the PG before re-raising. The router catches the raise in
        add_shard and returns added=False without inserting a partial
        entry."""

        def __init__(self):
            self.worker_group = _FakeWorkerGroup([0, 2])  # 2 existing leaders
            self.killed_during_cleanup: list[str] = []
            self.pg_removed = False

        def add_dp_worker(self, pre_append_hook=None):
            # Simulate Phase A: spawn 2 actors. They live on the PG until
            # we explicitly kill them.
            spawned = [_FakeActor("new-leader"), _FakeActor("new-follower")]

            # Simulate the production cleanup path verbatim: kill every
            # spawned actor, remove the PG, then raise. The real
            # vllm_generation.add_dp_worker does this for the
            # prepare_refit_info / report_url failure paths.
            for actor in spawned:
                self.killed_during_cleanup.append(actor.name)
            self.pg_removed = True
            raise RuntimeError("simulated prepare_refit_info crash")

        def reset_collective(self):
            return []

    fake_gen = _FakeAddDpFailingGen()
    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-0", "http://a/v1"), ("dp-1", "http://b/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={
            "dp-0": [_FakeActor("dp0-l"), _FakeActor("dp0-h")],
            "dp-1": [_FakeActor("dp1-l"), _FakeActor("dp1-h")],
        },
        worker_indices_by_shard={"dp-0": [0, 1], "dp-1": [2, 3]},
        generation=fake_gen,
    )
    assert r.current_gen_world_size() == 4

    result = await r.add_shard(reason="test partial failure")
    # Add reported failure cleanly.
    assert result["added"] is False
    assert "add_dp_worker failed" in result["reason"]
    # Both spawned actors were killed in the cleanup path (NOT just
    # the leader).
    assert "new-leader" in fake_gen.killed_during_cleanup
    assert "new-follower" in fake_gen.killed_during_cleanup
    # PG was released so KubeRay's autoscaler can reclaim the pod.
    assert fake_gen.pg_removed is True
    # Router did NOT insert a half-built shard entry — survivors stay
    # at 2 shards.
    assert len(r._shards) == 2
    assert "dp-0" in r._shards
    assert "dp-1" in r._shards
    assert r.current_gen_world_size() == 4
    # Refit gate is open (gate cleared on the exception path).
    ready, _ = r.refit_ready_state()
    assert ready


@pytest.mark.asyncio
async def test_vllm_add_dp_worker_phase_a_failure_kills_all_spawned_actors(
    monkeypatch,
):
    """Direct test of the vllm_generation.add_dp_worker cleanup path:
    when prepare_refit_info raises after spawn_workers_for_shard
    succeeds, ray.kill must hit every spawned actor (leader and TP
    followers), then remove_placement_group must run.

    This is the unit-level guarantee for the orphaned-follower bug:
    Ray's GC alone is racy at TP>1 because followers reserve PG bundles
    independently of the leader and can outlive the leader's actor
    wrapper until the PG is fully reclaimed.
    """
    import sys
    import types

    # Stub ray + ray.util.placement_group BEFORE importing vllm_generation
    # so the in-module ``import ray`` resolves to our stub. Guard the
    # imports vllm_generation does internally too.
    killed_actor_names: list[str] = []
    pgs_removed: list[str] = []

    class _StubFuture:
        """Returned by leader_actor.<method>.remote(...)."""

        def __init__(self, op: str, raises: bool = False):
            self.op = op
            self.raises = raises

    class _StubLeader:
        """Stand-in vLLM leader actor. prepare_refit_info_async raises
        to simulate the Phase A failure."""

        def __init__(self, name: str):
            self.name = name
            self._method_pat = {
                "prepare_refit_info_async": True,  # raises
                "prepare_refit_info": True,
                "warmup_nccl_library_async": False,
                "warmup_nccl_library": False,
                "report_dp_openai_server_base_url": False,
            }

        def __getattr__(self, attr: str):
            raises = self._method_pat.get(attr, False)
            return _StubMethod(attr, raises)

    class _StubFollower:
        def __init__(self, name: str):
            self.name = name

    class _StubMethod:
        def __init__(self, name: str, raises: bool):
            self.name = name
            self.raises = raises

        def remote(self, *args, **kwargs):
            return _StubFuture(self.name, raises=self.raises)

    def fake_kill(actor, no_restart=True):
        killed_actor_names.append(getattr(actor, "name", repr(actor)))

    def fake_get(fut, timeout=None):  # noqa: ARG001
        if isinstance(fut, _StubFuture) and fut.raises:
            raise RuntimeError(f"simulated {fut.op} failure")
        return None

    class _StubPG:
        """Stand-in placement group. ``ready()`` returns a sentinel
        future; ``ray.get`` resolves it as None."""

        def __init__(self, name: str):
            self.name = name

        def ready(self):
            return _StubFuture("pg.ready", raises=False)

    fake_pg_obj = _StubPG("pg-new")

    def fake_pg(*args, **kwargs):
        return fake_pg_obj

    def fake_remove_pg(pg):
        pgs_removed.append(getattr(pg, "name", repr(pg)))

    stub_ray = types.SimpleNamespace(
        kill=fake_kill,
        get=fake_get,
        util=types.SimpleNamespace(
            placement_group=fake_pg,
        ),
    )
    stub_pg_mod = types.SimpleNamespace(
        remove_placement_group=fake_remove_pg,
    )
    monkeypatch.setitem(sys.modules, "ray", stub_ray)
    monkeypatch.setitem(
        sys.modules, "ray.util", types.SimpleNamespace(placement_group=fake_pg)
    )
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", stub_pg_mod)

    # Build a minimal VllmGeneration-shaped object with just enough
    # surface to drive add_dp_worker through the cleanup path.
    from nemo_rl.models.generation.vllm import vllm_generation as vg

    # Patch vg.ray to our stub so the module-level ``import ray`` we
    # already resolved gets shadowed inside the function body.
    monkeypatch.setattr(vg, "ray", stub_ray, raising=True)

    leader = _StubLeader("new-leader")
    followers = [_StubFollower("new-follower")]

    spawned_tuples = [
        (leader, "new-leader-name", (0, [0, 1]), 2, {}),
        (followers[0], "new-follower-name", None, 2, {}),
    ]

    class _StubCluster:
        max_colocated_worker_groups = 1
        use_gpus = False

    class _StubWG:
        def __init__(self):
            self.cluster = _StubCluster()
            self.dp_size = 2
            self._spawn_called = False

        def spawn_workers_for_shard(self, **kwargs):
            self._spawn_called = True
            return spawned_tuples

        def append_spawned_shard_workers(self, **kwargs):
            raise AssertionError(
                "Phase B append should never run when prepare_refit_info raises"
            )

    class _StubVllmGeneration:
        """Smallest shape of VllmGeneration that add_dp_worker reads."""

        def __init__(self):
            self.worker_group = _StubWG()
            self.ep_size = 1
            self.tp_size = 2
            self.pp_size = 1
            self.model_parallel_size = 2
            self.dp_size = 2
            self.cfg = {"vllm_cfg": {"async_engine": True}}
            self._cached_state_dict_info = {"some_key": "some_value"}
            self.dp_openai_server_base_urls = []

    instance = _StubVllmGeneration()

    # Bind the real add_dp_worker to our stub instance and invoke.
    add_method = vg.VllmGeneration.add_dp_worker.__get__(
        instance, vg.VllmGeneration
    )
    with pytest.raises(RuntimeError, match="prepare_refit_info"):
        add_method()
    # All spawned actors got ray.kill'd (leader + follower).
    assert "new-leader" in killed_actor_names
    assert "new-follower" in killed_actor_names
    # PG was removed in the same cleanup path.
    assert "pg-new" in pgs_removed


# =====================================================================
# Zero-shards collective endpoints: train side already retries on 503
# but treats 500 as fatal. Cascade-to-zero (every shard evicted by
# init_collective failures) must surface a recoverable status.
# =====================================================================


@pytest.mark.asyncio
async def test_router_collective_endpoints_handle_zero_shards_gracefully(
    monkeypatch,
):
    """When the router has 0 shards alive in the NCCL group,
    /init_collective, /update_weights_from_collective, and
    /reset_collective must NOT return 500. They must return a
    recoverable status (503 for init/update, 2xx no-op for reset)
    so the train side's retry path treats it as transient and waits
    for add_shard recovery instead of fast-failing."""
    import asyncio as _asyncio

    import aiohttp
    import httpx

    import nemo_rl.models.generation.generation_router as gr

    # Build a router with NO shards alive — register_shards then evict
    # to drop shard_count_alive_for_collective() to 0. Use a
    # _FakeGeneration so /reset_collective has something to dispatch
    # to (but it should short-circuit before that).
    fake_gen = _FakeGeneration()
    r = gr.GenerationRouter(
        port=0,
        health_poll_interval_s=10.0,
        proxy_timeout_s=2.0,
    )
    r.register_shards([], per_shard_world_size=2, generation=fake_gen)
    assert r.shard_count_alive_for_collective() == 0
    assert r.current_gen_world_size() == 0

    # Bring up the FastAPI app via httpx.ASGITransport on the test
    # event loop so we can hit the route handlers directly without a
    # real uvicorn.
    r._http_session = aiohttp.ClientSession()
    transport = httpx.ASGITransport(app=r.get_app())
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://router.test"
        ) as client:
            # /init_collective → 503 (recoverable), not 500.
            resp = await client.post(
                "/init_collective",
                json={
                    "ip": "127.0.0.1",
                    "port": 12345,
                    "world_size": 0,
                    "train_world_size": 1,
                },
            )
            assert resp.status_code == 503, resp.text
            body = resp.json()
            assert body["success"] is False
            assert body["current_gen_world_size"] == 0
            # ``error`` describes the cascade, not a generic Python
            # exception traceback (500 path). Train's retry treats
            # this as transient and re-polls /refit_ready.
            assert "no shards" in body["error"].lower()

            # /update_weights_from_collective → 503 (recoverable),
            # not 500.
            resp = await client.post("/update_weights_from_collective")
            assert resp.status_code == 503, resp.text
            body = resp.json()
            assert body["success"] is False
            assert body["current_gen_world_size"] == 0
            # Make sure the response has the keys train side reads.
            assert body.get("evicted_shard_ids") == []
            assert body.get("promoted_shards") == []

            # /reset_collective → 200 success no-op (idempotent
            # contract). Train side already calls this defensively
            # before each rendezvous; raising on empty world is
            # what was killing the recovery path.
            resp = await client.post("/reset_collective")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["success"] is True
            # Reset_collective must NOT have dispatched to the
            # underlying generation when no shards exist.
            assert fake_gen.reset_calls == 0
    finally:
        await r._http_session.close()
        # Best-effort drain of any tasks the lifespan would normally
        # cancel.
        await _asyncio.sleep(0)


@pytest.mark.asyncio
async def test_refit_ready_state_zero_shards_returns_false(monkeypatch):
    """Sanity: when shard_count_alive_for_collective is 0, refit_ready_state
    returns (False, "no_shards_alive") so the train side waits for
    add_shard rather than dispatching init_collective on an empty
    world."""
    import nemo_rl.models.generation.generation_router as gr

    r = gr.GenerationRouter(port=0)
    # Empty router.
    ready, reason = r.refit_ready_state()
    assert ready is False
    assert reason == "no_shards_alive"


@pytest.mark.asyncio
async def test_fire_cordon_hook_skips_auto_remove_when_last_alive(monkeypatch):
    """When cordoning the LAST alive shard, _fire_cordon_hook must NOT
    auto-remove (which would drop fleet to 0). Instead it leaves the
    cordoned shard parked so traffic abatement / manual uncordon can
    recover it, and so the train side doesn't burn the full add_shard
    recovery budget (~7-10 min on 30B) before progress is possible.

    Regression test for the live-run failure mode where dp-3 (lone
    survivor after a 3-shard burst kill) saturated under full rollout
    QPS, /health probes timed out → cordoned → auto-removed →
    ActorDiedError → 0 shards → no train progress until next FaultInjector
    add_shard cycle.
    """
    import nemo_rl.models.generation.generation_router as gr

    fake_gen = _FakeGeneration()
    actor_a = _FakeActor("dp-a-leader")
    pg_a = _FakePG("pg-a")

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={"dp-a": [actor_a]},
        pg_by_shard={"dp-a": pg_a},
        generation=fake_gen,
    )

    # Drive dp-a into cordoned status (mimic the health-poll cordon
    # transition — set status before _fire_cordon_hook, since the
    # hook itself is dispatched after the status flip).
    r._shards["dp-a"].status = "cordoned"
    assert r.shard_count_alive_for_collective() == 0

    # Patch ray.kill / remove_placement_group so any inadvertent
    # remove_shard call would surface as kills we can detect.
    killed: list[str] = []
    import sys
    import types
    monkeypatch.setitem(
        sys.modules,
        "ray",
        types.SimpleNamespace(
            kill=lambda actor, no_restart=True: killed.append(actor.name),
            get=lambda fs: [],
        ),
    )
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "ray.util.placement_group",
        types.SimpleNamespace(remove_placement_group=lambda pg: None),
    )

    # Fire the cordon hook. With the last-alive guard, this should
    # SKIP auto-remove and leave dp-a in the table.
    await r._fire_cordon_hook("dp-a", "health threshold breached")

    # dp-a still in the table (auto-remove skipped).
    assert "dp-a" in r._shards
    # No actors got ray.kill'd.
    assert killed == []


@pytest.mark.asyncio
async def test_fire_cordon_hook_auto_removes_when_other_shards_alive(monkeypatch):
    """Sanity check the inverse: when there ARE other alive shards, the
    hook DOES auto-remove the cordoned shard (the original behavior).
    This protects against the last-alive guard accidentally suppressing
    auto-remove in the common case.
    """
    import nemo_rl.models.generation.generation_router as gr

    fake_gen = _FakeGeneration()
    actor_a = _FakeActor("dp-a-leader")
    actor_b = _FakeActor("dp-b-leader")
    pg_a = _FakePG("pg-a")
    pg_b = _FakePG("pg-b")

    r = gr.GenerationRouter(port=0, health_poll_interval_s=10.0)
    r.register_shards(
        [("dp-a", "http://a/v1"), ("dp-b", "http://b/v1")],
        per_shard_world_size=2,
        actor_handles_by_shard={"dp-a": [actor_a], "dp-b": [actor_b]},
        pg_by_shard={"dp-a": pg_a, "dp-b": pg_b},
        generation=fake_gen,
    )

    # Cordon dp-a; dp-b stays ready.
    r._shards["dp-a"].status = "cordoned"
    assert r.shard_count_alive_for_collective() == 1  # dp-b still ready

    killed: list[str] = []
    import sys
    import types
    monkeypatch.setitem(
        sys.modules,
        "ray",
        types.SimpleNamespace(
            kill=lambda actor, no_restart=True: killed.append(actor.name),
            get=lambda fs: [],
        ),
    )
    monkeypatch.setitem(sys.modules, "ray.util", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "ray.util.placement_group",
        types.SimpleNamespace(remove_placement_group=lambda pg: None),
    )

    await r._fire_cordon_hook("dp-a", "health threshold breached")

    # dp-a was removed from table (auto-remove fired).
    assert "dp-a" not in r._shards
    assert "dp-b" in r._shards
    # dp-a's actor got ray.kill'd; dp-b untouched.
    assert "dp-a-leader" in killed
    assert "dp-b-leader" not in killed


# =====================================================================
# RL-412 auto-backfill reconciler
# =====================================================================
import time as _time


def _seed_shards(r: GenerationRouter, statuses: dict) -> None:
    """Seed router._shards with minimal ShardEntry rows. statuses maps
    shard_id -> status string."""
    r._shards = {
        sid: ShardEntry(
            shard_id=sid, url="", status=st, last_health_ok_at=_time.monotonic()
        )
        for sid, st in statuses.items()
    }


def test_backfill_deficit_counts_ready_joining_and_inflight():
    r = GenerationRouter(port=0, backfill_target=8)
    _seed_shards(r, {"dp-0": "ready", "dp-1": "ready", "dp-2": "joining",
                     "dp-3": "cordoned"})
    # alive (ready+joining) = 3; cordoned excluded; target 8 -> deficit 5
    assert r._backfill_deficit() == 5
    r._inflight_adds = 2
    # 3 alive + 2 in-flight => deficit 3
    assert r._backfill_deficit() == 3


def test_backfill_target_defaults_to_bootstrap():
    r = GenerationRouter(port=0)  # no explicit target
    r._total_shards_at_bootstrap = 8
    _seed_shards(r, {"dp-0": "ready"})
    assert r._backfill_target_count() == 8
    assert r._backfill_deficit() == 7


@pytest.mark.asyncio
async def test_reconcile_launches_capped_by_max_concurrent_and_inflight():
    r = GenerationRouter(port=0, auto_backfill=True, backfill_target=8,
                         backfill_max_concurrent=3)
    r._generation = object()  # non-None so reconciler is active
    _seed_shards(r, {"dp-0": "ready", "dp-1": "ready"})  # alive=2, deficit=6

    calls = {"n": 0}

    async def _fake_add_shard(reason="manual"):
        calls["n"] += 1
        await asyncio.sleep(0.01)  # simulate PG boot
        return {"added": True, "shard_id": f"dp-x{calls['n']}"}

    r.add_shard = _fake_add_shard  # type: ignore[assignment]

    launched = r._reconcile_backfill()
    assert launched == 3            # capped by max_concurrent
    assert r._inflight_adds == 3    # reserved immediately
    # second tick while 3 in-flight => no new launches (slots full)
    assert r._reconcile_backfill() == 0
    await asyncio.sleep(0.05)       # let the 3 tasks finish
    assert calls["n"] == 3
    assert r._inflight_adds == 0    # released after completion


@pytest.mark.asyncio
async def test_reconcile_noop_when_disabled_or_no_generation():
    r = GenerationRouter(port=0, auto_backfill=False, backfill_target=8)
    r._generation = object()
    _seed_shards(r, {"dp-0": "ready"})
    assert r._reconcile_backfill() == 0   # disabled
    r.auto_backfill = True
    r._generation = None
    assert r._reconcile_backfill() == 0   # no backing generation


@pytest.mark.asyncio
async def test_health_loop_calls_reconcile_each_tick(monkeypatch):
    r = GenerationRouter(port=0, auto_backfill=True, health_poll_interval_s=0.05,
                         backfill_target=8)
    r._generation = object()
    _seed_shards(r, {"dp-0": "ready"})
    r._http_session = aiohttp.ClientSession()
    ticks = {"n": 0}
    monkeypatch.setattr(
        r, "_reconcile_backfill",
        lambda: ticks.__setitem__("n", ticks["n"] + 1) or 0,
    )
    task = asyncio.create_task(r._health_poll_loop())
    await asyncio.sleep(0.2)   # ~3-4 ticks
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    await r._http_session.close()
    assert ticks["n"] >= 2


def test_metrics_snapshot_includes_backfill_fields():
    r = GenerationRouter(port=0, auto_backfill=True, backfill_target=8)
    _seed_shards(r, {"dp-0": "ready", "dp-1": "joining"})
    r._inflight_adds = 1
    snap = r.metrics_snapshot()
    assert snap["backfill_target"] == 8
    assert snap["backfill_in_flight"] == 1
    assert snap["backfill_deficit"] == 8 - 2 - 1  # target - (ready+joining) - inflight


# =====================================================================
# RL-412 elastic re-init: joinable set + stability signal
# =====================================================================


def _seed_shards_with_successes(r, rows):
    """rows maps shard_id -> (status, consecutive_successes)."""
    r._shards = {
        sid: ShardEntry(
            shard_id=sid, url="", status=st,
            last_health_ok_at=_time.monotonic(), consecutive_successes=succ,
        )
        for sid, (st, succ) in rows.items()
    }


def test_joinable_world_size_excludes_unhealthy_joining():
    # joinable_min_age_s=0.0 isolates the threshold gate from the warm-up age
    # gate (tested separately below).
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=0.0)
    r._per_shard_world_size = 1
    _seed_shards_with_successes(r, {
        "dp-0": ("ready", 5),      # joinable
        "dp-1": ("ready", 0),      # joinable (ready is always joinable)
        "dp-2": ("joining", 2),    # joinable (>= threshold)
        "dp-3": ("joining", 1),    # NOT joinable (still booting/unproven)
        "dp-4": ("cordoned", 9),   # NOT joinable
    })
    assert r.shard_count_alive_for_collective() == 4   # ready+joining
    assert r.joinable_world_size() == 3                # ready,ready,joining(2)


def test_joinable_stable_resets_when_count_changes():
    r = GenerationRouter(port=0, join_success_threshold=1, joinable_min_age_s=0.0)
    r._per_shard_world_size = 1
    _seed_shards_with_successes(r, {"dp-0": ("ready", 1)})
    r._joinable_changed_at = _time.monotonic() - 100.0
    r._last_joinable_count = 1
    # No change → timer NOT reset → still ~100s stable.
    r._refresh_joinable_stability()
    assert r.joinable_stable_for_s() >= 100.0
    # A new joinable shard appears → timer resets to ~0.
    r._shards["dp-1"] = ShardEntry(
        shard_id="dp-1", url="", status="ready",
        last_health_ok_at=_time.monotonic(), consecutive_successes=1,
    )
    r._refresh_joinable_stability()
    assert r.joinable_stable_for_s() < 1.0
    assert r._last_joinable_count == 2


def test_metrics_snapshot_includes_joinable_fields():
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=0.0)
    r._per_shard_world_size = 1
    _seed_shards_with_successes(r, {"dp-0": ("ready", 3), "dp-1": ("joining", 1)})
    snap = r.metrics_snapshot()
    assert snap["joinable_world_size"] == 1            # dp-1 not yet joinable
    assert "joinable_stable_for_s" in snap
    assert snap["joinable_stable_for_s"] >= 0.0


def test_joinable_warmup_age_gate_excludes_fresh_unproven():
    # A fresh (just-added), health-responsive, UNPROVEN joining shard must NOT
    # count as joinable until it ages past the warm-up window (cold RoCE route).
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=90.0)
    r._per_shard_world_size = 1
    now = _time.monotonic()
    r._shards = {
        "dp-0": ShardEntry(shard_id="dp-0", url="", status="ready",
                           last_health_ok_at=now, consecutive_successes=5),
        # fresh + healthy + unproven → excluded by the age gate
        "dp-1": ShardEntry(shard_id="dp-1", url="", status="joining",
                           last_health_ok_at=now, consecutive_successes=3,
                           joined_at=now, proven=False),
    }
    assert r.joinable_world_size() == 1                # only dp-0 (ready)
    # Age it past the warm-up window → now joinable.
    r._shards["dp-1"].joined_at = now - 100.0
    assert r.joinable_world_size() == 2


def test_joinable_proven_bypasses_age_gate():
    # A PROVEN shard (rendezvoused before → route warm) rejoins immediately,
    # even if freshly re-added.
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=90.0)
    r._per_shard_world_size = 1
    now = _time.monotonic()
    r._shards = {
        "dp-0": ShardEntry(shard_id="dp-0", url="", status="joining",
                           last_health_ok_at=now, consecutive_successes=3,
                           joined_at=now, proven=True),
    }
    assert r.joinable_world_size() == 1                # proven bypasses age


def test_joinable_cohort_snapshot_excludes_cold_and_cordoned():
    # The frozen cohort the rendezvous dispatches: ready + warm-joining only,
    # with their flat worker_indices. Cold/booting joining + cordoned excluded.
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=90.0)
    r._per_shard_world_size = 1
    now = _time.monotonic()
    r._shards = {
        "dp-0": ShardEntry(shard_id="dp-0", url="", status="ready",
                           last_health_ok_at=now, consecutive_successes=5),
        "dp-1": ShardEntry(shard_id="dp-1", url="", status="joining",
                           last_health_ok_at=now, consecutive_successes=3,
                           joined_at=now - 200.0, proven=False),   # warm by age
        "dp-2": ShardEntry(shard_id="dp-2", url="", status="joining",
                           last_health_ok_at=now, consecutive_successes=3,
                           joined_at=now, proven=False),            # cold → excluded
        "dp-3": ShardEntry(shard_id="dp-3", url="", status="cordoned",
                           last_health_ok_at=now, consecutive_successes=9),
    }
    r._shards["dp-0"].worker_indices = [0]
    r._shards["dp-1"].worker_indices = [1]
    r._shards["dp-2"].worker_indices = [2]
    r._shards["dp-3"].worker_indices = [3]
    sids, indices = r._joinable_cohort()
    assert set(sids) == {"dp-0", "dp-1"}      # cold dp-2 + cordoned dp-3 excluded
    assert indices == [0, 1]                  # flat, sorted worker indices


def test_joinable_ready_shard_ignores_age_gate():
    # ``ready`` shards are joinable unconditionally (already promoted by a prior
    # successful refit → warm), regardless of joined_at / proven.
    r = GenerationRouter(port=0, join_success_threshold=2, joinable_min_age_s=90.0)
    r._per_shard_world_size = 1
    now = _time.monotonic()
    r._shards = {
        "dp-0": ShardEntry(shard_id="dp-0", url="", status="ready",
                           last_health_ok_at=now, consecutive_successes=0,
                           joined_at=now, proven=False),
    }
    assert r.joinable_world_size() == 1


@pytest.mark.asyncio
async def test_current_gen_world_size_endpoint_exposes_joinable(router, aclient):
    # The router fixture registers dp-a, dp-b as ready with per_shard_ws=2.
    resp = await aclient.get("/current_gen_world_size")
    assert resp.status_code == 200
    body = resp.json()
    assert body["world_size"] == 4
    assert body["joinable_world_size"] == 4          # both ready → joinable
    assert "joinable_stable_for_s" in body
    assert body["joinable_stable_for_s"] >= 0.0
    # The trainer's quiesce-wait reads this; steady fixture → not re-initing.
    assert body["nccl_reinit_in_progress"] is False


def test_eligible_promote_excludes_not_in_comm_joining():
    """A `joining` shard that never joined the comm (in_comm=False, the
    debounce window) must NOT be eligible for promotion — promoting it would
    route the data plane to stale dummy weights. Once a grow re-init pulls it
    into the comm (in_comm=True), it becomes eligible."""
    r = GenerationRouter(port=0)
    r._per_shard_world_size = 1
    _seed_shards_with_successes(r, {
        "dp-0": ("ready", 5),      # ready, not joining → not in promote set
        "dp-1": ("joining", 3),    # joining + in_comm below → eligible
        "dp-2": ("joining", 3),    # joining, NOT in_comm → debounced → excluded
    })
    r._shards["dp-0"].in_comm = True
    r._shards["dp-1"].in_comm = True
    r._shards["dp-2"].in_comm = False
    assert r._eligible_promote_sids() == {"dp-1"}
    # After a grow re-init pulls dp-2 in:
    r._shards["dp-2"].in_comm = True
    assert r._eligible_promote_sids() == {"dp-1", "dp-2"}
    # promote_all_joining honors the eligible set (dp-2 excluded first time).
    r._shards["dp-2"].in_comm = False
    promoted = r.promote_all_joining(eligible_sids=r._eligible_promote_sids())
    assert promoted == ["dp-1"]
    assert r._shards["dp-2"].status == "joining"   # debounced shard untouched


@pytest.mark.asyncio
async def test_health_poll_cordons_unhealthy_joining_shard():
    """A `joining` shard whose engine goes unhealthy is cordoned after the
    failure threshold (so `alive` converges back to `joinable`). Before this
    change the health poll only ever cordoned `ready` shards, so a `joining`
    shard that lost its engine would pin `alive > joinable` forever and wedge
    the trainer's quiescence check."""
    import nemo_rl.models.generation.generation_router as gr
    r = gr.GenerationRouter(
        port=0, health_poll_interval_s=0.05, health_timeout_s=0.2,
        failure_threshold=2, join_success_threshold=1,
    )
    # A `joining` shard that never answers health (unroutable url). No actor
    # handles, so the proxy-side _probe short-circuits and only the HTTP
    # /openapi.json probe runs — which fails on this address.
    r._shards["dp-bad"] = ShardEntry(
        shard_id="dp-bad", url="http://127.0.0.1:1/v1", status="joining",
        last_health_ok_at=_time.monotonic(),
    )
    r._http_session = aiohttp.ClientSession()
    task = asyncio.create_task(r._health_poll_loop())
    await asyncio.sleep(0.6)   # > failure_threshold poll intervals
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    await r._http_session.close()
    assert r._shards["dp-bad"].status == "cordoned"
