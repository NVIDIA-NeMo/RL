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
"""GenerationRouter — unified ingress for disaggregated vLLM generation.

Exposes data plane (``/v1/{completions,chat/completions}`` proxied to per-shard
vLLM with cordon + replay) and control plane (weight sync, lifecycle, metrics)
on a single port (8089). Both training and NemoGym point here; shard failures
are invisible to clients via round-robin routing with replay.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Optional

import aiohttp
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

ShardStatus = Literal["ready", "draining", "cordoned", "joining"]

# vLLM's HTTP server exposes /openapi.json as a stable always-200 liveness probe.
_HEALTH_PATH = "/openapi.json"
_DEFAULT_HEALTH_INTERVAL_S = 2.0
# 5s timeout: vLLM's event loop can stall 2-3s under heavy burst load.
_DEFAULT_HEALTH_TIMEOUT_S = 5.0
# 10 failures required: ~50s of sustained unresponsiveness before cordon.
_DEFAULT_FAILURE_THRESHOLD = 10
_DEFAULT_JOIN_SUCCESS_THRESHOLD = 2
# Warm-up gate default (env NRL_JOINABLE_MIN_AGE_S). In ft_constants so the default lives alongside other FT knobs.
from nemo_rl.models.generation.ft_constants import (  # noqa: E402
    JOINABLE_MIN_AGE_S as _DEFAULT_JOINABLE_MIN_AGE_S,
)
# Require N consecutive proxy errors before cordoning; prevents burst 5xx from evicting a healthy shard.
_DEFAULT_PROXY_FAILURE_THRESHOLD = 5
# Raise connection limits well above aiohttp's default (100) to handle large RL batches.
_DEFAULT_PROXY_POOL_LIMIT_TOTAL = 4096
_DEFAULT_PROXY_POOL_LIMIT_PER_HOST = 1024
# Explicit backlog for tuning; keep_alive 120s amortizes TCP setup.
_DEFAULT_UVICORN_BACKLOG = 4096
_DEFAULT_UVICORN_KEEP_ALIVE_S = 120
_DEFAULT_PROXY_TIMEOUT_S = 300.0
# Bounds NCCL teardown per survivor after a peer is killed; prevents deadlock if abort wedges.
_DEFAULT_RESET_COLLECTIVE_TIMEOUT_S = 30.0


@dataclass
class ShardEntry:
    """Router's view of a single vLLM DP shard.

    `url` is the OpenAI-compatible base URL. `actor_handles` let `remove_shard`
    kill the actors. `node_id` records the SLURM node so a replacement can be
    pinned there.
    """

    shard_id: str
    url: str
    status: ShardStatus = "joining"
    last_health_ok_at: float = 0.0
    inflight: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    # Separate from health-poll failures so burst 5xx doesn't cordon a shard before the probe notices.
    proxy_consecutive_failures: int = 0
    # Gates joining→ready: requires >=2 successful refits to avoid post-join weight corruption
    # from an incomplete NCCL bootstrap on the first refit.
    successful_refits_since_join: int = 0
    # True once this shard is in the live weight-sync comm. Freshly added shards start False;
    # the data plane skips them until in_comm is set by /init_collective success.
    in_comm: bool = False
    # Monotonic timestamp this entry was created. A cold backfill shard must age past
    # JOINABLE_MIN_AGE_S before counting as joinable (unless already proven).
    joined_at: float = 0.0
    # Sticky: True once this shard completed >=1 successful init_collective (route is warm).
    # A proven shard bypasses the age gate on re-add.
    proven: bool = False
    actor_handles: list[Any] = field(default_factory=list, repr=False)
    # SLURM node the shard ran on — used by add_shard to pin the replacement
    # actor back to the same node via NodeAffinitySchedulingStrategy.
    node_id: str = ""
    # Indices of these actors in the underlying VllmGeneration's worker_group.
    # Used by remove_shard to call mark_workers_dead so subsequent
    # init_collective / reset_collective skip them.
    worker_indices: list[int] = field(default_factory=list, repr=False)
    _health_url: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        base = self.url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._health_url = f"{base}{_HEALTH_PATH}"
        # Stamp creation time for the warm-up age gate, unless the caller
        # supplied one (tests pass an explicit aged value to simulate warm-up).
        if self.joined_at == 0.0:
            self.joined_at = time.monotonic()


class GenerationRouter:
    """Unified data plane + control plane for the gen cluster.

    Owns the authoritative shard table (populated by ``register_shards()``,
    mutated by the health poller and admin endpoints) and gen-side NCCL
    lifecycle. The train side has a single HTTP target for data, status,
    lifecycle, and weight sync.
    """

    def __init__(
        self,
        port: int = 8089,
        *,
        generation: Any = None,
        health_poll_interval_s: float = _DEFAULT_HEALTH_INTERVAL_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        join_success_threshold: int = _DEFAULT_JOIN_SUCCESS_THRESHOLD,
        proxy_timeout_s: float = _DEFAULT_PROXY_TIMEOUT_S,
        proxy_failure_threshold: int = _DEFAULT_PROXY_FAILURE_THRESHOLD,
        proxy_pool_limit_total: int = _DEFAULT_PROXY_POOL_LIMIT_TOTAL,
        proxy_pool_limit_per_host: int = _DEFAULT_PROXY_POOL_LIMIT_PER_HOST,
        uvicorn_backlog: int = _DEFAULT_UVICORN_BACKLOG,
        uvicorn_keep_alive_s: int = _DEFAULT_UVICORN_KEEP_ALIVE_S,
        reset_collective_timeout_s: float = _DEFAULT_RESET_COLLECTIVE_TIMEOUT_S,
        joinable_min_age_s: float = _DEFAULT_JOINABLE_MIN_AGE_S,
    ):
        self.port = port
        self.health_poll_interval_s = health_poll_interval_s
        self.health_timeout_s = health_timeout_s
        self.failure_threshold = failure_threshold
        self.join_success_threshold = join_success_threshold
        # Unproven joining shards must age past this before counting as joinable (cold route warm-up).
        self._joinable_min_age_s = float(joinable_min_age_s)
        self.proxy_failure_threshold = proxy_failure_threshold
        self.proxy_pool_limit_total = proxy_pool_limit_total
        self.proxy_pool_limit_per_host = proxy_pool_limit_per_host
        self.uvicorn_backlog = uvicorn_backlog
        self.uvicorn_keep_alive_s = uvicorn_keep_alive_s
        self.proxy_timeout_s = proxy_timeout_s
        self._reset_collective_timeout_s = reset_collective_timeout_s

        # Tracks when the joinable shard count last changed; used by the trainer's debounce logic.
        self._joinable_changed_at: float = time.monotonic()
        self._last_joinable_count: int = 0

        # Bumped when an in-comm shard is removed; trainer re-inits proactively instead of
        # reusing a wedged group (handles the "backfill restored count" case).
        self._comm_reset_epoch: int = 0

        self._shards: dict[str, ShardEntry] = {}
        self._rr_index: int = 0
        self._table_lock = asyncio.Lock()
        # Serializes add_shard / remove_shard. NCCL re-init is global, so
        # only one lifecycle op may be in flight at a time.
        self._lifecycle_lock = asyncio.Lock()

        # Flips false while shard lifecycle is mid-flight; /refit_ready reads
        # it. Set inside add_shard / remove_shard, cleared once the gen-side
        # workers have completed reset_collective + init_collective.
        self._nccl_reinit_in_progress: bool = False
        # Count of refit broadcasts handled. FaultInjector waits for >=1 before triggering.
        self._refit_attempts: int = 0
        # Set at bring-up via register_shards; updated on add/remove.
        self._per_shard_world_size: int = 1
        # Backing VllmGeneration; drives control-plane and lifecycle. None in unit-test / cordon-only mode.
        self._generation: Any = generation

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._health_task: Optional[asyncio.Task[None]] = None
        # Set during lifespan teardown so background tasks short-circuit instead of
        # scheduling new executor work after the event loop starts shutting down.
        self._shutting_down: bool = False
        self._app = self._build_app()

        # uvicorn runs in a daemon thread; the loop is owned by uvicorn.
        self._server_thread: Optional[threading.Thread] = None
        # Hook for tests / step 2: called after each cordon transition.
        self.on_cordon: Optional[Callable[[str, str], Awaitable[None]]] = None

        # Fault-tolerance metrics scraped by RemoteGeneration via /step_metrics_snapshot.
        # Reset when the gen daemon restarts.
        self._total_shards_at_bootstrap: int = 0
        self._cumulative_shards_removed: int = 0
        self._cumulative_shards_added: int = 0
        self._last_fault_event: Optional[dict[str, Any]] = None

    # =====================================================================
    # Shard table mutation
    # =====================================================================

    def register_shards(
        self,
        shards: list[tuple[str, str]],
        per_shard_world_size: int = 1,
        actor_handles_by_shard: Optional[dict[str, list[Any]]] = None,
        node_id_by_shard: Optional[dict[str, str]] = None,
        worker_indices_by_shard: Optional[dict[str, list[int]]] = None,
        generation: Any = None,
    ) -> None:
        """Seed the shard table at bring-up (synchronous, call before ``start()``).

        Initial status is ``ready`` since VllmGeneration bring-up already waits for
        vLLM health before returning. ``per_shard_world_size`` (TP*PP) is used to
        derive ``current_gen_world_size``. ``generation`` may also be passed at
        construction time.
        """
        self._per_shard_world_size = per_shard_world_size
        actor_handles_by_shard = actor_handles_by_shard or {}
        node_id_by_shard = node_id_by_shard or {}
        worker_indices_by_shard = worker_indices_by_shard or {}
        for shard_id, url in shards:
            self._shards[shard_id] = ShardEntry(
                shard_id=shard_id,
                url=url,
                status="ready",
                last_health_ok_at=time.monotonic(),
                actor_handles=list(actor_handles_by_shard.get(shard_id, [])),
                node_id=node_id_by_shard.get(shard_id, ""),
                worker_indices=list(worker_indices_by_shard.get(shard_id, [])),
            )
        # Capture bootstrap fleet size for wandb "lost X of Y shards" metric.
        self._total_shards_at_bootstrap = max(
            self._total_shards_at_bootstrap, len(self._shards)
        )
        if generation is not None:
            self._generation = generation

    async def cordon(self, shard_id: str, reason: str) -> None:
        async with self._table_lock:
            entry = self._shards.get(shard_id)
            if entry is None or entry.status == "cordoned":
                return
            entry.status = "cordoned"
            entry.consecutive_successes = 0
            self._last_fault_event = {
                "kind": "cordon",
                "shard_id": shard_id,
                "reason": reason,
                "monotonic_ts": time.monotonic(),
            }
        if self.on_cordon is not None:
            try:
                await self.on_cordon(shard_id, reason)
            except Exception as e:  # noqa: BLE001 - hook isolation
                print(f"[router] on_cordon hook raised for {shard_id}: {e}", flush=True)

    async def uncordon(self, shard_id: str) -> None:
        async with self._table_lock:
            entry = self._shards.get(shard_id)
            if entry is None or entry.status == "ready":
                return
            # Flip to joining (not ready): a cordoned shard may have missed the last
            # broadcast. Data plane is gated until the next refit promotes it.
            entry.status = "joining"
            entry.consecutive_failures = 0
            entry.consecutive_successes = 0
            # Reset refit counter — NCCL state may be wedged; re-apply warmup gate.
            entry.successful_refits_since_join = 0
            entry.last_health_ok_at = time.monotonic()

    # =====================================================================
    # Shard lifecycle (add/remove, NCCL re-init coordination)
    # =====================================================================

    async def remove_shard(
        self,
        shard_id: str,
        reason: str = "manual",
        drain_timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Remove a shard from the rotation, kill its actors, free its PG.

        Returns a dict summarising the outcome. The train driver's
        ``ensure_collective_synced`` re-rendezvouses at the new world size
        on the next refit via ``/init_collective``.
        """
        async with self._lifecycle_lock:
            async with self._table_lock:
                entry = self._shards.get(shard_id)
                if entry is None:
                    return {"shard_id": shard_id, "removed": False, "reason": "not found"}
                entry.status = "draining"
                self._nccl_reinit_in_progress = True
                actor_handles = list(entry.actor_handles)
                node_id = entry.node_id
                worker_indices = list(entry.worker_indices)

            # Drain inflight outside the lock so the proxy hot path keeps
            # decrementing inflight for in-progress requests.
            deadline = time.monotonic() + drain_timeout_s
            while time.monotonic() < deadline:
                async with self._table_lock:
                    if entry.inflight == 0:
                        break
                await asyncio.sleep(0.05)

            async with self._table_lock:
                entry.status = "cordoned"

            # Kill the actors. Blocking Ray op; run in executor so the
            # FastAPI loop keeps serving health checks etc.
            loop = asyncio.get_running_loop()

            def _kill_actors() -> None:
                import ray

                for actor in actor_handles:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[router] ray.kill on {shard_id} actor raised {e}", flush=True)

            await loop.run_in_executor(None, _kill_actors)

            # Drop the shard from the table. Health poller no longer probes
            # it; routing skips it. The autoscaler v2 + minReplicas:0
            # combination is what reclaims the pod from here.
            async with self._table_lock:
                removed_entry = self._shards.pop(shard_id, None)
                self._cumulative_shards_removed += 1
                # Bump epoch if the removed shard was in the live comm; forces proactive re-init.
                if removed_entry is not None and removed_entry.in_comm:
                    self._comm_reset_epoch += 1
                self._last_fault_event = {
                    "kind": "remove",
                    "shard_id": shard_id,
                    "reason": reason,
                    "node_id": node_id,
                    "monotonic_ts": time.monotonic(),
                }

            # Tell the underlying VllmGeneration's worker_group to skip these
            # actor indices on subsequent dispatches (init_collective /
            # reset_collective / generate). Without this, the next call from
            # either side would dispatch onto a ray.kill'd actor and raise
            # RayActorError.
            if self._generation is not None and worker_indices:
                wg = getattr(self._generation, "worker_group", None)
                if wg is not None and hasattr(wg, "mark_workers_dead"):
                    wg.mark_workers_dead(worker_indices)
                # VllmGeneration tracks dp_size separately for init_collective
                # rank computation; pull it down to the surviving count.
                if hasattr(self._generation, "dp_size") and wg is not None:
                    self._generation.dp_size = wg.dp_size

            # reset_collective is NOT dispatched here — redundant with per-refit self-teardown
            # and caused cascade evictions under burst kills. The next refit's
            # ensure_collective_synced handles recovery at the new world size.
            self._nccl_reinit_in_progress = False
            return {
                "shard_id": shard_id,
                "removed": True,
                "reason": reason,
                "world_size": self.current_gen_world_size(),
            }

    async def add_shard(self, reason: str = "manual") -> dict[str, Any]:
        """Allocate a new DP shard and register it as ``joining``.

        Spawns a vLLM worker and registers it as ``joining``; the data plane
        skips joining shards until the next refit promotes them to ``ready``.
        ``_nccl_reinit_in_progress`` gates ``refit_ready`` until refit succeeds.
        """
        async with self._lifecycle_lock:
            if self._generation is None:
                return {
                    "shard_id": None,
                    "added": False,
                    "reason": "no generation wired",
                }

            shard_id = self._next_shard_id()
            loop = asyncio.get_running_loop()

            # Gate goes UP (via pre_append_hook) just before the worker is appended
            # so train sees refit_ready=False during actor init. Cleared once done.

            def _gate_up() -> None:
                self._nccl_reinit_in_progress = True

            # Heavy work: vLLM worker init (~1-2min for 4B). Run on a thread so
            # the asyncio loop stays responsive (health poll, /metrics, /shards).
            def _do_add() -> tuple[list[Any], Any, list[int], Optional[str]]:
                return self._generation.add_dp_worker(pre_append_hook=_gate_up)

            try:
                actor_handles, _pg, worker_indices, base_url = (
                    await loop.run_in_executor(None, _do_add)
                )
            except Exception as e:  # noqa: BLE001 - bubble up a structured error
                # If the gate was raised by the hook, drop it so refit
                # isn't permanently wedged by a failed add_shard.
                self._nccl_reinit_in_progress = False
                print(
                    f"[router] add_shard failed during add_dp_worker: {e}",
                    flush=True,
                )
                return {
                    "shard_id": shard_id,
                    "added": False,
                    "reason": f"add_dp_worker failed: {e}",
                }

            if base_url is None:
                # Sync engines have no OpenAI server — the proxy can't route
                # to the new shard, but it still participates in NCCL. We
                # still register it so init_collective sees the new worker.
                proxy_url = ""
            else:
                proxy_url = base_url

            async with self._table_lock:
                # After compaction, re-map surviving shard worker_indices to avoid kill-wrong-actor bugs.
                wg = self._generation.worker_group
                leaders = list(wg.dp_leader_worker_indices)
                per_shard_ws = max(self._per_shard_world_size, 1)
                survivor_leaders = leaders[:-1]  # leaders[-1] is the new shard
                for entry, leader_new_idx in zip(
                    self._shards.values(), survivor_leaders
                ):
                    entry.worker_indices = list(
                        range(leader_new_idx, leader_new_idx + per_shard_ws)
                    )
                # Retrieve the Ray node ID for this shard so that a future
                # replacement can be pinned to the same SLURM node.
                try:
                    import ray
                    new_node_id: str = ray.get(actor_handles[0].get_node_id.remote()) if actor_handles else ""
                except Exception:  # noqa: BLE001
                    new_node_id = ""
                self._shards[shard_id] = ShardEntry(
                    shard_id=shard_id,
                    url=proxy_url,
                    status="joining",
                    last_health_ok_at=time.monotonic(),
                    actor_handles=list(actor_handles),
                    node_id=new_node_id,
                    worker_indices=list(worker_indices),
                )
                self._cumulative_shards_added += 1
                print(
                    f"[RECOVERY] shard={shard_id} status=joining "
                    f"unix_ts={time.time():.6f} reason={reason}",
                    flush=True,
                )
                self._last_fault_event = {
                    "kind": "add",
                    "shard_id": shard_id,
                    "reason": reason,
                    "monotonic_ts": time.monotonic(),
                }

            # reset_collective not dispatched here — same reasoning as remove_shard.
            # The next refit's init_collective destroys the old comm naturally.
            self._nccl_reinit_in_progress = False
            return {
                "shard_id": shard_id,
                "added": True,
                "reason": reason,
                "world_size": self.current_gen_world_size(),
                "status": "joining",
            }

    def _next_shard_id(self) -> str:
        """Return the next unused dp-N id, where N is monotonically increasing.

        Uses (max existing N) + 1 instead of (count) so that recycled ids
        don't collide with shards that may still appear in metrics
        (cumulative counters track lifetime adds/removes).
        """
        import re

        used: set[int] = set()
        for sid in self._shards.keys():
            m = re.match(r"^dp-(\d+)$", sid)
            if m:
                used.add(int(m.group(1)))
        n = (max(used) + 1) if used else 0
        while f"dp-{n}" in self._shards:
            n += 1
        return f"dp-{n}"

    def _eligible_promote_sids(self) -> set[str]:
        """Joining shards safe to promote to ``ready`` after a broadcast.

        Only joining shards that are IN the current comm (``in_comm``)
        received this broadcast and have fresh weights. A joining shard that
        is NOT in_comm was added/registered while the trainer was debouncing
        the grow — the gen-side broadcast skipped it (no ``model_update_group``)
        so its weights are still stale ``load_format=dummy``. Promoting it
        would route the data plane to garbage; it promotes on the refit AFTER
        a grow re-init pulls it into the comm. (Also subsumes the old
        "added after dispatch" guard: such a shard is never in_comm.)"""
        return {
            sid
            for sid, e in self._shards.items()
            if e.status == "joining" and e.in_comm
        }

    def promote_all_joining(
        self, eligible_sids: Optional[set[str]] = None
    ) -> list[str]:
        """Promote every ``joining`` shard to ``ready``.

        Called from the ``/update_weights_from_collective`` handler after
        a successful refit — refit has pushed weights to the new shard, so
        the data plane can now route to it without serving stale completions.

        ``eligible_sids``: optional set of shard ids that were ``joining``
        at the moment the refit dispatch began. A shard added via
        ``/admin/add_shard`` AFTER dispatch did NOT participate in the
        broadcast — its weights are still the random ``load_format=dummy``
        bring-up state — so promoting it would let the data plane serve
        garbage (rewards observed to collapse to 0 in this case). When
        ``eligible_sids`` is supplied, we only promote shards in that set
        and leave any later-arrived joining shards for the NEXT refit.
        Pass ``None`` to retain the legacy "promote everyone" behavior.

        The ``_nccl_reinit_in_progress`` gate was already cleared by
        ``add_shard``; the only state change here is the per-shard status.
        """
        promoted: list[str] = []
        for sid, entry in self._shards.items():
            if entry.status != "joining":
                continue
            if eligible_sids is not None and sid not in eligible_sids:
                continue
            entry.status = "ready"
            entry.last_health_ok_at = time.monotonic()
            entry.consecutive_failures = 0
            entry.consecutive_successes = 0
            promoted.append(sid)
            print(
                f"[RECOVERY] shard={sid} status=ready "
                f"unix_ts={time.time():.6f}",
                flush=True,
            )
        return promoted

    # =====================================================================
    # Routing
    # =====================================================================

    async def _pick_shard(
        self, sticky_key: Optional[str], skip: set[str]
    ) -> Optional[ShardEntry]:
        async with self._table_lock:
            ready = [s for s in self._shards.values() if s.status == "ready" and s.shard_id not in skip]
            if not ready:
                return None
            if sticky_key:
                idx = int(hashlib.blake2b(sticky_key.encode(), digest_size=8).hexdigest(), 16) % len(ready)
                chosen = ready[idx]
            else:
                self._rr_index = (self._rr_index + 1) % len(ready)
                chosen = ready[self._rr_index]
            chosen.inflight += 1
            return chosen

    async def _release_shard(self, entry: ShardEntry) -> None:
        async with self._table_lock:
            entry.inflight = max(0, entry.inflight - 1)

    async def _note_proxy_failure(self, shard_id: str, reason: str) -> None:
        """Record a /v1 proxy failure on this shard; cordon iff threshold breached.

        Distinct from health-poll's `consecutive_failures` because the
        proxy and the health-poller see different signals (an overloaded
        worker can return 5xx on /v1/completions while still answering
        /openapi.json instantly). Burst-induced 5xx no longer evicts a
        live shard until it's failed `proxy_failure_threshold` proxy
        round-trips in a row.
        """
        async with self._table_lock:
            entry = self._shards.get(shard_id)
            if entry is None or entry.status != "ready":
                return
            entry.proxy_consecutive_failures += 1
            if entry.proxy_consecutive_failures < self.proxy_failure_threshold:
                return
            # Don't cordon during NCCL re-init; event loop stall causes transient 5xx.
            if self._nccl_reinit_in_progress:
                return
            entry.status = "cordoned"
            entry.proxy_consecutive_failures = 0
        # Fire cordon hook outside the lock to avoid re-entrance.
        asyncio.create_task(self._fire_cordon_hook(shard_id, reason))

    async def _reset_proxy_failures(self, shard_id: str) -> None:
        async with self._table_lock:
            entry = self._shards.get(shard_id)
            if entry is None:
                return
            entry.proxy_consecutive_failures = 0

    async def _proxy(self, request: Request, suffix: str, *, strip_v1: bool = False) -> Response:
        if self._http_session is None or self._http_session.closed:
            raise HTTPException(status_code=503, detail="router not started")

        sticky_key = request.headers.get("X-NRL-Session-Id")
        body = await request.body()
        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() in {"content-type", "accept"}
        }

        attempted: set[str] = set()
        last_status = 502
        last_text = "no ready shard"
        timeout = aiohttp.ClientTimeout(total=self.proxy_timeout_s)

        while True:
            entry = await self._pick_shard(sticky_key, attempted)
            if entry is None:
                return JSONResponse(
                    status_code=last_status,
                    content={"error": last_text},
                )
            attempted.add(entry.shard_id)
            base = entry.url.rstrip("/")
            if strip_v1 and base.endswith("/v1"):
                base = base[:-3]
            target = f"{base}{suffix}"
            try:
                async with self._http_session.post(
                    target,
                    data=body,
                    headers=forward_headers,
                    timeout=timeout,
                ) as resp:
                    text = await resp.read()
                    if resp.status >= 500:
                        last_status, last_text = resp.status, text.decode("utf-8", "replace")
                        await self._note_proxy_failure(
                            entry.shard_id,
                            f"5xx from /v1: status={resp.status}",
                        )
                        await self._release_shard(entry)
                        continue
                    # 2xx/4xx counts as a successful round-trip — reset the
                    # per-shard proxy failure counter so a future single 5xx
                    # doesn't piggyback on an old burst.
                    await self._reset_proxy_failures(entry.shard_id)
                    await self._release_shard(entry)
                    return Response(
                        content=text,
                        status_code=resp.status,
                        media_type=resp.headers.get("Content-Type", "application/json"),
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_status, last_text = 502, f"transport error: {type(e).__name__}: {e}"
                await self._note_proxy_failure(entry.shard_id, last_text)
                await self._release_shard(entry)
                continue

    # =====================================================================
    # Health poller
    # =====================================================================

    async def _health_poll_loop(self) -> None:
        assert self._http_session is not None
        timeout = aiohttp.ClientTimeout(total=self.health_timeout_s)
        while True:
            try:
                async with self._table_lock:
                    # Probe ready/joining AND cordoned shards. Cordoned ones
                    # are eligible for auto-recovery: if a shard was cordoned
                    # by a transient probe failure (e.g. cross-cluster NCCL
                    # rendezvous blocking the worker's event loop) and is now
                    # answering /openapi.json again, promote it back to ready.
                    targets = [
                        (s.shard_id, s._health_url, s.status)
                        for s in self._shards.values()
                        if s.status in ("ready", "joining", "cordoned")
                    ]
                results = await asyncio.gather(
                    *(self._probe(self._http_session, url, timeout) for _, url, _ in targets),
                    return_exceptions=False,
                )
                async with self._table_lock:
                    nccl_paused = self._nccl_reinit_in_progress
                    # Count alive shards for deadlock detection (all cordoned = auto-recovery path).
                    n_alive_or_joining = sum(
                        1 for s in self._shards.values()
                        if s.status in ("ready", "joining")
                    )
                    for (shard_id, _, prev_status), ok in zip(targets, results):
                        entry = self._shards.get(shard_id)
                        if entry is None:
                            continue
                        if ok:
                            entry.consecutive_failures = 0
                            entry.consecutive_successes += 1
                            entry.last_health_ok_at = time.monotonic()
                            # Health poller never auto-promotes to ready. joining→ready
                            # requires a successful refit broadcast. cordoned→joining
                            # only when ALL shards are cordoned (deadlock escape hatch).
                            if (
                                entry.status == "cordoned"
                                and n_alive_or_joining == 0
                                and entry.consecutive_successes
                                >= self.join_success_threshold
                            ):
                                successes = entry.consecutive_successes
                                entry.status = "joining"
                                entry.consecutive_failures = 0
                                entry.consecutive_successes = 0
                                # Reset refit counter; next refit re-certifies weights.
                                entry.successful_refits_since_join = 0
                                n_alive_or_joining = 1
                                print(
                                    f"[router] auto-uncordoned {shard_id} "
                                    f"(0 alive shards, {successes} "
                                    f"successful probes); status=joining; "
                                    f"will re-promote on next refit",
                                    flush=True,
                                )
                        else:
                            entry.consecutive_successes = 0
                            entry.consecutive_failures += 1
                            # Skip cordoning during NCCL re-init (event loop stall causes probe misses).
                            if nccl_paused:
                                continue
                            if (
                                entry.status in ("ready", "joining")
                                and entry.consecutive_failures >= self.failure_threshold
                            ):
                                entry.status = "cordoned"
                                # Fire cordon hook outside the lock to avoid
                                # re-entrance from on_cordon callbacks.
                                asyncio.create_task(
                                    self._fire_cordon_hook(shard_id, "health threshold breached")
                                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep loop alive across transient errors
                print(f"[router] health poll iteration failed: {e}", flush=True)
            # Recompute the joinable set's stability AFTER this tick's status
            # updates so the trainer's debounce sees a current timer.
            try:
                self._refresh_joinable_stability()
            except Exception as e:  # noqa: BLE001
                print(f"[router] joinable refresh failed: {e}", flush=True)
            await asyncio.sleep(self.health_poll_interval_s)

    async def _per_worker_results(
        self,
        futures: list[Any],
        rendezvous_timeout_s: float,
        fast_fail_s: float = 8.0,
    ) -> tuple[list[int], list[Any], list[Optional[str]], list[int]]:
        """Wait on per-worker futures, identify failures.

        Returns ``(failed_indices, results, exception_types, failed_fast)``.
        A worker is "failed" if its future either (a) raised an exception or
        (b) was still pending when ``rendezvous_timeout_s`` elapsed. Pending
        futures are ``ray.cancel`` ed so they don't pin the actor's task queue.

        ``results`` / ``exception_types`` are one-per-future (value/None on
        success; class name or ``"PENDING"`` on failure). Callers use the
        exception types to distinguish "rendezvous-master failure" (every gen
        client raises ``DistStoreError``/``DistNetworkError`` because rank 0
        timed out) from individual worker death.

        ``failed_fast`` is the subset of ``failed_indices`` whose future RAISED
        within ``fast_fail_s`` seconds. This is the robust discriminator for
        "this worker's own engine/NCCL is dead/poisoned" (EngineCore died, or
        the in-band ``comm_abort`` left NCCL state poisoned so the next
        ``comm_init_rank_scalable`` raises immediately, in <1s) vs "healthy
        survivor that merely TIMED OUT (~30s NCCL bootstrap) waiting on the
        rendezvous for a dead/poisoned peer". ``_evict_failed_workers_inline``
        force-evicts ONLY the fast-failers, so a single fault never cascades
        into mass-eviction of the healthy-but-blocked survivors (which would
        collapse the gen world toward zero).
        """
        import ray

        loop = asyncio.get_running_loop()

        # Two-phase wait: anything that resolves in the first ``fast_fail_s``
        # is a fast-failer (poisoned/dead engine raises immediately); the rest
        # are slow (healthy survivors hit the ~30s bootstrap timeout) or stay
        # pending.
        def _wait_fast() -> tuple[list[Any], list[Any]]:
            return ray.wait(futures, num_returns=len(futures), timeout=fast_fail_s)

        ready_fast, pending_after_fast = await loop.run_in_executor(None, _wait_fast)
        if pending_after_fast:
            rem = max(0.0, rendezvous_timeout_s - fast_fail_s)

            def _wait_slow() -> tuple[list[Any], list[Any]]:
                return ray.wait(
                    pending_after_fast,
                    num_returns=len(pending_after_fast),
                    timeout=rem,
                )

            _ready_slow, pending = await loop.run_in_executor(None, _wait_slow)
        else:
            pending = []
        fast_set = set(ready_fast)

        # Cancel any straggler futures up-front. ``ray.cancel`` is best-
        # effort against a wedged actor method but it does drop the future
        # from the actor's queue once the method returns/raises.
        for fut in pending:
            try:
                ray.cancel(fut, force=True)
            except Exception:  # noqa: BLE001
                pass

        failed_indices: list[int] = []
        results: list[Any] = [None] * len(futures)
        exception_types: list[Optional[str]] = [None] * len(futures)
        failed_fast: list[int] = []
        pending_set = set(pending)
        for i, fut in enumerate(futures):
            if fut in pending_set:
                failed_indices.append(i)
                exception_types[i] = "PENDING"
                continue
            try:
                results[i] = await loop.run_in_executor(
                    None, lambda f=fut: ray.get(f)
                )
            except Exception as e:  # noqa: BLE001 - per-worker fault isolation
                failed_indices.append(i)
                exception_types[i] = type(e).__name__
                is_fast = fut in fast_set
                if is_fast:
                    failed_fast.append(i)
                print(
                    f"[router] per-worker future idx={i} raised "
                    f"{type(e).__name__}: {e} "
                    f"({'fast-fail→poisoned/dead' if is_fast else 'slow→waiting/healthy'})",
                    flush=True,
                )

        return failed_indices, results, exception_types, failed_fast

    @staticmethod
    def _is_rendezvous_master_failure(
        failed_idxs: list[int], exception_types: list[Optional[str]]
    ) -> bool:
        """Detect "the rendezvous master timed out" pattern.

        When the StatelessProcessGroup master (RefitWorker, rank 0) fails
        to bind its TCPStore in time, EVERY gen-side client raises
        ``DistNetworkError`` ("client socket has timed out trying to
        connect"). Mass-evicting all gen workers on that signal is wrong:
        the gen workers were healthy, the master was the problem.
        Heuristic: if every failed worker raised
        ``DistStoreError``/``DistNetworkError`` AND that's all the workers
        we dispatched to, treat it as a transient rendezvous-master
        failure and let the train-side retry loop handle it.

        See ``docs/scratch/tcpstore-fix-report.md`` for the empirical
        repro that motivated this.
        """
        if not failed_idxs:
            return False
        rendezvous_excs = {"DistStoreError", "DistNetworkError"}
        for idx in failed_idxs:
            if exception_types[idx] not in rendezvous_excs:
                return False
        # All failures are rendezvous-related. If even one worker
        # succeeded, the cluster was partially functional — cascade was
        # local, evict normally. Only skip eviction if EVERY dispatched
        # worker raised a rendezvous error (master-side single-point).
        return len(failed_idxs) == len(exception_types)

    async def _filter_targets_by_liveness(
        self, candidate_sids: list[str], probe_timeout_s: float = 5.0
    ) -> tuple[list[str], list[str]]:
        """Split candidate shards into (confirmed_dead, alive) by ``is_alive`` ping.

        At TP>1, when a burst kill removes multiple shards simultaneously,
        the surviving leaders' ``init_collective`` futures fail for two
        VERY different reasons that look the same to the dispatch layer:

        1. **Actually dead**: the leader's Ray actor was killed by the FI.
           Future raises ``RayActorError``. ``is_alive`` raises
           ``RayActorError`` too — confirmed dead.

        2. **Alive but blocked on a dead peer**: the leader's TCPStore
           rendezvous timed out waiting for the killed shards' TP partners
           to connect. Future raises ``DistNetworkError`` /
           ``DistStoreError`` (or a ``RayActorError`` if a peer's
           ``cudaErrorLaunchFailure`` propagates back via vLLM's
           ``collective_rpc``). The actor process is still healthy and
           ``is_alive`` returns True — we MUST NOT evict.

        Without this filter, the eviction path treats every failed future
        the same and mass-evicts the survivors, cascading the world to
        zero. The TP=1 path got away with the cascade-without-this-check
        only because a single FI kill leaves ≥1 survivor whose future
        still succeeds (rendezvous reaches the smaller-world quorum
        organically); at TP>1 the per-shard world is multi-rank and a
        single kill takes out ALL ranks in that shard, so quorum can't
        be reached and EVERY survivor's future fails.

        Returns ``(dead_shard_ids, alive_shard_ids)``. Caller evicts only
        the dead set; the alive set retries via the train-side
        ``ensure_collective_synced`` loop, which re-reads the world size
        (still N) and rendezvouses again now that the dead shards have
        been evicted.
        """
        import ray

        loop = asyncio.get_running_loop()

        # Snapshot the shard table once. Probing is unordered (we use
        # asyncio.gather), so a small race with concurrent
        # add_shard/remove_shard is fine — the lifecycle lock outside
        # this call serializes mass-eviction with shard table churn.
        async with self._table_lock:
            handles = {
                sid: list(self._shards[sid].actor_handles)
                for sid in candidate_sids
                if sid in self._shards
            }

        async def _probe(sid: str) -> tuple[str, bool]:
            actor_handles = handles.get(sid, [])
            if not actor_handles:
                # No handles registered (unit-test or manual cordon path);
                # assume alive — we'd rather skip eviction than evict
                # blind. The health poller will catch a truly-dead shard
                # via the HTTP /openapi.json probe.
                return sid, True
            # Ping the LEADER actor only. If the leader is dead, the
            # whole TP group is gone (kai-scheduler / FI tears the pod
            # down atomically); if the leader is alive, the TP partners
            # are alive too because they share the same Pod / placement-
            # group bundle.
            leader = actor_handles[0]
            try:
                fut = leader.is_alive.remote()
            except Exception as e:  # noqa: BLE001 - any RPC dispatch failure → assume dead
                print(
                    f"[router] _filter_targets_by_liveness: {sid} "
                    f"is_alive dispatch raised {type(e).__name__}: {e}",
                    flush=True,
                )
                return sid, False
            try:
                # ray.get with timeout in an executor — asyncio loop
                # stays responsive for parallel probes.
                ok = await loop.run_in_executor(
                    None, lambda f=fut: ray.get(f, timeout=probe_timeout_s)
                )
                return sid, bool(ok)
            except Exception as e:  # noqa: BLE001 - RayActorError, GetTimeoutError, etc.
                print(
                    f"[router] _filter_targets_by_liveness: {sid} "
                    f"is_alive raised {type(e).__name__}: {e}; "
                    f"classifying as dead",
                    flush=True,
                )
                return sid, False

        results = await asyncio.gather(
            *[_probe(sid) for sid in candidate_sids if sid in handles],
            return_exceptions=False,
        )

        dead: list[str] = []
        alive: list[str] = []
        seen: set[str] = set()
        for sid, is_alive in results:
            seen.add(sid)
            if is_alive:
                alive.append(sid)
            else:
                dead.append(sid)
        # Candidates that weren't in the table at snapshot time (already
        # removed) are silently dropped — nothing to evict.
        return dead, alive

    async def _evict_failed_workers_inline(
        self,
        failed_idxs: list[int],
        reason: str,
        exception_types: Optional[list[Optional[str]]] = None,
        force_evict_idxs: Optional[list[int]] = None,
        shard_ids_in_order: Optional[list[str]] = None,
    ) -> tuple[int, list[str]]:
        """Cordon + remove the gen workers whose futures failed.

        Maps ``failed_idxs`` (per-DP-leader future indices) to shard ids and
        evicts confirmed-dead shards. Liveness probe (``is_alive``) prevents
        mass-eviction of survivors blocked on a dead peer's rendezvous (TP>1
        cascade fix). Force-evicts shards with ``RayActorError`` or fast-fail
        (<8s raise) without probing. Returns ``(num_evicted, evicted_shard_ids)``.
        """
        if not failed_idxs:
            return 0, []

        # ``shard_ids_in_order`` maps failed future index → shard id (caller must pass
        # the exact dispatch order). Falls back to full shard insertion order.
        if shard_ids_in_order is None:
            async with self._table_lock:
                shard_ids_in_order = list(self._shards.keys())
        targets: list[str] = []
        # Force-evict (bypass liveness probe): RayActorError/ActorDiedError (Ray confirmed
        # dead) or fast-fail (<8s raise, NCCL poisoned). All other failures route through
        # is_alive probe to retain healthy survivors blocked on a dead peer.
        force_dead: set[str] = set()
        seen: set[str] = set()
        _ACTOR_DEAD_TYPES = {"RayActorError", "ActorDiedError"}
        force_idx_set = set(force_evict_idxs or [])
        for idx in failed_idxs:
            if 0 <= idx < len(shard_ids_in_order):
                sid = shard_ids_in_order[idx]
                # Dedupe: if the same shard appeared twice (shouldn't with
                # per-leader dispatch but defensive), only evict once.
                if sid not in seen:
                    seen.add(sid)
                    targets.append(sid)
                et = (
                    exception_types[idx]
                    if exception_types is not None
                    and 0 <= idx < len(exception_types)
                    else None
                )
                if idx in force_idx_set or (et in _ACTOR_DEAD_TYPES):
                    force_dead.add(sid)
            else:
                print(
                    f"[router] _evict_failed_workers_inline: future idx={idx} "
                    f"out of bounds (have {len(shard_ids_in_order)} shards)",
                    flush=True,
                )

        # TP>1 cascade fix: probe each candidate's leader actor before
        # evicting. Survivors blocked on a dead peer's rendezvous look
        # identical to actually-dead shards at the dispatch layer; a
        # fresh ``is_alive`` ping disambiguates.
        # Force-dead targets (RayActorError / ActorDiedError) bypass the
        # probe — Ray confirmed those actors are gone.
        if targets:
            probe_targets = [sid for sid in targets if sid not in force_dead]
            dead_from_probe: list[str] = []
            alive_targets: list[str] = []
            if probe_targets:
                dead_from_probe, alive_targets = (
                    await self._filter_targets_by_liveness(probe_targets)
                )
            if alive_targets:
                print(
                    f"[router] _evict_failed_workers_inline: skipping "
                    f"{len(alive_targets)} alive survivors "
                    f"(reason={reason}, alive={alive_targets}, "
                    f"dead={dead_from_probe}, "
                    f"force_dead={sorted(force_dead)}); "
                    f"train will retry rendezvous",
                    flush=True,
                )
            # Evict force_dead first (preserves caller-supplied order
            # by re-iterating ``targets``), then probe-confirmed dead.
            dead_set = force_dead.union(dead_from_probe)
            targets = [sid for sid in targets if sid in dead_set]

        evicted: list[str] = []
        for sid in targets:
            try:
                # drain_timeout_s=0: shard is already wedged in NCCL/CUDA,
                # no real inflight to drain. ``remove_shard`` already
                # acquires ``_lifecycle_lock`` and runs reset_collective on
                # survivors after killing the dead actor.
                res = await self.remove_shard(
                    sid, reason=reason, drain_timeout_s=0.0
                )
                if res.get("removed"):
                    evicted.append(sid)
                    print(
                        f"[router] evicted poisoned shard {sid}: {reason}",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[router] failed to evict {sid} ({reason}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        return len(evicted), evicted

    async def _fire_cordon_hook(self, shard_id: str, reason: str) -> None:
        # Skip auto-remove during teardown to avoid executor errors; next daemon starts fresh.
        if self._shutting_down:
            print(
                f"[router] skipping cordon hook for {shard_id} ({reason}); "
                f"router is shutting down",
                flush=True,
            )
            return

        # Custom override path: fault injector or tests want full control
        # over what happens after a cordon (e.g. http-error mode keeps the
        # cordon reversible, no removal).
        if self.on_cordon is not None:
            try:
                await self.on_cordon(shard_id, reason)
            except Exception as e:  # noqa: BLE001
                print(f"[router] on_cordon hook raised for {shard_id}: {e}", flush=True)
            return

        # Default: drive remove_shard so the actor + PG come down and the pod is reclaimed.
        if self._generation is None:
            # No lifecycle ownership — this is a unit-test or cordon-only
            # mode where the router doesn't know how to kill anything.
            # Leave the cordon as-is and surface a log so an operator can
            # see why traffic stopped flowing.
            print(
                f"[router] cordoned {shard_id} ({reason}); "
                f"no generation wired, can't auto-remove",
                flush=True,
            )
            return
        # Last-alive guard: don't auto-remove if this was the last alive shard (would drop world to 0).
        if self.shard_count_alive_for_collective() == 0:
            print(
                f"[router] skipping auto-remove of {shard_id} "
                f"({reason}): would drop fleet to 0 alive shards. "
                f"Cordoned shard left in table; manual /admin/uncordon "
                f"or remove via /admin/remove_shard if truly dead.",
                flush=True,
            )
            return
        try:
            await self.remove_shard(
                shard_id, reason=f"auto-remove on cordon: {reason}"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[router] auto-remove on cordon for {shard_id} raised {e}",
                flush=True,
            )

    @staticmethod
    async def _probe(
        session: aiohttp.ClientSession, url: str, timeout: aiohttp.ClientTimeout
    ) -> bool:
        try:
            async with session.get(url, timeout=timeout) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    # =====================================================================
    # FastAPI app — data plane + status + admin + control plane.
    # =====================================================================

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
            # Explicit limits to avoid bottleneck under burst load; limit_per_host prevents one slow shard starving others.
            connector = aiohttp.TCPConnector(
                limit=self.proxy_pool_limit_total,
                limit_per_host=self.proxy_pool_limit_per_host,
                keepalive_timeout=60,
            )
            self._http_session = aiohttp.ClientSession(connector=connector)
            self._health_task = asyncio.create_task(self._health_poll_loop())
            try:
                yield
            finally:
                # Set flag first so in-flight cordon hooks short-circuit before hitting
                # run_in_executor (which would raise "cannot schedule new futures") and leave
                # the shard table in a half-removed state. The next
                # daemon process starts from a clean register_shards call
                # so we don't need to finish in-flight removes here.
                self._shutting_down = True
                if self._health_task is not None:
                    self._health_task.cancel()
                    try:
                        await self._health_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if self._http_session is not None:
                    await self._http_session.close()

        app = FastAPI(title="Generation Router", lifespan=_lifespan)

        # ---- Status / health ----------------------------------------------

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok"}

        @app.get("/shards")
        async def shards() -> list[dict[str, Any]]:
            async with self._table_lock:
                return [
                    {
                        "shard_id": s.shard_id,
                        "url": s.url,
                        "status": s.status,
                        "inflight": s.inflight,
                        "last_health_ok_at": s.last_health_ok_at,
                        "consecutive_failures": s.consecutive_failures,
                    }
                    for s in self._shards.values()
                ]

        @app.get("/refit_ready")
        async def refit_ready() -> dict[str, Any]:
            ready, reason = self.refit_ready_state()
            return {"ready": ready, "reason": reason}

        @app.get("/refit_count")
        async def refit_count() -> dict[str, Any]:
            """Number of weight-refit broadcasts handled so far. The
            FaultInjector polls this and waits for >=1 before starting its
            trigger countdown, anchoring fault timing to 'training has started
            refitting' instead of a wall-clock guess (setup time varies)."""
            return {"refit_attempts": self._refit_attempts}

        @app.get("/current_gen_world_size")
        async def current_gen_world_size() -> dict[str, Any]:
            """Sum of per-shard world sizes for shards in the NCCL group.

            ``world_size`` is the live dispatch set (``ready`` + ``joining``);
            the training driver rendezvouses at this. ``joinable_world_size``
            is the subset whose engines are health-responsive, and
            ``joinable_stable_for_s`` is how long that subset has held — the
            trainer uses both to debounce comm growth (only grow once the
            joinable set == the dispatch set and has settled).
            """
            return {
                "world_size": self.current_gen_world_size(),
                "joinable_world_size": self.joinable_world_size(),
                "joinable_stable_for_s": self.joinable_stable_for_s(),
                "comm_epoch": self._comm_reset_epoch,
                # The trainer's quiesce-wait reads this to avoid retrying a
                # rendezvous while a lifecycle op (add/remove/reset) is mid-flight.
                "nccl_reinit_in_progress": self._nccl_reinit_in_progress,
            }

        @app.get("/dp_openai_server_base_urls")
        async def get_dp_urls() -> list[str]:
            """Per-shard OpenAI base URLs for clients that want direct routing.

            Round-robin clients (legacy direct-shard mode) read this once at
            bring-up. The unified router prefers clients to send to
            ``/v1/{completions,chat/completions}`` so cordon + replay is
            invisible, but this endpoint is preserved for back-compat /
            diagnostics. Returns the URLs of every shard the router knows
            about — including non-ready ones — because the legacy clients
            don't track status.
            """
            return [s.url for s in self._shards.values() if s.url]

        @app.get("/config")
        async def get_config() -> dict[str, Any]:
            """Return the gen-side ``policy.cfg`` so train can fetch it.

            ``RemoteGeneration._fetch_remote_config`` polls this during
            bring-up to discover sampling defaults, the model name, etc.
            503 when no generation is wired (fake / cordon-only mode).
            """
            if self._generation is None:
                raise HTTPException(status_code=503, detail="no generation wired")
            return dict(self._generation.cfg)

        @app.get("/metrics")
        async def metrics() -> dict[str, Any]:
            """Router-side fault-tolerance + traffic snapshot for wandb."""
            return self.metrics_snapshot()

        # ---- Admin --------------------------------------------------------

        @app.post("/admin/cordon")
        async def admin_cordon(payload: dict[str, str]) -> dict[str, str]:
            shard_id = payload.get("shard_id")
            reason = payload.get("reason", "manual")
            if not shard_id:
                raise HTTPException(status_code=400, detail="shard_id required")
            await self.cordon(shard_id, reason)
            return {"shard_id": shard_id, "status": "cordoned"}

        @app.post("/admin/uncordon")
        async def admin_uncordon(payload: dict[str, str]) -> dict[str, str]:
            shard_id = payload.get("shard_id")
            if not shard_id:
                raise HTTPException(status_code=400, detail="shard_id required")
            await self.uncordon(shard_id)
            return {"shard_id": shard_id, "status": "joining"}

        @app.post("/admin/remove_shard")
        async def admin_remove_shard(payload: dict[str, Any]) -> dict[str, Any]:
            shard_id = payload.get("shard_id")
            reason = payload.get("reason", "admin /admin/remove_shard")
            if not shard_id:
                raise HTTPException(status_code=400, detail="shard_id required")
            # ``drain_timeout_s`` is opt-in via payload; default keeps the
            # graceful 30s drain. Stress-test/fault-injection callers
            # pass a smaller value (e.g. 5s) so in-flight rollouts on
            # the dying shard fail fast and the proxy replays them on
            # survivors instead of blocking the next training step.
            kwargs: dict[str, Any] = {"reason": reason}
            if "drain_timeout_s" in payload:
                kwargs["drain_timeout_s"] = float(payload["drain_timeout_s"])
            return await self.remove_shard(shard_id, **kwargs)

        @app.post("/admin/add_shard")
        async def admin_add_shard(
            payload: Optional[dict[str, str]] = None,
        ) -> dict[str, Any]:
            """Allocate + spawn a new DP shard.

            Body (optional): ``{"reason": "..."}``. Reason is logged into
            the router's last_fault_event for wandb attribution.
            """
            reason = (payload or {}).get("reason", "admin /admin/add_shard")
            return await self.add_shard(reason=reason)

        # ---- Data plane ---------------------------------------------------

        @app.post("/v1/completions")
        async def v1_completions(request: Request) -> Response:
            return await self._proxy(request, "/completions")

        @app.post("/v1/chat/completions")
        async def v1_chat_completions(request: Request) -> Response:
            return await self._proxy(request, "/chat/completions")

        # /tokenize and /detokenize live at the vLLM server root (no /v1
        # prefix). NemoGym's vllm_model gym sub-server hits /tokenize after
        # every chat completion to recover prompt token ids — see
        # responses_api_models/vllm_model/app.py:402.
        @app.post("/tokenize")
        async def tokenize(request: Request) -> Response:
            return await self._proxy(request, "/tokenize", strip_v1=True)

        @app.post("/detokenize")
        async def detokenize(request: Request) -> Response:
            return await self._proxy(request, "/detokenize", strip_v1=True)

        # ---- Control plane (weight sync + lifecycle) ---------------------

        # All control-plane handlers use run_in_executor to avoid blocking
        # the uvicorn event loop. Blocking ray.get() or synchronous GPU
        # ops in async handlers would deadlock the NCCL warmup broadcast
        # (uvicorn's loop is also where the proxy + health poll run).
        async def _run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn, *args)

        def _require_generation() -> Any:
            if self._generation is None:
                raise HTTPException(status_code=503, detail="no generation wired")
            return self._generation

        @app.post("/init_collective")
        async def init_collective(request: Request) -> Any:
            generation = _require_generation()
            body = await request.json()
            # Cascade-to-zero guard: if no shards are alive in the NCCL
            # group, init_collective on the underlying generation would
            # raise "Data parallel size is zero" → caller catches as 500
            # → train treats as fatal. Return 503 (recoverable) so the
            # train-side ensure_collective_synced retry loop polls
            # /refit_ready (which is also gated on shard_count > 0) and
            # waits for add_shard recovery instead of fast-failing.
            if self.shard_count_alive_for_collective() == 0:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "error": "no shards alive in NCCL group; retry after add_shard",
                        "current_gen_world_size": 0,
                        "failed_indices": [],
                        "evicted_shard_ids": [],
                    },
                )
            # Pause cordon transitions for the duration of the cross-cluster
            # NCCL rendezvous: the gen workers' event loop can stall while
            # waiting for the train side to enter init, missing /openapi.json
            # probes; without this gate the router would cordon every shard
            # and the next request would get a 502.
            prev_flag = self._nccl_reinit_in_progress
            self._nccl_reinit_in_progress = True
            try:
                # Dispatch to only the JOINABLE cohort (cold backfills excluded) so a
                # booting shard can't sabotage the handshake. Per-future tracking
                # avoids a single bad worker sinking the whole rendezvous.
                async with self._table_lock:
                    cohort_sids, cohort_indices = self._joinable_cohort()
                dispatched_sids = set(cohort_sids)
                if not cohort_indices:
                    # No shard is joinable yet (e.g. survivors all cold mid-
                    # recovery). Don't rendezvous an empty cohort — 503 so the
                    # train retries after the quiesce-wait, by which point a
                    # shard has warmed.
                    self._nccl_reinit_in_progress = prev_flag
                    return JSONResponse(
                        status_code=503,
                        content={
                            "success": False,
                            "error": "no joinable shards yet; retry after warm-up",
                            "current_gen_world_size": self.current_gen_world_size(),
                            "failed_indices": [],
                            "evicted_shard_ids": [],
                        },
                    )

                def _dispatch() -> list[Any]:
                    return generation.init_collective(
                        ip=body["ip"],
                        port=body["port"],
                        world_size=body["world_size"],
                        train_world_size=body["train_world_size"],
                        include_worker_indices=cohort_indices,
                    )

                futures = await _run_blocking(_dispatch)

                # Wait long enough for stragglers to surface as
                # DistStoreError (TCPStore timeout in
                # stateless_process_group is 30s) but bound the total so
                # we don't pin train indefinitely. 90s = 3x TCPStore
                # timeout + slack for the actor's task queue to drain.
                failed_idxs, _results, exc_types, failed_fast = await self._per_worker_results(
                    futures, rendezvous_timeout_s=90.0
                )

                if not failed_idxs:
                    # Rendezvous succeeded: these shards are now in the live
                    # weight-sync comm. Promotion of a joining shard is gated
                    # on this (see /update_weights_from_collective).
                    async with self._table_lock:
                        for sid in dispatched_sids:
                            entry = self._shards.get(sid)
                            if entry is not None:
                                entry.in_comm = True
                                # Sticky: this shard's cross-cluster route has
                                # now rendezvoused → warm. Bypasses the warm-up
                                # age gate on any future re-add/rejoin.
                                entry.proven = True
                    return {"success": True}

                # Rendezvous-master failure short-circuit: when EVERY gen
                # worker raises DistStoreError / DistNetworkError, the
                # train-side TCPStore master timed out. Mass-evicting all
                # gen workers in this case is wrong — they were healthy,
                # the master was the single point of failure. Return 503
                # without evicting; train's retry loop will re-rendezvous
                # at the same world after a brief settle.
                if self._is_rendezvous_master_failure(failed_idxs, exc_types):
                    print(
                        f"[router] /init_collective: rendezvous-master "
                        f"failure (all {len(failed_idxs)} gen workers "
                        f"raised {exc_types[failed_idxs[0]]}); skipping "
                        f"mass-eviction, returning 503 for retry",
                        flush=True,
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "success": False,
                            "error": "rendezvous master timed out; retry",
                            "current_gen_world_size": self.current_gen_world_size(),
                            "failed_indices": failed_idxs,
                            "evicted_shard_ids": [],
                            "rendezvous_master_failure": True,
                        },
                    )

                # Per-worker failure path: evict the bad workers, drain
                # NCCL state on survivors via ``remove_shard`` (which
                # internally calls ``reset_collective``), and tell train
                # to re-rendezvous at the smaller world. 503 is a
                # "transient — try again with the new world" signal;
                # train's ``ensure_collective_synced`` retry already
                # re-reads ``/current_gen_world_size`` on each attempt.
                num_evicted, evicted_ids = await self._evict_failed_workers_inline(
                    failed_idxs,
                    reason="init_collective: per-worker failure",
                    exception_types=exc_types,
                    force_evict_idxs=failed_fast,
                    # Futures cover only the frozen cohort, in cohort order —
                    # map failed indices back through the SAME order.
                    shard_ids_in_order=cohort_sids,
                )
                new_ws = self.current_gen_world_size()
                msg = (
                    f"{len(failed_idxs)} of {len(futures)} workers failed "
                    f"rendezvous; evicted {num_evicted} (ids={evicted_ids}); "
                    f"current_gen_world_size={new_ws}"
                )
                print(f"[router] /init_collective: {msg}", flush=True)
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "error": msg,
                        "current_gen_world_size": new_ws,
                        "failed_indices": failed_idxs,
                        "evicted_shard_ids": evicted_ids,
                    },
                )
            except Exception as e:  # noqa: BLE001 - structured error to caller
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )
            finally:
                self._nccl_reinit_in_progress = prev_flag

        @app.post("/reset_collective")
        async def reset_collective() -> Any:
            """Tear down the weight-sync NCCL group so a new run can re-init.

            Idempotent — safe to call when no collective is currently held.
            """
            generation = _require_generation()
            # 0-shards short-circuit: nothing to tear down (every worker
            # was evicted), so don't dispatch to the underlying
            # generation (whose worker_group may be empty and raise on
            # dispatch). Return 200 success with a no-op note so the
            # train side's idempotent reset path doesn't trip on a 500.
            if self.shard_count_alive_for_collective() == 0:
                return {"success": True, "note": "no shards alive; nothing to reset"}
            try:
                import ray

                def _do() -> None:
                    futures = generation.reset_collective()
                    if futures:
                        ray.get(futures)

                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                # 503 (not 500) so train-side retry treats it as
                # recoverable. The underlying NCCL state may have been
                # torn down by the per-refit destroy() in vllm_backend
                # already; a downstream init_collective will rebuild
                # cleanly. Surface the actual error for diagnostics.
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "error": f"reset_collective dispatch raised: {e}",
                    },
                )

        @app.post("/update_weights_from_collective")
        async def update_weights_from_collective() -> Any:
            generation = _require_generation()
            # Cascade-to-zero guard (mirror of /init_collective): no
            # shards in the NCCL group → return 503 so train's retry
            # path waits for add_shard recovery instead of fast-failing
            # on a generic 500.
            if self.shard_count_alive_for_collective() == 0:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "error": "no shards alive in NCCL group; retry after add_shard",
                        "current_gen_world_size": 0,
                        "failed_indices": [],
                        "evicted_shard_ids": [],
                        "promoted_shards": [],
                    },
                )
            try:
                # Mark that training has reached (at least) its first refit
                # broadcast — the FaultInjector waits on this via /refit_count.
                self._refit_attempts += 1
                # Per-worker error tracking: a single worker hitting
                # cudaErrorLaunchFailure during the broadcast would
                # otherwise sink the whole refit (ray.get raises first
                # exception → generic 500 → train can't tell which
                # worker is bad → next refit poisons everyone). Same
                # pattern as /init_collective above.
                # Snapshot which shards were ``joining`` at the moment we
                # dispatched the broadcast. Only THESE shards are eligible
                # for promotion when the refit succeeds — a shard added
                # via ``/admin/add_shard`` AFTER dispatch did not
                # participate in the broadcast and still has stale
                # ``load_format=dummy`` weights; promoting it would let
                # the data plane route to it and serve garbage (rewards
                # collapse to 0 — observed in production once when an
                # add_shard landed during a long ~410s refit window).
                async with self._table_lock:
                    eligible_promote = self._eligible_promote_sids()

                def _dispatch() -> list[Any]:
                    return generation.update_weights_from_collective()

                futures = await _run_blocking(_dispatch)
                # Broadcast itself is fast (~3-5s for 30B); a 60s window
                # is enough for any peer-loss DistStoreError or NCCL
                # timeout to surface.
                failed_idxs, results, exc_types, failed_fast = await self._per_worker_results(
                    futures, rendezvous_timeout_s=60.0
                )

                if failed_idxs and self._is_rendezvous_master_failure(
                    failed_idxs, exc_types
                ):
                    print(
                        f"[router] /update_weights_from_collective: "
                        f"rendezvous-master failure (all {len(failed_idxs)} "
                        f"gen workers raised {exc_types[failed_idxs[0]]}); "
                        f"skipping mass-eviction, returning 503 for retry",
                        flush=True,
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "success": False,
                            "error": "rendezvous master timed out; retry",
                            "current_gen_world_size": self.current_gen_world_size(),
                            "failed_indices": failed_idxs,
                            "evicted_shard_ids": [],
                            "promoted_shards": [],
                            "rendezvous_master_failure": True,
                        },
                    )

                if failed_idxs:
                    num_evicted, evicted_ids = await self._evict_failed_workers_inline(
                        failed_idxs,
                        reason="update_weights_from_collective: per-worker error",
                        exception_types=exc_types,
                        force_evict_idxs=failed_fast,
                    )
                    new_ws = self.current_gen_world_size()
                    msg = (
                        f"{len(failed_idxs)} of {len(futures)} workers failed "
                        f"weight broadcast; evicted {num_evicted} (ids={evicted_ids}); "
                        f"current_gen_world_size={new_ws}"
                    )
                    print(
                        f"[router] /update_weights_from_collective: {msg}",
                        flush=True,
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "success": False,
                            "error": msg,
                            "current_gen_world_size": new_ws,
                            "failed_indices": failed_idxs,
                            "evicted_shard_ids": evicted_ids,
                            "promoted_shards": [],
                        },
                    )

                success = all(r for r in results if r is not None)
                if not success:
                    # All workers responded but at least one returned a
                    # falsy result (no exception, just refused). Same
                    # treatment — surface to train so it retries.
                    raise RuntimeError(
                        f"One or more workers reported broadcast failure. Results: {results}"
                    )
                # Promote joining shards to ready only after refit succeeds (weights are now current).
                promoted: list[str] = []
                if success:
                    promoted = self.promote_all_joining(
                        eligible_sids=eligible_promote
                    )
                    if promoted:
                        print(
                            f"[router] refit complete; promoted "
                            f"{len(promoted)} joining shard(s) -> ready: "
                            f"{promoted}",
                            flush=True,
                        )
                return {"success": success, "promoted_shards": promoted}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/prepare_for_generation")
        async def prepare_for_generation() -> Any:
            generation = _require_generation()
            try:
                result = await _run_blocking(generation.prepare_for_generation)
                return {"success": bool(result)}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/finish_generation")
        async def finish_generation() -> Any:
            generation = _require_generation()
            try:
                result = await _run_blocking(generation.finish_generation)
                return {"success": bool(result)}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/prepare_refit_info")
        async def prepare_refit_info(request: Request) -> Any:
            generation = _require_generation()
            try:
                body_bytes = await request.body()

                def _do() -> None:
                    state_dict_info = torch.load(
                        io.BytesIO(body_bytes), weights_only=True
                    )
                    generation.prepare_refit_info(state_dict_info)

                await _run_blocking(_do)
                return {"success": True}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/invalidate_kv_cache")
        async def invalidate_kv_cache() -> Any:
            generation = _require_generation()
            try:
                result = await _run_blocking(generation.invalidate_kv_cache)
                return {"success": bool(result)}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"success": False, "error": str(e)}
                )

        @app.post("/clear_logger_metrics")
        async def clear_logger_metrics() -> dict[str, bool]:
            generation = _require_generation()
            generation.clear_logger_metrics()
            return {"success": True}

        @app.get("/get_logger_metrics")
        async def get_logger_metrics() -> dict[str, Any]:
            generation = _require_generation()
            return generation.get_logger_metrics()

        @app.post("/snapshot_step_metrics")
        async def snapshot_step_metrics() -> dict[str, bool]:
            generation = _require_generation()
            if hasattr(generation, "snapshot_step_metrics"):
                generation.snapshot_step_metrics()
            return {"success": True}

        @app.get("/get_step_metrics")
        async def get_step_metrics() -> dict[str, Any]:
            generation = _require_generation()
            if hasattr(generation, "get_step_metrics"):
                return generation.get_step_metrics()
            return {}

        @app.get("/step_metrics_snapshot")
        async def step_metrics_snapshot() -> dict[str, Any]:
            """Consolidated, non-destructive gen-side metrics snapshot.

            Bundles two sources the train cluster can't reach directly:
              * ``vllm_logger_metrics`` — vLLM's PrometheusStatLogger samples
                (inflight batch sizes, num_pending, kv_cache_usage_perc,
                generation_tokens) collected on each model-owner actor.
              * ``router_metrics`` — fault-tolerance counters: number of
                ready/cordoned/joining shards, cumulative removals, last
                fault event.

            Spec-decode counters (``vllm/spec_*``) are NOT included here —
            they're a destructive delta-since-snapshot read consumed by
            grpo via the existing ``GET /get_step_metrics`` endpoint. The
            ``spec_decode_metrics`` key is left as ``{}`` so the response
            shape is stable for clients that index into it.

            Single network round-trip per step. Response is KB-scale (no
            tensors, no large vectors — vllm_logger lists are timeline
            samples capped by training step duration).
            """
            generation = _require_generation()
            return {
                "vllm_logger_metrics": generation.get_logger_metrics(),
                "spec_decode_metrics": {},
                "router_metrics": self.metrics_snapshot(),
            }

        return app

    # =====================================================================
    # State accessors
    # =====================================================================

    def refit_ready_state(self) -> tuple[bool, str]:
        if self._nccl_reinit_in_progress:
            return False, "nccl_reinit_in_progress"
        # 0 shards alive → no gen workers in the NCCL world. ``init_collective``
        # would raise "Data parallel size is zero" instantly; the train side's
        # ensure_collective_synced retry budget (~6 fast retries) blows past
        # in <10s and the run dies. Hold the gate until at least one shard
        # is back so train waits for ``add_shard`` recovery (typically 5-10
        # min for vLLM-30B init) instead of fast-failing on a transient
        # cascade-to-zero between burst kill and recovery.
        if self.shard_count_alive_for_collective() == 0:
            return False, "no_shards_alive"
        # joining does NOT block refit — the next refit pushes fresh weights to joining shards.
        return True, "ok"

    def shard_count_ready(self) -> int:
        return sum(1 for s in self._shards.values() if s.status == "ready")

    def shard_count_alive_for_collective(self) -> int:
        """Number of shards in the cross-cluster NCCL group (``ready`` + ``joining``).

        Excludes ``cordoned``/``draining`` shards whose actors are being torn down.
        """
        return sum(
            1
            for s in self._shards.values()
            if s.status in ("ready", "joining")
        )

    def current_gen_world_size(self) -> int:
        """World size to advertise to the train side for init_collective.

        Sums per-shard world size over shards that participate in the
        cross-cluster NCCL group (``ready`` + ``joining``). Cordoned /
        draining shards are excluded — their actors are being torn down
        on the next ``init_collective``.
        """
        return self.shard_count_alive_for_collective() * self._per_shard_world_size

    def _is_joinable(self, s: "ShardEntry", now: float) -> bool:
        """Whether a shard can complete a cross-cluster rendezvous *right now*.

        ``ready`` always qualifies (promoted by a prior successful refit → its
        route is warm by construction). A ``joining`` shard qualifies only once
        it has passed ``join_success_threshold`` health probes AND is either
        ``proven`` (rendezvoused before → warm) or has aged past the warm-up
        window — a freshly-backfilled shard answers /openapi.json long before
        its COLD cross-cluster route can finish a TCPStore rendezvous, and
        counting it too early is what sabotaged the grow ("7/9 clients joined").
        """
        if s.status == "ready":
            return True
        if (
            s.status == "joining"
            and s.consecutive_successes >= self.join_success_threshold
        ):
            return s.proven or (now - s.joined_at) >= self._joinable_min_age_s
        return False

    def joinable_shard_count(self) -> int:
        """Number of shards that can complete a rendezvous right now."""
        now = time.monotonic()
        return sum(1 for s in self._shards.values() if self._is_joinable(s, now))

    def _joinable_cohort(self) -> tuple[list[str], list[int]]:
        """Snapshot the joinable cohort: (shard_ids, worker_indices).

        The frozen set the cross-cluster ``init_collective`` rendezvouses — a
        booting/cold backfill shard (in the worker group but not yet joinable)
        is excluded so it can't sabotage the handshake.

        Both lists are ordered by the shard's LEADER worker index so they line
        up with the dispatch order of ``run_all_workers_multiple_data`` (which
        iterates workers in ascending index): ``shard_ids[i]`` is the shard
        whose leader future is the i-th in the returned futures list, which is
        what ``_evict_failed_workers_inline`` relies on to map a failed index
        back to its shard. ``worker_indices`` is the flat, sorted dispatch set."""
        now = time.monotonic()
        joinable = [
            (min(s.worker_indices) if s.worker_indices else 0, sid, s)
            for sid, s in self._shards.items()
            if self._is_joinable(s, now)
        ]
        joinable.sort(key=lambda t: t[0])
        sids = [sid for _, sid, _ in joinable]
        indices = sorted(i for _, _, s in joinable for i in s.worker_indices)
        return sids, indices

    def joinable_world_size(self) -> int:
        """Joinable shard count × per-shard world size (subset of
        ``current_gen_world_size``; equal to it at a settled point)."""
        return self.joinable_shard_count() * self._per_shard_world_size

    def _refresh_joinable_stability(self) -> None:
        """Bump ``_joinable_changed_at`` whenever the joinable count changes.

        Called once per health-poll tick (2 s granularity is plenty for a
        ~45 s debounce). Idempotent: a tick with no change leaves the timer
        alone so ``joinable_stable_for_s`` keeps growing."""
        count = self.joinable_shard_count()
        if count != self._last_joinable_count:
            self._last_joinable_count = count
            self._joinable_changed_at = time.monotonic()

    def joinable_stable_for_s(self) -> float:
        """Seconds since the joinable count last changed."""
        return max(0.0, time.monotonic() - self._joinable_changed_at)

    def metrics_snapshot(self) -> dict[str, Any]:
        """Cheap point-in-time snapshot of router state for wandb.

        Synchronous accessor — reads the shard table without taking
        ``_table_lock``. The shard dict is mutated only from the asyncio
        event loop (health poll loop, proxy hot path, lifecycle ops); a
        synchronous reader on the same thread sees a consistent enough view
        for wandb's per-step rollup. Callers must NOT block on this from
        inside an asyncio coroutine where ``_table_lock`` is held.

        Per-shard ``last_health_ok_age_s`` is exposed so wandb can flag a
        shard whose health is going stale even before the failure threshold
        flips it to cordoned.

        Counters are over the daemon's lifetime; they reset when the gen
        daemon restarts (e.g. ``--replace``). Documented limitation.
        """
        now = time.monotonic()
        ready = joining = cordoned = draining = 0
        per_shard: list[dict[str, Any]] = []
        for s in self._shards.values():
            if s.status == "ready":
                ready += 1
            elif s.status == "joining":
                joining += 1
            elif s.status == "cordoned":
                cordoned += 1
            elif s.status == "draining":
                draining += 1
            per_shard.append(
                {
                    "shard_id": s.shard_id,
                    "status": s.status,
                    "inflight": s.inflight,
                    "consecutive_failures": s.consecutive_failures,
                    "last_health_ok_age_s": (
                        max(0.0, now - s.last_health_ok_at)
                        if s.last_health_ok_at > 0
                        else None
                    ),
                }
            )
        return {
            "num_ready_shards": ready,
            "num_joining_shards": joining,
            "num_cordoned_shards": cordoned,
            "num_draining_shards": draining,
            "num_total_shards": len(self._shards),
            "total_shards_at_bootstrap": self._total_shards_at_bootstrap,
            "cumulative_shards_removed": self._cumulative_shards_removed,
            "cumulative_shards_added": self._cumulative_shards_added,
            "per_shard_world_size": self._per_shard_world_size,
            "current_gen_world_size": self.current_gen_world_size(),
            "joinable_world_size": self.joinable_world_size(),
            "joinable_stable_for_s": self.joinable_stable_for_s(),
            "comm_epoch": self._comm_reset_epoch,
            "nccl_reinit_in_progress": self._nccl_reinit_in_progress,
            "last_fault_event": (
                dict(self._last_fault_event)
                if self._last_fault_event is not None
                else None
            ),
            "per_shard": per_shard,
        }

    # =====================================================================
    # Server lifecycle
    # =====================================================================

    def start(self) -> None:
        """Start uvicorn in a background thread."""
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            timeout_keep_alive=self.uvicorn_keep_alive_s,
            # Bump TCP listen backlog from uvicorn's default 2048. Under
            # an RL burst (2048 concurrent samples), the kernel can drop
            # incoming SYNs if backlog fills before uvicorn accepts them,
            # surfacing as `aiohttp.ServerDisconnectedError` on the
            # client. Doubling gives headroom.
            backlog=self.uvicorn_backlog,
            # Cap the request line + headers size at something well above
            # vLLM's expected payload — defaults are fine, but explicit so
            # nothing surprising happens with large prompts.
            h11_max_incomplete_event_size=64 * 1024 * 1024,
        )
        server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=server.run, daemon=True)
        self._server_thread.start()
        print(
            f"GenerationRouter started on port {self.port} "
            f"(shards={len(self._shards)}, ready={self.shard_count_ready()})",
            flush=True,
        )

    def get_app(self) -> FastAPI:
        """Expose the FastAPI app (for testing or custom server setup)."""
        return self._app
