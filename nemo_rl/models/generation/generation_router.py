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

A single FastAPI service exposes both the data plane (proxied
``/v1/{completions,chat/completions}`` to per-shard vLLM endpoints with
cordon + replay) and the control plane (weight sync, lifecycle, refit
gate, metrics). Both training (``RemoteGeneration._http_mode``) and
NemoGym point at this server, so cordoning of failed DP shards is
invisible to clients.

The router proxies requests to a per-shard vLLM OpenAI endpoint
(round-robin, optionally sticky via ``X-NRL-Session-Id``) and replays on
a different shard if the chosen one returns 5xx / connection error.

The router also owns the gen-side lifecycle: placement-group + Ray-actor
add/remove and weight-sync NCCL coordination. Control endpoints
(``/init_collective``, ``/update_weights_from_collective``, etc.) wrap
the underlying ``VllmGeneration`` and run on the same uvicorn process /
event loop, so there is exactly one HTTP service and one port (8089) per
gen daemon.
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

# vLLM's OpenAI HTTP server (in nemo-rl's nightly image) exposes only
# /v1/{completions,chat/completions}, /tokenize, and /openapi.json — no
# /health, /v1/models, or /healthz. /openapi.json is the only stable
# always-200 GET we can use as a liveness probe; if a future vLLM build
# adds /health we can flip this without changing the rest of the router.
_HEALTH_PATH = "/openapi.json"
_DEFAULT_HEALTH_INTERVAL_S = 2.0
# Bumped from 1s — vLLM's FastAPI event loop can stall 2-3s under heavy
# inflight load (hundreds of concurrent /v1/completions for 30B-class
# models with TP>1). A 1s probe timeout misclassifies that as failure.
_DEFAULT_HEALTH_TIMEOUT_S = 5.0
# Bumped from 3 — combined with the longer timeout above, cordon now
# requires ~50s of sustained unresponsiveness instead of ~6s. Genuine
# failures (pod death, OOM, NCCL hang) still trip cordon within a
# bounded window; transient probe lag during burst load doesn't.
_DEFAULT_FAILURE_THRESHOLD = 10
_DEFAULT_JOIN_SUCCESS_THRESHOLD = 2
# Data-plane (proxy) error threshold. The proxy used to cordon on a
# single 5xx or transport error from a shard, which under burst load
# turned an overloaded vLLM into a permanently-removed shard. Now we
# require N consecutive proxy errors — same shape as the health-poll
# threshold but accumulated on /v1/completions failures instead of
# /openapi.json probe failures.
_DEFAULT_PROXY_FAILURE_THRESHOLD = 5
# aiohttp's default TCPConnector(limit=100) caps the router at 100
# in-flight downstream connections — 20x too small for a 2048-prompt
# burst (32 generations × 64 prompts). Excess requests queue in the
# router and time out / get connection-reset before reaching a shard.
# 4096 total + 1024/host gives plenty of headroom for typical RL
# batches (2048-4096) without exhausting OS file handles.
_DEFAULT_PROXY_POOL_LIMIT_TOTAL = 4096
_DEFAULT_PROXY_POOL_LIMIT_PER_HOST = 1024
# uvicorn's default backlog (2048) is fine for typical loads, but
# explicit so it's tunable. timeout_keep_alive 120 stays — long
# enough to amortize TLS/TCP setup across many requests.
_DEFAULT_UVICORN_BACKLOG = 4096
_DEFAULT_UVICORN_KEEP_ALIVE_S = 120
_DEFAULT_PROXY_TIMEOUT_S = 300.0
# Per-worker timeout for reset_collective during add_shard / remove_shard.
# Bounds how long we wait on a survivor's NCCL teardown after a peer was
# actor-killed. NCCL abort can wedge if a dead peer's collective was
# in-flight; without this bound, ray.get(futures) deadlocks the lifecycle
# lock and `_nccl_reinit_in_progress` stays True forever, killing the train
# run via ensure_collective_synced's 600s timeout. 30s is well above
# legitimate abort+reset wall time (~1-2s on healthy survivors); a hung
# survivor is treated as failed and the next init_collective rebuilds
# from a clean rendezvous.
_DEFAULT_RESET_COLLECTIVE_TIMEOUT_S = 30.0


@dataclass
class ShardEntry:
    """Router's view of a single vLLM DP shard.

    `url` is the OpenAI-compatible base, e.g. http://<pod-ip>:8000/v1.
    `_health_url` is derived once (strip trailing /v1, append /health) so we
    don't pay string ops on the hot path.

    `actor_handles` and `placement_group` are populated when the router owns
    the lifecycle (post bring-up); they let `remove_shard` kill the actors
    and free the PG so KubeRay's autoscaler can release the pod.
    """

    shard_id: str
    url: str
    status: ShardStatus = "joining"
    last_health_ok_at: float = 0.0
    inflight: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    # Data-plane (proxy) error counter — distinct from the health-poll
    # `consecutive_failures` so a transient burst of 5xx/transport errors
    # on /v1/completions doesn't cordon a shard before the health-poll
    # has even noticed.
    proxy_consecutive_failures: int = 0
    # Number of successful refits this shard has participated in since
    # being added (or since the last fault). Used to gate joining→ready
    # transition: the FIRST refit including a freshly-joined shard pays
    # the full NCCL connection bootstrap cost (~20s on 30B-class hardware
    # — IB queue pairs, GPU memory IPC mappings, NCCL topology cache),
    # and during that bootstrap window the broadcast can land on the new
    # shard while NCCL is still finishing setup, leaving its weights
    # *almost* correct but with subtle byte corruption that surfaces as
    # KL=1.4+ on the next training step. Requiring >=2 successful refits
    # means the second refit lands on a fully-warm comm with clean bytes,
    # and only then do we let the data plane route requests to this
    # shard — eliminates the post-join KL spike.
    successful_refits_since_join: int = 0
    actor_handles: list[Any] = field(default_factory=list, repr=False)
    placement_group: Any = field(default=None, repr=False)
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


class GenerationRouter:
    """Unified data plane + control plane for the gen cluster.

    Owns the only authoritative shard table for the gen cluster, plus
    placement-group + Ray-actor lifecycle and gen-side NCCL reset/init.
    The shard table is populated once at bring-up via ``register_shards()``
    and mutated by the health poll loop, the data-plane proxy on
    threshold-breaching errors, and admin endpoints (cordon /
    add_shard / remove_shard).

    Control-plane endpoints (init_collective, update_weights_from_collective,
    reset_collective, prepare_refit_info, etc.) wrap the underlying
    ``VllmGeneration`` so the train side has a single HTTP target for
    everything: data, status, lifecycle, weight sync.
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
        auto_backfill: bool = False,
        backfill_target: Optional[int] = None,
        backfill_max_concurrent: int = 4,
        pg_ready_timeout_s: float = 600.0,
    ):
        self.port = port
        self.health_poll_interval_s = health_poll_interval_s
        self.health_timeout_s = health_timeout_s
        self.failure_threshold = failure_threshold
        self.join_success_threshold = join_success_threshold
        self.proxy_failure_threshold = proxy_failure_threshold
        self.proxy_pool_limit_total = proxy_pool_limit_total
        self.proxy_pool_limit_per_host = proxy_pool_limit_per_host
        self.uvicorn_backlog = uvicorn_backlog
        self.uvicorn_keep_alive_s = uvicorn_keep_alive_s
        self.proxy_timeout_s = proxy_timeout_s
        self._reset_collective_timeout_s = reset_collective_timeout_s

        # RL-412 auto-backfill (state-driven recovery). When enabled, the
        # health-poll loop restores the gen world to ``_backfill_target_count``
        # by calling add_shard for the deficit — covering collateral
        # poison-evicted shards the fault injector never replaces.
        self.auto_backfill = auto_backfill
        self._backfill_target = backfill_target
        self.backfill_max_concurrent = max(1, int(backfill_max_concurrent))
        self.pg_ready_timeout_s = float(pg_ready_timeout_s)
        # add_shard calls launched by the reconciler that have not yet
        # registered a ``joining`` shard (PG still booting). Counted in the
        # deficit so repeated ticks don't over-provision.
        self._inflight_adds: int = 0

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
        # Count of weight-refit broadcasts the router has handled. The
        # FaultInjector gates its FIRST kill on this reaching >=1 ("training
        # has actually started refitting") instead of a wall-clock guess —
        # setup time varies wildly (warm reuse ~4min vs cold Megatron-30B
        # load ~13min), so a fixed delay-from-daemon-start fires during setup
        # on a slow start or after the run finished on a fast one.
        self._refit_attempts: int = 0
        # Set at bring-up via register_shards; updated on add/remove.
        self._per_shard_world_size: int = 1
        # Reference to the underlying VllmGeneration. Drives both the
        # control-plane endpoints (init/reset/update_weights_from_collective,
        # prepare_for_generation, etc.) and the lifecycle path
        # (reset_collective after add/remove). None when the router runs
        # without a backing VllmGeneration (unit tests, manual /admin
        # operations on a fake setup); control endpoints then 503.
        self._generation: Any = generation

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._health_task: Optional[asyncio.Task[None]] = None
        # Set true in the lifespan teardown. Background tasks that schedule
        # blocking work via ``loop.run_in_executor`` (the default thread-pool)
        # will fail with "cannot schedule new futures after shutdown" once
        # the loop is being torn down — observed during gen daemon restart
        # / ``--replace`` when a cordon-driven auto-remove fires concurrently
        # with shutdown. Hooks read this flag and short-circuit so the next
        # daemon starts with a clean shard table instead of inheriting
        # half-removed actors / dangling cordons.
        self._shutting_down: bool = False
        self._app = self._build_app()

        # uvicorn runs in a daemon thread; the loop is owned by uvicorn.
        self._server_thread: Optional[threading.Thread] = None
        # Hook for tests / step 2: called after each cordon transition.
        self.on_cordon: Optional[Callable[[str, str], Awaitable[None]]] = None

        # =====================================================================
        # Fault-tolerance metrics (surfaced via metrics_snapshot()).
        #
        # Counters track shard fleet churn over the daemon's lifetime so wandb
        # can plot how many shards we've lost / regained over the full run.
        # Note: these reset when the gen daemon restarts (e.g. on `--replace`);
        # we accept that limitation rather than persist counters across a
        # daemon recycle. RemoteGeneration scrapes these once per training
        # step via /step_metrics_snapshot.
        # =====================================================================
        self._total_shards_at_bootstrap: int = 0
        self._cumulative_shards_removed: int = 0
        self._cumulative_shards_added: int = 0
        # Most recent fault event ({"reason": str, "shard_id": str,
        # "monotonic_ts": float, "kind": "remove"|"add"|"cordon"}).
        # Train side flips a per-step fault marker when this changes.
        self._last_fault_event: Optional[dict[str, Any]] = None

    # =====================================================================
    # Shard table mutation
    # =====================================================================

    def register_shards(
        self,
        shards: list[tuple[str, str]],
        per_shard_world_size: int = 1,
        actor_handles_by_shard: Optional[dict[str, list[Any]]] = None,
        pg_by_shard: Optional[dict[str, Any]] = None,
        worker_indices_by_shard: Optional[dict[str, list[int]]] = None,
        generation: Any = None,
    ) -> None:
        """Seed the shard table at bring-up.

        Synchronous; intended to be called before `start()`. Initial status
        is `ready` because the existing VllmGeneration bring-up already
        waits for vLLM `/health` before returning; the health poller demotes
        on subsequent failure.

        Args:
            shards: List of ``(shard_id, url)`` tuples.
            per_shard_world_size: TP*PP for one DP shard. Multiplied by the
                ready-shard count to derive ``current_gen_world_size``,
                which the train driver reads on each refit to pick up
                shrink/grow without redeploying.
            actor_handles_by_shard: Optional ``{shard_id: [actor, ...]}``.
                Required for ``remove_shard`` to kill the right actors.
                Omit for unit tests / manual cordon-only usage.
            pg_by_shard: Optional ``{shard_id: PlacementGroup}``. Required
                for ``remove_shard`` to free the PG so KubeRay's autoscaler
                can reclaim the pod.
            generation: Optional reference to the underlying
                ``VllmGeneration`` instance. The router calls
                ``generation.reset_collective()`` on the surviving workers
                after a remove_shard, and uses it to drive control-plane
                endpoints. May also be passed at construction time.
        """
        self._per_shard_world_size = per_shard_world_size
        actor_handles_by_shard = actor_handles_by_shard or {}
        pg_by_shard = pg_by_shard or {}
        worker_indices_by_shard = worker_indices_by_shard or {}
        for shard_id, url in shards:
            self._shards[shard_id] = ShardEntry(
                shard_id=shard_id,
                url=url,
                status="ready",
                last_health_ok_at=time.monotonic(),
                actor_handles=list(actor_handles_by_shard.get(shard_id, [])),
                placement_group=pg_by_shard.get(shard_id),
                worker_indices=list(worker_indices_by_shard.get(shard_id, [])),
            )
        # Capture the bootstrap fleet size on first registration so wandb can
        # compute "lost X of Y shards" over the run. Subsequent calls (which
        # we don't currently make, but keep correct just in case) only update
        # if we observe a larger fleet — never decrease.
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
            # Stash the most recent cordon as a fault event so wandb can mark
            # the step boundary where the fleet shrank. Cheap dict assignment
            # under the existing lock — no new network or compute.
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
            # Flip cordoned -> joining, NOT directly to ready. We can't
            # prove a cordoned shard's weights are current: cordon could
            # have fired because the actor was hung mid-broadcast, in
            # which case it missed the last weight sync and would serve
            # stale completions if we routed traffic to it now. `joining`
            # gates the data plane and the refit_ready signal until the
            # next /update_weights_from_collective rebroadcasts and
            # promote_all_joining moves it back to ready.
            entry.status = "joining"
            entry.consecutive_failures = 0
            entry.consecutive_successes = 0
            # Reset the warmup-refit counter — a cordoned shard's NCCL
            # state may be wedged, so the same N-refit warmup gate
            # applies as for a fresh add_shard.
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

        After this returns, the gen-side workers have torn down their
        weight-sync NCCL group via ``generation.reset_collective()``. The
        train driver's ``ensure_collective_synced`` notices the new
        ``current_gen_world_size`` on its next refit and drives the
        rendezvous via the existing ``/init_collective`` control endpoint;
        until that completes ``/refit_ready`` returns false.

        Returns a small dict summarising the outcome — useful for the
        ``--fault-mode`` injector and integration tests.
        """
        async with self._lifecycle_lock:
            async with self._table_lock:
                entry = self._shards.get(shard_id)
                if entry is None:
                    return {"shard_id": shard_id, "removed": False, "reason": "not found"}
                entry.status = "draining"
                self._nccl_reinit_in_progress = True
                actor_handles = list(entry.actor_handles)
                pg = entry.placement_group
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

            # Kill the actors and free the PG. Both calls are blocking Ray
            # ops; run them in the executor so the FastAPI loop keeps
            # serving health checks etc.
            loop = asyncio.get_running_loop()

            def _kill_and_free() -> None:
                import ray
                from ray.util.placement_group import remove_placement_group

                for actor in actor_handles:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception as e:  # noqa: BLE001
                        # Log + continue: a dead actor is fine, we're tearing it down.
                        print(f"[router] ray.kill on {shard_id} actor raised {e}", flush=True)
                if pg is not None:
                    try:
                        remove_placement_group(pg)
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"[router] remove_placement_group on {shard_id} raised {e}",
                            flush=True,
                        )

            await loop.run_in_executor(None, _kill_and_free)

            # Drop the shard from the table. Health poller no longer probes
            # it; routing skips it. The autoscaler v2 + minReplicas:0
            # combination is what reclaims the pod from here.
            async with self._table_lock:
                self._shards.pop(shard_id, None)
                self._cumulative_shards_removed += 1
                self._last_fault_event = {
                    "kind": "remove",
                    "shard_id": shard_id,
                    "reason": reason,
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

            # NB: we used to dispatch ``generation.reset_collective()`` on
            # surviving workers here so they'd tear down the cross-cluster
            # NCCL group. That was redundant with per-refit self-teardown
            # (RefitWorker + vllm_backend both ``destroy()`` at the end of
            # every refit) AND it triggered a cascade-eviction failure
            # mode under burst kills:
            #   1. dp-X killed → ``remove_shard(dp-X)`` → reset_collective
            #      on N-1 survivors with a 30s timeout.
            #   2. ONE survivor is mid-broadcast (its update_weights_from_
            #      collective is still draining the NCCL kernel that was
            #      waiting on the now-dead peer). Its reset_collective
            #      queues behind that and exceeds the 30s budget.
            #   3. Router (incorrectly) treats the timeout as "poisoned"
            #      and evicts that survivor → triggers another
            #      remove_shard → another reset round → another timeout
            #      on a different survivor (which had its own pending
            #      drain because of the eviction). Cascade to world=0.
            #
            # With the reset call dropped, the next refit's
            # ``ensure_collective_synced`` retry path in
            # ``grpo.refit_policy_generation`` handles this cleanly:
            # the broadcast raises (peer death), the train driver
            # ray.kills RefitWorker, calls ensure_collective_synced
            # again at the new (smaller) world. ``vllm_backend.init_
            # collective`` on each survivor destroys its wedged old
            # comm before creating the new one — so the destroy
            # happens in the natural lifecycle, not pre-emptively
            # from this remove_shard handler.

            # Refit gate stays false until the train driver re-inits the
            # collective. We can't observe that directly from the gen side;
            # the simplest signal is "no more `joining` shards AND the gen
            # workers have rendezvoused with the train side". For now, lift
            # the gate immediately — the train side's
            # ensure_collective_synced is idempotent and will block on the
            # actual rendezvous before unblocking refit. The /refit_ready
            # gate then guards subsequent refits from the train side's
            # perspective.
            self._nccl_reinit_in_progress = False
            return {
                "shard_id": shard_id,
                "removed": True,
                "reason": reason,
                "world_size": self.current_gen_world_size(),
            }

    async def add_shard(self, reason: str = "manual") -> dict[str, Any]:
        """Allocate a new DP shard and append it as ``joining``.

        RL-412 scale-up: drives a fresh placement-group request (KubeRay
        autoscaler v2 schedules a new pod into the gpu-shard worker group),
        spawns a vLLM async worker, and registers it here as a ``joining``
        shard. The data plane skips ``joining`` shards (their weights are
        stale) until the train side completes the next refit — at which
        point ``/update_weights_from_collective`` calls
        ``promote_all_joining`` and the new shard flips to ``ready``.

        ``reset_collective`` is also driven against surviving workers so
        that the next ``init_collective`` from the train side rendezvouses
        at the new (world_size + 1).

        The ``_nccl_reinit_in_progress`` flag is held True until refit
        succeeds (cleared by ``promote_all_joining``), gating ``refit_ready``
        for the next training step.
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

            # Refit-gate timing — three phases inside ``add_dp_worker``:
            #   1. ``pg.ready()`` (~90-120s): autoscaler v2 schedules the
            #      pod. Worker_group is UNCHANGED. Train can keep
            #      refitting at the smaller (post-remove) world freely.
            #      Gate stays DOWN.
            #   2. Spawn actor (instant) → ``pre_append_hook`` fires.
            #      Gate goes UP here, BEFORE the worker is appended to
            #      ``_workers`` / ``dp_leader_worker_indices``. After
            #      this point, any ``init_collective`` dispatch from the
            #      gen side would include the new worker, so train's
            #      ``ensure_collective_synced`` must see ``refit_ready=False``
            #      until reset_collective publishes the new world size.
            #   3. Append + prepare_refit_info + base_url + return
            #      (~30-60s while actor's __init__ finishes). Gate is
            #      UP, train blocks on ``/refit_ready`` if it hits this
            #      window.
            # The router then inserts into the shard table, dispatches
            # reset_collective on all gen workers (incl. new joining),
            # and clears the gate. Train's next refit picks up the new
            # world via /current_gen_world_size and rendezvouses cleanly.

            def _gate_up() -> None:
                self._nccl_reinit_in_progress = True

            # Heavy work: PG allocation (autoscaler waits ~3-5min for new
            # pod), vLLM worker init (~1-2min for 4B). Run on a thread so the
            # asyncio loop stays responsive (health poll, /metrics, /shards).
            def _do_add() -> tuple[list[Any], Any, list[int], Optional[str]]:
                return self._generation.add_dp_worker(pre_append_hook=_gate_up)

            try:
                actor_handles, pg, worker_indices, base_url = (
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
                # ``spawn_workers_for_shard`` compacts dead indices before
                # spawning, which shifts the indices of the surviving
                # workers. Re-map the existing _shards entries'
                # worker_indices so a future ``remove_shard`` on a
                # survivor doesn't kill the wrong actor.
                #
                # For TP/PP > 1, each shard owns N=tp*pp consecutive
                # worker indices: ``[leader, leader+1, ..., leader+N-1]``.
                # After compaction the alive workers stay in their
                # original relative order, so each surviving leader's
                # NEW index is just its position in
                # ``dp_leader_worker_indices``, and the followers are
                # the next N-1 indices. The router preserves shard
                # insertion order so we can pair leader-list and shard
                # table positionally.
                wg = self._generation.worker_group
                leaders = list(wg.dp_leader_worker_indices)
                per_shard_ws = max(self._per_shard_world_size, 1)
                # leaders[-1] is the just-added shard's leader;
                # leaders[:-1] map 1:1 to existing _shards entries in
                # insertion order.
                survivor_leaders = leaders[:-1]
                for entry, leader_new_idx in zip(
                    self._shards.values(), survivor_leaders
                ):
                    entry.worker_indices = list(
                        range(leader_new_idx, leader_new_idx + per_shard_ws)
                    )
                self._shards[shard_id] = ShardEntry(
                    shard_id=shard_id,
                    url=proxy_url,
                    status="joining",
                    last_health_ok_at=time.monotonic(),
                    actor_handles=list(actor_handles),
                    placement_group=pg,
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

            # NB: previously dispatched ``generation.reset_collective()``
            # here for symmetry with remove_shard (tear down existing
            # comm so the next init_collective rebuilds at the new
            # world). Removed for the same reasons as the remove_shard
            # version: redundant with per-refit self-teardown, AND
            # creates a cascade-eviction failure mode if any survivor
            # is mid-refit. The next refit's init_collective on each
            # worker destroys the old comm naturally before creating
            # the new one at world=N+1.

            # Clear the gate now that the gen-side teardown is complete.
            # The train driver's ``ensure_collective_synced`` reads
            # ``/refit_ready`` (this gate) at the top of the next refit
            # and then re-rendezvouses the cross-cluster NCCL group at
            # the new world size (which now includes the joining shard).
            # ``promote_all_joining`` runs after refit completes and
            # flips the new shard to ``ready``.
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
            # Mirror the health-poll guard: don't cordon (and don't fire
            # the auto-remove hook) while a cross-cluster NCCL re-init
            # is in flight. The gen workers' event loop stalls during
            # rendezvous, so any rollout traffic in that window times
            # out and lands here as 5xx — but the shard is healthy, it's
            # just busy. Cordoning + auto-removing the whole surviving
            # fleet over a transient NCCL stall is exactly the cascade
            # that took down the 4B run on the first stress-test pass.
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
                    # Count alive (ready/joining) shards so we can detect
                    # the deadlock-on-flap case: if EVERY shard is
                    # cordoned, auto-recovery on probe-success is the
                    # only path forward (the on_cordon → remove_shard
                    # hook is blocked by the last-alive guard, and the
                    # operator path is too slow for runs in flight).
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
                            # The health poller never auto-promotes a shard
                            # to `ready`. Both joining and cordoned states
                            # require explicit signals to leave:
                            #
                            # - `joining`  -> `ready` only via
                            #   `promote_all_joining()` (called from the
                            #   /update_weights_from_collective handler
                            #   after a successful broadcast). Auto-
                            #   promoting on /health alone would route
                            #   real rollouts to a shard with STALE
                            #   weights (vLLM answers /health as soon as
                            #   the engine is up off the model file on
                            #   disk, well before the next refit catches
                            #   it up).
                            #
                            # - `cordoned` -> `ready` only via admin
                            #   `/admin/uncordon` UNLESS the cordoned
                            #   shard is the last way out of a stuck
                            #   state (no ready or joining shards left).
                            #   That deadlock case can happen when a
                            #   transient probe flap on the last alive
                            #   shard trips its cordon, the on_cordon
                            #   → remove_shard hook is suppressed by the
                            #   last-alive guard (which prevents dropping
                            #   the fleet to zero), AND no operator
                            #   intervenes with /admin/uncordon. Without
                            #   the auto-recovery path below, the run
                            #   wedges forever: refit_ready stays false,
                            #   train can't refit, joining shards never
                            #   promote. So when ALL shards are
                            #   cordoned and a cordoned shard's probes
                            #   recover for ``join_success_threshold``
                            #   consecutive ticks, demote it to
                            #   ``joining`` (NOT directly to ``ready`` —
                            #   we still can't prove its weights are
                            #   current; the next refit's
                            #   ``promote_all_joining`` is what
                            #   certifies that). This is the same
                            #   semantics as the explicit
                            #   ``/admin/uncordon`` endpoint.
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
                                # Reset warmup-refit counter: this shard
                                # has been out of the fleet for
                                # potentially several refits, so its
                                # weights are stale. promote_all_joining
                                # will re-certify after the next refit.
                                entry.successful_refits_since_join = 0
                                # n_alive_or_joining now becomes 1 from
                                # this transition; no further auto-
                                # uncordons in this iteration so we
                                # don't bulk-promote everything that
                                # happened to flap together.
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
                            # Skip cordon transitions while a cross-cluster
                            # NCCL re-init is in flight. The gen workers'
                            # event loop can stall during rendezvous and
                            # miss /openapi.json probes; cordoning here would
                            # break a healthy run that's mid-refit.
                            if nccl_paused:
                                continue
                            if (
                                entry.status == "ready"
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
            # RL-412 auto-backfill: restore the gen world to target after any
            # shard loss (killed or poison-evicted). Guarded so a reconcile
            # failure never kills the health loop.
            try:
                self._reconcile_backfill()
            except Exception as e:  # noqa: BLE001
                print(f"[router] reconcile tick failed: {e}", flush=True)
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
    ) -> tuple[int, list[str]]:
        """Cordon + remove the gen workers whose futures failed.

        Synchronous from caller's perspective: by the time we return, the
        evicted shards have been popped from the router table, their
        actors killed, their PGs released, and ``reset_collective`` has
        been driven against the survivors. Train's next read of
        ``/current_gen_world_size`` will reflect the smaller world.

        Mapping ``failed_idxs`` (indices into the dispatched futures list)
        to shard ids: ``init_collective`` / weight-broadcast dispatches
        one future per DP-leader (``run_rank_0_only_axes=["tensor_parallel",
        "pipeline_parallel"]``), in worker_group insertion order. The
        router preserves shard insertion order in ``self._shards``, so
        positional pairing is correct at any TP/PP. With TP>1, the
        leader's ``init_collective`` internally fans out to TP partners
        via vLLM's ``collective_rpc``; a single failed leader future thus
        encompasses the whole TP group, and ``remove_shard`` (which
        ``ray.kill`` s every actor in the worker_group) is the right
        unit of eviction.

        Liveness gate (TP>1 cascade fix): a burst kill of M shards leaves
        survivors blocked on the TCPStore rendezvous waiting for the dead
        shards' ranks. Their futures fail with mixed
        ``DistNetworkError`` / ``RayActorError`` (the latter from
        ``collective_rpc`` propagating a TP-partner crash) — looking
        identical to the dead shards' futures. Without filtering, the
        eviction path mass-evicts EVERY failed-future shard, dropping
        the world to zero. We probe each candidate's leader actor with
        ``is_alive`` (5s timeout) and only evict shards confirmed dead.
        Survivors stay; the train-side ``ensure_collective_synced`` retry
        re-rendezvouses at the same N now that the dead shards have been
        cleared.

        Force-evict shortcut: two kinds of candidate bypass the liveness
        probe. (1) Actor death — the future raised ``RayActorError`` /
        ``ActorDiedError``; Ray confirmed the actor is gone. (2) Fast-fail —
        ``force_evict_idxs`` lists workers whose future RAISED within the
        fast-fail window (<8s); their engine_core died or their NCCL state is
        poisoned (next ``comm_init`` raises immediately), yet their actor
        wrapper still answers ``is_alive`` — so the probe would wrongly keep
        them. Both must be evicted: a poisoned/dead worker can never rejoin
        the rendezvous and a collective needs all ranks. For TP>1, the death
        of any actor in the shard's TP group means the shard is unrecoverable
        and must be removed wholesale — ``remove_shard`` ray.kills the whole
        actor list, including survivors, so a half-dead TP group doesn't get
        orphaned. Slow-failers and PENDING timeouts (healthy survivors blocked
        on a dead peer) are NOT force-evicted — they route through the probe
        and are kept.

        Returns ``(num_evicted, evicted_shard_ids)``.
        """
        if not failed_idxs:
            return 0, []

        # Dispatch goes per-DP-leader (run_rank_0_only_axes covers
        # tensor_parallel + pipeline_parallel), so the futures list is one
        # entry per DP shard regardless of TP/PP. failed_idxs map directly
        # to shard insertion order. With TP>1, one failed leader future
        # implies the whole TP group is evicted via remove_shard (which
        # ray.kills every actor in the worker_group).
        async with self._table_lock:
            shard_ids_in_order = list(self._shards.keys())
        targets: list[str] = []
        # ``force_dead`` shards are confirmed dead by Ray (the future
        # raised RayActorError/ActorDiedError). These bypass the
        # liveness probe — Ray told us the actor is gone, no need to
        # double-check, and the probe can race-classify them as alive
        # if a sibling actor in the TP group is still up.
        force_dead: set[str] = set()
        seen: set[str] = set()
        # Two ways a shard force-evicts (bypassing the is_alive probe):
        #
        #   1. Actor death — the future raised RayActorError / ActorDiedError.
        #      Ray itself confirmed the actor process is gone; no need to probe.
        #
        #   2. Fast-fail (idx in ``force_evict_idxs``) — the future RAISED
        #      within the fast-fail window (<8s, see ``_per_worker_results``).
        #      The worker's own engine/NCCL blew up immediately: a dead
        #      EngineCore, or NCCL state POISONED by a prior in-band
        #      ``comm_abort`` so the next ``comm_init_rank_scalable`` raises at
        #      once. Such a worker is still POD-ALIVE (``is_alive`` returns
        #      True), so the liveness probe would wrongly KEEP it — but it can
        #      NEVER rejoin the rendezvous, and a collective needs ALL ranks,
        #      so it blocks recovery for everyone. Force-evict + replace.
        #
        # Everything else routes through the is_alive probe, which RETAINS the
        # healthy. The critical case: a survivor that merely TIMED OUT
        # ("PENDING") or hit the ~30s NCCL bootstrap timeout (slow-fail) while
        # blocked on the rendezvous for a dead/poisoned peer. Timing — not the
        # exception type — is what separates it from a poisoned worker (whose
        # comm_init raises in <1s). Evicting these survivors would cascade the
        # gen world toward zero; instead they rejoin on the next attempt once
        # the bad peers are gone. (DistStoreError / DistNetworkError from the
        # rendezvous master are already short-circuited by the caller's
        # ``_is_rendezvous_master_failure`` before we get here.)
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
        # Daemon is being torn down (--replace, SIGTERM, etc). Skip
        # auto-remove: it dispatches blocking Ray ops onto the asyncio
        # executor, which raises "cannot schedule new futures after
        # shutdown" once the loop is closing. The half-finished cordon
        # can leave the shard table in a state where the next daemon
        # bring-up sees zombie shards. The next process starts fresh
        # anyway via register_shards, so dropping the hook here is the
        # safest cleanup.
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

        # Default: a cordoned shard is no use to anyone. Its weights are
        # now uncertain, the data plane skips it, auto-recovery is gone,
        # and it occupies a placement-group slot that KubeRay's autoscaler
        # could be giving to a fresh replacement. Drive remove_shard so
        # the actor + PG come down and the pod is reclaimed; if the run
        # is configured for recovery, FaultInjector / a separate policy
        # will follow up with add_shard.
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
        # Last-alive guard: the shard we just cordoned was excluded from
        # ``shard_count_alive_for_collective`` (cordoned status). If that
        # count is now 0, this was the LAST alive shard. Auto-removing it
        # would drop the gen world to 0 and force the train side to wait
        # the full add_shard recovery budget (~7-10 min on 30B) before
        # any progress is possible — and during that window every refit
        # endpoint returns 503 and the rollout collector burns retries.
        # The shard's actor MIGHT recover (e.g., the cordon was driven by
        # a transient health-probe timeout from a vLLM engine that's just
        # busy serving a backlog, not actually dead) — keeping it parked
        # leaves a path to recovery via ``/admin/uncordon`` or future
        # health-poll auto-promotion. If it's truly dead, the next
        # init_collective will surface ActorDiedError → the force-evict
        # path in ``_evict_failed_workers_inline`` will reap it cleanly.
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
            # Explicit TCPConnector — aiohttp's default `limit=100` is the
            # router's #1 bottleneck under RL burst load (2048+ concurrent
            # /v1/completions). limit_per_host caps connections to a single
            # shard so one slow shard can't starve the others.
            connector = aiohttp.TCPConnector(
                limit=self.proxy_pool_limit_total,
                limit_per_host=self.proxy_pool_limit_per_host,
                # Force-close stale keepalive connections faster than the
                # default to avoid mid-request resets when a shard's TCP
                # state changes (e.g. NCCL re-init briefly stalls uvicorn).
                keepalive_timeout=60,
            )
            self._http_session = aiohttp.ClientSession(connector=connector)
            self._health_task = asyncio.create_task(self._health_poll_loop())
            try:
                yield
            finally:
                # Flip the flag FIRST so any in-flight cordon-fired
                # _fire_cordon_hook tasks (created with asyncio.create_task
                # earlier in the loop) short-circuit before they hit
                # remove_shard's run_in_executor — which would raise
                # "cannot schedule new futures after shutdown" and leave
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

            The training driver reads this immediately before init_collective
            so it picks up shrink/grow without redeploying. Includes
            ``ready`` + ``joining`` shards (both participate in NCCL) and
            excludes ``cordoned``/``draining``.
            """
            return {"world_size": self.current_gen_world_size()}

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
                # Dispatch init_collective to all gen workers and capture
                # the returned futures WITHOUT blocking on ray.get — we
                # need per-future error tracking so a single bad worker
                # (cudaErrorLaunchFailure / poisoned NCCL state /
                # DistStoreError) doesn't sink the whole rendezvous.
                def _dispatch() -> list[Any]:
                    return generation.init_collective(
                        ip=body["ip"],
                        port=body["port"],
                        world_size=body["world_size"],
                        train_world_size=body["train_world_size"],
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
                    eligible_promote = {
                        sid for sid, e in self._shards.items()
                        if e.status == "joining"
                    }

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
                # RL-412 scale-up: only after refit succeeds do we promote
                # `joining` shards to `ready`. Until now the data plane was
                # routing past them because their weights were stale (vLLM
                # came up with the model file but doesn't yet have the
                # latest training-side weights).
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
        # Note: ``joining`` does NOT block refit. RL-412 scale-up registers
        # a new shard as joining (stale weights) and the *next* refit is
        # exactly what we want to push fresh weights to it. The data plane
        # skips joining shards (see ``_pick_shard``), so no client traffic
        # hits stale weights. Initial bring-up's joining state is also
        # cleared by the health poller without depending on this gate.
        return True, "ok"

    def shard_count_ready(self) -> int:
        return sum(1 for s in self._shards.values() if s.status == "ready")

    def shard_count_alive_for_collective(self) -> int:
        """Number of shards present in the cross-cluster NCCL group.

        Includes ``ready`` AND ``joining`` (RL-412 scale-up: a freshly
        added shard joins NCCL via ``init_collective`` before being
        promoted to ``ready`` by the next successful refit). Excludes
        ``cordoned``/``draining`` (the actor is being killed) and any
        shard already popped on ``remove_shard``.
        """
        return sum(
            1
            for s in self._shards.values()
            if s.status in ("ready", "joining")
        )

    def _backfill_target_count(self) -> int:
        """Target number of gen shards to maintain. Explicit override, else the
        bootstrap cohort size (recorded by ``register_shards``)."""
        if self._backfill_target is not None:
            return int(self._backfill_target)
        return int(self._total_shards_at_bootstrap)

    def _backfill_deficit(self) -> int:
        """How many shards short of target, counting shards already booting.

        ``shard_count_alive_for_collective`` counts ``ready`` + ``joining``
        shards; ``_inflight_adds`` counts add_shard calls whose PG is still
        booting (not yet ``joining``). Both are shard counts, matching the
        shard-count target."""
        in_world = self.shard_count_alive_for_collective() + self._inflight_adds
        return max(0, self._backfill_target_count() - in_world)

    def _reconcile_backfill(self) -> int:
        """If under target, launch up to ``backfill_max_concurrent`` (minus
        already in-flight) ``add_shard`` tasks to restore the gen world —
        regardless of whether the missing shards were killed or poison-evicted.

        Non-blocking: each add_shard is fired as a background task (its PG boot
        happens off the request path). ``_inflight_adds`` is reserved here and
        released in ``_backfill_one`` so repeated ticks don't over-provision.
        Returns the number of tasks launched this tick."""
        if not self.auto_backfill or self._generation is None:
            return 0
        deficit = self._backfill_deficit()
        slots = max(0, self.backfill_max_concurrent - self._inflight_adds)
        launch = min(deficit, slots)
        for _ in range(launch):
            self._inflight_adds += 1
            asyncio.create_task(self._backfill_one())
        if launch:
            print(
                f"[router] auto-backfill: launching {launch} add_shard(s) "
                f"(deficit={deficit}, in_flight={self._inflight_adds}, "
                f"target={self._backfill_target_count()})",
                flush=True,
            )
        return launch

    async def _backfill_one(self) -> None:
        try:
            await self.add_shard(reason="auto-backfill")
        except Exception as e:  # noqa: BLE001 - keep the reconciler alive
            print(f"[router] auto-backfill add_shard failed: {e}", flush=True)
        finally:
            self._inflight_adds -= 1

    def current_gen_world_size(self) -> int:
        """World size to advertise to the train side for init_collective.

        Sums per-shard world size over shards that participate in the
        cross-cluster NCCL group (``ready`` + ``joining``). Cordoned /
        draining shards are excluded — their actors are being torn down
        on the next ``init_collective``.
        """
        return self.shard_count_alive_for_collective() * self._per_shard_world_size

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
            "backfill_target": self._backfill_target_count(),
            "backfill_in_flight": self._inflight_adds,
            "backfill_deficit": self._backfill_deficit(),
            "cumulative_shards_removed": self._cumulative_shards_removed,
            "cumulative_shards_added": self._cumulative_shards_added,
            "per_shard_world_size": self._per_shard_world_size,
            "current_gen_world_size": self.current_gen_world_size(),
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
