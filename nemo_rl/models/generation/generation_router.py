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
"""GenerationRouter — shard table + control plane for disaggregated vLLM.

Owns the authoritative shard table (health poller, NCCL lifecycle, fault
eviction) and exposes async control-plane methods (``run_init_collective``,
``run_update_weights_from_collective``, etc.) that the training side calls
directly via Ray — no HTTP required when everything runs in the same cluster.

The health poller still uses aiohttp to probe per-shard vLLM HTTP endpoints
(``/openapi.json``), which is a lightweight in-cluster HTTP call, not an
inter-cluster hop.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import aiohttp

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
        *,
        generation: Any = None,
        health_poll_interval_s: float = _DEFAULT_HEALTH_INTERVAL_S,
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        join_success_threshold: int = _DEFAULT_JOIN_SUCCESS_THRESHOLD,
        reset_collective_timeout_s: float = _DEFAULT_RESET_COLLECTIVE_TIMEOUT_S,
        joinable_min_age_s: float = _DEFAULT_JOINABLE_MIN_AGE_S,
        auto_recover: bool = True,
    ):
        self.health_poll_interval_s = health_poll_interval_s
        self.health_timeout_s = health_timeout_s
        self.failure_threshold = failure_threshold
        self.join_success_threshold = join_success_threshold
        # Unproven joining shards must age past this before counting as joinable (cold route warm-up).
        self._joinable_min_age_s = float(joinable_min_age_s)
        self._reset_collective_timeout_s = reset_collective_timeout_s
        # When True, the health poll reconciler restores lost shards automatically.
        self.auto_recover = auto_recover
        # Bootstrap fleet size — reconciler restores toward this target.
        # Set by register_shards(); not updated on add/remove so it always
        # reflects the intended fleet size.
        self._target_shard_count: int = 0
        # add_shard calls launched by the reconciler that haven't yet registered
        # a joining shard. Counted against the deficit so consecutive ticks
        # don't over-provision.
        self._inflight_adds: int = 0

        # Tracks when the joinable shard count last changed; used by the trainer's debounce logic.
        self._joinable_changed_at: float = time.monotonic()
        self._last_joinable_count: int = 0

        # Bumped when an in-comm shard is removed; trainer re-inits proactively instead of
        # reusing a wedged group (handles the "backfill restored count" case).
        self._comm_reset_epoch: int = 0

        self._shards: dict[str, ShardEntry] = {}
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
        # Set during shutdown so background tasks short-circuit instead of
        # scheduling new executor work after the event loop starts shutting down.
        self._shutting_down: bool = False

        # Background asyncio event loop (set by start_background()).
        self._loop: Optional[Any] = None

        # Fault-tolerance metrics for wandb; reset when the process restarts.
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
        # Target fleet size for the reconciler: restore toward this count on failure.
        self._target_shard_count = len(self._shards)
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
            # event loop keeps serving health checks etc.
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
            target_node_id = (self._last_fault_event or {}).get("node_id") or None

            def _do_add() -> tuple[list[Any], Any, list[int], Optional[str]]:
                return self._generation.add_dp_worker(
                    pre_append_hook=_gate_up, node_id=target_node_id
                )

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
                                # re-entrance from cordon callbacks.
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
            # Reconcile fleet size: spawn replacements for lost shards.
            try:
                await self._reconcile_recovery()
            except Exception as e:  # noqa: BLE001
                print(f"[router] reconcile recovery failed: {e}", flush=True)
            await asyncio.sleep(self.health_poll_interval_s)

    async def _reconcile_recovery(self) -> None:
        """Restore the fleet to _target_shard_count after shard loss.

        Called once per health poll tick. Each unit of deficit launches one
        background add_shard task. _inflight_adds prevents consecutive ticks
        from over-provisioning while a long vLLM init is in progress.
        """
        if not self.auto_recover or self._generation is None or self._target_shard_count == 0:
            return
        alive = self.shard_count_alive_for_collective()
        deficit = self._target_shard_count - (alive + self._inflight_adds)
        if deficit <= 0:
            return
        print(
            f"[router] reconcile: deficit={deficit} "
            f"(target={self._target_shard_count}, alive={alive}, "
            f"inflight={self._inflight_adds}); launching {deficit} add_shard(s)",
            flush=True,
        )
        for _ in range(deficit):
            self._inflight_adds += 1
            asyncio.create_task(self._recover_one())

    async def _recover_one(self) -> None:
        """Launch one add_shard and decrement _inflight_adds when done."""
        try:
            await self.add_shard(reason="auto-recover")
        except Exception as e:  # noqa: BLE001
            print(f"[router] auto-recover add_shard failed: {e}", flush=True)
        finally:
            self._inflight_adds -= 1

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
        for sid, is_alive in results:
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

        # Drive remove_shard so the actor + PG come down and the pod is reclaimed.
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
    # Control-plane methods (called directly via Ray, no HTTP).
    # Each is an async coroutine that mirrors the body of the old HTTP
    # endpoint handler.  Call them via call_async() from a non-async context.
    # =====================================================================

    async def run_init_collective(
        self, ip: str, port: int, world_size: int, train_world_size: int
    ) -> dict:
        """Gen-side init_collective: rendezvous the joinable cohort.

        Returns ``{"success": True}`` on success.  On recoverable failure
        (503 in the old HTTP path) returns ``{"success": False, "error": ...}``.
        Raises ``RuntimeError`` for unrecoverable errors.
        """
        generation = self._generation
        if generation is None:
            raise RuntimeError("no generation wired")

        if self.shard_count_alive_for_collective() == 0:
            return {
                "success": False,
                "error": "no shards alive in NCCL group; retry after add_shard",
                "current_gen_world_size": 0,
                "failed_indices": [],
                "evicted_shard_ids": [],
            }

        prev_flag = self._nccl_reinit_in_progress
        self._nccl_reinit_in_progress = True
        try:
            async with self._table_lock:
                cohort_sids, cohort_indices = self._joinable_cohort()
            dispatched_sids = set(cohort_sids)

            if not cohort_indices:
                self._nccl_reinit_in_progress = prev_flag
                return {
                    "success": False,
                    "error": "no joinable shards yet; retry after warm-up",
                    "current_gen_world_size": self.current_gen_world_size(),
                    "failed_indices": [],
                    "evicted_shard_ids": [],
                }

            loop = asyncio.get_running_loop()

            def _dispatch() -> list:
                return generation._raw_init_collective(
                    ip=ip,
                    port=port,
                    world_size=world_size,
                    train_world_size=train_world_size,
                    include_worker_indices=cohort_indices,
                )

            futures = await loop.run_in_executor(None, _dispatch)

            failed_idxs, _results, exc_types, failed_fast = (
                await self._per_worker_results(futures, rendezvous_timeout_s=90.0)
            )

            if not failed_idxs:
                async with self._table_lock:
                    for sid in dispatched_sids:
                        entry = self._shards.get(sid)
                        if entry is not None:
                            entry.in_comm = True
                            entry.proven = True
                return {"success": True}

            if self._is_rendezvous_master_failure(failed_idxs, exc_types):
                print(
                    f"[router] run_init_collective: rendezvous-master failure "
                    f"(all {len(failed_idxs)} gen workers raised "
                    f"{exc_types[failed_idxs[0]]}); skipping mass-eviction, "
                    f"returning failure for retry",
                    flush=True,
                )
                return {
                    "success": False,
                    "error": "rendezvous master timed out; retry",
                    "current_gen_world_size": self.current_gen_world_size(),
                    "failed_indices": failed_idxs,
                    "evicted_shard_ids": [],
                    "rendezvous_master_failure": True,
                }

            num_evicted, evicted_ids = await self._evict_failed_workers_inline(
                failed_idxs,
                reason="init_collective: per-worker failure",
                exception_types=exc_types,
                force_evict_idxs=failed_fast,
                shard_ids_in_order=cohort_sids,
            )
            new_ws = self.current_gen_world_size()
            msg = (
                f"{len(failed_idxs)} of {len(futures)} workers failed "
                f"rendezvous; evicted {num_evicted} (ids={evicted_ids}); "
                f"current_gen_world_size={new_ws}"
            )
            print(f"[router] run_init_collective: {msg}", flush=True)
            return {
                "success": False,
                "error": msg,
                "current_gen_world_size": new_ws,
                "failed_indices": failed_idxs,
                "evicted_shard_ids": evicted_ids,
            }
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise RuntimeError(f"run_init_collective failed: {e}") from e
        finally:
            self._nccl_reinit_in_progress = prev_flag

    async def run_update_weights_from_collective(self) -> dict:
        """Gen-side update_weights_from_collective: broadcast weights to all shards.

        Returns ``{"success": True, "promoted_shards": [...]}`` on success.
        """
        generation = self._generation
        if generation is None:
            raise RuntimeError("no generation wired")

        if self.shard_count_alive_for_collective() == 0:
            return {
                "success": False,
                "error": "no shards alive in NCCL group; retry after add_shard",
                "current_gen_world_size": 0,
                "failed_indices": [],
                "evicted_shard_ids": [],
                "promoted_shards": [],
            }

        try:
            self._refit_attempts += 1

            async with self._table_lock:
                eligible_promote = self._eligible_promote_sids()

            loop = asyncio.get_running_loop()

            def _dispatch() -> list:
                return generation._raw_update_weights_from_collective()

            futures = await loop.run_in_executor(None, _dispatch)

            failed_idxs, results, exc_types, failed_fast = (
                await self._per_worker_results(futures, rendezvous_timeout_s=60.0)
            )

            if failed_idxs and self._is_rendezvous_master_failure(
                failed_idxs, exc_types
            ):
                print(
                    f"[router] run_update_weights_from_collective: "
                    f"rendezvous-master failure (all {len(failed_idxs)} "
                    f"gen workers raised {exc_types[failed_idxs[0]]}); "
                    f"skipping mass-eviction, returning failure for retry",
                    flush=True,
                )
                return {
                    "success": False,
                    "error": "rendezvous master timed out; retry",
                    "current_gen_world_size": self.current_gen_world_size(),
                    "failed_indices": failed_idxs,
                    "evicted_shard_ids": [],
                    "promoted_shards": [],
                    "rendezvous_master_failure": True,
                }

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
                print(f"[router] run_update_weights_from_collective: {msg}", flush=True)
                return {
                    "success": False,
                    "error": msg,
                    "current_gen_world_size": new_ws,
                    "failed_indices": failed_idxs,
                    "evicted_shard_ids": evicted_ids,
                    "promoted_shards": [],
                }

            success = all(r for r in results if r is not None)
            if not success:
                raise RuntimeError(
                    f"One or more workers reported broadcast failure. Results: {results}"
                )
            promoted: list[str] = self.promote_all_joining(
                eligible_sids=eligible_promote
            )
            if promoted:
                print(
                    f"[router] refit complete; promoted "
                    f"{len(promoted)} joining shard(s) -> ready: {promoted}",
                    flush=True,
                )
            return {"success": success, "promoted_shards": promoted}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise RuntimeError(f"run_update_weights_from_collective failed: {e}") from e

    async def run_reset_collective(self) -> None:
        """Gen-side reset_collective: tear down weight-sync NCCL group on all shards."""
        generation = self._generation
        if generation is None:
            return
        if self.shard_count_alive_for_collective() == 0:
            return  # no-op: nothing to tear down

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, generation._raw_reset_collective)

    async def run_prepare_for_generation(self) -> bool:
        """Gen-side prepare_for_generation."""
        generation = self._generation
        if generation is None:
            raise RuntimeError("no generation wired")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, generation.prepare_for_generation)
        return bool(result)

    async def run_finish_generation(self) -> bool:
        """Gen-side finish_generation."""
        generation = self._generation
        if generation is None:
            raise RuntimeError("no generation wired")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, generation.finish_generation)
        return bool(result)

    async def run_prepare_refit_info(self, state_dict_info: dict) -> None:
        """Gen-side prepare_refit_info: deserialise and push to all workers."""
        generation = self._generation
        if generation is None:
            raise RuntimeError("no generation wired")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: generation.prepare_refit_info(state_dict_info)
        )

    # =====================================================================
    # Background loop lifecycle
    # =====================================================================

    def start_background(self) -> None:
        """Start the health-poll loop in a background daemon thread.

        Creates a dedicated asyncio event loop, initialises the aiohttp
        session for health probing, and runs ``_health_poll_loop`` until
        the process exits.  ``call_async`` submits coroutines to this loop
        from any thread.
        """

        def _run() -> None:
            import asyncio as _asyncio

            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            self._loop = loop

            async def _main() -> None:
                connector = aiohttp.TCPConnector(
                    limit=256,
                    limit_per_host=64,
                    keepalive_timeout=60,
                )
                self._http_session = aiohttp.ClientSession(connector=connector)
                try:
                    await self._health_poll_loop()
                finally:
                    self._shutting_down = True
                    if self._http_session is not None:
                        await self._http_session.close()

            loop.run_until_complete(_main())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        print(
            f"GenerationRouter background loop starting "
            f"(shards={len(self._shards)}, ready={self.shard_count_ready()})",
            flush=True,
        )

    def call_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously from a non-async context.

        Submits ``coro`` to the background event loop started by
        ``start_background()`` and blocks until it completes.  Propagates
        exceptions raised by the coroutine.
        """
        import asyncio as _asyncio

        # Wait for the loop to be ready (start_background may still be
        # starting the thread when the first call arrives).
        deadline = time.monotonic() + 30
        while self._loop is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._loop is None:
            raise RuntimeError(
                "GenerationRouter background loop did not start within 30s"
            )
        future = _asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=600)

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

    def get_shards_list(self) -> list[dict]:
        """Return a snapshot of the shard table as a list of dicts (thread-safe read)."""
        return [
            {
                "shard_id": s.shard_id,
                "status": s.status,
                "url": s.url,
                "consecutive_failures": s.consecutive_failures,
                "consecutive_successes": s.consecutive_successes,
                "last_health_ok_age_s": None,
            }
            for s in self._shards.values()
        ]

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

