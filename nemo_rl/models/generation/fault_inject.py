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
"""Fault injection driver for the RL-412 demo.

Triggers a chosen failure mode against a target vLLM DP shard so we can
demonstrate the router's cordon-and-replay path on a live cluster:

  * ``http-error`` — POST /admin/cordon on the router. The underlying
    vLLM is left alone; the router temporarily marks the shard
    ``cordoned`` so all subsequent requests replay elsewhere. After
    ``recover_after_s`` we /admin/uncordon. No NCCL re-init.
  * ``actor-kill`` — POST /admin/remove_shard on the router. The router
    ``ray.kill``s the shard's actors, frees its placement group, and calls
    ``reset_collective()`` on surviving workers. World size shrinks; train
    driver picks it up via ``ensure_collective_synced`` on the next refit.
  * ``pod-kill`` — As above, plus a ``kubernetes.client`` call to
    ``delete_namespaced_pod`` on the underlying pod (requires
    ``pods.delete`` RBAC on the gen-namespace SA). Demonstrates that
    KubeRay's autoscaler v2 reclaims the GPU when the PG releases.

Invoked via the demo recipe by setting ``fault_inject.enabled=true``;
runs as a Ray actor on the gen cluster head so it can address the router
locally without DNS plumbing.
"""

from __future__ import annotations

import random
import time
from typing import Any, Literal, Optional

import ray
import requests

FaultMode = Literal["http-error", "actor-kill", "pod-kill"]


@ray.remote(num_cpus=0)
class FaultInjector:
    """Ray actor that fires fault(s) against the gen cluster.

    Default: one-shot — sleep ``trigger_after_s``, fire, optionally
    recover, then exit. Set ``repeat_every_s`` to loop indefinitely
    (kill → recover → wait → kill → ...) for stress testing.
    """

    def __init__(
        self,
        router_url: str,
        mode: FaultMode,
        target_shard: str,
        trigger_after_s: float = 60.0,
        recover_after_s: Optional[float] = None,
        pod_namespace: Optional[str] = None,
        pod_name: Optional[str] = None,
        recover: bool = False,
        repeat_every_s: Optional[float] = None,
        rotate_target: bool = True,
        max_cycles: Optional[int] = None,
        new_shard_grace_period_s: float = 300.0,
        burst_size: int = 1,
        burst_every_n_cycles: int = 0,
        burst_size_random_max: Optional[int] = None,
        recover_batch_size: Optional[int] = None,
        recover_batch_delay_s: float = 600.0,
    ):
        """Initialize a fault injector.

        Args:
            router_url: Base URL of the unified GenerationRouter
                (e.g. http://...:8089).
            mode: ``http-error`` (cordon), ``actor-kill`` (remove_shard), or
                ``pod-kill`` (remove_shard + delete the pod).
            target_shard: DP shard id (e.g. ``dp-0``) to fault.
            trigger_after_s: Sleep before firing the fault.
            recover_after_s: For ``http-error``, time to sleep before
                uncordon. For ``actor-kill``/``pod-kill``, time to sleep
                before calling ``/admin/add_shard`` (only when
                ``recover=True``). When ``None``, no recovery occurs.
            pod_namespace, pod_name: Optional explicit pod identity for
                ``pod-kill``; otherwise resolved from the router /shards.
            recover: When ``True`` and ``mode`` is ``actor-kill`` or
                ``pod-kill``, wait ``recover_after_s`` seconds after the
                kill then call ``/admin/add_shard`` to bring up a
                replacement DP shard.
            repeat_every_s: When set (and > 0), after each kill+recover
                cycle completes, sleep this many seconds and start
                another cycle. Stress-tests the FT path under repeated
                churn. Use ``None`` (default) for one-shot fire.
            rotate_target: When ``True`` and ``repeat_every_s`` is set,
                each cycle queries ``/shards`` and picks the lowest-
                numbered ready shard as the next target (the previous
                target's id is gone after recover, so we'd OOB on cycle
                2 without rotation). When ``False``, stays on the same
                ``target_shard`` (which only makes sense for
                ``http-error``, where the shard isn't actually removed).
            max_cycles: Cap the number of repeat cycles. ``None`` =
                unbounded (the actor only exits when its parent daemon
                exits).
        """
        self.router_url = router_url.rstrip("/")
        self.mode = mode
        self.target_shard = target_shard
        self.trigger_after_s = trigger_after_s
        self.recover_after_s = recover_after_s
        self.pod_namespace = pod_namespace
        self.pod_name = pod_name
        self.recover = recover
        self.repeat_every_s = repeat_every_s
        self.rotate_target = rotate_target
        self.max_cycles = max_cycles
        # Grace period: a freshly-promoted shard is excluded from kill
        # rotation for this many seconds after first observed as ``ready``.
        # Without this, the rotation immediately re-targets the just-
        # joined replacement on the next cycle, so each new pod barely
        # serves a single training step before being killed again. The
        # grace period gives every new pod at least a few step-times of
        # productive contribution to the rollout pool.
        self.new_shard_grace_period_s = float(new_shard_grace_period_s)
        # Multi-shard burst: every N cycles, kill ``burst_size`` shards
        # in quick succession (still respecting the grace period and
        # _wait_for_steady_state between bursts). Setting both to the
        # defaults (size=1, every=0) preserves the legacy single-kill
        # cadence. ``burst_every_n_cycles=3`` means cycles 3, 6, 9, …
        # fire bursts; other cycles fire single kills.
        self.burst_size = max(1, int(burst_size))
        self.burst_every_n_cycles = max(0, int(burst_every_n_cycles))
        # When set, each cycle's burst size is a uniform random pick from
        # ``[1, burst_size_random_max]`` instead of the static
        # ``burst_size`` / ``burst_every_n_cycles`` schedule. Useful for
        # stress-testing the FT path against varying burst pressure
        # without any deterministic schedule the system can game.
        self.burst_size_random_max = (
            max(1, int(burst_size_random_max))
            if burst_size_random_max is not None
            else None
        )
        self.recover_batch_size = (
            max(1, int(recover_batch_size))
            if recover_batch_size is not None
            else None
        )
        self.recover_batch_delay_s = float(recover_batch_delay_s)
        # Use SystemRandom (os.urandom-backed) so we don't inherit a
        # deterministic seed that some upstream module (torch / numpy /
        # vLLM) may have called ``random.seed()`` with — observed three
        # consecutive ``randint(1, 4)`` calls all returning 2, which is
        # 1/64 probability under true uniform sampling. SystemRandom
        # bypasses the global state.
        self._rng = random.SystemRandom()
        # Tracks first-seen-ready monotonic timestamp per shard_id.
        # Populated lazily by _pick_first_ready_shard via /shards probes.
        self._first_seen_ready: dict[str, float] = {}

    def run(self) -> dict[str, Any]:
        cycles: list[dict[str, Any]] = []
        cycle_n = 0
        # First cycle uses trigger_after_s; subsequent cycles fire
        # immediately after the inter-cycle sleep (repeat_every_s).
        delay_before_kill = self.trigger_after_s
        # Anchor the FIRST fault to "training has started refitting" rather
        # than a wall-clock guess from daemon start. Setup time varies wildly
        # (warm cluster reuse ~4 min vs cold Megatron-30B load ~13 min), so a
        # fixed delay would fire during setup on a slow start (startup is NOT
        # fault-tolerant yet) or after the run already finished on a fast one.
        # Block until the router reports >=1 refit, THEN apply trigger_after_s
        # as a delay measured from "training is genuinely stepping".
        self._wait_for_training_started(timeout_s=1800.0)
        while True:
            cycle_n += 1
            if cycle_n > 1:
                # Wait for the previous cycle's recover to fully settle
                # before firing the next kill. Without this gate, cycle
                # N+1's kill races cycle N's post-recover refit: the
                # train-side `init_collective(WS = old + 1)` rendezvous
                # is in-flight when we kill another peer, leaving NCCL
                # blocked forever waiting for the now-dead rank.
                #
                # "Settled" means: no `joining` shards in the router
                # (the post-recover broadcast has completed and
                # promote_all_joining flipped them to `ready`) AND the
                # refit gate is open. Bounded wait so a stuck refit
                # doesn't pin the FaultInjector forever.
                self._wait_for_steady_state(timeout_s=600)
                if self.rotate_target:
                    # Re-pick the target each loop. The previous cycle's
                    # target was removed and replaced with a new dp-N id;
                    # querying /shards always gives a current, valid id.
                    picked = self._pick_first_ready_shard()
                    if picked is not None:
                        self.target_shard = picked

            # Decide burst size for this cycle. If
            # ``burst_size_random_max`` is set, sample uniformly from
            # ``[1, burst_size_random_max]`` (random pressure each cycle).
            # Otherwise fall back to the static
            # ``burst_size`` / ``burst_every_n_cycles`` schedule.
            if self.burst_size_random_max is not None:
                this_cycle_burst = self._rng.randint(
                    1, self.burst_size_random_max
                )
            else:
                this_cycle_burst = (
                    self.burst_size
                    if self.burst_every_n_cycles > 0
                    and cycle_n % self.burst_every_n_cycles == 0
                    else 1
                )

            print(
                f"[fault-inject] cycle={cycle_n}: sleeping {delay_before_kill}s before "
                f"firing mode={self.mode} target={self.target_shard} "
                f"recover={self.recover} burst_size={this_cycle_burst}",
                flush=True,
            )
            time.sleep(delay_before_kill)

            fault_ts = time.time()
            print(
                f"[FAULT-EVENT] unix_ts={fault_ts:.6f} cycle={cycle_n} "
                f"mode={self.mode} target={self.target_shard} "
                f"burst_size={this_cycle_burst}",
                flush=True,
            )

            # Track shards killed in this burst so the rotation in
            # ``_pick_first_ready_shard`` doesn't re-pick them.
            killed_in_burst: set[str] = set()
            burst_results: list[dict[str, Any]] = []
            for burst_idx in range(this_cycle_burst):
                if burst_idx > 0:
                    # Re-pick from remaining ready shards, excluding
                    # the ones we just killed in this burst. No
                    # _wait_for_steady_state between burst kills —
                    # that's the point of "burst".
                    picked = self._pick_first_ready_shard(
                        exclude=killed_in_burst
                    )
                    if picked is None:
                        print(
                            f"[fault-inject] cycle={cycle_n} burst aborted "
                            f"after {burst_idx}/{this_cycle_burst} kills "
                            f"(no more eligible ready shards)",
                            flush=True,
                        )
                        break
                    self.target_shard = picked
                    print(
                        f"[fault-inject] cycle={cycle_n} burst kill "
                        f"{burst_idx + 1}/{this_cycle_burst}: target="
                        f"{self.target_shard}",
                        flush=True,
                    )

                # Fire just the kill here. Do NOT call _maybe_recover
                # mid-burst: the recovery path waits ``recover_after_s``
                # (~7 min on 30B) before calling /admin/add_shard, which
                # would serialize the burst into back-to-back single
                # kills with multi-minute gaps — defeating the whole
                # purpose. We accumulate kills first, then fire ONE
                # _maybe_recover-equivalent at the end of the burst
                # that brings back replacements for ALL killed shards.
                try:
                    if self.mode == "http-error":
                        kill_result = self._fire_http_error()
                    elif self.mode == "actor-kill":
                        kill_result = self._fire_actor_kill()
                    elif self.mode == "pod-kill":
                        kill_result = self._fire_pod_kill()
                    else:
                        raise ValueError(f"unknown fault mode {self.mode!r}")
                except ValueError:
                    raise
                except Exception as e:  # noqa: BLE001
                    # A single kill stalling (e.g. the router's remove_shard
                    # blocked on its lifecycle lock mid-cascade → HTTP read
                    # timeout) must NOT abort the rest of the burst or the run
                    # loop (which would skip later cycles). The pod delete in
                    # pod-kill already fired before any router call, so the
                    # fault landed; record + continue to the next burst kill.
                    print(
                        f"[fault-inject] cycle={cycle_n} burst kill "
                        f"{burst_idx + 1}/{this_cycle_burst} on "
                        f"{self.target_shard} raised {type(e).__name__}: {e} "
                        f"— continuing burst",
                        flush=True,
                    )
                    kill_result = {
                        "mode": self.mode,
                        "error": str(e),
                        "target": self.target_shard,
                    }
                killed_in_burst.add(self.target_shard)
                burst_results.append(kill_result)

            # After all kills fire, run the recovery path (sleeps
            # ``recover_after_s`` then calls /admin/add_shard once per
            # killed shard). This replaces the per-kill _maybe_recover.
            # http-error mode has its own uncordon flow inside
            # _fire_http_error, so it's already self-contained — skip.
            if self.recover and self.mode in ("actor-kill", "pod-kill"):
                if self.recover_after_s is not None:
                    print(
                        f"[fault-inject] cycle={cycle_n} sleeping "
                        f"{self.recover_after_s}s before /admin/add_shard "
                        f"x{len(killed_in_burst)} (recover from "
                        f"{self.mode} on {sorted(killed_in_burst)})",
                        flush=True,
                    )
                    time.sleep(float(self.recover_after_s))
                sorted_killed = sorted(killed_in_burst)
                print(
                    f"[RECOVERY-START] unix_ts={time.time():.6f} "
                    f"recovering {len(sorted_killed)} shards: {sorted_killed}",
                    flush=True,
                )
                batch_sz = self.recover_batch_size or len(sorted_killed)
                for batch_start in range(0, len(sorted_killed), batch_sz):
                    batch = sorted_killed[batch_start:batch_start + batch_sz]
                    if batch_start > 0:
                        print(
                            f"[fault-inject] staged recovery: sleeping "
                            f"{self.recover_batch_delay_s}s before next "
                            f"batch of {len(batch)} add_shard(s)",
                            flush=True,
                        )
                        time.sleep(self.recover_batch_delay_s)
                    for sid in batch:
                        try:
                            resp = requests.post(
                                f"{self.router_url}/admin/add_shard",
                                json={},
                                timeout=600,
                            )
                            resp.raise_for_status()
                            body = resp.json()
                            print(
                                f"[fault-inject] add_shard succeeded for "
                                f"replacement of {sid}: {body}",
                                flush=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(
                                f"[fault-inject] add_shard failed for "
                                f"replacement of {sid}: {e}",
                                flush=True,
                            )

            result = (
                burst_results[0]
                if len(burst_results) == 1
                else {"burst": burst_results, "burst_size": len(burst_results)}
            )
            cycles.append({"cycle": cycle_n, **result})

            if not self.repeat_every_s or self.repeat_every_s <= 0:
                # One-shot — return the single cycle's result directly
                # to preserve the existing single-fault contract.
                return result
            if self.max_cycles is not None and cycle_n >= self.max_cycles:
                print(
                    f"[fault-inject] hit max_cycles={self.max_cycles}; exiting",
                    flush=True,
                )
                return {"cycles": cycles}
            print(
                f"[fault-inject] cycle={cycle_n} done; sleeping "
                f"{self.repeat_every_s}s before next cycle",
                flush=True,
            )
            time.sleep(float(self.repeat_every_s))
            # After the first cycle, the kill should fire as soon as we
            # re-pick a target (no extra trigger delay).
            delay_before_kill = 0.0

    def _wait_for_training_started(
        self, timeout_s: float = 1800.0, poll_every_s: float = 5.0
    ) -> None:
        """Block until the router reports >=1 weight refit (training has
        started refitting), so the first fault is anchored to real training
        progress rather than a wall-clock guess from daemon start.

        Best-effort: if ``/refit_count`` never reports a refit within the
        window (or the endpoint is missing on an older router), we log and
        fall through to the old wall-clock behavior rather than pinning the
        actor forever.
        """
        deadline = time.monotonic() + timeout_s
        last_logged = 0.0
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{self.router_url}/refit_count", timeout=5)
                if resp.status_code == 200:
                    n = int(resp.json().get("refit_attempts", 0))
                    if n >= 1:
                        print(
                            f"[fault-inject] training started "
                            f"(refit_attempts={n}); beginning trigger countdown",
                            flush=True,
                        )
                        return
            except Exception as e:  # noqa: BLE001
                if time.monotonic() - last_logged > 30:
                    print(
                        f"[fault-inject] waiting for first refit "
                        f"(/refit_count probe: {e})",
                        flush=True,
                    )
                    last_logged = time.monotonic()
            time.sleep(poll_every_s)
        print(
            "[fault-inject] WARN: no refit observed within "
            f"{timeout_s:.0f}s; proceeding with trigger countdown anyway",
            flush=True,
        )

    def _wait_for_steady_state(
        self, timeout_s: float = 600.0, poll_every_s: float = 5.0
    ) -> bool:
        """Block until the router shows no `joining` shards AND
        ``/refit_ready`` is true.

        After a recover, the new shard sits in `joining` until the
        train side's post-recover refit completes a broadcast and the
        `/update_weights_from_collective` handler calls
        ``promote_all_joining``. Firing another kill while a shard is
        still `joining` means there's an `init_collective` rendezvous
        in flight — removing a peer mid-rendezvous deadlocks the NCCL
        group. This gate prevents that.

        Returns True when steady state is reached, False on timeout
        (which still proceeds to fire — the caller logs and accepts
        whatever consequence follows, rather than pinning the actor).
        """
        deadline = time.monotonic() + timeout_s
        last_logged = 0.0
        while time.monotonic() < deadline:
            try:
                shards = requests.get(
                    f"{self.router_url}/shards", timeout=5
                ).json()
                refit = requests.get(
                    f"{self.router_url}/refit_ready", timeout=5
                ).json()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[fault-inject] steady-state probe failed: {e}; retrying",
                    flush=True,
                )
                time.sleep(poll_every_s)
                continue
            joining = [s for s in shards if s.get("status") == "joining"]
            ready_for_refit = bool(refit.get("ready"))
            if not joining and ready_for_refit:
                print(
                    f"[fault-inject] steady state reached: "
                    f"{len(shards)} shards, all ready, refit_ready=True",
                    flush=True,
                )
                return True
            now = time.monotonic()
            if now - last_logged > 30:
                print(
                    f"[fault-inject] waiting for steady state: "
                    f"joining={[s['shard_id'] for s in joining]}, "
                    f"refit_ready={ready_for_refit}",
                    flush=True,
                )
                last_logged = now
            time.sleep(poll_every_s)
        print(
            f"[fault-inject] steady-state wait timed out after {timeout_s}s; "
            f"proceeding anyway (run may be wedged)",
            flush=True,
        )
        return False

    def _pick_first_ready_shard(
        self, exclude: Optional[set[str]] = None
    ) -> Optional[str]:
        """Query the router for a kill candidate.

        Filters:
          1. Excludes shards in the caller's ``exclude`` set (used by
             burst-mode to avoid picking the same shard twice in one
             burst).
          2. Excludes shards within ``new_shard_grace_period_s`` of
             their first-seen ready time, so freshly-promoted
             replacements get to serve traffic for at least a few
             training steps before being killed again. Falls back
             to picking from the youngest shard pool if NO shard is
             eligible (rare — would only happen if grace_period is
             longer than the full kill cycle).

        Returns the lowest-numbered eligible shard, or None.
        """
        try:
            resp = requests.get(f"{self.router_url}/shards", timeout=10)
            resp.raise_for_status()
            shards = resp.json()
        except Exception as e:  # noqa: BLE001
            print(
                f"[fault-inject] /shards lookup failed: {e}; "
                f"keeping previous target {self.target_shard}",
                flush=True,
            )
            return None
        ready = [s for s in shards if s.get("status") == "ready"]
        if not ready:
            print(
                "[fault-inject] no ready shards in router; "
                "keeping previous target",
                flush=True,
            )
            return None

        # Update first_seen_ready for newly-observed ready shards;
        # forget shards that have left the fleet so a re-add starts a
        # fresh grace period.
        now = time.monotonic()
        live_ids = {s["shard_id"] for s in ready}
        for sid in list(self._first_seen_ready.keys()):
            if sid not in live_ids:
                self._first_seen_ready.pop(sid, None)
        for s in ready:
            sid = s["shard_id"]
            self._first_seen_ready.setdefault(sid, now)

        exclude = exclude or set()

        def _eligible(s: dict[str, Any]) -> bool:
            sid = s["shard_id"]
            if sid in exclude:
                return False
            age_s = now - self._first_seen_ready.get(sid, now)
            return age_s >= self.new_shard_grace_period_s

        def _idx(s: dict[str, Any]) -> int:
            sid = s.get("shard_id", "dp-0")
            try:
                return int(sid.split("-")[-1])
            except ValueError:
                return 0

        eligible = [s for s in ready if _eligible(s)]
        if not eligible:
            # All ready shards are in the grace window. Fall back to
            # picking the OLDEST one (longest first_seen_ready) so we
            # at least don't go round-robin on the freshest replacement.
            print(
                "[fault-inject] all ready shards are within grace "
                f"period ({self.new_shard_grace_period_s:.0f}s); "
                "falling back to oldest-ready",
                flush=True,
            )
            ready.sort(key=lambda s: self._first_seen_ready.get(s["shard_id"], now))
            return ready[0]["shard_id"] if exclude.symmetric_difference(
                {s["shard_id"] for s in ready}
            ) else None
        eligible.sort(key=_idx)
        return eligible[0]["shard_id"]

    def _maybe_recover(
        self, kill_result: dict[str, Any], kind: str
    ) -> dict[str, Any]:
        """If ``recover`` is set, wait then POST /admin/add_shard.

        Returns the merged kill+add result so the caller can see both.
        """
        if not self.recover:
            return kill_result
        delay = self.recover_after_s if self.recover_after_s is not None else 60.0
        print(
            f"[fault-inject] sleeping {delay}s before /admin/add_shard "
            f"(recover from {kind} on {self.target_shard})",
            flush=True,
        )
        time.sleep(float(delay))
        try:
            resp = requests.post(
                f"{self.router_url}/admin/add_shard",
                json={"reason": f"fault-inject recover from {kind}"},
                timeout=600,  # vLLM startup on a fresh pod can take ~3-5min
            )
            resp.raise_for_status()
            add_body = resp.json()
            print(
                f"[fault-inject] add_shard succeeded: {add_body}",
                flush=True,
            )
            return {**kill_result, "add_shard": add_body, "recovered": True}
        except Exception as e:  # noqa: BLE001 - record + continue, don't crash actor
            print(f"[fault-inject] add_shard raised {e}", flush=True)
            return {**kill_result, "add_shard_error": str(e), "recovered": False}

    # ------------------------------------------------------------------

    def _fire_http_error(self) -> dict[str, Any]:
        resp = requests.post(
            f"{self.router_url}/admin/cordon",
            json={"shard_id": self.target_shard, "reason": "fault-inject http-error"},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[fault-inject] cordoned {self.target_shard}: {resp.json()}", flush=True)
        if self.recover_after_s and self.recover_after_s > 0:
            time.sleep(self.recover_after_s)
            resp = requests.post(
                f"{self.router_url}/admin/uncordon",
                json={"shard_id": self.target_shard},
                timeout=10,
            )
            resp.raise_for_status()
            print(
                f"[fault-inject] uncordoned {self.target_shard}: {resp.json()}",
                flush=True,
            )
        return {"mode": "http-error", "target_shard": self.target_shard}

    def _fire_actor_kill(self, timeout: float = 60.0) -> dict[str, Any]:
        # Pass drain_timeout_s=5 so in-flight rollouts on the dying
        # shard fail fast (~5s) instead of blocking up to 30s for
        # graceful drain. The proxy replays the failed requests on
        # surviving shards, which is faster end-to-end than waiting
        # for the dying shard's vLLM to finish them. Cuts the
        # step-time bump on kill+recover from ~97s to ~30-40s.
        #
        # ``timeout`` is bounded by the caller: pod-kill passes a short
        # value because the remove is best-effort bookkeeping (the pod is
        # already deleted; the health poll reaps router-side state), and a
        # long block here under cascade load would serialize a burst.
        resp = requests.post(
            f"{self.router_url}/admin/remove_shard",
            json={
                "shard_id": self.target_shard,
                "reason": "fault-inject actor-kill",
                "drain_timeout_s": 5.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        print(f"[fault-inject] removed {self.target_shard}: {body}", flush=True)
        return {"mode": "actor-kill", "result": body}

    def _fire_pod_kill(self) -> dict[str, Any]:
        # Resolve the target pod by IP before tearing things down on the
        # router (after remove_shard the shard URL is gone). When pod_name
        # / pod_namespace are not pre-set on the actor, look them up from
        # the router's /shards entry (URL → IP) and the k8s pod list.
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        v1 = client.CoreV1Api()

        ns = self.pod_namespace
        pod_name = self.pod_name
        if not pod_name:
            shard_url = self._lookup_shard_url(self.target_shard)
            if not shard_url:
                raise RuntimeError(
                    f"pod-kill: could not find shard {self.target_shard} on router"
                )
            shard_ip = shard_url.split("//", 1)[1].split(":", 1)[0]
            # Default namespace = whatever the gen daemon is running in.
            if not ns:
                try:
                    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
                        ns = f.read().strip()
                except Exception:
                    ns = "default"
            pods = v1.list_namespaced_pod(
                namespace=ns,
                label_selector="ray.io/node-type=worker",
            )
            for p in pods.items:
                if p.status.pod_ip == shard_ip:
                    pod_name = p.metadata.name
                    break
            if not pod_name:
                raise RuntimeError(
                    f"pod-kill: no pod in {ns} matched shard IP {shard_ip}"
                )
            print(
                f"[fault-inject] resolved {self.target_shard} → "
                f"{ns}/{pod_name} (IP {shard_ip})",
                flush=True,
            )

        # Delete the pod FIRST — that is the actual fault. Doing it before
        # the router-side remove_shard means a burst of kills lands the real
        # pod deaths immediately even when the router is busy (its lifecycle
        # lock is held mid-cascade and remove_shard would block for the full
        # HTTP timeout). The router's health poll detects the dead pod within
        # ``failure_threshold`` ticks and reaps its table entry; the
        # best-effort remove below just makes that immediate when the router
        # is free. With minReplicas=0 the autoscaler reclaims the freed GPU
        # once the PG releases.
        try:
            v1.delete_namespaced_pod(
                name=pod_name,
                namespace=ns,
                grace_period_seconds=0,
            )
            print(
                f"[fault-inject] deleted pod {ns}/{pod_name}",
                flush=True,
            )
        except client.exceptions.ApiException as e:
            print(f"[fault-inject] pod delete raised {e}", flush=True)

        # Best-effort router bookkeeping with a SHORT timeout: under cascade
        # load the lifecycle lock may be held, so don't block the burst on it
        # — the health poll already covers detection. A timeout/error here is
        # non-fatal (caught) so the burst proceeds to the next kill.
        try:
            actor_kill_result = self._fire_actor_kill(timeout=10.0)
        except Exception as e:  # noqa: BLE001
            print(
                f"[fault-inject] pod-kill: best-effort remove_shard for "
                f"{self.target_shard} failed ({type(e).__name__}: {e}); "
                f"health poll will reap the dead pod",
                flush=True,
            )
            actor_kill_result = {"mode": "actor-kill", "error": str(e)}
        return {
            "mode": "pod-kill",
            "actor_kill": actor_kill_result,
            "pod": f"{ns}/{pod_name}",
        }

    def _lookup_shard_url(self, shard_id: str) -> Optional[str]:
        try:
            resp = requests.get(f"{self.router_url}/shards", timeout=10)
            resp.raise_for_status()
            for entry in resp.json():
                if entry.get("shard_id") == shard_id:
                    return entry.get("url")
        except Exception as e:  # noqa: BLE001
            print(f"[fault-inject] /shards lookup failed: {e}", flush=True)
        return None


def maybe_launch_fault_injector(
    config: dict[str, Any],
    router_url: str,
) -> list[ray.ObjectRef]:
    """Convenience hook for the gen daemon's main(): if the recipe's
    ``fault_inject.enabled`` is true, spawn one FaultInjector actor per
    scheduled fault and return their run-futures. Caller does NOT need
    to ``ray.get()`` — the actors run independently and print their
    outcomes.

    Two config shapes:
      1) Single fault (legacy)::

           fault_inject:
             enabled: true
             mode: pod-kill
             target_shard: dp-0
             trigger_after_s: 240

      2) Multiple faults staggered in time::

           fault_inject:
             enabled: true
             mode: pod-kill
             schedule:
               - target_shard: dp-0
                 trigger_after_s: 240
               - target_shard: dp-1
                 trigger_after_s: 480

         ``mode`` / ``recover_after_s`` / ``pod_namespace`` / ``pod_name``
         from the top-level ``fault_inject`` block apply to every entry
         unless overridden inline.
    """
    fi_cfg = (config or {}).get("fault_inject") or {}
    if not fi_cfg.get("enabled"):
        return []

    schedule = fi_cfg.get("schedule")
    if not schedule:
        # Back-compat: a single-fault config becomes a one-entry schedule.
        schedule = [
            {
                "target_shard": fi_cfg.get("target_shard", "dp-0"),
                "trigger_after_s": fi_cfg.get("trigger_after_s", 60),
            }
        ]

    # Pin all FaultInjector actors to the daemon's node (the gen head pod,
    # not the GPU shard pods). Without this, Ray can place the actor onto
    # any node with spare CPU — including a shard pod that's about to be
    # the *target* of a pod-kill fault. When the actor lands on its own
    # target, `delete_namespaced_pod` kills the actor before
    # ``_maybe_recover`` can call ``/admin/add_shard``, breaking the
    # scale-up half of the FT demo.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    head_node_id = ray.get_runtime_context().get_node_id()

    futures: list[ray.ObjectRef] = []
    for entry in schedule:
        target = entry.get("target_shard", "dp-0")
        trigger = float(entry.get("trigger_after_s", 60))
        mode = entry.get("mode", fi_cfg.get("mode", "http-error"))
        # The kubernetes client isn't in the base nemo-rl image's venv, so
        # install it into the actor's runtime_env. Required for pod-kill;
        # harmless for actor-kill / http-error.
        #
        # NOTE: do NOT pass lifetime="detached". A detached actor outlives
        # the gen daemon and keeps references to the router + per-shard
        # actor handles + placement groups, so Ray cannot GC those when
        # the daemon exits — `nrl-k8s run --replace` then races against
        # zombie PGs and the autoscaler keeps both old + new pods alive.
        recover = bool(entry.get("recover", fi_cfg.get("recover", False)))
        repeat_every_s = entry.get(
            "repeat_every_s", fi_cfg.get("repeat_every_s")
        )
        rotate_target = bool(
            entry.get("rotate_target", fi_cfg.get("rotate_target", True))
        )
        max_cycles = entry.get("max_cycles", fi_cfg.get("max_cycles"))
        new_shard_grace_period_s = float(
            entry.get(
                "new_shard_grace_period_s",
                fi_cfg.get("new_shard_grace_period_s", 300.0),
            )
        )
        burst_size = int(
            entry.get("burst_size", fi_cfg.get("burst_size", 1))
        )
        burst_every_n_cycles = int(
            entry.get(
                "burst_every_n_cycles",
                fi_cfg.get("burst_every_n_cycles", 0),
            )
        )
        burst_size_random_max_raw = entry.get(
            "burst_size_random_max",
            fi_cfg.get("burst_size_random_max"),
        )
        burst_size_random_max = (
            int(burst_size_random_max_raw)
            if burst_size_random_max_raw is not None
            else None
        )
        recover_batch_size_raw = entry.get(
            "recover_batch_size",
            fi_cfg.get("recover_batch_size"),
        )
        recover_batch_size = (
            int(recover_batch_size_raw)
            if recover_batch_size_raw is not None
            else None
        )
        recover_batch_delay_s = float(
            entry.get(
                "recover_batch_delay_s",
                fi_cfg.get("recover_batch_delay_s", 600.0),
            )
        )
        actor = FaultInjector.options(
            name=f"fault-inject-{target}",
            runtime_env={"pip": ["kubernetes"]},
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head_node_id, soft=False
            ),
        ).remote(
            router_url=router_url,
            mode=mode,
            target_shard=target,
            trigger_after_s=trigger,
            recover_after_s=entry.get(
                "recover_after_s", fi_cfg.get("recover_after_s")
            ),
            pod_namespace=entry.get(
                "pod_namespace", fi_cfg.get("pod_namespace")
            ),
            pod_name=entry.get("pod_name", fi_cfg.get("pod_name")),
            recover=recover,
            repeat_every_s=repeat_every_s,
            rotate_target=rotate_target,
            max_cycles=max_cycles,
            new_shard_grace_period_s=new_shard_grace_period_s,
            burst_size=burst_size,
            burst_every_n_cycles=burst_every_n_cycles,
            burst_size_random_max=burst_size_random_max,
            recover_batch_size=recover_batch_size,
            recover_batch_delay_s=recover_batch_delay_s,
        )
        print(
            f"[fault-inject] launched: mode={mode}, target={target}, "
            f"trigger_after_s={trigger}, recover={recover}, "
            f"repeat_every_s={repeat_every_s}, "
            f"rotate_target={rotate_target}, max_cycles={max_cycles}, "
            f"new_shard_grace_period_s={new_shard_grace_period_s}, "
            f"burst_size={burst_size}, burst_every_n_cycles={burst_every_n_cycles}, "
            f"burst_size_random_max={burst_size_random_max}",
            flush=True,
        )
        futures.append(actor.run.remote())
    return futures
