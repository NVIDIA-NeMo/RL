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
"""Fault injection for SLURM vLLM fault-tolerance testing.

Triggers a chosen failure mode against a target vLLM DP shard to exercise
the router's cordon-and-remove path on a live training run.

Modes:
  * ``cordon``      — cordon the shard in the router (simulate transient
                      health failure). After ``recover_after_s`` the shard
                      is uncordoned. No NCCL re-init; tests the proxy replay
                      path only.
  * ``actor-kill``  — call router.remove_shard(), which ray.kills the shard's
                      actors and triggers NCCL world-size shrink on next refit.
                      The router's reconciler automatically spawns a replacement.
  * ``ray-kill``    — directly ray.kill the shard's leader actor and let the
                      health poll detect it naturally. More realistic than
                      actor-kill since it exercises the full detection path.
                      Replacement is also spawned automatically by the reconciler.

Recovery (scale-up) after actor-kill / ray-kill is handled automatically by
the GenerationRouter's reconciler loop — no explicit ``recover`` flag needed.
For cordon mode, ``recover_after_s`` controls when the shard is uncordoned.

Usage — add to the recipe YAML:

    fault_inject:
      enabled: true
      mode: actor-kill
      target_shard: dp-0
      trigger_after_s: 120

Multiple staggered faults:

    fault_inject:
      enabled: true
      mode: actor-kill
      schedule:
        - target_shard: dp-0
          trigger_after_s: 120
        - target_shard: dp-1
          trigger_after_s: 360

Then call ``maybe_launch_fault_injector(config, vllm_generation)`` from the
training setup after VllmGeneration is initialized. The injector runs in a
background daemon thread and does not block training.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Literal, Optional

FaultMode = Literal["cordon", "actor-kill", "ray-kill"]
TriggerOn = Literal["time", "during_generation", "during_refit"]


class FaultInjector:
    """Injects faults against a live VllmGeneration instance.

    Runs in a background daemon thread. All router interactions go through
    ``vllm_gen._router.call_async()`` (direct in-process call, no HTTP).
    """

    def __init__(
        self,
        vllm_gen: Any,
        mode: FaultMode,
        target_shard: str,
        trigger_after_s: float = 60.0,
        trigger_on: TriggerOn = "time",
        recover_after_s: Optional[float] = None,
        repeat_every_s: Optional[float] = None,
        rotate_target: bool = True,
        max_cycles: Optional[int] = None,
        new_shard_grace_period_s: float = 300.0,
        burst_size: int = 1,
        burst_every_n_cycles: int = 0,
        burst_size_random_max: Optional[int] = None,
    ):
        self._gen = vllm_gen
        self.mode = mode
        self.target_shard = target_shard
        self.trigger_after_s = trigger_after_s
        self.trigger_on = trigger_on
        self.recover_after_s = recover_after_s
        self.repeat_every_s = repeat_every_s
        self.rotate_target = rotate_target
        self.max_cycles = max_cycles
        self.new_shard_grace_period_s = float(new_shard_grace_period_s)
        self.burst_size = max(1, int(burst_size))
        self.burst_every_n_cycles = max(0, int(burst_every_n_cycles))
        self.burst_size_random_max = (
            max(1, int(burst_size_random_max)) if burst_size_random_max is not None else None
        )
        self._rng = random.SystemRandom()
        self._first_seen_ready: dict[str, float] = {}

    @property
    def _router(self):
        return self._gen._router

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True, name="fault-injector")
        t.start()
        return t

    def run(self) -> dict[str, Any]:
        cycles: list[dict[str, Any]] = []
        cycle_n = 0
        delay_before_kill = self.trigger_after_s
        self._wait_for_training_started(timeout_s=1800.0)
        while True:
            cycle_n += 1
            if cycle_n > 1:
                self._wait_for_steady_state(timeout_s=600)
                if self.rotate_target:
                    picked = self._pick_first_ready_shard()
                    if picked is not None:
                        self.target_shard = picked

            if self.burst_size_random_max is not None:
                this_cycle_burst = self._rng.randint(1, self.burst_size_random_max)
            else:
                this_cycle_burst = (
                    self.burst_size
                    if self.burst_every_n_cycles > 0 and cycle_n % self.burst_every_n_cycles == 0
                    else 1
                )

            print(
                f"[fault-inject] cycle={cycle_n}: waiting to fire "
                f"(trigger_on={self.trigger_on}, delay={delay_before_kill}s) "
                f"mode={self.mode} target={self.target_shard} "
                f"burst_size={this_cycle_burst}",
                flush=True,
            )
            self._wait_for_trigger(delay_before_kill)

            fault_ts = time.time()
            print(
                f"[FAULT-EVENT] unix_ts={fault_ts:.6f} cycle={cycle_n} "
                f"mode={self.mode} target={self.target_shard} "
                f"burst_size={this_cycle_burst}",
                flush=True,
            )

            killed_in_burst: set[str] = set()
            burst_results: list[dict[str, Any]] = []
            for burst_idx in range(this_cycle_burst):
                if burst_idx > 0:
                    picked = self._pick_first_ready_shard(exclude=killed_in_burst)
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
                        f"{burst_idx + 1}/{this_cycle_burst}: target={self.target_shard}",
                        flush=True,
                    )

                try:
                    if self.mode == "cordon":
                        kill_result = self._fire_cordon()
                    elif self.mode == "actor-kill":
                        kill_result = self._fire_actor_kill()
                    elif self.mode == "ray-kill":
                        kill_result = self._fire_ray_kill()
                    else:
                        raise ValueError(f"unknown fault mode {self.mode!r}")
                except ValueError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[fault-inject] cycle={cycle_n} burst kill "
                        f"{burst_idx + 1}/{this_cycle_burst} on "
                        f"{self.target_shard} raised {type(e).__name__}: {e} "
                        f"— continuing burst",
                        flush=True,
                    )
                    kill_result = {"mode": self.mode, "error": str(e), "target": self.target_shard}
                killed_in_burst.add(self.target_shard)
                burst_results.append(kill_result)

            result = (
                burst_results[0]
                if len(burst_results) == 1
                else {"burst": burst_results, "burst_size": len(burst_results)}
            )
            cycles.append({"cycle": cycle_n, **result})

            if not self.repeat_every_s or self.repeat_every_s <= 0:
                return result
            if self.max_cycles is not None and cycle_n >= self.max_cycles:
                print(f"[fault-inject] hit max_cycles={self.max_cycles}; exiting", flush=True)
                return {"cycles": cycles}
            print(
                f"[fault-inject] cycle={cycle_n} done; sleeping "
                f"{self.repeat_every_s}s before next cycle",
                flush=True,
            )
            time.sleep(float(self.repeat_every_s))
            delay_before_kill = 0.0

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def _wait_for_trigger(self, delay_s: float, phase_timeout_s: float = 1800.0) -> None:
        """Wait according to trigger_on, then sleep delay_s before firing.

        - ``"time"``: plain sleep(delay_s), same as K8s behaviour.
        - ``"during_generation"``: block until VllmGeneration._generating is
          set (generate() is in progress), then sleep delay_s within that window.
        - ``"during_refit"``: block until VllmGeneration._refitting is set
          (update_weights_from_collective() is in progress), then sleep delay_s.
        """
        if self.trigger_on == "time":
            time.sleep(delay_s)
            return

        event_name = (
            "_generating" if self.trigger_on == "during_generation" else "_refitting"
        )
        event = getattr(self._gen, event_name, None)
        if event is None:
            print(
                f"[fault-inject] {event_name} event not found on generation object; "
                f"falling back to time-based trigger",
                flush=True,
            )
            time.sleep(delay_s)
            return

        print(
            f"[fault-inject] waiting for phase '{self.trigger_on}' to start "
            f"(timeout={phase_timeout_s:.0f}s) ...",
            flush=True,
        )
        fired = event.wait(timeout=phase_timeout_s)
        if not fired:
            print(
                f"[fault-inject] timed out waiting for phase '{self.trigger_on}'; "
                f"proceeding anyway",
                flush=True,
            )
        else:
            print(
                f"[fault-inject] phase '{self.trigger_on}' started; "
                f"sleeping {delay_s}s then firing",
                flush=True,
            )
        time.sleep(delay_s)

    # ------------------------------------------------------------------
    # Steady-state gates
    # ------------------------------------------------------------------

    def _wait_for_training_started(
        self, timeout_s: float = 1800.0, poll_every_s: float = 5.0
    ) -> None:
        """Block until at least one weight refit has completed (training is stepping)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                refits = getattr(self._router, "_refit_attempts", 0)
                if refits >= 1:
                    print(
                        f"[fault-inject] training started (refit_count={refits}); "
                        f"fault injection armed",
                        flush=True,
                    )
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(poll_every_s)
        print(
            "[fault-inject] timed out waiting for training to start; proceeding anyway",
            flush=True,
        )

    def _wait_for_steady_state(
        self, timeout_s: float = 600.0, poll_every_s: float = 5.0
    ) -> bool:
        """Block until no joining shards remain and refit gate is open."""
        deadline = time.monotonic() + timeout_s
        last_logged = 0.0
        while time.monotonic() < deadline:
            try:
                shards = self._router.get_shards_list()
                ready, _ = self._router.refit_ready_state()
                joining = [s for s in shards if s.get("status") == "joining"]
                if not joining and ready:
                    print(
                        f"[fault-inject] steady state: {len(shards)} shards, "
                        f"all ready, refit_ready=True",
                        flush=True,
                    )
                    return True
                now = time.monotonic()
                if now - last_logged > 30:
                    print(
                        f"[fault-inject] waiting for steady state: "
                        f"joining={[s['shard_id'] for s in joining]}, "
                        f"refit_ready={ready}",
                        flush=True,
                    )
                    last_logged = now
            except Exception as e:  # noqa: BLE001
                print(f"[fault-inject] steady-state probe failed: {e}; retrying", flush=True)
            time.sleep(poll_every_s)
        print(
            f"[fault-inject] steady-state wait timed out after {timeout_s}s; proceeding",
            flush=True,
        )
        return False

    def _pick_first_ready_shard(self, exclude: Optional[set[str]] = None) -> Optional[str]:
        """Return the lowest-numbered eligible ready shard, respecting the grace period."""
        try:
            shards = self._router.get_shards_list()
        except Exception as e:  # noqa: BLE001
            print(
                f"[fault-inject] get_shards_list failed: {e}; "
                f"keeping previous target {self.target_shard}",
                flush=True,
            )
            return None

        ready = [s for s in shards if s.get("status") == "ready"]
        if not ready:
            print("[fault-inject] no ready shards; keeping previous target", flush=True)
            return None

        now = time.monotonic()
        live_ids = {s["shard_id"] for s in ready}
        for sid in list(self._first_seen_ready.keys()):
            if sid not in live_ids:
                self._first_seen_ready.pop(sid, None)
        for s in ready:
            self._first_seen_ready.setdefault(s["shard_id"], now)

        exclude = exclude or set()

        def _eligible(s: dict[str, Any]) -> bool:
            sid = s["shard_id"]
            if sid in exclude:
                return False
            return (now - self._first_seen_ready.get(sid, now)) >= self.new_shard_grace_period_s

        def _idx(s: dict[str, Any]) -> int:
            try:
                return int(s.get("shard_id", "dp-0").split("-")[-1])
            except ValueError:
                return 0

        eligible = [s for s in ready if _eligible(s)]
        if not eligible:
            print(
                f"[fault-inject] all ready shards within grace period "
                f"({self.new_shard_grace_period_s:.0f}s); falling back to oldest-ready",
                flush=True,
            )
            ready.sort(key=lambda s: self._first_seen_ready.get(s["shard_id"], now))
            candidates = [s for s in ready if s["shard_id"] not in exclude]
            return candidates[0]["shard_id"] if candidates else None
        eligible.sort(key=_idx)
        return eligible[0]["shard_id"]

    # ------------------------------------------------------------------
    # Fault modes
    # ------------------------------------------------------------------

    def _fire_cordon(self) -> dict[str, Any]:
        """Cordon the shard (mark unhealthy in router without killing actors)."""
        self._router.call_async(
            self._router.cordon(self.target_shard, "fault-inject cordon")
        )
        print(f"[fault-inject] cordoned {self.target_shard}", flush=True)
        if self.recover_after_s and self.recover_after_s > 0:
            time.sleep(self.recover_after_s)
            self._router.call_async(self._router.uncordon(self.target_shard))
            print(f"[fault-inject] uncordoned {self.target_shard}", flush=True)
        return {"mode": "cordon", "target_shard": self.target_shard}

    def _fire_actor_kill(self) -> dict[str, Any]:
        """Call router.remove_shard() — kills actors and shrinks NCCL world."""
        result = self._router.call_async(
            self._router.remove_shard(
                self.target_shard,
                reason="fault-inject actor-kill",
                drain_timeout_s=5.0,
            )
        )
        print(f"[fault-inject] removed {self.target_shard}: {result}", flush=True)
        return {"mode": "actor-kill", "result": result}

    def _fire_ray_kill(self) -> dict[str, Any]:
        """Directly ray.kill the shard's leader actor; health poll detects it."""
        import ray

        shards = self._router.get_shards_list()
        shard = next((s for s in shards if s["shard_id"] == self.target_shard), None)
        if shard is None:
            raise RuntimeError(f"ray-kill: shard {self.target_shard} not found in router")

        # Get actor handles directly from the router's shard table.
        entry = self._router._shards.get(self.target_shard)
        if entry is None or not entry.actor_handles:
            raise RuntimeError(f"ray-kill: no actor handles for {self.target_shard}")

        for actor in entry.actor_handles:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as e:  # noqa: BLE001
                print(f"[fault-inject] ray.kill on {self.target_shard} raised {e}", flush=True)

        print(
            f"[fault-inject] ray.killed {len(entry.actor_handles)} actor(s) "
            f"for {self.target_shard}; health poll will detect and evict",
            flush=True,
        )
        return {"mode": "ray-kill", "target_shard": self.target_shard}


def maybe_launch_fault_injector(
    config: dict[str, Any],
    vllm_gen: Any,
) -> list[threading.Thread]:
    """Convenience hook: if ``fault_inject.enabled`` is true in config, spawn
    one FaultInjector background thread per scheduled fault.

    Single fault::

        fault_inject:
          enabled: true
          mode: actor-kill
          target_shard: dp-0
          trigger_after_s: 120

    Multiple staggered faults::

        fault_inject:
          enabled: true
          mode: actor-kill
          schedule:
            - target_shard: dp-0
              trigger_after_s: 120
            - target_shard: dp-1
              trigger_after_s: 360
    """
    fi_cfg = (config or {}).get("fault_inject") or {}
    if not fi_cfg.get("enabled"):
        return []

    if vllm_gen is None or getattr(vllm_gen, "_router", None) is None:
        print(
            "[fault-inject] WARNING: fault_inject.enabled=true but VllmGeneration "
            "has no router; fault injection disabled",
            flush=True,
        )
        return []

    schedule = fi_cfg.get("schedule")
    if not schedule:
        schedule = [
            {
                "target_shard": fi_cfg.get("target_shard", "dp-0"),
                "trigger_after_s": fi_cfg.get("trigger_after_s", 60),
            }
        ]

    threads: list[threading.Thread] = []
    for entry in schedule:
        target = entry.get("target_shard", "dp-0")
        trigger = float(entry.get("trigger_after_s", 60))
        mode = entry.get("mode", fi_cfg.get("mode", "actor-kill"))
        repeat_every_s = entry.get("repeat_every_s", fi_cfg.get("repeat_every_s"))
        rotate_target = bool(entry.get("rotate_target", fi_cfg.get("rotate_target", True)))
        max_cycles = entry.get("max_cycles", fi_cfg.get("max_cycles"))
        new_shard_grace_period_s = float(
            entry.get("new_shard_grace_period_s", fi_cfg.get("new_shard_grace_period_s", 300.0))
        )
        burst_size = int(entry.get("burst_size", fi_cfg.get("burst_size", 1)))
        burst_every_n_cycles = int(
            entry.get("burst_every_n_cycles", fi_cfg.get("burst_every_n_cycles", 0))
        )
        burst_size_random_max_raw = entry.get(
            "burst_size_random_max", fi_cfg.get("burst_size_random_max")
        )
        burst_size_random_max = (
            int(burst_size_random_max_raw) if burst_size_random_max_raw is not None else None
        )

        trigger_on = entry.get("trigger_on", fi_cfg.get("trigger_on", "time"))
        injector = FaultInjector(
            vllm_gen=vllm_gen,
            mode=mode,
            target_shard=target,
            trigger_after_s=trigger,
            trigger_on=trigger_on,
            recover_after_s=entry.get("recover_after_s", fi_cfg.get("recover_after_s")),
            repeat_every_s=repeat_every_s,
            rotate_target=rotate_target,
            max_cycles=max_cycles,
            new_shard_grace_period_s=new_shard_grace_period_s,
            burst_size=burst_size,
            burst_every_n_cycles=burst_every_n_cycles,
            burst_size_random_max=burst_size_random_max,
        )
        t = injector.start()
        print(
            f"[fault-inject] launched: mode={mode}, target={target}, "
            f"trigger_after_s={trigger}, repeat_every_s={repeat_every_s}, "
            f"rotate_target={rotate_target}, max_cycles={max_cycles}",
            flush=True,
        )
        threads.append(t)
    return threads
