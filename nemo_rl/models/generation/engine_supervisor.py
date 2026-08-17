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

"""Brings dead generation shards back.

Without this the fleet only shrinks: a shard lost at hour one is gone for the rest of the
run, and a job that sheds a few transient failures ends up permanently smaller. The
recovery path itself already handles both directions -- a rebuild can name more shards
than the last one -- so all that is missing is something to restart the engine and say
when it is ready.

The restart is deliberately *not* awaited by the caller. Reloading a model takes minutes,
and the control loop this runs from also drives the rollout pump, the watchdog and the
refit; blocking it for a restart would stall the training that the surviving shards are
still perfectly able to do.

Handover to the rest of the system is through fleet-health states, not through this class:

    DEAD --(restart starts)--> RESTARTING --(engine up)--> STALE --(next refit)--> HEALTHY

``RESTARTING`` is absent from collectives, so a rebuild that happens mid-restart correctly
leaves the shard out. ``STALE`` is *present* but not serving, which is what lets the next
refit write current weights into it before it takes traffic again.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Optional

from nemo_rl.models.generation.fleet_health import GenerationFleetHealth, ShardState


class EngineSupervisor:
    """Drives restarts of dead generation shards, one background task per shard."""

    def __init__(
        self,
        generation: Any,
        monitor: GenerationFleetHealth,
    ) -> None:
        self._generation = generation
        self._monitor = monitor
        self._in_flight: dict[int, asyncio.Task] = {}
        self._restarts_started = 0
        self._restarts_succeeded = 0
        self._restarts_failed = 0

    def metrics(self) -> dict[str, float]:
        return {
            "supervisor/restarts_started": float(self._restarts_started),
            "supervisor/restarts_succeeded": float(self._restarts_succeeded),
            "supervisor/restarts_failed": float(self._restarts_failed),
            "supervisor/restarts_in_flight": float(len(self._in_flight)),
        }

    def tick(self) -> None:
        """Start a restart for any shard that needs one. Returns immediately.

        Safe to call on every watchdog tick: a shard already being restarted is skipped,
        and a shard whose attempts are exhausted has been RETIRED by the monitor and is
        no longer DEAD, so it is never picked up again.
        """
        for shard_idx in self._restartable_shards():
            self._begin_restart(shard_idx)

    def _restartable_shards(self) -> list[int]:
        return [
            health.dp_shard_idx
            for health in self._monitor.snapshot()
            if health.state is ShardState.DEAD
            and health.dp_shard_idx not in self._in_flight
        ]

    def _begin_restart(self, shard_idx: int) -> None:
        # mark_restarting owns the attempt budget: it increments the count and retires
        # the shard when the budget is spent, which is also what stops this from looping
        # forever on a node that is never coming back.
        self._monitor.mark_restarting(shard_idx)
        if self._monitor.state_of(shard_idx) is ShardState.RETIRED:
            print(
                f"  supervisor: shard {shard_idx} retired, not restarting again",
                flush=True,
            )
            return

        self._restarts_started += 1
        print(f"  supervisor: restarting generation shard {shard_idx}", flush=True)
        task = asyncio.get_running_loop().create_task(self._restart(shard_idx))
        self._in_flight[shard_idx] = task
        task.add_done_callback(partial(self._forget, shard_idx))

    def _forget(self, shard_idx: int, task: "asyncio.Task") -> None:
        """Drop the finished task so a later tick can retry this shard."""
        del task
        self._in_flight.pop(shard_idx, None)

    async def _restart(self, shard_idx: int) -> None:
        try:
            # to_thread: every step below is a blocking Ray call, and this coroutine
            # shares a loop with the rollout pump and the watchdog.
            url = await asyncio.to_thread(self._generation.restart_shard, shard_idx)
        except Exception as e:  # noqa: BLE001 - a failed restart must not kill the run
            self._restarts_failed += 1
            print(f"  supervisor: shard {shard_idx} restart failed: {e!r}", flush=True)
            # Back to DEAD rather than stuck in RESTARTING, so the next tick can retry
            # until the attempt budget retires it. Not report_failure: probes are ignored
            # for non-serving states, so that would leave it stuck.
            self._monitor.mark_restart_failed(shard_idx)
            return

        self._restarts_succeeded += 1
        # STALE, not HEALTHY: the engine is up but holds whatever weights it loaded from
        # disk. It is eligible for the next refit and not for traffic until that refit
        # lands, which is exactly the ordering that keeps stale weights out of rollouts.
        self._monitor.mark_loaded(shard_idx, base_url=url)
        print(f"  supervisor: shard {shard_idx} back up at {url}", flush=True)

    async def drain(self, timeout_s: Optional[float] = None) -> None:
        """Wait for in-flight restarts, for orderly shutdown."""
        if not self._in_flight:
            return
        await asyncio.wait(list(self._in_flight.values()), timeout=timeout_s)
