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
import threading
import time
from functools import partial
from typing import Any, Optional

from nemo_rl.models.generation.fleet_health import GenerationFleetHealth, ShardState

# Budget for one restart, and the wait between a failed attempt and the next. Defaults
# only: the controller passes the configured values through. See
# ``async_rl.generation_fleet_health.restart_timeout_s`` / ``restart_backoff_s``.
DEFAULT_RESTART_TIMEOUT_S = 1800.0
DEFAULT_RESTART_BACKOFF_S = 60.0


class EngineSupervisor:
    """Drives restarts of dead generation shards, one background task per shard."""

    def __init__(
        self,
        generation: Any,
        monitor: GenerationFleetHealth,
        restart_timeout_s: float = DEFAULT_RESTART_TIMEOUT_S,
        restart_backoff_s: float = DEFAULT_RESTART_BACKOFF_S,
        clock: Any = time.monotonic,
    ) -> None:
        self._generation = generation
        self._monitor = monitor
        self._restart_timeout_s = restart_timeout_s
        self._restart_backoff_s = restart_backoff_s
        self._clock = clock
        self._in_flight: dict[int, asyncio.Task] = {}
        # When each shard may be tried again. Only a *failed* attempt writes here.
        self._retry_not_before: dict[int, float] = {}
        self._restarts_started = 0
        self._restarts_succeeded = 0
        self._restarts_failed = 0
        self._restarts_timed_out = 0

    def metrics(self) -> dict[str, float]:
        return {
            "supervisor/restarts_started": float(self._restarts_started),
            "supervisor/restarts_succeeded": float(self._restarts_succeeded),
            "supervisor/restarts_failed": float(self._restarts_failed),
            "supervisor/restarts_timed_out": float(self._restarts_timed_out),
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
        """DEAD, not already being restarted, and past its cooldown.

        The cooldown is the whole reason this is not just a state test. A failed restart
        returns the shard to DEAD and the next probe tick picks it straight back up, so
        with a 5s probe interval the entire five-attempt budget can be spent inside 25s --
        on one cause, none of the attempts having waited for it to clear. The cause that
        motivated the feature is exactly this shape: an orphaned EngineCore held its GPU
        for 370s, so every attempt inside that window was guaranteed to fail on placement.
        """
        now = self._clock()
        return [
            health.dp_shard_idx
            for health in self._monitor.snapshot()
            if health.state is ShardState.DEAD
            and health.dp_shard_idx not in self._in_flight
            and now >= self._retry_not_before.get(health.dp_shard_idx, 0.0)
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
        # The attempt number, not just the shard: the [GPU_DIAG] lines this run produces
        # are otherwise uncorrelatable with a particular attempt, and five failed attempts
        # against one cause read identically to five against five.
        attempt = self._monitor.snapshot()[shard_idx].restart_attempts
        print(
            f"  supervisor: restarting generation shard {shard_idx} (attempt {attempt})",
            flush=True,
        )
        task = asyncio.get_running_loop().create_task(self._restart(shard_idx, attempt))
        self._in_flight[shard_idx] = task
        task.add_done_callback(partial(self._forget, shard_idx))

    def _forget(self, shard_idx: int, task: "asyncio.Task") -> None:
        """Drop the finished task so a later tick can retry this shard."""
        del task
        self._in_flight.pop(shard_idx, None)

    async def _restart(self, shard_idx: int, attempt: int) -> None:
        try:
            url = await self._restart_off_loop(shard_idx, attempt)
        except Exception as e:  # noqa: BLE001 - a failed restart must not kill the run
            self._restarts_failed += 1
            if isinstance(e, asyncio.TimeoutError):
                self._restarts_timed_out += 1
            reason = f"restart failed: {type(e).__name__}: {e}"
            print(f"  supervisor: shard {shard_idx} {reason}", flush=True)
            # Back to DEAD rather than stuck in RESTARTING, so the next tick can retry
            # until the attempt budget retires it. Not report_failure: probes are ignored
            # for non-serving states, so that would leave it stuck.
            self._monitor.mark_restart_failed(shard_idx, error=reason)
            self._retry_not_before[shard_idx] = self._clock() + self._restart_backoff_s
            return

        self._restarts_succeeded += 1
        self._retry_not_before.pop(shard_idx, None)
        # STALE, not HEALTHY: the engine is up but holds whatever weights it loaded from
        # disk. It is eligible for the next refit and not for traffic until that refit
        # lands, which is exactly the ordering that keeps stale weights out of rollouts.
        self._monitor.mark_loaded(shard_idx, base_url=url)
        print(f"  supervisor: shard {shard_idx} back up at {url}", flush=True)

    async def _restart_off_loop(self, shard_idx: int, attempt: int) -> Optional[str]:
        """Run one restart on a dedicated daemon thread, and stop waiting after the budget.

        A DEDICATED DAEMON THREAD, not asyncio.to_thread, for the same reason
        ``_sync_weights_within`` gives: to_thread runs on the default ThreadPoolExecutor,
        whose workers are not daemons and are joined at interpreter exit with no timeout,
        and a worker only checks for the stop sentinel *between* items. A thread parked
        inside a reload never sees it, so the join never returns and the process cannot
        exit. A model reload takes minutes, which makes this the call in the feature most
        likely to be that thread. ``drain`` cannot help: the join happens whether or not
        anyone calls it.

        BOUNDED, because nothing else in the chain is. ``restart_shard`` ends in ``ray.get``
        calls with no timeout, and ``create_worker`` returns *before* the actor is
        scheduled -- so a placement-group bundle that can never be filled, which is what a
        lost node looks like from here, does not raise: it blocks in ``post_init``. Then
        the ``except`` above never fires and the shard sits in RESTARTING for the rest of
        the run, never retried because it is no longer DEAD and never retired because
        retirement is driven by restart attempts. Timing out converts that silent park into
        a failed restart that spends an attempt and eventually retires the shard.

        The orphaned thread is a real cost, not a free win: the reload may still be running,
        and if it later succeeds it does so into a fleet that has written the shard off. It
        is bounded by the attempt budget, and the alternative is a shard that disappears.
        """
        loop = asyncio.get_running_loop()
        settled: asyncio.Future = loop.create_future()

        def _settle(setter, value) -> None:
            # wait_for cancels `settled` on timeout, and setting a result on a cancelled
            # future raises InvalidStateError inside the loop callback.
            if not settled.done():
                setter(value)

        def _run() -> None:
            try:
                # On this thread, not in _begin_restart: the reading has to happen before
                # the restart is issued, but the controller's event loop also drives the
                # rollout pump, and this is a bounded Ray call to another node. It never
                # raises, so it cannot turn a restartable shard into a failed attempt.
                self._generation.log_shard_gpu_state(
                    shard_idx, label=f"pre_restart_shard{shard_idx}_attempt{attempt}"
                )
                url = self._generation.restart_shard(shard_idx)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop below
                loop.call_soon_threadsafe(_settle, settled.set_exception, exc)
            else:
                loop.call_soon_threadsafe(_settle, settled.set_result, url)

        threading.Thread(
            target=_run, name=f"sc-restart-{shard_idx}", daemon=True
        ).start()

        try:
            return await asyncio.wait_for(settled, self._restart_timeout_s)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"the engine did not come back within {self._restart_timeout_s}s "
                "(restart_timeout_s). Nothing below this has a timeout of its own, so a "
                "bundle that can never be filled -- a lost node -- would otherwise block "
                "here for the rest of the run."
            ) from None

    async def drain(self, timeout_s: Optional[float] = None) -> None:
        """Wait briefly for in-flight restarts so shutdown does not abandon one.

        Bounded on purpose: restart_shard can block on a placement-group bundle that will
        never be filled again, so giving up here is what lets the process exit. The thread
        it gave up on is a daemon (see _restart_off_loop), which is what makes giving up
        harmless rather than a leak.
        """
        if not self._in_flight:
            return
        _, pending = await asyncio.wait(
            list(self._in_flight.values()), timeout=timeout_s
        )
        if pending:
            print(
                f"  supervisor: {len(pending)} restart(s) still running after "
                f"{timeout_s}s; shutting down without them",
                flush=True,
            )
