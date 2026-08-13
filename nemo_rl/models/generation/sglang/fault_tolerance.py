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

import logging
import threading
import time
from typing import Optional

import ray

from nemo_rl.models.generation.sglang.config import SGLangConfig

logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, sglang_generation, sglang_cfg: SGLangConfig):
        self._sglang_generation = sglang_generation

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = sglang_cfg["sglang_cfg"]["rollout_health_check_interval"]
        self._check_timeout = sglang_cfg["sglang_cfg"]["rollout_health_check_timeout"]
        self._check_first_wait = sglang_cfg["sglang_cfg"][
            "rollout_health_check_first_wait"
        ]
        # Absolute monotonic deadline before which no probe may run, giving a
        # booting engine time to become ready. It is a DEADLINE rather than a
        # "wait once" flag so that a pause part-way through cannot restart the
        # clock -- see ``arm_first_wait``.
        self._first_check_after: Optional[float] = None
        self._is_checking_enabled = False  # Track if health checking should be active
        # Held for the duration of one check round so pause() can wait it out.
        self._check_lock = threading.Lock()

    def arm_first_wait(self) -> None:
        """Hold off probing until ``rollout_health_check_first_wait`` has elapsed.

        Call this whenever an engine is about to boot -- at startup, and after
        ``_recover`` restarts a dead one -- so a fresh engine is not killed for
        being slow to load weights.

        The deadline is absolute. It deliberately does NOT reset on
        ``resume()``: resume runs once per training step, and re-arming there
        meant that any generation phase shorter than the first-wait left the
        monitor permanently in its grace period, so no probe ever ran.
        """
        self._first_check_after = time.monotonic() + self._check_first_wait

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._sglang_generation.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self.arm_first_wait()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "Rollout health monitor thread did not terminate within %.1fs", timeout
            )
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False
        # Wait out an in-flight round so callers may offload/refit right after.
        with self._check_lock:
            pass

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        # The first-wait deadline is deliberately NOT re-armed here; see
        # ``arm_first_wait``. It is armed at start() and after a recovery.
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Hold off the first probe until the boot deadline passes, so a
            # still-loading engine is not killed for answering slowly. The
            # deadline is absolute, so time spent paused still counts and a
            # pause part-way through cannot restart it.
            if self._first_check_after is not None:
                remaining = self._first_check_after - time.monotonic()
                if remaining > 0:
                    logger.info(
                        f"Health monitor waiting {remaining:.1f}s before the first check."
                    )
                    if self._stop_event.wait(remaining):
                        logger.info("Health monitor stopped during first wait.")
                        break
                    if self._pause_event.is_set():
                        # Paused mid-wait: go back to the pause gate rather than
                        # probing. The deadline is unchanged, so the elapsed
                        # time is not lost and the next resume picks up where
                        # this left off.
                        logger.info("Health monitor paused during first wait.")
                        continue
                self._first_check_after = None

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                with self._check_lock:
                    self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._sglang_generation.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            # Inner timeout bounds the HTTP probe, outer one a wedged actor process.
            ray.get(
                engine.health_generate.remote(timeout=self._check_timeout),
                timeout=2 * self._check_timeout,
            )
        except Exception as e:
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (timeout or error). Killing actor. Exception: {e}"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)
        else:
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing server group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._sglang_generation.nodes_per_engine,
            (rollout_engine_id + 1) * self._sglang_generation.nodes_per_engine,
        ):
            engine = self._sglang_generation.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote(), timeout=self._check_timeout)
                except Exception as e:
                    logger.warning(
                        f"Graceful shutdown failed for engine at index {i} (e: {e})"
                    )
                try:
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._sglang_generation.all_engines[i] = None
