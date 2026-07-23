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
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active
        self._check_condition = threading.Condition()
        self._checks_in_flight = 0

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
            logging.warning(
                "Rollout health monitor thread did not terminate within %.1fs", timeout
            )
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self, timeout_s: float | None = None) -> None:
        """Pause health checks and synchronously drain any in-flight check.

        Setting the pause event and admitting a new check use the same
        condition, so once this method returns no health check can overlap a
        refit, recovery, or memory transition.
        """
        if self._pause_event is None:
            return
        with self._check_condition:
            logger.info("Pausing health monitor...")
            self._pause_event.set()
            self._is_checking_enabled = False
            drained = self._check_condition.wait_for(
                lambda: self._checks_in_flight == 0,
                timeout=timeout_s,
            )
        if not drained:
            raise TimeoutError(
                "Timed out waiting for an in-flight rollout health check to "
                "finish; health monitoring remains paused."
            )

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        with self._check_condition:
            logger.info("Resuming health monitor...")
            self._need_first_wait = True  # Need to wait after each resume
            self._pause_event.clear()
            self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    @property
    def default_quiesce_timeout_s(self) -> float:
        """Bound a drain across one health RPC and its failure cleanup."""
        return 2 * self._check_timeout + 5

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(
                    f"Health monitor doing first wait after resume: {self._check_first_wait}s"
                )
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info(
                        "Health monitor paused during first wait, will wait again next resume."
                    )
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
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

        with self._check_condition:
            if (
                self._stop_event is None
                or self._stop_event.is_set()
                or self._pause_event is None
                or self._pause_event.is_set()
            ):
                return
            self._checks_in_flight += 1

        try:
            try:
                ray.get(
                    engine.health_generate.remote(timeout=self._check_timeout),
                    # Leave a small scheduling margin beyond the worker's
                    # own HTTP timeout so a timely HTTP failure is observed
                    # instead of misclassified as a Ray timeout.
                    timeout=self._check_timeout + 1,
                )
            except Exception as e:
                logger.error(
                    f"Health check failed for rollout engine {rollout_engine_id} "
                    f"(ray timeout or error). Killing actor. Exception: {e}"
                )
                self._kill_engine(rollout_engine_id=rollout_engine_id)
            else:
                logger.debug(
                    f"Health check passed for rollout engine {rollout_engine_id}"
                )
        finally:
            with self._check_condition:
                self._checks_in_flight -= 1
                self._check_condition.notify_all()

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing server group {rollout_engine_id}...")
        cleanup_failures = []
        for i in range(
            rollout_engine_id * self._sglang_generation.nodes_per_engine,
            (rollout_engine_id + 1) * self._sglang_generation.nodes_per_engine,
        ):
            engine = self._sglang_generation.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    shutdown_confirmed = ray.get(
                        engine.shutdown.remote(timeout_s=self._check_timeout),
                        timeout=self._check_timeout + 1,
                    )
                    if shutdown_confirmed is not True:
                        cleanup_failures.append(
                            f"engine index {i} returned {shutdown_confirmed!r} "
                            "from bounded shutdown"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to shut down engine at index {i} cleanly (e: {e})"
                    )
                    cleanup_failures.append(
                        f"engine index {i} shutdown was not confirmed: {e!r}"
                    )
                try:
                    ray.kill(engine, no_restart=True)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Failed to kill engine at index {i} (e: {e})")
                    cleanup_failures.append(
                        f"engine index {i} Ray actor kill failed: {e!r}"
                    )
            else:
                logger.info(f"Engine at index {i} is already None")
            self._sglang_generation.all_engines[i] = None
        if cleanup_failures:
            self._sglang_generation._latch_engine_cleanup_failure(
                "; ".join(cleanup_failures)
            )
