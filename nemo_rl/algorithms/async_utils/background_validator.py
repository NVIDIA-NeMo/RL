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

"""Run async GRPO validations in the background while training continues."""

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BackgroundValidationResult:
    """A finished background validation.

    Attributes:
        val_step: Training step the validation started at.
        metrics: Metrics returned by the validation function.
        timings: Timings returned by the validation function.
    """

    val_step: int
    metrics: dict[str, Any]
    timings: dict[str, Any]


class BackgroundValidator:
    """Runs validations on driver threads and hands back their results.

    Used by ``grpo.async_grpo.validation_engines``. The training loop calls
    ``launch()`` at a validation step, ``collect_finished()`` once per step, and
    ``wait_all()`` before training stops. Results are returned to the caller,
    which logs them on the training thread; this class owns only the threads
    and their bookkeeping, and is meant to be driven from a single (training)
    thread.

    Each validation runs ``validate_fn`` on its own daemon thread, so a
    validation still running when the process exits does not hold up exit.
    The thread spends most of its time waiting on generation and environment
    workers, which releases the GIL, so the training loop keeps running.
    """

    def __init__(
        self, validate_fn: Callable[[int], tuple[dict[str, Any], dict[str, Any]]]
    ) -> None:
        """Initialize the validator.

        Args:
            validate_fn: Runs one validation for a training step and returns its
                metrics and timings. Called on a background thread.
        """
        self._validate_fn = validate_fn
        self._futures: dict[int, Future] = {}

    @property
    def pending_steps(self) -> list[int]:
        """Steps whose validation has been launched but not yet collected."""
        return sorted(self._futures)

    def is_running(self) -> bool:
        """Return whether any launched validation has not finished yet."""
        return any(not future.done() for future in self._futures.values())

    def launch(self, step: int) -> None:
        """Start the validation for ``step`` on a background thread.

        Args:
            step: Training step the validation starts at.
        """
        if step in self._futures:
            raise ValueError(f"A validation for step {step} is already running")
        future: Future = Future()

        def run() -> None:
            try:
                future.set_result(self._validate_fn(step))
            except BaseException as e:  # re-raised by collect_finished()/wait_all()
                future.set_exception(e)

        threading.Thread(target=run, name=f"validation-step{step}", daemon=True).start()
        self._futures[step] = future

    def collect_finished(self) -> list[BackgroundValidationResult]:
        """Return the validations that have finished, oldest first.

        Raises:
            BaseException: Whatever a finished validation raised.
        """
        done = [step for step in self.pending_steps if self._futures[step].done()]
        return [self._pop_result(step) for step in done]

    def wait_all(self) -> list[BackgroundValidationResult]:
        """Wait for every running validation and return them, oldest first.

        Raises:
            BaseException: Whatever a validation raised.
        """
        if self._futures:
            print(
                f"Waiting for validation(s) started at step(s) {self.pending_steps} "
                "to finish...",
                flush=True,
            )
        return [self._pop_result(step) for step in self.pending_steps]

    def _pop_result(self, step: int) -> BackgroundValidationResult:
        metrics, timings = self._futures.pop(step).result()
        return BackgroundValidationResult(
            val_step=step, metrics=metrics, timings=timings
        )
