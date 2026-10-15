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
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BackgroundValidationResult:
    """A finished background validation.

    Attributes:
        val_step: Training step the validation started at.
        end_step: Training step reached when the validation finished.
        metrics: Metrics returned by the validation function.
        timings: Timings returned by the validation function.
        checkpoint_pending: Whether a checkpoint was saved at ``val_step``
            before this result came in, so its score still has to be recorded.
    """

    val_step: int
    end_step: int
    metrics: dict[str, Any]
    timings: dict[str, Any]
    checkpoint_pending: bool


class BackgroundValidator:
    """Runs validations on driver threads and hands back their results.

    Used by ``grpo.async_grpo.overlap_validation``. The training loop calls
    ``launch()`` at a validation step, ``collect_finished()`` once per step, and
    ``wait_all()`` before training stops. Results are returned to the caller,
    which logs them and records them in checkpoints on the training thread;
    this class owns only the threads and their bookkeeping, and is meant to be
    driven from a single (training) thread.

    Each validation runs ``validate_fn`` on its own daemon thread, so a
    validation still running when the process exits does not hold up exit.
    The thread spends most of its time waiting on generation and environment
    workers, which releases the GIL, so the training loop keeps running.
    ``validate_fn`` shares the generation engine with training: it must accept
    requests while training refits weights (async vLLM with in-flight weight
    updates), and anything else it touches must be safe to use concurrently
    with the training loop.
    """

    def __init__(
        self,
        validate_fn: Callable[[int], tuple[dict[str, Any], dict[str, Any]]],
        current_step_fn: Callable[[], int],
    ) -> None:
        """Initialize the validator.

        Args:
            validate_fn: Runs one validation for a training step and returns its
                metrics and timings. Called on a background thread.
            current_step_fn: Returns the training step reached so far. Called
                on the background thread when a validation finishes.
        """
        self._validate_fn = validate_fn
        self._current_step_fn = current_step_fn
        self._futures: dict[int, Future] = {}
        self._checkpoint_pending: set[int] = set()

    @property
    def pending_steps(self) -> list[int]:
        """Steps whose validation has been launched but not yet collected."""
        return sorted(self._futures)

    def is_pending(self, step: int) -> bool:
        """Return whether the validation launched at ``step`` is uncollected."""
        return step in self._futures

    def launch(self, step: int) -> None:
        """Start the validation for ``step`` on a background thread.

        Args:
            step: Training step the validation starts at.
        """
        if step in self._futures:
            raise ValueError(f"A validation for step {step} is already running")
        if self._futures:
            warnings.warn(
                f"Validation from step(s) {self.pending_steps} is still running at "
                f"step {step}; grpo.val_period is shorter than a validation.",
                stacklevel=2,
            )
        future: Future = Future()

        def run() -> None:
            try:
                metrics, timings = self._validate_fn(step)
                future.set_result((metrics, timings, self._current_step_fn()))
            except BaseException as e:  # re-raised by collect_finished()/wait_all()
                future.set_exception(e)

        threading.Thread(target=run, name=f"validation-step{step}", daemon=True).start()
        self._futures[step] = future

    def mark_checkpoint_pending(self, step: int) -> None:
        """Note that ``step`` was checkpointed before its validation finished.

        Args:
            step: Training step of the checkpoint; its validation must be pending.
        """
        if step not in self._futures:
            raise ValueError(f"No validation is running for step {step}")
        self._checkpoint_pending.add(step)
        print(
            f"Checkpoint at step {step} saved before its validation finished; "
            "its score is recorded when the validation finishes.",
            flush=True,
        )

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
        metrics, timings, end_step = self._futures.pop(step).result()
        checkpoint_pending = step in self._checkpoint_pending
        self._checkpoint_pending.discard(step)
        return BackgroundValidationResult(
            val_step=step,
            end_step=end_step,
            metrics=metrics,
            timings=timings,
            checkpoint_pending=checkpoint_pending,
        )
