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

"""Write debug output (training-data dump, W&B tables) from a separate process."""

import logging
import multiprocessing
import os
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WandbRun:
    """The controller's W&B run that writers attach to in shared mode."""

    id: str
    project: str
    entity: str | None

    @classmethod
    def from_config(cls, wandb_cfg: Any) -> "WandbRun":
        """From ``logger.wandb``, whose id setup pins for shared mode."""
        return cls(id=wandb_cfg.id, project=wandb_cfg.project, entity=wandb_cfg.entity)


@dataclass(frozen=True)
class WandbTable:
    """Where a writer logs its table: the run, the table's key and its step axis."""

    run: WandbRun
    key: str
    step_key: str


class DebugSinks:
    """A writer process's outputs: an optional dump and an optional W&B table.

    Lives in the writer process; every submitted call receives it as its first
    argument.
    """

    def __init__(self, label: str, table: WandbTable | None, dump: Any) -> None:
        self.label, self.table, self.dump = label, table, dump
        self._run: Any = None

    def log_table(self, step: float, rows: list[dict[str, Any]]) -> None:
        """Log ``rows`` as one table at ``step``. A failure is a warning only."""
        assert self.table is not None
        try:
            # Deferred: wandb loads only in processes that log tables.
            import wandb

            if self._run is None:
                # A writer spawned by the controller inherits the env pointing at
                # the controller's W&B service, which already holds this run id
                # as its primary and refuses a second attach. Start our own.
                os.environ.pop(wandb.env.SERVICE, None)
                self._run = wandb.init(
                    id=self.table.run.id,
                    project=self.table.run.project,
                    entity=self.table.run.entity,
                    settings=wandb.Settings(
                        mode="shared",
                        x_primary=False,
                        x_label=self.label,
                        x_update_finish_state=False,
                    ),
                )
                self._run.define_metric(self.table.key, step_metric=self.table.step_key)
            columns = list(rows[0])
            table = wandb.Table(
                columns=columns,
                data=[[row[c] for c in columns] for row in rows],
                # Type inference off: W&B otherwise checks every cell.
                allow_mixed_types=True,
            )
            self._run.log({self.table.key: table, self.table.step_key: step})
        except Exception:  # noqa: BLE001 - a debug table must not stop training
            _log.warning("%s not logged to W&B", self.table.key, exc_info=True)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


def _serve(calls: Any, done: Any, sinks: DebugSinks) -> None:
    """The writer process: run each ``(fn, args)`` as ``fn(sinks, *args)``, in order.

    Reports one result per call on ``done``: None, or the failure's traceback as
    text (an exception may not pickle). A None call closes the sinks and ends the
    loop.
    """
    while (call := calls.get()) is not None:
        fn, args = call
        try:
            fn(sinks, *args)
        except Exception:  # noqa: BLE001 - reported to the parent's flush
            done.put(traceback.format_exc())
        else:
            done.put(None)
    sinks.close()
    done.put(None)


class DebugWriter:
    """Debug output written by a dedicated process, off the caller's GIL.

    One per writing process: each advantage-stage process (the controller without
    a pool, every pool actor with one) and each NemoGym actor. ``submit(fn, *args)``
    runs ``fn(sinks, *args)`` in the writer process, in order, and admits at most
    two pending calls: it blocks beyond that, so callers run it off their event
    loop. Tensors in ``args`` reach the process through shared memory. ``flush``
    waits for every submitted call and raises the first failure; ``close`` drains,
    finishes the W&B attachment and stops the process. The process starts on the
    first ``submit``.
    """

    def __init__(
        self, *, label: str, table: WandbTable | None = None, dump: Any = None
    ) -> None:
        self._sinks = DebugSinks(label, table, dump)
        self._process: Any = None
        self._calls: Any = None
        self._done: Any = None
        self._outstanding = 0
        self._error: str | None = None
        # submit and flush run on different threads (to_thread callers).
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., None], *args: Any) -> None:
        """Queue ``fn(sinks, *args)``; blocks while two calls are pending."""
        with self._lock:
            if self._process is None:
                # spawn: never fork a process holding Ray and CUDA state.
                context = multiprocessing.get_context("spawn")
                self._calls, self._done = context.Queue(), context.Queue()
                self._process = context.Process(
                    target=_serve,
                    args=(self._calls, self._done, self._sinks),
                    name=self._sinks.label,
                    daemon=True,
                )
                self._process.start()
            if self._outstanding == 2:
                self._collect()
            self._calls.put((fn, args))
            self._outstanding += 1

    def flush(self) -> None:
        """Wait for every submitted call; raise the first failure since the last flush."""
        with self._lock:
            while self._outstanding:
                self._collect()
            error, self._error = self._error, None
        if error is not None:
            raise RuntimeError(f"{self._sinks.label} failed to write:\n{error}")

    def close(self, timeout_s: float = 300.0) -> None:
        """Drain (bounded per call), finish the W&B attachment, stop the process."""
        with self._lock:
            if self._process is None:
                return
            try:
                self._calls.put(None)
                self._outstanding += 1
                while self._outstanding:
                    self._collect(timeout_s)
                if self._error is not None:
                    raise RuntimeError(self._error)
            except Exception:  # noqa: BLE001 - teardown must not mask the run's outcome
                _log.warning(
                    "%s did not drain cleanly", self._sinks.label, exc_info=True
                )
            finally:
                self._process.join(timeout_s)
                if self._process.is_alive():
                    self._process.terminate()
                self._process = None

    def _collect(self, timeout_s: float | None = None) -> None:
        """Wait for the oldest outstanding call; keep its failure for flush."""
        error = self._done.get(timeout=timeout_s)
        self._outstanding -= 1
        if error is not None and self._error is None:
            self._error = error
