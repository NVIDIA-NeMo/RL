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

"""Asynchronous task sampling, prefetch, and resumable rollout tickets."""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from logra.osworld import Episode


@dataclass
class TaskSampler:
    size: int
    seed: int
    generator: torch.Generator = field(init=False)
    order: list[int] = field(default_factory=list, init=False)
    cursor: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)

    def next(self, count: int) -> list[int]:
        selected = []
        while len(selected) < count:
            if self.cursor == len(self.order):
                self.order = torch.randperm(
                    self.size, generator=self.generator
                ).tolist()
                self.cursor = 0
            take = min(count - len(selected), self.size - self.cursor)
            selected.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "generator": self.generator.get_state(),
            "size": self.size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["size"] != self.size:
            raise ValueError("Training dataset size changed on resume")
        self.order, self.cursor = state["order"], state["cursor"]
        self.generator.set_state(state["generator"])


@dataclass(frozen=True)
class RolloutTicket:
    """Identify an unconsumed task independently of when its desktop finishes."""

    serial: int
    task_index: int
    seed: int
    dispatch_step: int


@dataclass
class RolloutPoolState:
    sampler: dict[str, Any]
    pending: list[RolloutTicket]
    next_serial: int


@dataclass
class RolloutBatch:
    episodes: list[tuple[RolloutTicket, Episode]]
    generation_seconds: float
    dropped_no_action: int = 0


class RollingTaskPool:
    """Prefetch finished task groups while retaining and refilling slow desktops.

    The next batch is collected while the caller trains the current one.
    Checkpoints retain every unconsumed task ticket. Resume reruns those tasks;
    live desktops and sampled responses are not claimed to resume bit for bit.
    """

    def __init__(
        self,
        generate: Callable[[RolloutTicket], Episode],
        sampler: TaskSampler,
        *,
        batch_size: int,
        inflight: int,
        seed: int,
        start_step: int,
        resume: RolloutPoolState | None,
    ) -> None:
        if batch_size <= 0 or inflight < batch_size:
            raise ValueError("Rollout capacity must cover a positive training batch")
        self._generate = generate
        self._sampler = sampler
        self._batch_size = batch_size
        self._capacity = inflight
        self._seed = seed
        self._step = start_step
        self._lock = Lock()
        self._closed = False
        self._running: dict[Future[Episode], RolloutTicket] = {}
        self._pending: dict[int, RolloutTicket] = {}
        self._restored: deque[RolloutTicket] = deque()
        self._next_serial = 0
        self._next_batch: Future[RolloutBatch] | None = None
        if resume is not None:
            sampler.load_state_dict(resume.sampler)
            self._pending = {ticket.serial: ticket for ticket in resume.pending}
            if len(self._pending) != len(resume.pending) or any(
                ticket.serial >= resume.next_serial
                or not 0 <= ticket.task_index < sampler.size
                for ticket in resume.pending
            ):
                raise ValueError("Invalid pending rollout checkpoint")
            self._restored.extend(resume.pending)
            self._next_serial = resume.next_serial
        self._workers = ThreadPoolExecutor(max_workers=inflight)
        self._collector = ThreadPoolExecutor(max_workers=1)

    def _fill(self) -> None:
        # Called under the lock: checkpoint data order and pending tasks agree.
        while len(self._running) < self._capacity:
            if self._restored:
                ticket = self._restored.popleft()
            else:
                ticket = RolloutTicket(
                    self._next_serial,
                    self._sampler.next(1)[0],
                    self._seed + self._next_serial,
                    self._step,
                )
                self._next_serial += 1
                self._pending[ticket.serial] = ticket
            future = self._workers.submit(self._generate, ticket)
            self._running[future] = ticket

    def _collect(self) -> RolloutBatch:
        started = time.perf_counter()
        finished = []
        dropped_no_action = 0
        while len(finished) < self._batch_size:
            with self._lock:
                if self._closed:
                    raise CancelledError("Rollout collection closed")
                self._fill()
                running = list(self._running)
            ready, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in sorted(ready, key=lambda item: self._running[item].serial):
                with self._lock:
                    ticket = self._running.pop(future)
                episode = future.result()
                # The reference rejects empty trajectories before counting prompt
                # groups. Runtime failures with action tokens still enter masked.
                if not any(part.token_mask.any() for part in episode.segments):
                    with self._lock:
                        del self._pending[ticket.serial]
                    dropped_no_action += 1
                    continue
                finished.append((ticket, episode))
                if len(finished) == self._batch_size:
                    break
        return RolloutBatch(finished, time.perf_counter() - started, dropped_no_action)

    def next_batch(self, *, step: int, prefetch_next: bool) -> RolloutBatch:
        with self._lock:
            if self._closed:
                raise RuntimeError("Rollout pool is closed")
            self._step = step
        if self._next_batch is None:
            self._next_batch = self._collector.submit(self._collect)
        finished = self._next_batch.result()
        with self._lock:
            for ticket, _ in finished.episodes:
                del self._pending[ticket.serial]
        self._next_batch = (
            self._collector.submit(self._collect) if prefetch_next else None
        )
        return finished

    def state_dict(self) -> RolloutPoolState:
        with self._lock:
            return RolloutPoolState(
                copy.deepcopy(self._sampler.state_dict()),
                list(self._pending.values()),
                self._next_serial,
            )

    def close(self) -> None:
        """Stop refilling, then let dispatched desktops finish and release resources."""
        with self._lock:
            self._closed = True
        self._collector.shutdown(wait=True, cancel_futures=True)
        self._workers.shutdown(wait=True, cancel_futures=True)
