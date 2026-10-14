# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Benchmark prototype: bounded asynchronous capture, separate from serving.

Records own immutable buffers. Threads borrow references; multiprocessing.Queue
serializes them in its feeder thread. Neither path waits for tree insertion.
Queue overflow invalidates the rollout, and drain reports every failure. This is
an experiment, not a durable storage service or a production capture adapter.
"""

from __future__ import annotations

import multiprocessing as mp
import hashlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from tools.http_profile.tracing import Trace


@dataclass(frozen=True)
class CallRecord:
    rollout_id: str
    call_id: str
    token_bytes: bytes  # Packed native-endian uint32 prompt + completion.
    prompt_tokens: int
    logprob_bytes: bytes  # Packed float32 for generated tokens only.
    submitted_at: float
    trace_id: str = ""

    @property
    def nbytes(self) -> int:
        return len(self.token_bytes) + len(self.logprob_bytes)


@dataclass(frozen=True)
class StoredCall:
    leaf: int
    tail: bytes
    prompt_tokens: int
    logprob_bytes: bytes


class PrefixTree:
    """Intern exact token blocks; retain independent action records at leaves.

    The single resolver owns the tree. Reordered arrivals do not require a parent
    generation to have committed. Roots are per rollout, not per GRPO task group.
    """

    def __init__(self, block_tokens: int = 16) -> None:
        if block_tokens < 1:
            raise ValueError("block_tokens must be positive")
        self.block_bytes = 4 * block_tokens
        self.roots: dict[str, int] = {}
        self.edges: dict[tuple[int, bytes], int] = {}
        self.parents: dict[int, tuple[int, bytes]] = {}
        self.calls: dict[tuple[str, str], StoredCall] = {}
        self.logical_tokens = 0
        self.generated_tokens = 0
        self.next_node = 1

    def insert(self, record: CallRecord) -> None:
        n_tokens = len(record.token_bytes) // 4
        if len(record.token_bytes) % 4 or not 0 <= record.prompt_tokens <= n_tokens:
            raise ValueError("invalid token buffer or prompt boundary")
        if len(record.logprob_bytes) != 4 * (n_tokens - record.prompt_tokens):
            raise ValueError("generated token/logprob lengths differ")
        key = (record.rollout_id, record.call_id)
        if key in self.calls:
            raise ValueError("duplicate call identity")
        node = self.roots.get(record.rollout_id)
        if node is None:
            node = self.next_node
            self.next_node += 1
            self.roots[record.rollout_id] = node
        full_end = len(record.token_bytes) // self.block_bytes * self.block_bytes
        for start in range(0, full_end, self.block_bytes):
            block = record.token_bytes[start : start + self.block_bytes]
            edge = (node, block)
            child = self.edges.get(edge)
            if child is None:
                child = self.next_node
                self.next_node += 1
                self.edges[edge] = child
                self.parents[child] = edge
            node = child
        self.calls[key] = StoredCall(
            node,
            record.token_bytes[full_end:],
            record.prompt_tokens,
            record.logprob_bytes,
        )
        self.logical_tokens += n_tokens
        self.generated_tokens += n_tokens - record.prompt_tokens

    def materialize(self, rollout_id: str, call_id: str) -> bytes:
        """Dense fallback, used to check that interning loses no context."""
        call = self.calls[(rollout_id, call_id)]
        pieces = [call.tail]
        node = call.leaf
        while node in self.parents:
            node, block = self.parents[node]
            pieces.append(block)
        return b"".join(reversed(pieces))

    def stats(self) -> dict[str, int]:
        return {
            "stored_calls": len(self.calls),
            "logical_tokens": self.logical_tokens,
            "generated_tokens": self.generated_tokens,
            "unique_token_bytes": len(self.edges) * self.block_bytes
            + sum(len(call.tail) for call in self.calls.values()),
        }


def _resolve_loop(
    inputs: Any,
    reports: Any,
    counters: Any,
    delay_s: float,
    block_tokens: int,
    tracing: bool,
    separate_process: bool,
) -> None:
    trace = Trace(
        tracing,
        "capture resolver",
        track_offset=1_000_000,
        process_metadata=separate_process,
    )
    cpu_clock = time.process_time if separate_process else time.thread_time
    cpu_start = cpu_clock()
    tree = PrefixTree(block_tokens)
    failures: list[str] = []
    lag_sum = 0.0
    lag_max = 0.0
    resolver_cpu = 0.0
    completed = 0
    while True:
        item = inputs.get()
        if item == "stop":
            return
        if item == "report":
            reports.put(
                {
                    **tree.stats(),
                    "completed": completed,
                    "failures": failures,
                    "resolver_cpu_s": resolver_cpu,
                    "resolver_total_cpu_s": cpu_clock() - cpu_start,
                    "queue_lag_mean_ms": 1000 * lag_sum / max(1, completed),
                    "queue_lag_max_ms": 1000 * lag_max,
                    "trace_events": trace.events,
                }
            )
            continue
        if item == "reset":
            tree = PrefixTree(block_tokens)
            failures = []
            lag_sum = lag_max = resolver_cpu = 0.0
            completed = 0
            trace.reset()
            cpu_start = cpu_clock()
            reports.put(True)
            continue
        started = time.thread_time()
        received = time.perf_counter()
        lag = received - item.submitted_at
        tid = trace.track("resolver: " + item.trace_id)
        trace.span("queue wait", item.submitted_at, received, tid, bytes=item.nbytes)
        trace.flow("capture handoff", item.trace_id, "f", received, tid)
        lag_sum += lag
        lag_max = max(lag_max, lag)
        try:
            if delay_s:
                time.sleep(delay_s)
                trace.span(
                    "injected resolver delay", received, time.perf_counter(), tid
                )
            tree.insert(item)
        except Exception as error:
            # A worker failure must invalidate capture, not terminate silently.
            failures.append(f"{item.rollout_id}/{item.call_id}: {error}")
        finally:
            completed += 1
            resolver_cpu += time.thread_time() - started
            with counters.get_lock():
                counters[0] -= 1
                counters[1] -= item.nbytes
                pending, pending_bytes = counters[:2]
            trace.span(
                "resolver work",
                received,
                time.perf_counter(),
                tid,
                bytes=item.nbytes,
                injected_delay_ms=delay_s * 1000,
            )
            trace.counter("capture pending", records=pending, bytes=pending_bytes)


class CaptureQueue:
    """Nonblocking admission with bounded pending records and buffer bytes.

    Call drain() outside the model-response path, before publishing a rollout for
    training. Failure/overflow is reported explicitly; this prototype is memory-only.
    """

    def __init__(
        self,
        mode: Literal["thread", "process"],
        *,
        max_records: int = 256,
        max_bytes: int = 64 * 1024 * 1024,
        delay_s: float = 0.0,
        block_tokens: int = 16,
        tracing: bool = False,
    ) -> None:
        if max_records < 1 or max_bytes < 1 or delay_s < 0:
            raise ValueError("invalid capture queue limits")
        context = mp.get_context("spawn")
        # pending records, pending bytes, peak records, peak bytes
        self.counters = context.Array("q", 4)
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.mode = mode
        self.submitted = 0
        self.rejected = 0
        self.invalid_rollouts: set[str] = set()
        self.closed = False
        self.inputs = (
            context.Queue(max_records)
            if mode == "process"
            else queue.Queue(max_records)
        )
        self.reports = context.Queue() if mode == "process" else queue.Queue()
        args = (
            self.inputs,
            self.reports,
            self.counters,
            delay_s,
            block_tokens,
            tracing,
            mode == "process",
        )
        if mode == "process":
            self.worker = context.Process(target=_resolve_loop, args=args)
        else:
            self.worker = threading.Thread(target=_resolve_loop, args=args)
        self.worker.start()

    def submit(self, record: CallRecord) -> bool:
        if self.closed:
            raise RuntimeError("capture queue is closed")
        with self.counters.get_lock():
            available = (
                self.counters[0] < self.max_records
                and self.counters[1] + record.nbytes <= self.max_bytes
            )
            if available:
                self.counters[0] += 1
                self.counters[1] += record.nbytes
                self.counters[2] = max(self.counters[2], self.counters[0])
                self.counters[3] = max(self.counters[3], self.counters[1])
        if available:
            try:
                self.inputs.put_nowait(record)
            except queue.Full:
                with self.counters.get_lock():
                    self.counters[0] -= 1
                    self.counters[1] -= record.nbytes
            else:
                self.submitted += 1
                return True
        self.rejected += 1
        self.invalid_rollouts.add(record.rollout_id)
        return False

    def drain(self) -> dict[str, Any]:
        """Barrier after prior submissions; caller must stop admitting new work."""
        started = time.perf_counter()
        self.inputs.put("report", timeout=60)
        report = self.reports.get(timeout=60)
        with self.counters.get_lock():
            pending, pending_bytes, peak, peak_bytes = self.counters[:]
        return {
            **report,
            "submitted": self.submitted,
            "rejected": self.rejected,
            "invalid_rollouts": sorted(self.invalid_rollouts),
            "pending": pending,
            "pending_bytes": pending_bytes,
            "peak_pending": peak,
            "peak_pending_bytes": peak_bytes,
            "drain_ms": 1000 * (time.perf_counter() - started),
        }

    def reset(self) -> None:
        self.drain()
        self.inputs.put("reset", timeout=60)
        self.reports.get(timeout=60)
        with self.counters.get_lock():
            self.counters[:] = [0, 0, 0, 0]
        self.submitted = self.rejected = 0
        self.invalid_rollouts.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.inputs.put("stop", timeout=60)
        self.worker.join(timeout=60)
        if self.worker.is_alive():
            raise RuntimeError("capture resolver did not stop")
        self.closed = True
        if self.mode == "process":
            self.inputs.close()
            self.inputs.join_thread()
            self.reports.close()
            self.reports.join_thread()


class ShardedCaptureQueue:
    """Route all calls of a rollout to one process; bound each shard's backlog."""

    def __init__(
        self,
        workers: int,
        *,
        max_records: int,
        max_bytes: int,
        delay_s: float = 0.0,
        tracing: bool = False,
    ) -> None:
        if not 1 <= workers <= min(max_records, max_bytes):
            raise ValueError(
                "each resolver needs at least one record and byte of capacity"
            )
        self.shards = [
            CaptureQueue(
                "process",
                max_records=max_records // workers,
                max_bytes=max_bytes // workers,
                delay_s=delay_s,
                tracing=tracing,
            )
            for _ in range(workers)
        ]

    def submit(self, record: CallRecord) -> bool:
        # Hash only the short rollout identifier, never the token payload.
        digest = hashlib.blake2b(record.rollout_id.encode(), digest_size=8).digest()
        shard = int.from_bytes(digest, "little") % len(self.shards)
        return self.shards[shard].submit(record)

    def drain(self) -> dict[str, Any]:
        started = time.perf_counter()
        shards = [shard.drain() for shard in self.shards]
        total_keys = (
            "stored_calls",
            "logical_tokens",
            "generated_tokens",
            "unique_token_bytes",
            "completed",
            "resolver_cpu_s",
            "resolver_total_cpu_s",
            "submitted",
            "rejected",
            "pending",
            "pending_bytes",
            "peak_pending",
            "peak_pending_bytes",
        )
        return {
            **{key: sum(shard[key] for shard in shards) for key in total_keys},
            "failures": [failure for shard in shards for failure in shard["failures"]],
            "invalid_rollouts": sorted(
                {rollout for shard in shards for rollout in shard["invalid_rollouts"]}
            ),
            "queue_lag_mean_ms": sum(
                shard["queue_lag_mean_ms"] * shard["completed"] for shard in shards
            )
            / max(1, sum(shard["completed"] for shard in shards)),
            "queue_lag_max_ms": max(shard["queue_lag_max_ms"] for shard in shards),
            "trace_events": [
                event for shard in shards for event in shard["trace_events"]
            ],
            "drain_ms": 1000 * (time.perf_counter() - started),
            "peak_pending_is_upper_bound": True,
            "shards": [
                {key: value for key, value in shard.items() if key != "trace_events"}
                for shard in shards
            ],
        }

    def reset(self) -> None:
        for shard in self.shards:
            shard.reset()

    def close(self) -> None:
        for shard in self.shards:
            shard.close()
