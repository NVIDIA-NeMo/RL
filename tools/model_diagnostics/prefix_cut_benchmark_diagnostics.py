# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Bounded, worker-local diagnostics for the standalone prefix-cut benchmark."""

from __future__ import annotations

import asyncio
import faulthandler
import gc
import json
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NamedTuple


class _GCStart(NamedTuple):
    wall: float
    thread_cpu: float
    process_cpu: float
    owner: int
    thresholds: tuple[int, int, int]
    counts: tuple[int, int, int]


class _GCCollection(NamedTuple):
    generation: int
    wall_s: float
    thread_cpu_s: float
    process_cpu_s: float
    owner: int
    finished_unix_s: float
    collected: int
    uncollectable: int
    thresholds: tuple[int, int, int]
    counts_before: tuple[int, int, int]


class BenchmarkDiagnostics:
    """Record CPU phases and lock waits without writing on a serving loop.

    Only slow events enter a bounded writer queue. A separate watchdog captures
    all Python thread stacks when either loop's heartbeat stops advancing.
    GC callbacks identify pauses that can also prevent the watchdog running.
    Native timed dumps are deliberately omitted: their lock-free frame walk is
    suspected in a worker crash. The Python watchdog dumps while holding the GIL.
    """

    def __init__(self, output_dir: str) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
        self._guard = threading.RLock()
        self._phases: dict[str, dict[str, float]] = {}
        self._heartbeats: dict[str, tuple[int, float]] = {}
        self._gc_started: dict[int, _GCStart] = {}
        self._gc_finished: deque[_GCCollection] = deque(maxlen=512)
        self._gc_totals: dict[int, Counter[str]] = {}
        self._gc_dropped = 0
        self._closed = threading.Event()
        self._dropped = 0
        self._stacks = (directory / f"stacks-{os.getpid()}.log").open("a")
        self._writer = threading.Thread(
            target=self._write,
            args=(directory,),
            name="cut-benchmark-writer",
            daemon=True,
        )
        self._watchdog = threading.Thread(
            target=self._watch, name="cut-benchmark-watchdog", daemon=True
        )
        self._writer.start()
        self._watchdog.start()
        self._gc_callback_ref = self._gc_callback
        gc.callbacks.append(self._gc_callback_ref)

    def label(self) -> str:
        current = threading.get_ident()
        with self._guard:
            for name, (thread_id, _) in self._heartbeats.items():
                if current == thread_id:
                    return name
        return "executor"

    def record(self, name: str, elapsed: float, cpu: float = 0.0) -> None:
        with self._guard:
            phase = self._phases.setdefault(
                name, {"count": 0, "wall_s": 0, "cpu_s": 0, "max_s": 0}
            )
            phase["count"] += 1
            phase["wall_s"] += elapsed
            phase["cpu_s"] += cpu
            phase["max_s"] = max(phase["max_s"], elapsed)
        if elapsed >= 0.05:
            self.event("slow_phase", phase=name, wall_s=elapsed, thread_cpu_s=cpu)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            self.record(
                name, time.perf_counter() - started, time.thread_time() - cpu_started
            )

    def event(self, event: str, **fields: Any) -> None:
        item = {
            "event": event,
            "unix_s": time.time(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            **fields,
        }
        try:
            self._events.put_nowait(item)
        except queue.Full:
            with self._guard:
                self._dropped += 1

    async def heartbeat(self, name: str) -> None:
        try:
            while not self._closed.is_set():
                started, cpu_started = time.perf_counter(), time.thread_time()
                with self._guard:
                    self._heartbeats[name] = (threading.get_ident(), started)
                await asyncio.sleep(0.25)
                lag = max(0.0, time.perf_counter() - started - 0.25)
                self.record(f"{name}.loop_lag", lag, time.thread_time() - cpu_started)
        finally:
            with self._guard:
                self._heartbeats.pop(name, None)

    def summary(self) -> dict[str, Any]:
        self._drain_gc()
        with self._guard:
            return {
                "pid": os.getpid(),
                "phases": {key: dict(value) for key, value in self._phases.items()},
                "dropped_events": self._dropped,
                "gc": {str(key): dict(value) for key, value in self._gc_totals.items()},
                "dropped_gc_events": self._gc_dropped,
                "loop_threads": {
                    key: value[0] for key, value in self._heartbeats.items()
                },
            }

    def _gc_callback(self, phase: str, info: dict[str, Any]) -> None:
        generation = info["generation"]
        if phase == "start":
            self._gc_started[generation] = _GCStart(
                time.perf_counter(),
                time.thread_time(),
                time.process_time(),
                threading.get_ident(),
                gc.get_threshold(),
                gc.get_count(),
            )
        else:
            started = self._gc_started.pop(generation, None)
            if started is not None:
                # GC can interrupt queue.put or a diagnostic lock acquisition.
                # Never reenter those from a GC callback; drain off-loop later.
                if len(self._gc_finished) == self._gc_finished.maxlen:
                    self._gc_dropped += 1
                self._gc_finished.append(
                    _GCCollection(
                        generation,
                        time.perf_counter() - started.wall,
                        time.thread_time() - started.thread_cpu,
                        time.process_time() - started.process_cpu,
                        started.owner,
                        time.time(),
                        info.get("collected", 0),
                        info.get("uncollectable", 0),
                        started.thresholds,
                        started.counts,
                    )
                )

    def _drain_gc(self) -> None:
        # Only consumers take this lock. GC callbacks exclusively append records.
        with self._guard:
            while self._gc_finished:
                collection = self._gc_finished.popleft()
                self.record(
                    f"gc.generation_{collection.generation}",
                    collection.wall_s,
                    collection.thread_cpu_s,
                )
                totals = self._gc_totals.setdefault(collection.generation, Counter())
                totals.update(
                    {
                        "count": 1,
                        "wall_s": collection.wall_s,
                        "thread_cpu_s": collection.thread_cpu_s,
                        "process_cpu_s": collection.process_cpu_s,
                        "collected": collection.collected,
                        "uncollectable": collection.uncollectable,
                    }
                )
                if collection.generation == 2 or collection.wall_s >= 0.05:
                    self.event(
                        "gc_pause",
                        generation=collection.generation,
                        wall_s=collection.wall_s,
                        thread_cpu_s=collection.thread_cpu_s,
                        process_cpu_s=collection.process_cpu_s,
                        collected=collection.collected,
                        uncollectable=collection.uncollectable,
                        thresholds=collection.thresholds,
                        counts_before=collection.counts_before,
                        owner_thread_id=collection.owner,
                        started_unix_s=collection.finished_unix_s - collection.wall_s,
                        finished_unix_s=collection.finished_unix_s,
                    )

    def heap_summary(self) -> dict[str, Any]:
        """Census tracked objects between cuts; never call from a GC callback.

        The census itself pauses Python work and allocates a temporary reference
        list. Report its cost separately and retain only JSON values. NumPy data
        buffers are generally untracked and are not measured by this census.
        """
        started = time.perf_counter()
        types: Counter[str] = Counter()
        containers: dict[str, dict[str, int]] = {}
        objects = gc.get_objects()
        tracked_objects = len(objects)
        obj = None
        try:
            for obj in objects:
                kind = type(obj)
                types[f"{kind.__module__}.{kind.__qualname__}"] += 1
                if kind in (list, tuple, dict, set, frozenset):
                    entry = containers.setdefault(
                        kind.__name__,
                        {
                            "count": 0,
                            "items": 0,
                            "max_items": 0,
                            "shallow_bytes": 0,
                        },
                    )
                    length = len(obj)
                    entry["count"] += 1
                    entry["items"] += length
                    entry["max_items"] = max(entry["max_items"], length)
                    entry["shallow_bytes"] += sys.getsizeof(obj)
        finally:
            del obj, objects
        result = {
            "pid": os.getpid(),
            "unix_s": time.time(),
            "wall_s": time.perf_counter() - started,
            "tracked_objects": tracked_objects,
            "top_types": dict(types.most_common(30)),
            "containers": containers,
            "gc_counts": gc.get_count(),
            "gc_thresholds": gc.get_threshold(),
            "gc_stats": gc.get_stats(),
        }
        self.event("heap_summary", **result)
        return result

    def _write(self, directory: Path) -> None:
        with (directory / f"events-{os.getpid()}.jsonl").open(
            "a", buffering=1
        ) as output:
            while True:
                item = self._events.get()
                if item is None:
                    return
                output.write(json.dumps(item, sort_keys=True) + "\n")

    def _watch(self) -> None:
        last_dump: Counter[str] = Counter()
        while not self._closed.wait(0.5):
            self._drain_gc()
            now = time.perf_counter()
            with self._guard:
                heartbeats = dict(self._heartbeats)
            for name, (thread_id, beat) in heartbeats.items():
                if now - beat >= 2 and now - last_dump[name] >= 5:
                    last_dump[name] = now
                    self.event(
                        "loop_stalled",
                        loop=name,
                        owner_thread_id=thread_id,
                        heartbeat_age_s=now - beat,
                    )
                    self._stacks.write(
                        f"\n=== unix_s={time.time()} loop={name} thread={thread_id} age_s={now - beat:.3f} ===\n"
                    )
                    self._stacks.flush()
                    faulthandler.dump_traceback(file=self._stacks, all_threads=True)

    def close(self) -> None:
        self._closed.set()
        gc.callbacks.remove(self._gc_callback_ref)
        self._watchdog.join(timeout=2)
        self._drain_gc()
        self._events.put(None, timeout=2)
        self._writer.join(timeout=2)
        if not self._watchdog.is_alive():
            self._stacks.close()


class ObservedStateLock:
    """Observe ordinary capture-state lock acquisition without changing semantics."""

    def __init__(self, lock: Any, diagnostics: BenchmarkDiagnostics) -> None:
        self._lock = lock
        self._diagnostics = diagnostics
        self._owner: tuple[str, float] | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        label = self._diagnostics.label()
        started = time.perf_counter()
        acquired = self._lock.acquire(blocking, timeout)
        self._diagnostics.record(
            f"{label}.state_lock_wait", time.perf_counter() - started
        )
        if acquired:
            self._owner = (label, time.perf_counter())
        return acquired

    def release(self) -> None:
        owner = self._owner
        self._owner = None
        held = 0.0 if owner is None else time.perf_counter() - owner[1]
        self._lock.release()
        if owner is not None:
            self._diagnostics.record(f"{owner[0]}.state_lock_hold", held)

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> ObservedStateLock:
        self.acquire()
        return self

    def __exit__(self, *exception: Any) -> None:
        self.release()
