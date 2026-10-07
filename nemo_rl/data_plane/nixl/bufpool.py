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
"""Registered host buffers for one client process, reused across operations.

Registration is the expensive part of moving bytes with NIXL: a fresh 1 GiB
buffer costs ~70 ms to register and deregister on GB300, against ~10 ms to
copy it with torch's parallel ``copy_`` (scripts/bench_register_vs_copy.py).
So the pool keeps buffers registered:

- ``base_count`` buffers of ``base_bytes``, registered at start. An op that
  fits takes one; with several, independent ops proceed concurrently.
- Larger ops get a buffer rounded up to a power of two, registered on first
  use and **kept** (LRU, free buffers only) until ``max_oversize_bytes`` of
  oversize buffers are registered. Only then is anything deregistered.

The endpoint is not thread-safe, so register / deregister go through
``ep_lock``; the pool's own bookkeeping has its own lock.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


def _round_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


class BufferPool:
    def __init__(
        self,
        ep: Any,
        ep_lock: Any,
        *,
        base_bytes: int,
        base_count: int = 2,
        max_oversize_bytes: int = 8 << 30,
    ) -> None:
        self.ep = ep
        self.ep_lock = ep_lock
        self.base_bytes = int(base_bytes)
        self.max_oversize_bytes = int(max_oversize_bytes)
        self._cv = threading.Condition()
        self._free_base: list[tuple[int, np.ndarray]] = []
        for _ in range(max(1, int(base_count))):
            self._free_base.append(
                self._register(np.empty(self.base_bytes, dtype=np.uint8))
            )
        self._oversize_free: OrderedDict[int, list[tuple[int, np.ndarray]]] = (
            OrderedDict()
        )
        self._oversize_bytes = 0  # registered oversize bytes, free or in use
        self.stats = {
            "base_waits": 0,
            "oversize_hits": 0,
            "oversize_registers": 0,
            "oversize_evictions": 0,
        }

    def _register(self, buf: np.ndarray) -> tuple[int, np.ndarray]:
        with self.ep_lock:
            return self.ep.register(buf), buf

    def _deregister(self, addr: int) -> None:
        with self.ep_lock:
            self.ep.deregister(addr)

    @contextmanager
    def acquire(self, nbytes: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(registered_addr, uint8 array)`` with at least ``nbytes``."""
        if nbytes <= self.base_bytes:
            with self._cv:
                if not self._free_base:
                    self.stats["base_waits"] += 1
                while not self._free_base:
                    self._cv.wait()
                item = self._free_base.pop()
            try:
                yield item
            finally:
                with self._cv:
                    self._free_base.append(item)
                    self._cv.notify()
            return

        size = _round_pow2(nbytes)
        item = None
        with self._cv:
            bucket = self._oversize_free.get(size)
            if bucket:
                item = bucket.pop()
                self._oversize_free.move_to_end(size)
                self.stats["oversize_hits"] += 1
        if item is None:
            self._evict_for(size)
            item = self._register(np.empty(size, dtype=np.uint8))
            with self._cv:
                self._oversize_bytes += size
                self.stats["oversize_registers"] += 1
        try:
            yield item
        finally:
            with self._cv:
                self._oversize_free.setdefault(size, []).append(item)
                self._oversize_free.move_to_end(size)

    def _evict_for(self, incoming: int) -> None:
        """Deregister least-recently-used free oversize buffers to stay in budget."""
        victims: list[int] = []
        with self._cv:
            while (
                self._oversize_bytes + incoming > self.max_oversize_bytes
                and self._oversize_free
            ):
                size, bucket = next(iter(self._oversize_free.items()))
                addr, _ = bucket.pop()
                if not bucket:
                    del self._oversize_free[size]
                self._oversize_bytes -= size
                self.stats["oversize_evictions"] += 1
                victims.append(addr)
        for addr in victims:
            self._deregister(addr)

    def close(self) -> None:
        with self._cv:
            addrs = [a for a, _ in self._free_base]
            addrs += [a for bucket in self._oversize_free.values() for a, _ in bucket]
            self._free_base.clear()
            self._oversize_free.clear()
        for addr in addrs:
            self._deregister(addr)
