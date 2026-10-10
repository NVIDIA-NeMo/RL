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

A buffer that a failed transfer may still use (``ep.touches_unfinished``: a
READ writing into it, or a WRITE reading from it) is never handed out again. It stays referenced and registered in ``retired``
until the process exits; a lost base buffer is replaced with a fresh one. If
that replacement cannot be registered the pool shrinks; once no base buffer
is left, ``acquire`` raises :class:`PoolExhausted` instead of waiting forever.

The endpoint is not thread-safe, so register / deregister go through
``ep_lock``; the pool's own bookkeeping has its own lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


log = logging.getLogger(__name__)


class PoolExhausted(RuntimeError):
    """No base buffer is left, or none came free before the caller's timeout."""


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
        self._base_live = len(self._free_base)  # base buffers free or in use
        self._oversize_free: OrderedDict[int, list[tuple[int, np.ndarray]]] = (
            OrderedDict()
        )
        self._oversize_bytes = 0  # registered oversize bytes, free or in use
        self.retired: list[tuple[int, np.ndarray]] = []
        self.stats = {
            "base_waits": 0,
            "oversize_hits": 0,
            "oversize_registers": 0,
            "oversize_evictions": 0,
            "retired_bytes": 0,
            "replace_failures": 0,
        }

    def _register(self, buf: np.ndarray) -> tuple[int, np.ndarray]:
        with self.ep_lock:
            return self.ep.register(buf), buf

    def _deregister(self, addr: int) -> None:
        with self.ep_lock:
            self.ep.deregister(addr)

    @contextmanager
    def acquire(
        self, nbytes: int, timeout_s: float | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(registered_addr, uint8 array)`` with at least ``nbytes``.

        Waits at most ``timeout_s`` (forever if None) for a base buffer.
        """
        if nbytes <= self.base_bytes:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            with self._cv:
                if not self._free_base:
                    self.stats["base_waits"] += 1
                while not self._free_base:
                    if self._base_live == 0:
                        raise PoolExhausted(
                            "no base buffer left: retired buffers could not be "
                            f"replaced ({self.stats['replace_failures']} failures)"
                        )
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        raise PoolExhausted(f"no base buffer free within {timeout_s}s")
                    self._cv.wait(left)
                item = self._free_base.pop()
            try:
                yield item
            finally:
                if self._retire(item):
                    item = self._replace_base()
                with self._cv:
                    if item is None:
                        self._base_live -= 1
                        self._cv.notify_all()  # waiters may now have to give up
                    else:
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
            if self._retire(item):
                with self._cv:
                    self._oversize_bytes -= size
            else:
                with self._cv:
                    self._oversize_free.setdefault(size, []).append(item)
                    self._oversize_free.move_to_end(size)

    def _replace_base(self) -> tuple[int, np.ndarray] | None:
        """A fresh registered base buffer, or None if registration fails.

        Never raises: it runs in ``acquire``'s ``finally`` and must not mask
        the caller's own exception.
        """
        try:
            return self._register(np.empty(self.base_bytes, dtype=np.uint8))
        except Exception:  # noqa: BLE001 - out of memory or registration failure
            log.exception("could not replace a retired base buffer; pool shrinks")
            with self._cv:
                self.stats["replace_failures"] += 1
            return None

    def _retire(self, item: tuple[int, np.ndarray]) -> bool:
        """Keep ``item`` out of the pool if a failed transfer may still use it."""
        addr, buf = item
        if not self.ep.touches_unfinished(addr, buf.nbytes):
            return False
        with self._cv:
            self.retired.append(item)
            self.stats["retired_bytes"] += buf.nbytes
        return True

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
