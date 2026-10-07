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
"""First-fit slab allocator with aligned allocations and timed quarantine.

Blobs are immutable and freed whole, so the allocator only needs
``alloc(nbytes)`` and ``free(off, nbytes)``. A freed region may still be
the target of an in-flight one-sided read, so callers place it in
*quarantine* with a release time instead of freeing it outright;
``reclaim(now)`` moves expired regions back to the free list.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from nemo_rl.data_plane.nixl.errors import UnitFull


def align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


@dataclass(order=True)
class _Range:
    off: int
    size: int


class SlabAllocator:
    def __init__(self, capacity: int, align: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if align <= 0 or (align & (align - 1)):
            raise ValueError("align must be a positive power of two")
        self.capacity = capacity
        self.align = align
        self._free: list[_Range] = [
            _Range(0, capacity)
        ]  # sorted by off, non-overlapping
        self._quarantine: list[tuple[float, int, int]] = []  # (release_at, off, size)
        self.used_bytes = 0

    # ------------------------------------------------------------------ alloc / free
    def alloc(self, nbytes: int) -> int:
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        size = align_up(nbytes, self.align)
        for i, r in enumerate(self._free):
            if r.size >= size:
                off = r.off
                if r.size == size:
                    del self._free[i]
                else:
                    r.off += size
                    r.size -= size
                self.used_bytes += size
                return off
        raise UnitFull(
            f"need {size} B, free {self.free_bytes} B (largest {self.largest_free} B)"
        )

    def free(self, off: int, nbytes: int) -> None:
        size = align_up(nbytes, self.align)
        if off < 0 or off + size > self.capacity:
            raise ValueError(f"free out of range: off={off} size={size}")
        r = _Range(off, size)
        i = bisect.bisect_left(self._free, r)
        # merge with right neighbour
        if i < len(self._free) and self._free[i].off == off + size:
            r.size += self._free[i].size
            del self._free[i]
        # merge with left neighbour
        if i > 0 and self._free[i - 1].off + self._free[i - 1].size == off:
            self._free[i - 1].size += r.size
        else:
            self._free.insert(i, r)
        self.used_bytes -= size

    # ------------------------------------------------------------------ quarantine
    def quarantine(self, off: int, nbytes: int, release_at: float) -> None:
        """Defer ``free`` until ``release_at`` (monotonic seconds)."""
        self._quarantine.append((release_at, off, nbytes))

    def reclaim(self, now: float) -> int:
        """Free every quarantined region whose release time has passed."""
        if not self._quarantine:
            return 0
        keep: list[tuple[float, int, int]] = []
        n = 0
        for item in self._quarantine:
            if item[0] <= now:
                self.free(item[1], item[2])
                n += 1
            else:
                keep.append(item)
        self._quarantine = keep
        return n

    # ------------------------------------------------------------------ stats
    @property
    def free_bytes(self) -> int:
        return sum(r.size for r in self._free)

    @property
    def quarantined_bytes(self) -> int:
        return sum(align_up(s, self.align) for _, _, s in self._quarantine)

    @property
    def largest_free(self) -> int:
        return max((r.size for r in self._free), default=0)

    @property
    def fragments(self) -> int:
        return len(self._free)
