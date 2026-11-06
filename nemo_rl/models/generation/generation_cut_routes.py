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
"""Owned route chunks selected from append-only engine fragments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def concatenate_route_pieces(
    pieces: list[Any], *, dtype: np.dtype[Any], count: Callable[[str, int], None]
) -> np.ndarray:
    """Copy and validate integer fragments without an intermediate dtype cast."""
    if not pieces:
        raise ValueError("vLLM has not routed any token of the request yet")
    try:
        first_piece = pieces[0]
        if (
            not isinstance(first_piece, np.ndarray)
            or first_piece.dtype.kind not in "iu"
        ):
            raise TypeError("route fragments need integer dtypes")
        added = np.concatenate(pieces, axis=0, dtype=first_piece.dtype, casting="no")
    except (TypeError, ValueError):
        arrays = [np.asarray(piece) for piece in pieces]
        shape = tuple(arrays[0].shape[1:])
        if (
            any(
                array.ndim != 3
                or tuple(array.shape[1:]) != shape
                or array.dtype.kind not in "iu"
                for array in arrays
            )
            or not shape
            or min(shape) < 1
        ):
            raise ValueError(
                "vLLM route chunks must share shape [*, layers, topk]"
            ) from None
        added = np.concatenate(arrays, axis=0)
        count("route_history_bulk_fallbacks", 1)
    else:
        shape = tuple(added.shape[1:])
        if added.ndim != 3 or not shape or min(shape) < 1:
            raise ValueError("vLLM route chunks must share shape [*, layers, topk]")
        count("route_history_bulk_fast_calls", 1)
    if added.size and not np.can_cast(added.dtype, dtype, casting="safe"):
        limits = np.iinfo(dtype)
        if int(added.min()) < limits.min or int(added.max()) > limits.max:
            raise ValueError(
                f"vLLM routed expert IDs do not fit checkpoint dtype {dtype}"
            )
    return added


@dataclass
class RoutePieceIndex:
    """Small row offsets over immutable engine fragments; no route payload cache.

    Indexing is observation, not durability: failed writes may retain this
    metadata, while the next selection still uses the last sealed token count.
    One cut executor owns updates. The engine owns the append-only fragments.
    """

    ends: np.ndarray | None = field(default=None, repr=False)
    count: int = 0
    rows: int = 0
    last_piece: Any | None = field(default=None, repr=False)

    def select(self, pieces: list[Any], *, start: int, stop: int) -> list[Any]:
        """Return fragments covering exactly [start, stop), trimming two edges."""
        if not pieces:
            raise ValueError("vLLM has not routed any token of the request yet")
        if len(pieces) < self.count:
            raise ValueError("vLLM route-piece list became shorter")
        if self.count and pieces[self.count - 1] is not self.last_piece:
            raise ValueError("vLLM replaced a published route piece")
        if len(pieces) > self.count:
            lengths = np.fromiter(
                map(len, pieces[self.count :]),
                dtype=np.int64,
                count=len(pieces) - self.count,
            )
            ends = np.cumsum(lengths)
            ends += self.rows
            buffer = self.ends
            if buffer is None or len(pieces) > len(buffer):
                grown = np.empty(
                    max(len(pieces), 2 * len(buffer) if buffer is not None else 1),
                    dtype=np.int64,
                )
                if buffer is not None:
                    grown[: self.count] = buffer[: self.count]
                buffer = grown
            buffer[self.count : len(pieces)] = ends
            self.ends = buffer
            self.rows = int(ends[-1])
            self.count = len(pieces)
            self.last_piece = pieces[-1]
        if start < 0 or stop < start or stop > self.rows:
            raise ValueError(
                f"vLLM returned fewer routed experts than the frozen tokens need: range=[{start}, {stop}) rows={self.rows}"
            )
        if start == stop:
            # A one-token empty-prompt first cut needs only the final marker;
            # preserve model dimensions for an empty, validated route array.
            return [np.asarray(pieces[0])[:0]]
        assert self.ends is not None
        ends = self.ends[: self.count]
        first = int(np.searchsorted(ends, start, side="right"))
        last = int(np.searchsorted(ends, stop, side="left"))
        selected = pieces[first : last + 1]
        first_start = int(ends[first - 1]) if first else 0
        if first == last:
            selected[0] = selected[0][start - first_start : stop - first_start]
        else:
            if start != first_start:
                selected[0] = selected[0][start - first_start :]
            last_start = int(ends[last - 1]) if last else 0
            if stop != int(ends[last]):
                selected[-1] = selected[-1][: stop - last_start]
        return selected
