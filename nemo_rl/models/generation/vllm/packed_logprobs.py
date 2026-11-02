# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Packed selected-token histories for captured FINAL_ONLY vLLM requests."""

from __future__ import annotations

import operator
from array import array
from collections.abc import Callable, Sequence
from typing import Any, overload

import numpy as np


class PackedSelectedLogprobs(Sequence[dict[int, Any]]):
    """Own three numeric buffers; materialize native Logprob mappings on demand.

    Exactly one token is stored at each position, so position offsets are
    implicit. Binary64 preserves the Python floats produced by native float32
    and float64 logprob arrays. No view of a growable buffer escapes this class.
    Appends run under the worker's existing publication lock; cuts pin a length
    there and read only that append-only prefix afterward.
    """

    __slots__ = ("_ids", "_values", "_ranks", "_length", "_factory", "__weakref__")

    def __init__(self, logprob_factory: Callable[..., Any]) -> None:
        self._ids: array[int] = array("q")
        self._values: array[float] = array("d")
        self._ranks: array[int] = array("q")
        self._length = 0
        self._factory = logprob_factory
        if self._ids.itemsize != 8 or self._values.itemsize != 8:
            raise RuntimeError("Packed capture requires 64-bit integer/double buffers")

    def __len__(self) -> int:
        return self._length

    @property
    def payload_bytes(self) -> int:
        """Logical numeric payload size, excluding capacity and object overhead."""
        return self._length * 24

    def append_arrays(
        self, ids: np.ndarray, values: np.ndarray, ranks: np.ndarray
    ) -> None:
        """Append selected-token arrays, rolling back on conversion/allocation errors."""
        if (
            ids.ndim != 2
            or ids.shape[1] != 1
            or values.shape != ids.shape
            or ranks.shape != (ids.shape[0],)
        ):
            raise ValueError("Expected one selected token and rank per position")
        if ids.dtype.kind not in "iu" or ranks.dtype.kind not in "iu":
            raise ValueError("Packed token IDs and ranks must be integers")
        if values.dtype.kind != "f" or values.dtype.itemsize > 8:
            raise ValueError(
                "Packed logprobs require floating point values up to binary64"
            )
        for integers in (ids, ranks):
            if integers.dtype.kind == "u" and integers.dtype.itemsize == 8:
                if np.any(integers > np.iinfo(np.int64).max):
                    raise OverflowError(
                        "Packed token ID/rank exceeds signed 64-bit range"
                    )
        count = ids.shape[0]
        before = self._length
        try:
            if count == 1:
                self._ids.append(int(ids[0, 0]))
                self._values.append(float(values[0, 0]))
                self._ranks.append(int(ranks[0]))
            elif count:
                self._ids.frombytes(np.asarray(ids[:, 0], dtype=np.int64).tobytes())
                self._values.frombytes(
                    np.asarray(values[:, 0], dtype=np.float64).tobytes()
                )
                self._ranks.frombytes(np.asarray(ranks, dtype=np.int64).tobytes())
        except BaseException:
            # Never publish a partially appended position, including on OOM.
            del self._ids[before:]
            del self._values[before:]
            del self._ranks[before:]
            raise
        self._length = before + count

    def cumulative_after(self, previous: float, *, start: int) -> float:
        """Match native sequential Python float addition, including NaN/inf."""
        for position in range(start, self._length):
            previous += self._values[position]
        return previous

    def selected_values(
        self, token_ids: Sequence[int], *, start: int, stop: int, floor: float
    ) -> list[float]:
        """Copy a frozen prefix, checking alignment without per-token mappings."""
        if not 0 <= start <= stop <= self._length or len(token_ids) != stop - start:
            raise ValueError(
                "Selected logprob bounds do not match the published prefix"
            )
        selected = []
        for position, token_id in enumerate(token_ids, start):
            if self._ids[position] != token_id:
                raise RuntimeError("Packed logprobs do not include the selected token")
            selected.append(max(self._values[position], floor))
        return selected

    @overload
    def __getitem__(self, index: int) -> dict[int, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> PackedSelectedLogprobs: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[int, Any] | PackedSelectedLogprobs:
        if isinstance(index, slice):
            # Slice explicitly against published length, including negative steps.
            indices = range(*index.indices(self._length))
            result = PackedSelectedLogprobs(self._factory)
            result._ids = array("q", (self._ids[i] for i in indices))
            result._values = array("d", (self._values[i] for i in indices))
            result._ranks = array("q", (self._ranks[i] for i in indices))
            result._length = len(indices)
            return result
        position = operator.index(index)
        if position < 0:
            position += self._length
        if not 0 <= position < self._length:
            raise IndexError("Packed logprob position is out of range")
        return {
            self._ids[position]: self._factory(
                logprob=self._values[position],
                rank=self._ranks[position],
                decoded_token=None,
            )
        }
