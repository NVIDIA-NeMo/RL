# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal data contracts for retrieval providers and indexes."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]
IntMatrix = NDArray[np.int64]


@dataclass(frozen=True)
class Document:
    """One searchable document."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """One ranked document returned to the model."""

    document: Document
    score: float
    rank: int


@dataclass(frozen=True)
class SearchTimings:
    """Wall-clock time spent in each local retrieval stage."""

    queue_ms: float
    encode_ms: float
    index_ms: float
    fetch_ms: float
    total_ms: float
    cache_hits: int
    cache_misses: int
    batch_size: int


@dataclass(frozen=True)
class SearchResult:
    """Ranked hits and timing details for one query."""

    query: str
    hits: tuple[SearchHit, ...]
    timings: SearchTimings


class QueryEncoder(Protocol):
    """Text encoder contract used by a dense search provider."""

    @property
    def dimension(self) -> int:
        """Return the output embedding dimension."""
        ...

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        """Encode query strings into a row-major float32 matrix."""
        ...

    def encode_passages(self, texts: list[str]) -> FloatMatrix:
        """Encode document strings into a row-major float32 matrix."""
        ...

    def close(self) -> None:
        """Release encoder-owned device resources."""
        ...


class VectorIndex(Protocol):
    """Vector index contract; cuVS and reference implementations share it."""

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        ...

    @property
    def build_time_ms(self) -> float:
        """Return index construction or load time in milliseconds."""
        ...

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        """Return row indices and comparable similarity scores."""
        ...

    def close(self) -> None:
        """Release index-owned device resources."""
        ...


class SearchProvider(Protocol):
    """Backend-neutral contract that a future inverted index can implement."""

    def search_batch(self, queries: list[str], top_k: int) -> list[SearchResult]:
        """Search a batch of raw-text queries."""
        ...

    def close(self) -> None:
        """Release provider-owned resources."""
        ...
