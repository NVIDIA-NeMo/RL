# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuVS vector index and reference implementations."""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from resources_servers.ai_search.retrieval.config import IndexConfig
from resources_servers.ai_search.retrieval.types import (
    FloatMatrix,
    IntMatrix,
    VectorIndex,
)


class NumpyExactIndex:
    """CPU exhaustive-search reference used for tests and recall measurement."""

    def __init__(self, vectors: FloatMatrix, config: IndexConfig) -> None:
        started = time.perf_counter()
        self.config = config
        self._vectors = np.asarray(vectors, dtype=np.float32, order="C")
        self._build_time_ms = (time.perf_counter() - started) * 1000.0

    @property
    def size(self) -> int:
        return self._vectors.shape[0]

    @property
    def dimension(self) -> int:
        return self._vectors.shape[1]

    @property
    def build_time_ms(self) -> float:
        return self._build_time_ms

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        query_matrix = np.asarray(queries, dtype=np.float32, order="C")
        if self.config.metric == "cosine":
            query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            vector_norms = np.linalg.norm(self._vectors, axis=1, keepdims=True).T
            scores = (query_matrix @ self._vectors.T) / np.maximum(
                query_norms * vector_norms, 1e-12
            )
        else:
            differences = query_matrix[:, None, :] - self._vectors[None, :, :]
            distances = np.einsum("qnd,qnd->qn", differences, differences)
            scores = -distances

        candidate_indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        indices = np.take_along_axis(candidate_indices, order, axis=1)
        sorted_scores = np.take_along_axis(candidate_scores, order, axis=1)
        return (
            np.asarray(indices, dtype=np.int64, order="C"),
            np.asarray(sorted_scores, dtype=np.float32, order="C"),
        )

    def close(self) -> None:
        """Release resources (the NumPy index owns no device allocations)."""


class CuvsVectorIndex:
    """GPU vector search backed by NVIDIA cuVS brute-force or CAGRA."""

    def __init__(self, vectors: FloatMatrix, config: IndexConfig) -> None:
        self.config = config
        self._size, self._dimension = vectors.shape
        self._lock = threading.Lock()
        self._dataset: Any = None

        try:
            # Import CuPy before cuVS. RAPIDS' wheel loader then resolves the
            # CUDA libraries supplied by the same environment consistently.
            import cupy as cp
            from cuvs.neighbors import brute_force, cagra
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "cuVS backend requested but cuvs-cu13/cupy-cuda13x could not be "
                "imported. Run the AI-search environment setup script."
            ) from error

        self._cp = cp
        self._brute_force = brute_force
        self._cagra = cagra

        started = time.perf_counter()
        index_path = config.serialized_index_path
        if index_path is not None and index_path.is_file():
            self._index = self._load(index_path)
        else:
            self._dataset = cp.asarray(np.asarray(vectors, dtype=np.float32, order="C"))
            self._index = self._build()
            if index_path is not None and config.save_built_index:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                self._save(index_path)
        cp.cuda.Stream.null.synchronize()
        self._build_time_ms = (time.perf_counter() - started) * 1000.0

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def build_time_ms(self) -> float:
        return self._build_time_ms

    def _build(self) -> Any:
        if self.config.algorithm == "brute_force":
            return self._brute_force.build(
                self._dataset,
                metric=self.config.metric,
            )

        max_degree = max(2, self.size - 1)
        graph_degree = min(self.config.graph_degree, max_degree)
        intermediate_degree = min(
            max(self.config.intermediate_graph_degree, graph_degree), max_degree
        )
        params = self._cagra.IndexParams(
            metric=self.config.metric,
            graph_degree=graph_degree,
            intermediate_graph_degree=intermediate_degree,
            build_algo=self.config.build_algorithm,
        )
        return self._cagra.build(params, self._dataset)

    def _load(self, path: Path) -> Any:
        if self.config.algorithm == "brute_force":
            return self._brute_force.load(str(path))
        return self._cagra.load(str(path))

    def _save(self, path: Path) -> None:
        if self.config.algorithm == "brute_force":
            self._brute_force.save(str(path), self._index, include_dataset=True)
        else:
            self._cagra.save(str(path), self._index, include_dataset=True)

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        query_matrix = self._cp.asarray(
            np.asarray(queries, dtype=np.float32, order="C")
        )
        with self._lock:
            if self.config.algorithm == "brute_force":
                distances, neighbors = self._brute_force.search(
                    self._index, query_matrix, k=top_k
                )
            else:
                search_params = self._cagra.SearchParams(
                    itopk_size=max(self.config.itopk_size, top_k),
                    search_width=self.config.search_width,
                )
                distances, neighbors = self._cagra.search(
                    search_params, self._index, query_matrix, top_k
                )
            self._cp.cuda.Stream.null.synchronize()

        host_neighbors = np.asarray(
            self._cp.asnumpy(self._cp.asarray(neighbors)),
            dtype=np.int64,
            order="C",
        )
        host_distances = np.asarray(
            self._cp.asnumpy(self._cp.asarray(distances)),
            dtype=np.float32,
            order="C",
        )
        if self.config.metric == "cosine":
            scores = 1.0 - host_distances
        else:
            # With normalized vectors, squared L2 = 2 - 2*cosine.
            scores = 1.0 - 0.5 * host_distances
        return host_neighbors, np.asarray(scores, dtype=np.float32, order="C")

    def close(self) -> None:
        """Destroy cuVS objects before Python unloads their CUDA modules."""
        index = getattr(self, "_index", None)
        if index is None:
            return
        self._index = None
        del index
        self._dataset = None
        self._cp.get_default_memory_pool().free_all_blocks()


def build_vector_index(vectors: FloatMatrix, config: IndexConfig) -> VectorIndex:
    """Construct a supported vector index from an explicit registry."""
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(
            f"Embeddings must have non-zero shape [documents, dimension], got {vectors.shape}"
        )
    if config.kind == "cuvs":
        return CuvsVectorIndex(vectors, config)
    if config.kind == "numpy":
        return NumpyExactIndex(vectors, config)
    raise ValueError(f"Unsupported vector index kind: {config.kind}")
