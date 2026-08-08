# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense retrieval orchestration: encode, search, and fetch documents."""

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resources_servers.ai_search.retrieval.config import SearchRuntimeConfig
from resources_servers.ai_search.retrieval.corpus import (
    JsonlDocumentStore,
    sha256_file,
)
from resources_servers.ai_search.retrieval.encoder import (
    build_encoder,
    encoder_manifest,
)
from resources_servers.ai_search.retrieval.index import build_vector_index
from resources_servers.ai_search.retrieval.types import (
    FloatMatrix,
    SearchHit,
    SearchResult,
    SearchTimings,
)


@dataclass(frozen=True)
class EngineStats:
    """Static properties useful in logs and benchmark reports."""

    documents: int
    dimension: int
    index_kind: str
    index_algorithm: str
    index_build_ms: float


class DenseSearchEngine:
    """Backend-neutral dense search provider built around a vector index."""

    def __init__(self, config: SearchRuntimeConfig) -> None:
        self.config = config
        self._store = JsonlDocumentStore(config.corpus_path)
        self._embeddings = self._load_embeddings(config.embeddings_path)
        self._validate_artifacts()
        self._encoder = build_encoder(config.encoder)
        if self._encoder.dimension != self._embeddings.shape[1]:
            raise ValueError(
                "Encoder and embedding dimensions differ: "
                f"{self._encoder.dimension} != {self._embeddings.shape[1]}"
            )
        self._index = build_vector_index(self._embeddings, config.index)
        self._query_cache: OrderedDict[str, FloatMatrix] = OrderedDict()

    @staticmethod
    def _load_embeddings(path: Path) -> FloatMatrix:
        if not path.is_file():
            raise FileNotFoundError(
                f"Embedding file does not exist: {path}. Run prepare_index.py first."
            )
        embeddings = np.load(path, mmap_mode="r")
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Embeddings must be float32, got {embeddings.dtype} in {path}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a rank-2 matrix, got shape {embeddings.shape}"
            )
        return np.asarray(embeddings, dtype=np.float32, order="C")

    def _validate_artifacts(self) -> None:
        if len(self._store) != self._embeddings.shape[0]:
            raise ValueError(
                "Corpus and embedding row counts differ: "
                f"{len(self._store)} != {self._embeddings.shape[0]}"
            )

        manifest_path = self.config.embeddings_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Embedding manifest does not exist: {manifest_path}. "
                "Run prepare_index.py so corpus/encoder compatibility can be checked."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_encoder = encoder_manifest(self.config.encoder)
        if manifest.get("encoder") != expected_encoder:
            raise ValueError(
                "Embedding encoder metadata does not match the runtime config: "
                f"artifact={manifest.get('encoder')}, runtime={expected_encoder}"
            )
        if manifest.get("documents") != len(self._store):
            raise ValueError("Embedding manifest document count is stale")
        if manifest.get("dimension") != self._embeddings.shape[1]:
            raise ValueError("Embedding manifest dimension is stale")
        if self.config.verify_artifact_hash:
            actual_hash = sha256_file(self.config.corpus_path)
            if manifest.get("corpus_sha256") != actual_hash:
                raise ValueError(
                    "Corpus contents changed after embeddings were generated; "
                    "run prepare_index.py again"
                )

    @property
    def stats(self) -> EngineStats:
        return EngineStats(
            documents=len(self._store),
            dimension=self._embeddings.shape[1],
            index_kind=self.config.index.kind,
            index_algorithm=self.config.index.algorithm,
            index_build_ms=self._index.build_time_ms,
        )

    def _encode_queries(self, queries: list[str]) -> tuple[FloatMatrix, int, int]:
        vectors: list[FloatMatrix | None] = [None] * len(queries)
        misses: dict[str, list[int]] = {}
        cache_hits = 0
        for position, query in enumerate(queries):
            cached = self._query_cache.get(query)
            if cached is not None:
                cache_hits += 1
                self._query_cache.move_to_end(query)
                vectors[position] = cached
            else:
                misses.setdefault(query, []).append(position)

        if misses:
            missing_queries = list(misses)
            encoded = self._encoder.encode_queries(missing_queries)
            for query, vector in zip(missing_queries, encoded):
                row = np.asarray(vector, dtype=np.float32, order="C").reshape(1, -1)
                for position in misses[query]:
                    vectors[position] = row
                if self.config.query_cache_size > 0:
                    self._query_cache[query] = row
                    self._query_cache.move_to_end(query)
                    while len(self._query_cache) > self.config.query_cache_size:
                        self._query_cache.popitem(last=False)

        if any(vector is None for vector in vectors):  # pragma: no cover
            raise RuntimeError("Internal error while assembling query embeddings")
        matrix = np.concatenate(vectors, axis=0)  # type: ignore[arg-type]
        return matrix, cache_hits, len(misses)

    def search_batch(self, queries: list[str], top_k: int) -> list[SearchResult]:
        """Encode, retrieve, and materialize results for one query batch."""
        if not queries:
            return []
        if top_k < 1 or top_k > min(self.config.max_top_k, len(self._store)):
            raise ValueError(
                f"top_k must be in [1, {min(self.config.max_top_k, len(self._store))}], got {top_k}"
            )

        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        query_vectors, cache_hits, cache_misses = self._encode_queries(queries)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        index_started = time.perf_counter()
        indices, scores = self._index.search(query_vectors, top_k)
        index_ms = (time.perf_counter() - index_started) * 1000.0

        fetch_started = time.perf_counter()
        all_hits: list[tuple[SearchHit, ...]] = []
        for row_indices, row_scores in zip(indices, scores):
            hits: list[SearchHit] = []
            for rank, (row_index, score) in enumerate(
                zip(row_indices, row_scores), start=1
            ):
                document = self._store.get(int(row_index))
                if len(document.text) > self.config.max_passage_chars:
                    document = type(document)(
                        id=document.id,
                        title=document.title,
                        text=document.text[: self.config.max_passage_chars] + "…",
                    )
                hits.append(SearchHit(document=document, score=float(score), rank=rank))
            all_hits.append(tuple(hits))
        fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        timings = SearchTimings(
            queue_ms=0.0,
            encode_ms=encode_ms,
            index_ms=index_ms,
            fetch_ms=fetch_ms,
            total_ms=total_ms,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            batch_size=len(queries),
        )
        return [
            SearchResult(query=query, hits=hits, timings=timings)
            for query, hits in zip(queries, all_hits)
        ]

    def close(self) -> None:
        """Release encoder and vector-index device resources."""
        self._index.close()
        self._encoder.close()
