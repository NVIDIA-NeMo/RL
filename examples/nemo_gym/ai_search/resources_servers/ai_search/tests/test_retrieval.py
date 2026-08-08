# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the pluggable retrieval stack."""

import asyncio

import numpy as np
import pytest

from resources_servers.ai_search.retrieval.batching import AsyncSearchBatcher
from resources_servers.ai_search.retrieval.config import EncoderConfig, IndexConfig
from resources_servers.ai_search.retrieval.encoder import HashEncoder
from resources_servers.ai_search.retrieval.index import NumpyExactIndex
from resources_servers.ai_search.retrieval.types import (
    Document,
    SearchHit,
    SearchResult,
    SearchTimings,
)


def _encoder_config(dimension: int = 64) -> EncoderConfig:
    return EncoderConfig(
        kind="hash",
        model_name="deterministic-hash",
        device="cpu",
        dtype="float32",
        batch_size=8,
        max_length=64,
        dimension=dimension,
        query_prefix="",
        passage_prefix="",
        normalize=True,
        trust_remote_code=False,
    )


def _index_config() -> IndexConfig:
    return IndexConfig(
        kind="numpy",
        algorithm="brute_force",
        metric="cosine",
        serialized_index_path=None,
        save_built_index=False,
        graph_degree=8,
        intermediate_graph_degree=16,
        build_algorithm="ivf_pq",
        search_width=1,
        itopk_size=16,
    )


def test_hash_encoder_is_deterministic_and_normalized() -> None:
    encoder = HashEncoder(_encoder_config())
    first = encoder.encode_queries(["orchid relay", "cerulean quartz"])
    second = encoder.encode_queries(["orchid relay", "cerulean quartz"])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2))


def test_numpy_exact_index_returns_sorted_nearest_neighbors() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    index = NumpyExactIndex(vectors, _index_config())

    neighbors, scores = index.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32), top_k=2
    )

    assert neighbors.tolist() == [[0, 1]]
    assert scores[0, 0] >= scores[0, 1]


def test_numpy_backend_rejects_cagra() -> None:
    with pytest.raises(ValueError, match="only supports brute_force"):
        _index_config().model_copy(update={"algorithm": "cagra"}).model_validate(
            _index_config().model_dump() | {"algorithm": "cagra"}
        )


class _RecordingProvider:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def search_batch(self, queries: list[str], top_k: int) -> list[SearchResult]:
        self.batches.append(queries)
        timings = SearchTimings(
            queue_ms=0.0,
            encode_ms=1.0,
            index_ms=2.0,
            fetch_ms=3.0,
            total_ms=6.0,
            cache_hits=0,
            cache_misses=len(queries),
            batch_size=len(queries),
        )
        hits = tuple(
            SearchHit(
                document=Document(id=str(rank), title="title", text="text"),
                score=1.0 / rank,
                rank=rank,
            )
            for rank in range(1, top_k + 1)
        )
        return [
            SearchResult(query=query, hits=hits, timings=timings) for query in queries
        ]


@pytest.mark.asyncio
async def test_async_batcher_combines_concurrent_rollouts() -> None:
    provider = _RecordingProvider()
    batcher = AsyncSearchBatcher(provider, max_batch_size=8, wait_ms=5.0)

    first, second, third = await asyncio.gather(
        batcher.search("one", top_k=1),
        batcher.search("two", top_k=2),
        batcher.search("three", top_k=3),
    )
    await batcher.close()

    assert provider.batches == [["one", "two", "three"]]
    assert len(first.hits) == 1
    assert len(second.hits) == 2
    assert len(third.hits) == 3
    assert first.timings.queue_ms >= 0.0
