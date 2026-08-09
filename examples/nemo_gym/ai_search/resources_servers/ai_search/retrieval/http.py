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

"""HTTP retrieval provider used by controlled cross-framework benchmarks."""

import time
from typing import Any

import requests

from resources_servers.ai_search.retrieval.config import SearchRuntimeConfig
from resources_servers.ai_search.retrieval.types import (
    Document,
    SearchHit,
    SearchResult,
    SearchTimings,
)


class HttpSearchProvider:
    """Call a Search-R1-compatible batch retrieval endpoint."""

    def __init__(self, config: SearchRuntimeConfig) -> None:
        if config.http_url is None:
            raise ValueError("http_url is required for HttpSearchProvider")
        self._url = config.http_url
        self._timeout_s = config.http_timeout_s
        self._session = requests.Session()

    def search_batch(self, queries: list[str], top_k: int) -> list[SearchResult]:
        """Retrieve a batch and convert it to the local provider contract."""
        if not queries:
            return []
        started = time.perf_counter()
        response = self._session.post(
            self._url,
            json={"queries": queries, "topk": top_k, "return_scores": True},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        raw_batches = payload.get("result")
        if not isinstance(raw_batches, list) or len(raw_batches) != len(queries):
            raise ValueError(
                "Retriever response result count does not match query count: "
                f"{len(raw_batches) if isinstance(raw_batches, list) else 'invalid'} "
                f"!= {len(queries)}"
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings = SearchTimings(
            queue_ms=0.0,
            encode_ms=0.0,
            index_ms=0.0,
            fetch_ms=0.0,
            total_ms=elapsed_ms,
            cache_hits=0,
            cache_misses=len(queries),
            batch_size=len(queries),
        )

        results: list[SearchResult] = []
        for query, raw_hits in zip(queries, raw_batches):
            if not isinstance(raw_hits, list):
                raise ValueError("Retriever response contains a non-list result batch")
            hits: list[SearchHit] = []
            for rank, raw_hit in enumerate(raw_hits, start=1):
                raw_document = raw_hit["document"]
                hits.append(
                    SearchHit(
                        document=Document(
                            id=str(raw_document["id"]),
                            title=str(raw_document["title"]),
                            text=str(raw_document["text"]),
                        ),
                        score=float(raw_hit["score"]),
                        rank=rank,
                    )
                )
            results.append(
                SearchResult(query=query, hits=tuple(hits), timings=timings)
            )
        return results

    def close(self) -> None:
        """Close the persistent HTTP connection pool."""
        self._session.close()
