# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure end-to-end local retrieval and its encode/index/fetch stages."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np

from resources_servers.ai_search.prepare_index import _load_runtime_config
from resources_servers.ai_search.retrieval.batching import AsyncSearchBatcher
from resources_servers.ai_search.retrieval.engine import DenseSearchEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("resources_servers/ai_search/configs/ai_search.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("resources_servers/ai_search/data/train.jsonl"),
    )
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": statistics.mean(values),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


async def _measure_microbatch(
    engine: DenseSearchEngine,
    queries: list[str],
    repeats: int,
) -> dict[str, float]:
    batcher = AsyncSearchBatcher(
        engine,
        max_batch_size=len(queries),
        wait_ms=engine.config.batch_wait_ms,
    )
    latencies: list[float] = []
    queue_ms: list[float] = []
    try:
        for repeat in range(repeats):
            varied = [f"{query} profile iteration {repeat}" for query in queries]
            started = time.perf_counter()
            results = await asyncio.gather(
                *(
                    batcher.search(query, engine.config.default_top_k)
                    for query in varied
                )
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            queue_ms.extend(result.timings.queue_ms for result in results)
    finally:
        await batcher.close()
    return {
        "wall_p50_ms": _percentiles(latencies)["p50"],
        "wall_p95_ms": _percentiles(latencies)["p95"],
        "queue_mean_ms": statistics.mean(queue_ms),
        "queries_per_second": len(queries) / (statistics.mean(latencies) / 1000.0),
    }


def _measure_serial_single_queries(
    engine: DenseSearchEngine,
    queries: list[str],
    repeats: int,
) -> dict[str, float]:
    latencies: list[float] = []
    for repeat in range(repeats):
        started = time.perf_counter()
        for query in queries:
            engine.search_batch(
                [f"{query} serial iteration {repeat}"],
                engine.config.default_top_k,
            )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return {
        "wall_p50_ms": _percentiles(latencies)["p50"],
        "wall_p95_ms": _percentiles(latencies)["p95"],
        "queries_per_second": len(queries) / (statistics.mean(latencies) / 1000.0),
    }


def main() -> None:
    args = _parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_queries = [row["question"] for row in rows]
    if max(batch_sizes) > len(base_queries):
        raise ValueError("largest batch size exceeds the dataset row count")

    config = _load_runtime_config(args.config)
    config.query_cache_size = 0
    init_started = time.perf_counter()
    engine = DenseSearchEngine(config)
    init_ms = (time.perf_counter() - init_started) * 1000.0

    measurements = []
    try:
        for batch_size in batch_sizes:
            queries = base_queries[:batch_size]
            for warmup in range(args.warmup):
                engine.search_batch(
                    [f"{query} warmup {warmup}" for query in queries],
                    config.default_top_k,
                )

            wall_ms: list[float] = []
            encode_ms: list[float] = []
            index_ms: list[float] = []
            fetch_ms: list[float] = []
            for repeat in range(args.repeats):
                varied_queries = [
                    f"{query} measured iteration {repeat}" for query in queries
                ]
                started = time.perf_counter()
                results = engine.search_batch(varied_queries, config.default_top_k)
                wall_ms.append((time.perf_counter() - started) * 1000.0)
                timings = results[0].timings
                encode_ms.append(timings.encode_ms)
                index_ms.append(timings.index_ms)
                fetch_ms.append(timings.fetch_ms)

            microbatch = asyncio.run(
                _measure_microbatch(engine, queries, max(3, args.repeats // 2))
            )
            serial_single = _measure_serial_single_queries(
                engine, queries, max(3, args.repeats // 2)
            )
            measurements.append(
                {
                    "batch_size": batch_size,
                    "wall_ms": _percentiles(wall_ms),
                    "encode_ms": _percentiles(encode_ms),
                    "index_ms": _percentiles(index_ms),
                    "fetch_ms": _percentiles(fetch_ms),
                    "direct_queries_per_second": batch_size
                    / (statistics.mean(wall_ms) / 1000.0),
                    "microbatch": microbatch,
                    "serial_single": serial_single,
                    "microbatch_speedup_vs_serial": (
                        microbatch["queries_per_second"]
                        / serial_single["queries_per_second"]
                    ),
                }
            )

        config.query_cache_size = 128
        cache_engine = DenseSearchEngine(config)
        try:
            query = base_queries[0]
            uncached = cache_engine.search_batch([query], config.default_top_k)[
                0
            ].timings
            cached = cache_engine.search_batch([query], config.default_top_k)[0].timings
            cache_measurement = {
                "uncached_encode_ms": uncached.encode_ms,
                "cached_encode_ms": cached.encode_ms,
                "cached_total_ms": cached.total_ms,
                "cache_hits": cached.cache_hits,
            }
        finally:
            cache_engine.close()
    finally:
        engine.close()

    report = {
        "schema_version": 1,
        "engine_init_ms": init_ms,
        "backend": f"{config.index.kind}/{config.index.algorithm}",
        "encoder": config.encoder.model_name,
        "documents": engine.stats.documents,
        "top_k": config.default_top_k,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measurements": measurements,
        "query_cache": cache_measurement,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
