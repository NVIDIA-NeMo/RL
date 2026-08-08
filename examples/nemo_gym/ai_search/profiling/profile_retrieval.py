# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare exact and approximate vector-search backends on one workload."""

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import psutil

from resources_servers.ai_search.retrieval.config import IndexConfig
from resources_servers.ai_search.retrieval.index import build_vector_index
from resources_servers.ai_search.retrieval.types import FloatMatrix, IntMatrix


class _BenchmarkIndex(Protocol):
    """Small common interface used only by this benchmark."""

    build_time_ms: float

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        """Search one host-side query batch and return host-side arrays."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class _ConfiguredIndex:
    def __init__(self, values: FloatMatrix, config: IndexConfig) -> None:
        self._index = build_vector_index(values, config)
        self.build_time_ms = self._index.build_time_ms

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        return self._index.search(queries, top_k)

    def close(self) -> None:
        self._index.close()


class _FaissCpuIndex:
    def __init__(self, values: FloatMatrix) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "faiss-cpu is not installed; install the profile extra"
            ) from error

        started = time.perf_counter()
        self._index = faiss.IndexFlatIP(values.shape[1])
        self._index.add(values)
        self.build_time_ms = (time.perf_counter() - started) * 1000.0

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        scores, neighbors = self._index.search(queries, top_k)
        return (
            np.asarray(neighbors, dtype=np.int64, order="C"),
            np.asarray(scores, dtype=np.float32, order="C"),
        )

    def close(self) -> None:
        self._index = None


class _TorchCudaIndex:
    def __init__(self, values: FloatMatrix) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self._torch = torch
        started = time.perf_counter()
        self._values = torch.from_numpy(values).to("cuda")
        torch.cuda.synchronize()
        self.build_time_ms = (time.perf_counter() - started) * 1000.0

    def search(self, queries: FloatMatrix, top_k: int) -> tuple[IntMatrix, FloatMatrix]:
        torch = self._torch
        query_tensor = torch.from_numpy(queries).to("cuda")
        scores, neighbors = torch.topk(query_tensor @ self._values.T, k=top_k, dim=1)
        torch.cuda.synchronize()
        return (
            np.asarray(neighbors.cpu().numpy(), dtype=np.int64, order="C"),
            np.asarray(scores.cpu().numpy(), dtype=np.float32, order="C"),
        )

    def close(self) -> None:
        self._values = None
        self._torch.cuda.empty_cache()


@dataclass(frozen=True)
class SearchMeasurement:
    backend: str
    batch_size: int
    latency_p50_ms: float
    latency_p95_ms: float
    queries_per_second: float
    recall_at_k: float


@dataclass(frozen=True)
class BackendMeasurement:
    backend: str
    build_ms: float | None
    host_memory_delta_mib: float | None
    gpu_memory_delta_mib: float | None
    searches: list[SearchMeasurement]
    error: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--distribution",
        choices=["clustered", "isotropic"],
        default="clustered",
        help="Clustered is closer to semantic embeddings; isotropic is an ANN stress test.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def _index_config(backend: str) -> IndexConfig:
    is_cagra = backend == "cuvs_cagra"
    return IndexConfig(
        kind="numpy" if backend == "numpy" else "cuvs",
        algorithm="cagra" if is_cagra else "brute_force",
        metric="sqeuclidean" if is_cagra else "cosine",
        serialized_index_path=None,
        save_built_index=False,
        graph_degree=64,
        intermediate_graph_degree=128,
        build_algorithm="nn_descent",
        search_width=1,
        # cuVS documents this as CAGRA's primary accuracy/speed knob. This
        # value reaches full recall on the default clustered workload while
        # retaining a useful latency advantage at larger index sizes.
        itopk_size=256,
    )


def _build_backend(name: str, values: FloatMatrix) -> _BenchmarkIndex:
    if name in {"numpy", "cuvs_brute_force", "cuvs_cagra"}:
        return _ConfiguredIndex(values, _index_config(name))
    if name == "faiss_cpu":
        return _FaissCpuIndex(values)
    if name == "torch_cuda":
        return _TorchCudaIndex(values)
    raise ValueError(f"Unknown backend: {name}")


def _normalize(values: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.maximum(norms, 1e-12), dtype=np.float32, order="C")


def _generate_workload(
    rng: np.random.Generator,
    documents: int,
    dimension: int,
    queries: int,
    distribution: str,
) -> tuple[FloatMatrix, FloatMatrix]:
    if distribution == "isotropic":
        values = _normalize(
            rng.standard_normal((documents, dimension), dtype=np.float32)
        )
        query_values = _normalize(
            rng.standard_normal((queries, dimension), dtype=np.float32)
        )
        return values, query_values

    cluster_count = min(2048, max(32, documents // 32))
    centers = _normalize(
        rng.standard_normal((cluster_count, dimension), dtype=np.float32)
    )
    assignments = rng.integers(0, cluster_count, size=documents)
    values = rng.standard_normal((documents, dimension), dtype=np.float32)
    values *= 0.025
    values += centers[assignments]
    values = _normalize(values)

    selected_rows = rng.integers(0, documents, size=queries)
    query_values = values[selected_rows].copy()
    query_values += rng.standard_normal(query_values.shape, dtype=np.float32) * 0.005
    return values, _normalize(query_values)


def _recall_at_k(actual: IntMatrix, expected: IntMatrix) -> float:
    return float(
        np.mean(
            [
                len(set(row).intersection(reference)) / expected.shape[1]
                for row, reference in zip(actual, expected)
            ]
        )
    )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _gpu_free_bytes() -> int | None:
    try:
        import cupy as cp

        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        return int(free_bytes)
    except (ImportError, RuntimeError):
        return None


def _hardware() -> dict[str, str | int]:
    gpu = "unavailable"
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "machine": platform.node(),
        "python": platform.python_version(),
    }


def _markdown(report: dict) -> str:
    workload = report["workload"]
    lines = [
        "# Vector retrieval profile",
        "",
        (
            f"Workload: {workload['documents']:,} documents x {workload['dimension']} dimensions, "
            f"top-{workload['top_k']}, {workload['distribution']} vectors, "
            f"{workload['repeats']} measured repetitions."
        ),
        "",
        "| Backend | Build ms | Batch | p50 ms | p95 ms | queries/s | recall@k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend in report["backends"]:
        if backend["error"]:
            lines.append(f"| {backend['backend']} | error | - | - | - | - | - |")
            continue
        for index, search in enumerate(backend["searches"]):
            build = f"{backend['build_ms']:.2f}" if index == 0 else ""
            lines.append(
                f"| {backend['backend']} | {build} | {search['batch_size']} | "
                f"{search['latency_p50_ms']:.3f} | {search['latency_p95_ms']:.3f} | "
                f"{search['queries_per_second']:.1f} | {search['recall_at_k']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Latency includes query transfer and result transfer. Index build/load is reported separately.",
            "NumPy exact search supplies the ground-truth neighbors; CAGRA recall is measured against them.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if min(batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if args.top_k > args.documents:
        raise ValueError("top-k cannot exceed document count")

    rng = np.random.default_rng(args.seed)
    values, queries = _generate_workload(
        rng,
        documents=args.documents,
        dimension=args.dimension,
        queries=max(batch_sizes),
        distribution=args.distribution,
    )

    reference = _ConfiguredIndex(values, _index_config("numpy"))
    expected_by_batch = {
        batch_size: reference.search(queries[:batch_size], args.top_k)[0]
        for batch_size in batch_sizes
    }
    reference.close()

    process = psutil.Process()
    backends: list[BackendMeasurement] = []
    for backend_name in [
        "numpy",
        "faiss_cpu",
        "torch_cuda",
        "cuvs_brute_force",
        "cuvs_cagra",
    ]:
        host_before = process.memory_info().rss
        gpu_before = _gpu_free_bytes()
        index: _BenchmarkIndex | None = None
        try:
            index = _build_backend(backend_name, values)
            host_after = process.memory_info().rss
            gpu_after = _gpu_free_bytes()
            searches: list[SearchMeasurement] = []
            for batch_size in batch_sizes:
                query_batch = queries[:batch_size]
                for _ in range(args.warmup):
                    index.search(query_batch, args.top_k)

                latencies: list[float] = []
                actual = None
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    actual, _ = index.search(query_batch, args.top_k)
                    latencies.append((time.perf_counter() - started) * 1000.0)
                assert actual is not None
                p50 = statistics.median(latencies)
                searches.append(
                    SearchMeasurement(
                        backend=backend_name,
                        batch_size=batch_size,
                        latency_p50_ms=p50,
                        latency_p95_ms=_percentile(latencies, 95),
                        queries_per_second=batch_size
                        / (statistics.mean(latencies) / 1000.0),
                        recall_at_k=_recall_at_k(actual, expected_by_batch[batch_size]),
                    )
                )
            backends.append(
                BackendMeasurement(
                    backend=backend_name,
                    build_ms=index.build_time_ms,
                    host_memory_delta_mib=(host_after - host_before) / (1024**2),
                    gpu_memory_delta_mib=(
                        (gpu_before - gpu_after) / (1024**2)
                        if gpu_before is not None and gpu_after is not None
                        else None
                    ),
                    searches=searches,
                    error=None,
                )
            )
        except (
            Exception
        ) as error:  # Keep one missing optional backend from hiding other results.
            backends.append(
                BackendMeasurement(
                    backend=backend_name,
                    build_ms=None,
                    host_memory_delta_mib=None,
                    gpu_memory_delta_mib=None,
                    searches=[],
                    error=f"{type(error).__name__}: {error}",
                )
            )
        finally:
            if index is not None:
                index.close()

    report = {
        "schema_version": 1,
        "hardware": _hardware(),
        "workload": {
            "documents": args.documents,
            "dimension": args.dimension,
            "top_k": args.top_k,
            "batch_sizes": batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "distribution": args.distribution,
        },
        "backends": [asdict(backend) for backend in backends],
    }
    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = _markdown(report)
    print(rendered_markdown)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered_json, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(rendered_markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
