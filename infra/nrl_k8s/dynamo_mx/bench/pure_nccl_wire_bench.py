#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-rank, GPU-to-GPU NCCL broadcast benchmark with no vLLM engine.

The communicator is the repository's ``StatelessProcessGroup`` backed by
``nccl.core``. Rank 0 owns and preloads the source bytes; rank 1 allocates the
matching receive buffer. CUDA events measure only warm/measured broadcasts.
Allocation, preload, and communicator initialization are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PAYLOAD_BYTES = 61_064_245_248
SCHEMA_VERSION = "pure-nccl-wire-v1"
WORLD_SIZE = 2


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((percentile_value / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def statistics_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    normalized = [float(value) for value in values]
    return {
        "samples": len(normalized),
        "min": min(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "p95": percentile(normalized, 95),
        "max": max(normalized) if normalized else None,
    }


def build_result(
    *,
    byte_count: int,
    warmups: int,
    rank_seconds: dict[int, list[float]],
    init_seconds: dict[int, float],
    allocation_seconds: dict[int, float],
    preload_seconds: dict[int, float],
) -> dict[str, Any]:
    if set(rank_seconds) != {0, 1}:
        raise ValueError("result requires timing lists for ranks 0 and 1")
    sample_count = len(rank_seconds[0])
    if len(rank_seconds[1]) != sample_count:
        raise ValueError("result requires equally sized timing lists for ranks 0 and 1")
    critical_seconds = [
        max(rank_seconds[0][index], rank_seconds[1][index])
        for index in range(sample_count)
    ]
    if any(seconds <= 0 for seconds in critical_seconds):
        raise ValueError("wire timing samples must be positive")
    effective_gbps = [
        byte_count * 8 / seconds / 1e9 for seconds in critical_seconds
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "nccl",
        "benchmark": "pure_nccl_broadcast",
        "status": "ok",
        "world_size": WORLD_SIZE,
        "source_rank": 0,
        "receiver_rank": 1,
        "bytes": byte_count,
        "warmup_iterations": warmups,
        "measured_iterations": sample_count,
        "timing_scope": "CUDA-event broadcast completion; critical-path max across ranks",
        "communicator_init_seconds": {
            str(rank): init_seconds[rank] for rank in sorted(init_seconds)
        },
        "allocation_seconds": {
            str(rank): allocation_seconds[rank] for rank in sorted(allocation_seconds)
        },
        "source_preload_seconds": preload_seconds[0],
        "wire_seconds": statistics_summary(critical_seconds),
        "effective_gbps": statistics_summary(effective_gbps),
        "raw_iterations": [
            {
                "iteration": index,
                "rank_seconds": {
                    "0": rank_seconds[0][index],
                    "1": rank_seconds[1][index],
                },
                "wire_seconds": critical_seconds[index],
                "effective_gbps": effective_gbps[index],
            }
            for index in range(sample_count)
        ],
        "notes": [
            "Communicator initialization and source preload are excluded from wire timing.",
            "This is a two-rank transport microbenchmark and does not run vLLM.",
        ],
    }


def csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, unit in (("wire_seconds", "seconds"), ("effective_gbps", "Gbps")):
        summary = result[metric]
        rows.append(
            {
                "schema_version": result["schema_version"],
                "benchmark": result["benchmark"],
                "metric": metric,
                "unit": unit,
                "samples": summary["samples"],
                "min": summary["min"],
                "median": summary["median"],
                "p95": summary["p95"],
                "max": summary["max"],
                "bytes": result["bytes"],
                "warmup_iterations": result["warmup_iterations"],
            }
        )
    return rows


def write_outputs(result: dict[str, Any], json_path: str, csv_path: str) -> None:
    json_output = Path(json_path)
    csv_output = Path(csv_path) if csv_path else json_output.with_suffix(".csv")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = csv_rows(result)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "role",
        choices=("sender", "receiver", "torchrun"),
        help="sender=rank 0, receiver=rank 1, torchrun=read RANK/WORLD_SIZE",
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get("MASTER_ADDR", ""),
        help="rank-0 pod IP or hostname (defaults to MASTER_ADDR)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("NCCL_BENCH_MASTER_PORT", "29610")),
        help="benchmark TCPStore port; keep distinct from torchrun rendezvous port",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--fill-byte", type=int, default=165)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--result-csv", default="")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.master_address:
        parser.error("--master-address or MASTER_ADDR is required")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1, 65535]")
    if args.bytes < 1:
        parser.error("--bytes must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not 0 <= args.fill_byte <= 255:
        parser.error("--fill-byte must be in [0, 255]")
    return args


def resolve_rank_and_device(args: argparse.Namespace) -> tuple[int, int]:
    if args.role == "torchrun":
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        rank = int(os.environ.get("RANK", "-1"))
        if world_size != WORLD_SIZE or rank not in (0, 1):
            raise ValueError("torchrun mode requires WORLD_SIZE=2 and RANK=0 or 1")
        default_device = int(os.environ.get("LOCAL_RANK", str(rank)))
    else:
        rank = 0 if args.role == "sender" else 1
        default_device = 0
    return rank, args.device if args.device is not None else default_device


def _exchange_json(store: Any, rank: int, name: str, value: Any) -> dict[int, Any]:
    store.set(f"{name}/{rank}", json.dumps(value))
    keys = [f"{name}/{other}" for other in range(WORLD_SIZE)]
    store.wait(keys)
    return {
        other: json.loads(store.get(keys[other]).decode("utf-8"))
        for other in range(WORLD_SIZE)
    }


def _init_group(args: argparse.Namespace, rank: int, device: int, torch: Any):
    """Return ``(tcp_store, broadcast_fn)`` for a two-rank NCCL group.

    Prefers NeMo-RL's ``StatelessProcessGroup`` (matches the production refit
    path) and falls back to plain ``torch.distributed`` with a ``TCPStore`` so the
    baseline also runs in images that do not ship ``nemo_rl`` (e.g. the
    model-express-dev sender/receiver pods).
    """
    try:
        from nemo_rl.distributed.stateless_process_group import (
            StatelessProcessGroup,
        )
    except ModuleNotFoundError:
        import datetime as _dt

        import torch.distributed as dist

        timeout = _dt.timedelta(seconds=600)
        store = torch.distributed.TCPStore(
            args.master_address,
            args.master_port,
            WORLD_SIZE,
            rank == 0,
            timeout=timeout,
        )
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timeout,
        )

        def broadcast(payload, stream):
            with torch.cuda.stream(stream):
                dist.broadcast(payload, src=0)

        return store, broadcast

    group = StatelessProcessGroup(
        master_address=args.master_address,
        port=args.master_port,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    group.init_nccl_communicator(device=device)

    def broadcast(payload, stream):
        group.broadcast(payload, src=0, stream=stream)

    return group.tcp_store, broadcast


def run(args: argparse.Namespace) -> int:
    import torch

    rank, device = resolve_rank_and_device(args)
    torch.cuda.set_device(device)

    allocation_start = time.perf_counter()
    payload = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{device}")
    torch.cuda.synchronize(device)
    allocation_seconds = time.perf_counter() - allocation_start

    preload_seconds = 0.0
    if rank == 0:
        preload_start = time.perf_counter()
        payload.fill_(args.fill_byte)
        torch.cuda.synchronize(device)
        preload_seconds = time.perf_counter() - preload_start

    init_start = time.perf_counter()
    store, broadcast = _init_group(args, rank, device, torch)
    torch.cuda.synchronize(device)
    init_seconds = time.perf_counter() - init_start

    measured: list[float] = []
    for iteration in range(args.warmups + args.iterations):
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        broadcast(payload, stream)
        end.record(stream)
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        if iteration >= args.warmups:
            measured.append(seconds)
        label = "warmup" if iteration < args.warmups else "measured"
        print(
            f"NCCL_WIRE rank={rank} {label}={iteration} seconds={seconds:.6f}",
            flush=True,
        )

    rank_seconds = _exchange_json(store, rank, "measured_seconds", measured)
    init_by_rank = _exchange_json(store, rank, "init_seconds", init_seconds)
    allocation_by_rank = _exchange_json(
        store, rank, "allocation_seconds", allocation_seconds
    )
    preload_by_rank = _exchange_json(store, rank, "preload_seconds", preload_seconds)

    if rank == 0:
        result = build_result(
            byte_count=args.bytes,
            warmups=args.warmups,
            rank_seconds=rank_seconds,
            init_seconds=init_by_rank,
            allocation_seconds=allocation_by_rank,
            preload_seconds=preload_by_rank,
        )
        write_outputs(result, args.result_json, args.result_csv)
        print("NCCL_WIRE_RESULT " + json.dumps(result), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
