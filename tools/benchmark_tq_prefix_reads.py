# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare per-request and coalesced TQ reads on identical staged prefixes.

One Ray actor/client per simulated generation worker; all share simple TQ.
Writes and client setup are outside read timing. Read wall time includes arrival
pacing, client queueing, snapshot conversion and full payload verification, but
not lineage restore, cumulative cut-digest reconstruction or GPU prefill.
Multi-chunk requests group synthetic independent rows, exercising key fetching
and ordering rather than Gym's multi-cut lifecycle. Both modes read the same
data; alternate their order across repeats to expose warm-cache/order effects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import resource
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize pooled samples with nearest-rank percentiles."""
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
        "max": ordered[-1],
    }


async def read_workload(
    fetch: Callable[[list[str]], list[Any]],
    verify: Callable[[list[str], list[Any]], None],
    chains: list[list[str]],
    *,
    mode: str,
    concurrency: int,
    arrival_interval_ms: float,
    model_max_len: int,
    batch_size: int,
    batch_max_tokens: int,
    coalesce_ms: float,
) -> dict[str, Any]:
    """Exercise the production batcher, bounding outstanding rollout requests."""
    # Optional generation dependencies are not needed for CLI help/import.
    from nemo_rl.models.generation.prefix_read_batcher import PrefixReadBatcher

    if mode not in {"unbatched", "coalesced"} or concurrency < 1:
        raise ValueError("invalid mode or request concurrency")
    metrics: dict[str, list[float]] = {
        key: []
        for key in (
            "get_seconds",
            "keys_per_get",
            "queue_seconds",
            "request_seconds",
            "verify_seconds",
        )
    }
    submitted: dict[str, float] = {}
    lock = threading.Lock()

    def measured_fetch(keys: list[str]) -> list[Any]:
        start = time.perf_counter()
        rows = fetch(keys)
        elapsed = time.perf_counter() - start
        with lock:
            metrics["get_seconds"].append(elapsed)
            metrics["keys_per_get"].append(len(keys))
            metrics["queue_seconds"].extend(start - submitted[key] for key in keys)
        return rows

    reader = PrefixReadBatcher(
        measured_fetch,
        max_rows=batch_size,
        max_tokens=batch_max_tokens,
        wait_seconds=coalesce_ms / 1000,
    )
    slots = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def request(keys: list[str]) -> None:
        try:
            begin = time.perf_counter()
            with lock:
                submitted.update(dict.fromkeys(keys, begin))
            if mode == "coalesced":
                rows = await reader.fetch(keys, tokens_per_key=model_max_len)
            else:
                rows = await asyncio.to_thread(measured_fetch, keys)
            metrics["request_seconds"].append(time.perf_counter() - begin)
            verification_started = time.perf_counter()
            verify(keys, rows)
            metrics["verify_seconds"].append(time.perf_counter() - verification_started)
            with lock:
                for key in keys:
                    submitted.pop(key)
        finally:
            slots.release()

    try:
        async with asyncio.TaskGroup() as tasks:
            for index, chain in enumerate(chains):
                # Absolute arrival schedule: report wall time inclusive of pacing.
                due = started + index * arrival_interval_ms / 1000
                await asyncio.sleep(max(0, due - time.perf_counter()))
                await slots.acquire()
                tasks.create_task(request(chain))
    finally:
        await reader.aclose()
    return {
        "wall_seconds": time.perf_counter() - started,
        "verified_requests": len(chains),
        "verified_keys": sum(map(len, chains)),
        "samples": metrics,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


class ReadWorker:
    """Attach a process-local client, stage its rows, then replay read modes."""

    def __init__(
        self,
        dp_config: dict[str, Any],
        write_config: Any,
        args: dict[str, Any],
        start: int,
        stop: int,
    ) -> None:
        # Heavy TQ/Gym dependencies load in the container/actor, not CLI help.
        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
        from tools.benchmark_tq_prefix_writes import _benchmark_keys, _record_payload

        self.client = build_data_plane_client(dp_config, bootstrap=False)
        self.sink = TQTokenSink(
            self.client, staging_partition=write_config.partition_id
        )
        self.source = TQTokenSource(
            self.client, staging_partition=write_config.partition_id
        )
        self.config = write_config
        self.args = args
        chunks = args["chunks_per_rollout"]
        self.indices = range(start * chunks, stop * chunks)
        self.chains = [
            _benchmark_keys(write_config, list(range(i * chunks, (i + 1) * chunks)))
            for i in range(start, stop)
        ]
        self.expected = _record_payload(write_config.prefix_tokens)

    def ready(self) -> bool:
        return True

    def stage(self) -> int:
        # Reuse the existing production-sink write benchmark outside read timing.
        from tools.benchmark_tq_prefix_writes import _write_ranges

        _write_ranges(
            self.sink,
            self.indices,
            checkpoint_id=self.config.checkpoint_id,
            prefix_tokens=self.config.prefix_tokens,
            writers=1,
            batch_size=self.config.batch_size,
        )
        return len(self.indices)

    def verify(self, keys: list[str], rows: list[Any]) -> None:
        if len(keys) != len(rows):
            raise AssertionError("read count mismatch")
        for row in rows:
            if (
                row.token_ids_delta,
                row.token_mask_delta,
                row.generation_log_probs_delta,
            ) != self.expected:
                raise AssertionError("prefix payload mismatch")
        # TQTokenSource.fetch also validates each returned key's row identity.

    def read(self, mode: str) -> dict[str, Any]:
        return asyncio.run(
            read_workload(
                self.source.fetch,
                self.verify,
                self.chains,
                mode=mode,
                concurrency=self.args["requests_per_worker"],
                arrival_interval_ms=self.args["arrival_interval_ms"],
                model_max_len=self.args["model_max_len"],
                batch_size=self.args["batch_size"],
                batch_max_tokens=self.args["batch_max_tokens"],
                coalesce_ms=self.args["coalesce_ms"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in {
        "rows": 8192,
        "prefix-tokens": 16384,
        "workers": 128,
        "chunks-per-rollout": 1,
        "requests-per-worker": 64,
        "num-storage-units": 8,
        "batch-size": 256,
        "batch-max-tokens": 4_194_304,
        "write-batch-size": 256,
        "model-max-len": 32768,
        "repeats": 2,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--arrival-interval-ms", type=float, default=0)
    parser.add_argument("--coalesce-ms", type=float, default=2)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--progress-interval-s", type=float, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["both", "unbatched", "coalesced"],
        default="both",
        help="Read modes to measure; both alternates order across repeats.",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, (int, float)) and (
            not math.isfinite(value)
            or value < 0
            or (value == 0 and key not in {"arrival_interval_ms", "coalesce_ms"})
        ):
            parser.error(f"{key} must be positive (arrival/coalescing may be zero)")
    if args.prefix_tokens % args.chunks_per_rollout:
        parser.error("prefix-tokens must be divisible by chunks-per-rollout")
    if args.model_max_len < args.prefix_tokens:
        parser.error("model-max-len must cover prefix-tokens")

    # Optional container dependencies: keep --help available on login nodes.
    import ray
    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS
    from tools.benchmark_tq_prefix_writes import (
        BenchmarkConfig,
        _data_plane_config,
        _row_ranges,
        _wait_for_actor_results,
    )

    config = BenchmarkConfig(
        rows=args.rows * args.chunks_per_rollout,
        prefix_tokens=args.prefix_tokens // args.chunks_per_rollout,
        batch_size=args.write_batch_size,
        num_storage_units=args.num_storage_units,
        partition_id=f"prefix-read-{uuid.uuid4().hex}",
    )
    dp_config = _data_plane_config(config)
    client = build_data_plane_client(dp_config, bootstrap=True)
    actors = []
    output: dict[str, Any] = {
        "config": {**vars(args), "output_json": str(args.output_json)},
        "backend": "simple",
        "runs": [],
        "scope": "TQ fetch + conversion + verification; excludes writes, lineage restore and GPU prefill",
    }

    def collect(refs: list[Any], stage: str) -> list[Any]:
        return _wait_for_actor_results(
            refs,
            stage=stage,
            timeout_s=args.timeout_s,
            progress_interval_s=args.progress_interval_s,
        )

    def save() -> None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    try:
        client.register_partition(
            partition_id=config.partition_id,
            fields=list(STAGING_FIELDS),
            num_samples=config.rows,
            consumer_tasks=["benchmark"],
        )
        actor_type = ray.remote(num_cpus=0)(ReadWorker)
        actors = [
            actor_type.remote(dp_config, config, vars(args), part.start, part.stop)
            for part in _row_ranges(args.rows, args.workers)
        ]
        output["workers_used"] = len(actors)
        collect([actor.ready.remote() for actor in actors], "read_client_setup")
        start = time.perf_counter()
        staged = collect([actor.stage.remote() for actor in actors], "stage_prefixes")
        output["setup_write_seconds"] = time.perf_counter() - start
        if sum(staged) != config.rows:
            raise AssertionError("staged row count mismatch")
        for repeat in range(args.repeats):
            modes = (
                ["unbatched", "coalesced"]
                if repeat % 2 == 0
                else ["coalesced", "unbatched"]
            )
            if args.mode != "both":
                modes = [args.mode]
            for mode in modes:
                start = time.perf_counter()
                results = collect(
                    [actor.read.remote(mode) for actor in actors],
                    f"reads_{repeat}_{mode}",
                )
                elapsed = time.perf_counter() - start
                samples = {
                    key: [value for item in results for value in item["samples"][key]]
                    for key in results[0]["samples"]
                }
                record = {
                    "repeat": repeat,
                    "mode": mode,
                    "read_wall_seconds": elapsed,
                    "rollouts_per_second": args.rows / elapsed,
                    "get_calls": len(samples["get_seconds"]),
                    "verified_requests": sum(
                        item["verified_requests"] for item in results
                    ),
                    "verified_keys": sum(item["verified_keys"] for item in results),
                    "metrics": {
                        key: summarize(values) for key, values in samples.items()
                    },
                    "max_worker_lifetime_peak_rss_kib": max(
                        item["process_peak_rss_kib"] for item in results
                    ),
                }
                output["runs"].append(record)
                save()
                print(json.dumps(record), flush=True)
    finally:
        # Kill attached clients before clearing/closing the controller (same as
        # the write harness, avoiding teardown deadlock with attached clients).
        for actor in actors:
            ray.kill(actor, no_restart=True)
        client.clear_samples(sample_ids=None, partition_id=config.partition_id)
        client.close()


if __name__ == "__main__":
    main()
