# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU proxy: real TQ prefix writes inside Gym model prepare.

Includes production sink encoding, bounded concurrent PUTs, worker journals,
model-ledger commit, and post-timing lineage/coordinate restore verification.
Excludes vLLM cuts, agent state/ACKs, tools, TQ snapshot save/load, and networking
between physical nodes. Shared mode uses logical partitions in one process;
per-worker mode uses one synchronous Ray actor/client per generation worker.
Synthetic records are built before prepare; their immutable payload lists are
shared to avoid holding a second Python-list copy of the entire TQ dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import resource
import struct
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


class CachedRecordBuilder:
    """Benchmark-only v2 digest encoder for a fixed, validated payload.

    Owns a validated copy of the template. Returned records share its lists;
    callers must treat them as immutable. Only IDs and digest vary. No TQ
    tensors are precomputed: sink encoding remains inside prepare timing.
    """

    def __init__(self, template: Any) -> None:
        # Gym is optional for the dependency-light scheduling tests.
        from nemo_gym.token_id_capture.staging import digest as wire
        from nemo_gym.token_id_capture.staging.records import StagedCallRecord

        self._template = StagedCallRecord.model_validate(template.model_dump())
        record = self._template
        if (
            record.schema_version,
            record.digest_version,
            record.extras_digest_version,
        ) != (2, 2, 1):
            raise ValueError("Cached benchmark builder requires staging digest v2")
        self._header = wire._CALL_DIGEST_DOMAIN + struct.pack(
            ">BBB",
            record.schema_version,
            record.digest_version,
            record.extras_digest_version,
        )
        # This is exactly the suffix following the two IDs in the production
        # v2 wire layout. Reuse its encoders, including length/presence markers.
        self._suffix = b"".join(
            (
                wire._encode_optional_text(record.parent_call_id),
                wire._encode_text(record.mode),
                wire._encode_uint(record.prev_len, field="prev_len"),
                wire._encode_uint(record.delta_len, field="delta_len"),
                wire._encode_uint(record.cum_len, field="cum_len"),
                wire._encode_uint(record.weight_version, field="weight_version"),
                wire._encode_bytes(wire.encode_token_ids(record.token_ids_delta)),
                wire._encode_bytes(
                    wire._encode_float32_values(
                        record.token_mask_delta, field="token_mask_delta"
                    )
                ),
                wire._encode_bytes(
                    wire._encode_float32_values(
                        record.generation_log_probs_delta,
                        field="generation_log_probs_delta",
                    )
                ),
                bytes.fromhex(record.extras_digest),
                wire._encode_present_digest(record.chain_hash, field="chain_hash"),
                wire._encode_present_digest(
                    record.cumulative_hash, field="cumulative_hash"
                ),
            )
        )
        # Fail closed on wire-layout drift before generating a large inventory.
        if (
            self.build(
                rollout_id=record.rollout_id, model_call_id=record.model_call_id
            ).digest
            != record.digest
        ):
            raise ValueError(
                "Cached benchmark digest disagrees with production encoder"
            )
        probe = self.build(rollout_id="benchmark-parity-π", model_call_id="call-parity")
        StagedCallRecord.model_validate(probe.model_dump())

    def build(self, *, rollout_id: str, model_call_id: str) -> Any:
        """Reuse validated payload, hashing the exact ID-specific v2 bytes."""
        digest = hashlib.sha256(self._header)
        for value in (rollout_id, model_call_id):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "rollout_id and model_call_id must be non-empty strings"
                )
            encoded = value.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
        digest.update(self._suffix)
        return self._template.model_copy(
            update={
                "rollout_id": rollout_id,
                "model_call_id": model_call_id,
                "digest": digest.hexdigest(),
            }
        )


def worker_for_row(index: int, *, rows: int, workers: int) -> int:
    """Contiguous balanced ownership, including uneven and tiny workloads."""
    base, extra = divmod(rows, workers)
    boundary = (base + 1) * extra
    if index < boundary:
        return index // (base + 1)
    return extra + (index - boundary) // base


def latency_summary(values: list[float]) -> dict[str, float | int]:
    """Nearest-rank latency percentiles, in seconds."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "sum_s": sum(ordered),
        "mean_s": sum(ordered) / len(ordered),
        "p50_s": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95_s": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99_s": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "max_s": ordered[-1],
    }


class TimedPutClient:
    """Instrument the exact synchronous client API boundary used by the sink."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.lock = threading.Lock()
        self.latencies: list[float] = []
        self.batch_rows: list[int] = []
        self.failures = 0
        self.active = 0
        self.peak_active = 0

    def put_samples(self, **kwargs: Any) -> Any:
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        started = time.perf_counter()
        succeeded = False
        try:
            result = self.client.put_samples(**kwargs)
            succeeded = True
            return result
        finally:
            elapsed = time.perf_counter() - started
            with self.lock:
                self.active -= 1
                self.latencies.append(elapsed)
                self.batch_rows.append(len(kwargs["sample_ids"]))
                self.failures += int(not succeeded)


class BoundedStager:
    """Bound whole sink calls (encoding plus PUT), independently of owners."""

    def __init__(self, sink: Any, *, concurrency: int) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.sink = sink
        self.semaphore = asyncio.Semaphore(concurrency)
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.waits: list[float] = []
        self.stages: list[float] = []
        self.first_started: float | None = None
        self.last_finished: float | None = None

    async def stage(self, records: list[Any], checkpoint_id: str) -> list[Any]:
        queued = time.perf_counter()
        if self.first_started is None:
            self.first_started = queued
        async with self.semaphore:
            started = time.perf_counter()
            self.waits.append(started - queued)
            results = await asyncio.get_running_loop().run_in_executor(
                self.pool,
                lambda: self.sink.stage_generation_prefix_batch(
                    records,
                    checkpoint_id=checkpoint_id,
                    chunk_sequences=[0] * len(records),
                ),
            )
            self.last_finished = time.perf_counter()
            self.stages.append(self.last_finished - started)
            if len(results) != len(records) or any(not result.ok for result in results):
                raise RuntimeError(
                    "Prefix staging failed; no successful cut receipt will be published"
                )
            return results

    def close(self) -> None:
        """Join outstanding writes before the client/storage can be closed."""
        self.pool.shutdown(wait=True, cancel_futures=True)


class RealTQCutBackend:
    """Replace synthetic coordinates with acknowledged TQ prefix coordinates."""

    def __init__(
        self,
        owner: str,
        records: list[Any],
        stager: BoundedStager | PerWorkerStager,
        args: argparse.Namespace,
    ) -> None:
        self.owner = owner
        self.records = records
        self.stager = stager
        self.args = args
        self.keys: dict[int, str] = {}

    async def checkpoint_generation_cut(self, inventory: Any) -> Any:
        # Gym is optional; only the integrated runtime needs these contracts.
        from nemo_gym._checkpoint.model_control_contracts import (
            GenerationCutPrefixAck,
            GenerationCutReceipt,
        )

        groups: dict[int, list[Any]] = defaultdict(list)
        for prefix in inventory.active_prefixes:
            index = int(prefix.rollout_id.rsplit("-", 1)[1])
            groups[
                worker_for_row(
                    index, rows=self.args.rows, workers=self.args.generation_workers
                )
            ].append(prefix)

        async def write_owner(prefixes: list[Any]) -> list[Any]:
            acknowledgements = []
            for offset in range(0, len(prefixes), self.args.batch_size):
                batch = prefixes[offset : offset + self.args.batch_size]
                indices = [int(prefix.rollout_id.rsplit("-", 1)[1]) for prefix in batch]
                records = [self.records[index] for index in indices]
                results = await self.stager.stage(records, inventory.checkpoint_id)
                for prefix, index, record, result in zip(
                    batch, indices, records, results, strict=True
                ):
                    if (
                        prefix.model_call_id != record.model_call_id
                        or prefix.rollout_id != record.rollout_id
                    ):
                        raise RuntimeError(
                            "Cut inventory identity does not match staged record"
                        )
                    self.keys[index] = result.staging_key
                    acknowledgements.append(
                        GenerationCutPrefixAck(
                            **prefix.model_dump(mode="json"),
                            disposition="durable_prefix",
                            cut_kind="active_prefix",
                            frozen_buffer_id=f"active/{inventory.checkpoint_id}/{prefix.ticket_id}",
                            staging_keys=(result.staging_key,),
                            prefix_token_count=self.args.prefix_tokens,
                            prefix_digest=record.digest,
                            effective_output_limit=self.args.prefix_tokens + 1,
                        )
                    )
            return acknowledgements

        tasks = [asyncio.create_task(write_owner(group)) for group in groups.values()]
        try:
            chunks = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"cut-{self.owner}-{inventory.inventory_digest[:16]}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"snapshot-{self.owner}-{inventory.inventory_digest[:16]}",
            prefixes=tuple(prefix for chunk in chunks for prefix in chunk),
        )


class GenerationWriter:
    """Synchronous Ray actor implementation: one local client, one batch at a time."""

    def __init__(
        self, dp_config: Any, partition: str, start: int, stop: int, prefix_tokens: int
    ) -> None:
        # Ray/TQ dependencies are needed only inside actual writer processes.
        import ray
        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink
        from tools.benchmark_tq_prefix_writes import _build_record, _record_payload

        self.start = start
        self.stop = stop
        self.measured = TimedPutClient(
            build_data_plane_client(dp_config, bootstrap=False)
        )
        self.sink = TQTokenSink(self.measured, staging_partition=partition)
        started = time.perf_counter()
        builder = CachedRecordBuilder(_build_record(0, *_record_payload(prefix_tokens)))
        self.records = {
            i: builder.build(
                rollout_id=f"rollout-{i:09d}", model_call_id=f"call-{i:09d}"
            )
            for i in range(start, stop)
        }
        self.placement = {
            "pid": os.getpid(),
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "start": start,
            "stop": stop,
            "record_setup_seconds": time.perf_counter() - started,
        }

    def ready(self) -> dict[str, Any]:
        return self.placement

    def stage(self, indices: list[int], checkpoint_id: str) -> dict[str, Any]:
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(i not in self.records for i in indices)
        ):
            raise ValueError(
                "Batch must contain unique rows owned by this generation worker"
            )
        before = len(self.measured.latencies)
        started = time.perf_counter()
        results = self.sink.stage_generation_prefix_batch(
            [self.records[i] for i in indices],
            checkpoint_id=checkpoint_id,
            chunk_sequences=[0] * len(indices),
        )
        stage_seconds = time.perf_counter() - started
        if len(results) != len(indices) or any(not result.ok for result in results):
            raise RuntimeError(
                "Worker prefix staging failed; no cut receipt may be published"
            )
        return {
            "results": [result.model_dump() for result in results],
            "stage_seconds": stage_seconds,
            "put_latencies": self.measured.latencies[before:],
            "batch_rows": self.measured.batch_rows[before:],
        }


class PerWorkerStager:
    """Route ID-only batch requests to independent, pre-initialized clients.

    No shared eight-slot gate: each actor serializes its own batches while
    separate actors run concurrently. Ray RPC time is measured separately.
    """

    def __init__(self, measured: TimedPutClient, args: argparse.Namespace) -> None:
        self.measured = measured
        self.args = args
        self.actors: list[Any] = []
        self.placements: list[dict[str, Any]] = []
        self.setup_seconds = 0.0
        self.waits: list[float] = []
        self.stages: list[float] = []
        self.rpc_latencies: list[float] = []
        self.first_started: float | None = None
        self.last_finished: float | None = None
        self.active_requests = 0
        self.peak_requests = 0

    def setup(self, dp_config: Any, partition: str) -> None:
        # Optional Ray dependency only needed for the process-isolated mode.
        import ray
        from tools.benchmark_tq_prefix_writes import _wait_for_actor_results

        started = time.perf_counter()
        writer_class = ray.remote(
            num_cpus=self.args.worker_cpus,
            max_concurrency=1,
            max_restarts=0,
            max_task_retries=0,
        )(GenerationWriter)
        width = self.args.rows // self.args.generation_workers
        for owner in range(self.args.generation_workers):
            self.actors.append(
                writer_class.remote(
                    dp_config,
                    partition,
                    owner * width,
                    (owner + 1) * width,
                    self.args.prefix_tokens,
                )
            )
        self.placements = _wait_for_actor_results(
            [actor.ready.remote() for actor in self.actors],
            stage="proxy_worker_setup",
            timeout_s=self.args.client_setup_timeout_s,
            progress_interval_s=10,
        )
        self.setup_seconds = time.perf_counter() - started

    async def stage(self, records: list[Any], checkpoint_id: str) -> list[Any]:
        indices = [int(record.rollout_id.rsplit("-", 1)[1]) for record in records]
        owners = {
            worker_for_row(i, rows=self.args.rows, workers=self.args.generation_workers)
            for i in indices
        }
        if len(owners) != 1:
            raise ValueError("Batch crosses generation-worker ownership")
        started = time.perf_counter()
        if self.first_started is None:
            self.first_started = started
        self.active_requests += 1
        self.peak_requests = max(self.peak_requests, self.active_requests)
        try:
            payload = await self.actors[owners.pop()].stage.remote(
                indices, checkpoint_id
            )
        finally:
            self.active_requests -= 1
        self.last_finished = time.perf_counter()
        self.rpc_latencies.append(self.last_finished - started)
        self.stages.append(payload["stage_seconds"])
        self.measured.latencies.extend(payload["put_latencies"])
        self.measured.batch_rows.extend(payload["batch_rows"])
        return [SimpleNamespace(**result) for result in payload["results"]]

    def close(self) -> None:
        # Attached clients must not close the shared TQ deployment. Kill only
        # actors created by this harness, as in the existing multi-client test.
        import ray

        for actor in self.actors:
            ray.kill(actor, no_restart=True)
        self.actors.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("rows", 512),
        ("prefix-tokens", 16384),
        ("generation-workers", 128),
        ("gym-workers", 8),
        ("batch-size", 256),
        ("num-storage-units", 8),
        ("verify-rows", 256),
        ("timeout-s", 1800),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--client-mode", choices=("shared", "per-worker"), default="shared"
    )
    parser.add_argument(
        "--put-concurrency",
        type=int,
        default=None,
        help="Shared-client mode only; defaults to 8",
    )
    parser.add_argument(
        "--worker-cpus",
        type=float,
        default=0.25,
        help="Ray scheduling CPU reservation per writer actor; not a CPU thread cap",
    )
    parser.add_argument("--client-setup-timeout-s", type=int, default=300)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Artifact filesystem; use shared storage for representative commit timing",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.client_mode == "per-worker" and args.put_concurrency is not None:
        parser.error(
            "--put-concurrency applies only to shared mode; per-worker writes are sequential per actor"
        )
    if args.client_mode == "shared" and args.put_concurrency is None:
        args.put_concurrency = 8
    if not math.isfinite(args.worker_cpus) or args.worker_cpus <= 0:
        parser.error("worker-cpus must be finite and positive")
    for name, value in vars(args).items():
        if isinstance(value, int) and value < 1:
            parser.error(f"{name} must be positive")
    # Aligned contiguous policy inventories keep each generation owner within
    # exactly one policy participant, avoiding artificial cross-inventory batches.
    if (
        args.generation_workers % args.gym_workers
        or args.rows % args.generation_workers
    ):
        parser.error(
            "generation-workers must divide rows and be divisible by gym-workers"
        )
    return args


async def with_progress(
    operation: Awaitable[dict[str, Any]],
    measured: TimedPutClient,
    stager: BoundedStager | PerWorkerStager | None = None,
) -> dict[str, Any]:
    """Report progress through prepare, commit and post-timing ledger restore."""

    async def report() -> None:
        started = time.perf_counter()
        while True:
            await asyncio.sleep(10)
            with measured.lock:
                calls = len(measured.latencies)
                rows = sum(measured.batch_rows)
                active = measured.active
                failures = measured.failures
            activity = (
                f"remote_stage_requests={stager.active_requests}"
                if isinstance(stager, PerWorkerStager)
                else f"active_puts={active}"
            )
            print(
                f"proxy scope=prepare_commit_restore_verify puts_completed={calls} attempted_rows={rows} {activity} failed_puts={failures} pipeline_elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    reporter = asyncio.create_task(report())
    try:
        return await operation
    finally:
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)


def load_checkpoint_driver() -> ModuleType:
    """Load the adjacent Gym benchmark without relying on a scripts package."""
    gym_script = (
        Path(__file__).resolve().parents[1]
        / "3rdparty/Gym-workspace/Gym/scripts/benchmark_checkpoint_cpu.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gym_checkpoint_proxy_driver", gym_script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {gym_script}")
    gym = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gym
    spec.loader.exec_module(gym)
    return gym


def main() -> None:
    # Heavy Ray/Torch/Gym imports are optional for helper unit tests.
    from tools.benchmark_tq_prefix_writes import (
        BenchmarkConfig,
        _build_record,
        _data_plane_config,
        _record_payload,
        _verification_indices,
    )
    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.data_plane.tq_token_sink import (
        STAGING_FIELDS,
        TQTokenSink,
        TQTokenSource,
        generation_cut_staging_key,
    )

    args = parse_args()
    gym = load_checkpoint_driver()
    args.root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="prefix-proxy-", dir=args.root))
    config = BenchmarkConfig(
        rows=args.rows,
        prefix_tokens=args.prefix_tokens,
        num_storage_units=args.num_storage_units,
    )
    started = time.perf_counter()
    payload = _record_payload(args.prefix_tokens)
    records = []
    print("proxy stage=record_setup started (excluded from prepare)", flush=True)
    print(
        f"proxy token_tensor_bytes_lower_bound={args.rows * args.prefix_tokens * 16} (excludes metadata, staging copies and Ray overhead)",
        flush=True,
    )
    builder = CachedRecordBuilder(_build_record(0, *payload))
    for index in range(args.rows):
        record = builder.build(
            rollout_id=f"rollout-{index:09d}",
            model_call_id=f"call-{index:09d}",
        )
        records.append(record)
        if (index + 1) % 1024 == 0:
            print(f"proxy stage=record_setup rows={index + 1}/{args.rows}", flush=True)
    record_setup_seconds = time.perf_counter() - started
    client = build_data_plane_client(_data_plane_config(config), bootstrap=True)
    partition = config.partition_id
    measured = TimedPutClient(client)
    stager = (
        PerWorkerStager(measured, args)
        if args.client_mode == "per-worker"
        else BoundedStager(
            TQTokenSink(measured, staging_partition=partition),
            concurrency=args.put_concurrency,
        )
    )
    backends: list[RealTQCutBackend] = []

    def backend_factory(owner: str) -> RealTQCutBackend:
        backend = RealTQCutBackend(owner, records, stager, args)
        backends.append(backend)
        return backend

    try:
        client.register_partition(
            partition_id=partition,
            fields=list(STAGING_FIELDS),
            num_samples=args.rows,
            consumer_tasks=["benchmark"],
        )
        if isinstance(stager, PerWorkerStager):
            stager.setup(_data_plane_config(config), partition)
        print("proxy stage=checkpoint started", flush=True)
        checkpoint = asyncio.run(
            with_progress(
                gym.run_checkpoint_benchmark(
                    run_root,
                    workers=args.gym_workers,
                    cuts=args.rows,
                    hot_worker_fraction=1 / args.gym_workers,
                    prefix_tokens=args.prefix_tokens,
                    staging_key_bytes=0,
                    timeout_s=args.timeout_s,
                    backend_factory=backend_factory,
                    staging_keys_for_ticket=lambda ticket, checkpoint_id: (
                        generation_cut_staging_key(
                            checkpoint_id,
                            ticket.rollout_id,
                            ticket.model_call_id,
                            chunk_sequence=0,
                        ),
                    ),
                ),
                measured,
                stager,
            )
        )
        keys = {
            index: key for backend in backends for index, key in backend.keys.items()
        }
        if len(keys) != args.rows or set(client.list_sample_ids(partition)) != set(
            keys.values()
        ):
            raise RuntimeError(
                "TQ key inventory does not match successful cut receipts"
            )
        source = TQTokenSource(client, staging_partition=partition)
        indices = _verification_indices(args.rows, args.verify_rows)
        for offset in range(0, len(indices), 16):
            batch = indices[offset : offset + 16]
            if source.fetch_prefix_token_ids(
                [keys[index] for index in batch]
            ) != payload[0] * len(batch):
                raise RuntimeError("Stored prefix token verification failed")
        if stager.first_started is None or stager.last_finished is None:
            raise RuntimeError("No prefix staging was measured")
        write_window = stager.last_finished - stager.first_started
        result = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "scope": "model prepare + real prefix PUTs + model-ledger commit; excludes agent commit and TQ snapshot",
            "run_root": str(run_root),
            "clients": args.generation_workers
            if args.client_mode == "per-worker"
            else 1,
            "driver_bootstrap_client": True,
            "client_setup_seconds": stager.setup_seconds
            if isinstance(stager, PerWorkerStager)
            else 0.0,
            "writer_placements": stager.placements
            if isinstance(stager, PerWorkerStager)
            else [],
            "record_setup_seconds": record_setup_seconds,
            "record_builder": "cached_v2_payload",
            "checkpoint": checkpoint,
            "proxy_prepare_commit_seconds": checkpoint["prepare_commit_seconds"],
            "prefix_write_window_seconds": write_window,
            "prefix_rows_per_second": args.rows / write_window,
            "prefix_token_bytes_per_second": args.rows
            * args.prefix_tokens
            * 16
            / write_window,
            "put_samples_latency": latency_summary(measured.latencies),
            "batch_slot_wait": latency_summary(stager.waits),
            "sink_stage_latency": latency_summary(stager.stages),
            "put_calls": len(measured.latencies),
            "put_batch_rows": measured.batch_rows,
            "put_failures": measured.failures,
            "peak_concurrent_puts": None
            if isinstance(stager, PerWorkerStager)
            else measured.peak_active,
            "peak_remote_stage_requests": stager.peak_requests
            if isinstance(stager, PerWorkerStager)
            else None,
            "remote_stage_roundtrip_latency": latency_summary(stager.rpc_latencies)
            if isinstance(stager, PerWorkerStager)
            else None,
            "stored_keys": len(keys),
            "verified_rows": len(indices),
            "driver_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "token_tensor_bytes_lower_bound": args.rows * args.prefix_tokens * 16,
        }
    finally:
        try:
            stager.close()
        finally:
            try:
                client.clear_samples(sample_ids=None, partition_id=partition)
            finally:
                client.close()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
