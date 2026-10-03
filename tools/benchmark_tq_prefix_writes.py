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

"""Benchmark generation-prefix writes through the production TQ token sink.

Batch size one exercises the original single-prefix path. Larger batches call
``TQTokenSink.stage_generation_prefix_batch()``, issuing one synchronous
``DataPlaneClient.put_samples()`` per batch. ``--clients`` creates separate
Ray actor processes that attach to one shared TQ controller/storage deployment,
matching production's one process-local TQ client per generation model owner.
``--writers-per-client`` optionally adds concurrent callers inside each client
with independently bounded batches in each writer.

``stage_seconds_mean`` is the mean time a row waits for its batch write;
``stage_seconds_per_row_amortized`` divides total PUT time by row count.
``put_calls`` includes each writer's final partial batch.

Example:
    uv run --no-sync python tools/benchmark_tq_prefix_writes.py \
        --rows 10000 --prefix-tokens 128 --clients 32 \
        --writers-per-client 1 \
        --num-storage-units 8 --output-json /tmp/tq-prefix-131k.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import ray

from nemo_gym.token_id_capture.staging.digest import (
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import StagedCallRecord

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneClient
from nemo_rl.data_plane.tq_token_sink import (
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
    generation_cut_staging_key,
)

DEFAULT_PARTITION_ID = "tq_prefix_write_benchmark"
DEFAULT_CHECKPOINT_ID = "benchmark-checkpoint"
CONSUMER_TASK = "benchmark"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one TQ generation-prefix write benchmark."""

    rows: int = 10_000
    prefix_tokens: int = 128
    clients: int = 1
    writers_per_client: int = 1
    batch_size: int = 1
    num_storage_units: int = 8
    storage_capacity: int | None = None
    verify_rows: int = 16
    verify_batch_size: int = 16
    partition_id: str = DEFAULT_PARTITION_ID
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    keep_rows: bool = False
    client_setup_timeout_s: float = 180.0
    write_timeout_s: float = 1_800.0
    progress_interval_s: float = 5.0


@dataclass(frozen=True)
class _WriterResult:
    rows: int
    record_build_seconds: float
    stage_seconds: float
    put_calls: int
    row_wait_seconds: float


def _validate_config(config: BenchmarkConfig) -> None:
    """Reject invalid or misleading benchmark configurations."""
    positive_fields = {
        "rows": config.rows,
        "prefix_tokens": config.prefix_tokens,
        "clients": config.clients,
        "writers_per_client": config.writers_per_client,
        "batch_size": config.batch_size,
        "num_storage_units": config.num_storage_units,
        "verify_batch_size": config.verify_batch_size,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.storage_capacity is not None and config.storage_capacity < config.rows:
        raise ValueError(
            "storage_capacity must be at least rows: "
            f"{config.storage_capacity} < {config.rows}"
        )
    if config.verify_rows < 0:
        raise ValueError(f"verify_rows must be non-negative, got {config.verify_rows}")
    positive_float_fields = {
        "client_setup_timeout_s": config.client_setup_timeout_s,
        "write_timeout_s": config.write_timeout_s,
        "progress_interval_s": config.progress_interval_s,
    }
    for name, value in positive_float_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not config.partition_id:
        raise ValueError("partition_id must be non-empty")
    if not config.checkpoint_id:
        raise ValueError("checkpoint_id must be non-empty")


def _data_plane_config(config: BenchmarkConfig) -> DataPlaneConfig:
    """Build a simple-backend TQ configuration sized for this run."""
    storage_capacity = config.storage_capacity or max(1_024, config.rows)
    return cast(
        DataPlaneConfig,
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "claim_meta_poll_interval_s": 0.05,
            "simple": {
                "storage_capacity": storage_capacity,
                "num_storage_units": config.num_storage_units,
            },
        },
    )


def _record_payload(prefix_tokens: int) -> tuple[list[int], list[float], list[float]]:
    """Build deterministic token payloads shared by every benchmark row."""
    token_ids = [index % 32_000 for index in range(prefix_tokens)]
    return token_ids, [1.0] * prefix_tokens, [-0.5] * prefix_tokens


def _build_record(
    row_index: int,
    token_ids: list[int],
    token_mask: list[float],
    logprobs: list[float],
    *,
    rollout_id: str | None = None,
    model_call_id: str = "call-0",
) -> StagedCallRecord:
    """Build one valid production-shaped generation-prefix record."""
    rollout_id = rollout_id or f"benchmark-rollout-{row_index:09d}"
    extras_digest = compute_extras_digest(None)
    chain_hash = compute_chain_hash(None, token_ids)
    cumulative_hash = hash_token_ids(token_ids)
    values: dict[str, Any] = {
        "rollout_id": rollout_id,
        "model_call_id": model_call_id,
        "parent_call_id": None,
        "mode": "text",
        "prev_len": 0,
        "delta_len": len(token_ids),
        "cum_len": len(token_ids),
        "weight_version": 0,
        "token_ids_delta": token_ids,
        "token_mask_delta": token_mask,
        "generation_log_probs_delta": logprobs,
        "extras": None,
        "extras_digest": extras_digest,
        "chain_hash": chain_hash,
        "cumulative_hash": cumulative_hash,
    }
    digest_values = {key: value for key, value in values.items() if key != "extras"}
    return StagedCallRecord(
        **values,
        digest=compute_staging_digest(
            schema_version=2,
            digest_version=2,
            extras_digest_version=1,
            **digest_values,
        ),
    )


def _split_range(row_range: range, parts: int) -> list[range]:
    """Split one unit-step range into balanced, non-overlapping ranges."""
    if row_range.step != 1:
        raise ValueError("benchmark row ranges must have a step of one")
    part_count = min(len(row_range), parts)
    base, remainder = divmod(len(row_range), part_count)
    ranges: list[range] = []
    start = row_range.start
    for part_index in range(part_count):
        size = base + (1 if part_index < remainder else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


def _row_ranges(rows: int, writers: int) -> list[range]:
    """Split zero-based rows into balanced, non-overlapping writer ranges."""
    return _split_range(range(rows), writers)


def _write_range(
    sink: TQTokenSink,
    row_range: range,
    *,
    checkpoint_id: str,
    token_ids: list[int],
    token_mask: list[float],
    logprobs: list[float],
    batch_size: int = 1,
) -> _WriterResult:
    """Build and synchronously stage one writer's prefix rows."""
    build_seconds = 0.0
    stage_seconds = 0.0
    row_wait_seconds = 0.0
    put_calls = 0
    for offset in range(row_range.start, row_range.stop, batch_size):
        indices = range(offset, min(offset + batch_size, row_range.stop))
        started = time.perf_counter()
        records = [
            _build_record(index, token_ids, token_mask, logprobs) for index in indices
        ]
        build_seconds += time.perf_counter() - started

        started = time.perf_counter()
        if batch_size == 1:
            results = [
                sink.stage_generation_prefix(
                    records[0],
                    checkpoint_id=checkpoint_id,
                    chunk_sequence=0,
                )
            ]
        else:
            results = sink.stage_generation_prefix_batch(
                records,
                checkpoint_id=checkpoint_id,
                chunk_sequences=[0] * len(records),
            )
        elapsed = time.perf_counter() - started
        stage_seconds += elapsed
        row_wait_seconds += elapsed * len(records)
        put_calls += 1
        if len(results) != len(records):
            raise RuntimeError("TQ prefix batch returned an unexpected result count")
        for index, result in zip(indices, results, strict=True):
            if not result.ok:
                raise RuntimeError(
                    f"TQ prefix write failed for row {index}: {result.error}"
                )
    return _WriterResult(
        rows=len(row_range),
        record_build_seconds=build_seconds,
        stage_seconds=stage_seconds,
        put_calls=put_calls,
        row_wait_seconds=row_wait_seconds,
    )


def _write_ranges(
    sink: TQTokenSink,
    row_range: range,
    *,
    checkpoint_id: str,
    prefix_tokens: int,
    writers: int,
    batch_size: int = 1,
) -> list[_WriterResult]:
    """Stage one client's assigned rows with optional local concurrency."""
    token_ids, token_mask, logprobs = _record_payload(prefix_tokens)
    ranges = _split_range(row_range, writers)
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(
                _write_range,
                sink,
                writer_range,
                checkpoint_id=checkpoint_id,
                token_ids=token_ids,
                token_mask=token_mask,
                logprobs=logprobs,
                batch_size=batch_size,
            )
            for writer_range in ranges
        ]
        return [future.result() for future in futures]


class _AttachedTQClientWriter:
    """One production-shaped process-local client attached to shared TQ."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        partition_id: str,
        checkpoint_id: str,
    ) -> None:
        self._client = build_data_plane_client(dp_config, bootstrap=False)
        self._sink = TQTokenSink(self._client, staging_partition=partition_id)
        self._checkpoint_id = checkpoint_id

    def ready(self) -> bool:
        """Confirm that this process has attached its TQ client."""
        return True

    def write(
        self,
        start: int,
        stop: int,
        prefix_tokens: int,
        writers: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Write this client's assigned rows and return per-writer timings."""
        results = _write_ranges(
            self._sink,
            range(start, stop),
            checkpoint_id=self._checkpoint_id,
            prefix_tokens=prefix_tokens,
            writers=writers,
            batch_size=batch_size,
        )
        return [asdict(result) for result in results]


_AttachedTQClientWriterActor = ray.remote(num_cpus=0)(_AttachedTQClientWriter)


def _wait_for_actor_results(
    references: list[Any],
    *,
    stage: str,
    timeout_s: float,
    progress_interval_s: float,
) -> list[Any]:
    """Collect Ray results with bounded, periodic progress reporting."""
    pending = list(references)
    completed = 0
    results: list[Any] = []
    started = time.monotonic()
    print(
        f"tq_prefix_benchmark stage={stage} started total={len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    while pending:
        elapsed = time.monotonic() - started
        remaining = timeout_s - elapsed
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out during {stage}: completed={completed}, "
                f"pending={len(pending)}, timeout_s={timeout_s}"
            )
        ready, pending = ray.wait(
            pending,
            num_returns=len(pending),
            timeout=min(progress_interval_s, remaining),
        )
        if ready:
            results.extend(ray.get(ready))
            completed += len(ready)
        print(
            f"tq_prefix_benchmark stage={stage} completed={completed} "
            f"pending={len(pending)} elapsed_s={time.monotonic() - started:.1f}",
            file=sys.stderr,
            flush=True,
        )
    return results


def _run_single_client_writes(
    dp_client: DataPlaneClient,
    config: BenchmarkConfig,
) -> tuple[list[_WriterResult], float, int, int]:
    """Run the original one-client benchmark path."""
    sink = TQTokenSink(dp_client, staging_partition=config.partition_id)
    started = time.perf_counter()
    writer_results = _write_ranges(
        sink,
        range(config.rows),
        checkpoint_id=config.checkpoint_id,
        prefix_tokens=config.prefix_tokens,
        writers=config.writers_per_client,
        batch_size=config.batch_size,
    )
    return (
        writer_results,
        time.perf_counter() - started,
        1,
        len(writer_results),
    )


def _run_multi_client_writes(
    dp_config: DataPlaneConfig,
    config: BenchmarkConfig,
) -> tuple[list[_WriterResult], float, float, int, int]:
    """Write through process-isolated clients attached to one TQ deployment."""
    client_ranges = _row_ranges(config.rows, config.clients)
    setup_started = time.perf_counter()
    actors = [
        _AttachedTQClientWriterActor.remote(
            dp_config,
            config.partition_id,
            config.checkpoint_id,
        )
        for _ in client_ranges
    ]
    try:
        # Constructors finish before timing begins, just as production TQ
        # clients are already attached when checkpoint prepare starts.
        _wait_for_actor_results(
            [actor.ready.remote() for actor in actors],
            stage="client_setup",
            timeout_s=config.client_setup_timeout_s,
            progress_interval_s=config.progress_interval_s,
        )
        client_setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        payloads = _wait_for_actor_results(
            [
                actor.write.remote(
                    client_range.start,
                    client_range.stop,
                    config.prefix_tokens,
                    config.writers_per_client,
                    config.batch_size,
                )
                for actor, client_range in zip(actors, client_ranges)
            ],
            stage="prefix_writes",
            timeout_s=config.write_timeout_s,
            progress_interval_s=config.progress_interval_s,
        )
        write_seconds = time.perf_counter() - started
    finally:
        # Do not call DataPlaneClient.close() in attached actors: TQ's close
        # operation has deployment-level semantics and can contend with the
        # driver-owned controller/storage that must remain live for verification.
        # Terminating the client processes drops their local attachments without
        # asking them to close the shared deployment.
        print(
            f"tq_prefix_benchmark stage=client_teardown started total={len(actors)}",
            file=sys.stderr,
            flush=True,
        )
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        print(
            f"tq_prefix_benchmark stage=client_teardown completed={len(actors)}",
            file=sys.stderr,
            flush=True,
        )

    writer_results = [
        _WriterResult(**payload)
        for client_payloads in payloads
        for payload in client_payloads
    ]
    return (
        writer_results,
        write_seconds,
        client_setup_seconds,
        len(actors),
        len(writer_results),
    )


def _benchmark_keys(config: BenchmarkConfig, row_indices: list[int]) -> list[str]:
    """Return deterministic TQ keys for selected benchmark rows."""
    return [
        generation_cut_staging_key(
            config.checkpoint_id,
            f"benchmark-rollout-{row_index:09d}",
            "call-0",
            chunk_sequence=0,
        )
        for row_index in row_indices
    ]


def _verification_indices(rows: int, verify_rows: int) -> list[int]:
    """Select evenly spread rows, including both ends when possible."""
    count = min(rows, verify_rows)
    if count == 0:
        return []
    if count == 1:
        return [0]
    return [round(index * (rows - 1) / (count - 1)) for index in range(count)]


def _verify_prefix_tokens(
    source: TQTokenSource,
    config: BenchmarkConfig,
    expected_token_ids: list[int],
) -> tuple[int, float]:
    """Fetch a bounded sample of prefixes and verify token contents."""
    indices = _verification_indices(config.rows, config.verify_rows)
    keys = _benchmark_keys(config, indices)
    started = time.perf_counter()
    for offset in range(0, len(keys), config.verify_batch_size):
        batch_keys = keys[offset : offset + config.verify_batch_size]
        actual = source.fetch_prefix_token_ids(batch_keys)
        expected = expected_token_ids * len(batch_keys)
        if actual != expected:
            raise AssertionError(
                "TQ prefix token verification failed for keys "
                f"{batch_keys[0]!r} through {batch_keys[-1]!r}"
            )
    return len(keys), time.perf_counter() - started


def run_prefix_write_benchmark(
    dp_client: DataPlaneClient,
    config: BenchmarkConfig,
    *,
    dp_config: DataPlaneConfig | None = None,
) -> dict[str, Any]:
    """Run the benchmark against an initialized data-plane client."""
    _validate_config(config)
    token_ids, token_mask, logprobs = _record_payload(config.prefix_tokens)
    dp_client.register_partition(
        partition_id=config.partition_id,
        fields=list(STAGING_FIELDS),
        num_samples=config.rows,
        consumer_tasks=[CONSUMER_TASK],
    )
    source = TQTokenSource(dp_client, staging_partition=config.partition_id)

    cleanup_seconds: float | None = None
    try:
        client_setup_seconds = 0.0
        if config.clients == 1:
            (
                writer_results,
                write_seconds,
                clients_used,
                writers_used,
            ) = _run_single_client_writes(dp_client, config)
        else:
            if dp_config is None:
                raise ValueError(
                    "dp_config is required when clients is greater than one"
                )
            (
                writer_results,
                write_seconds,
                client_setup_seconds,
                clients_used,
                writers_used,
            ) = _run_multi_client_writes(dp_config, config)

        print(
            "tq_prefix_benchmark stage=list_stored_keys started",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        stored_keys = dp_client.list_sample_ids(config.partition_id)
        list_seconds = time.perf_counter() - started
        print(
            f"tq_prefix_benchmark stage=list_stored_keys completed "
            f"rows={len(stored_keys)} elapsed_s={list_seconds:.1f}",
            file=sys.stderr,
            flush=True,
        )
        if len(stored_keys) != config.rows:
            raise AssertionError(
                f"Expected {config.rows} stored prefixes, found {len(stored_keys)}"
            )

        print(
            "tq_prefix_benchmark stage=verify_prefixes started",
            file=sys.stderr,
            flush=True,
        )
        verified_rows, verify_seconds = _verify_prefix_tokens(
            source,
            config,
            token_ids,
        )
        print(
            f"tq_prefix_benchmark stage=verify_prefixes completed "
            f"rows={verified_rows} elapsed_s={verify_seconds:.1f}",
            file=sys.stderr,
            flush=True,
        )
        result: dict[str, Any] = {
            "backend": "simple",
            "rows": config.rows,
            "prefix_tokens": config.prefix_tokens,
            "clients_requested": config.clients,
            "clients_used": clients_used,
            "writers_per_client_requested": config.writers_per_client,
            "writers_requested_total": config.clients * config.writers_per_client,
            "writers_used": writers_used,
            "client_setup_seconds": client_setup_seconds,
            "num_storage_units": config.num_storage_units,
            "storage_capacity": config.storage_capacity or max(1_024, config.rows),
            "batch_size": config.batch_size,
            "put_calls": sum(item.put_calls for item in writer_results),
            "stage_batch_seconds_mean": sum(
                item.stage_seconds for item in writer_results
            )
            / sum(item.put_calls for item in writer_results),
            "write_seconds": write_seconds,
            "rows_per_second": config.rows / write_seconds,
            "tokens_per_second": config.rows * config.prefix_tokens / write_seconds,
            "record_build_seconds_sum": sum(
                item.record_build_seconds for item in writer_results
            ),
            "stage_seconds_sum": sum(item.stage_seconds for item in writer_results),
            "stage_seconds_mean": sum(item.row_wait_seconds for item in writer_results)
            / config.rows,
            "stage_seconds_per_row_amortized": sum(
                item.stage_seconds for item in writer_results
            )
            / config.rows,
            "stored_keys": len(stored_keys),
            "list_seconds": list_seconds,
            "verified_rows": verified_rows,
            "verify_seconds": verify_seconds,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rows_kept": config.keep_rows,
        }
    finally:
        if not config.keep_rows:
            print(
                "tq_prefix_benchmark stage=cleanup started",
                file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            dp_client.clear_samples(
                sample_ids=None,
                partition_id=config.partition_id,
            )
            cleanup_seconds = time.perf_counter() - started
            print(
                f"tq_prefix_benchmark stage=cleanup completed "
                f"elapsed_s={cleanup_seconds:.1f}",
                file=sys.stderr,
                flush=True,
            )
    result["cleanup_seconds"] = cleanup_seconds
    return result


def _parse_args() -> tuple[BenchmarkConfig, Path | None]:
    """Parse CLI arguments into a typed benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help=(
            "Process-isolated TQ clients attached to one shared deployment; "
            "use more than one to model separate generation model owners."
        ),
    )
    parser.add_argument(
        "--writers-per-client",
        "--writers",
        dest="writers_per_client",
        type=int,
        default=1,
        help=(
            "Concurrent writer threads inside each client. --writers is kept "
            "as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prefix rows per PUT in each writer; 1 preserves the baseline.",
    )
    parser.add_argument("--num-storage-units", type=int, default=8)
    parser.add_argument(
        "--storage-capacity",
        type=int,
        default=None,
        help="Maximum retained rows; defaults to max(1024, --rows).",
    )
    parser.add_argument("--verify-rows", type=int, default=16)
    parser.add_argument("--verify-batch-size", type=int, default=16)
    parser.add_argument("--partition-id", default=DEFAULT_PARTITION_ID)
    parser.add_argument("--checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--client-setup-timeout-s", type=float, default=180.0)
    parser.add_argument("--write-timeout-s", type=float, default=1_800.0)
    parser.add_argument("--progress-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--keep-rows",
        action="store_true",
        help="Do not clear benchmark rows before closing the TQ client.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Also write the final JSON result to this path.",
    )
    args = parser.parse_args()
    config = BenchmarkConfig(
        rows=args.rows,
        prefix_tokens=args.prefix_tokens,
        clients=args.clients,
        writers_per_client=args.writers_per_client,
        batch_size=args.batch_size,
        num_storage_units=args.num_storage_units,
        storage_capacity=args.storage_capacity,
        verify_rows=args.verify_rows,
        verify_batch_size=args.verify_batch_size,
        partition_id=args.partition_id,
        checkpoint_id=args.checkpoint_id,
        keep_rows=args.keep_rows,
        client_setup_timeout_s=args.client_setup_timeout_s,
        write_timeout_s=args.write_timeout_s,
        progress_interval_s=args.progress_interval_s,
    )
    try:
        _validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    return config, args.output_json


def main() -> None:
    """Run the CLI benchmark and print one machine-readable result."""
    config, output_json = _parse_args()
    dp_config = _data_plane_config(config)
    dp_client = build_data_plane_client(
        dp_config,
        bootstrap=True,
    )
    try:
        result = run_prefix_write_benchmark(
            dp_client,
            config,
            dp_config=dp_config,
        )
    finally:
        dp_client.close()

    payload = {"config": asdict(config), "result": result}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
