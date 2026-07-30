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

r"""Benchmark TransferQueue checkpoint save/load with rollout-shaped data.

The public invocation is an orchestrator. It runs save and load in separate
Python processes so a passing result proves that a checkpoint survives a full
TQ/Ray restart, rather than an in-process clear/load cycle.

Start with a small quiescent smoke test:

    uv run python tools/tq_checkpoint_benchmark.py \\
        --checkpoint-root /lustre/.../tq-bench \\
        --num-rows 1024 --min-seq-len 512 --max-seq-len 1024

Exercise concurrent writers sharing one process:

    uv run python tools/tq_checkpoint_benchmark.py \\
        --checkpoint-root /lustre/.../tq-bench \\
        --num-rows 8192 --min-seq-len 4096 --max-seq-len 4096 \\
        --payload-profile train-ready --producer-mode thread \\
        --num-producers 4

Exercise independent Ray worker processes:

    uv run python tools/tq_checkpoint_benchmark.py \\
        --checkpoint-root /lustre/.../tq-bench \\
        --num-rows 8192 --min-seq-len 4096 --max-seq-len 4096 \\
        --payload-profile train-ready --producer-mode process \\
        --num-producers 4 --num-storage-units 8

The benchmark intentionally uses unique checkpoint directories. TQ v0.1.9
does not atomically replace an existing checkpoint. It also requires the same
number of SimpleStorage units at load time and must not receive clears while a
checkpoint is in progress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PARTITION_ID = "benchmark"
BASE_KEY_PREFIX = "base-"
PRODUCER_KEY_PREFIX = "producer-"
PRODUCER_ROW_BASE = 1_000_000_000_000
PRODUCER_ROW_STRIDE = 1_000_000_000

PayloadProfile = Literal["generation", "train-ready"]
ProducerMode = Literal["quiescent", "thread", "process"]
VerifyMode = Literal["none", "sample", "all"]


@dataclass(frozen=True)
class BenchmarkConfig:
    checkpoint_root: str
    run_dir: str
    num_rows: int
    min_seq_len: int
    max_seq_len: int
    payload_profile: PayloadProfile
    batch_rows: int
    num_storage_units: int
    producer_mode: ProducerMode
    num_producers: int
    producer_batch_rows: int
    producer_max_rows: int
    producer_warmup_s: float
    producer_cooldown_s: float
    producer_sleep_ms: float
    verify_mode: VerifyMode
    verify_samples: int
    verify_batch_rows: int
    group_size: int
    weight_version: int
    seed: int
    ray_address: str
    phase_timeout_s: float
    torch_num_threads: int

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.run_dir) / "checkpoint"


def sequence_length(
    row_id: int,
    min_seq_len: int,
    max_seq_len: int,
    seed: int,
) -> int:
    """Return a deterministic, approximately uniform length for ``row_id``."""
    if min_seq_len == max_seq_len:
        return min_seq_len
    span = max_seq_len - min_seq_len + 1
    # SplitMix64-style integer mixing avoids allocating or sharing RNG state.
    value = (row_id + seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return min_seq_len + value % span


def payload_field_names(profile: PayloadProfile) -> list[str]:
    """Return fields matching NeMo-RL's generation or train-ready rows."""
    fields = [
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
    ]
    if profile == "train-ready":
        fields[3:3] = [
            "prev_logprobs",
            "reference_policy_logprobs",
            "advantages",
        ]
    return fields


def logical_tensor_bytes_for_length(length: int, profile: PayloadProfile) -> int:
    """Compute unpadded tensor bytes represented by one rollout row."""
    # input_ids int64 + generation_logprobs bf16 + token_mask int32
    per_token = 8 + 2 + 4
    if profile == "train-ready":
        # prev_logprobs, reference_policy_logprobs, advantages: bf16 each.
        per_token += 2 + 2 + 2
    # input_lengths int64 + sample_mask int32.
    return length * per_token + 8 + 4


def logical_tensor_bytes(
    row_ids: list[int],
    *,
    min_seq_len: int,
    max_seq_len: int,
    seed: int,
    profile: PayloadProfile,
) -> int:
    return sum(
        logical_tensor_bytes_for_length(
            sequence_length(row_id, min_seq_len, max_seq_len, seed),
            profile,
        )
        for row_id in row_ids
    )


def percentile(values: list[float], q: float) -> float | None:
    """Return a linearly interpolated percentile without NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize_lengths(lengths: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(lengths),
        "total_valid_tokens": sum(lengths),
        "mean": sum(lengths) / len(lengths) if lengths else None,
        "p50": percentile([float(v) for v in lengths], 0.50),
        "p95": percentile([float(v) for v in lengths], 0.95),
        "max": max(lengths) if lengths else None,
    }


def summarize_records(
    records: list[dict[str, float | int]],
    *,
    window_start: float,
    window_end: float,
) -> dict[str, float | int | None]:
    """Summarize puts whose acknowledgement time falls in a time window."""
    selected = [
        record
        for record in records
        if window_start < float(record["completed_at"]) <= window_end
    ]
    duration = max(0.0, window_end - window_start)
    rows = sum(int(record["rows"]) for record in selected)
    logical_bytes = sum(int(record["logical_tensor_bytes"]) for record in selected)
    latencies = [float(record["put_duration_s"]) for record in selected]
    return {
        "duration_s": duration,
        "batches": len(selected),
        "rows": rows,
        "logical_tensor_bytes": logical_bytes,
        "rows_per_s": rows / duration if duration else None,
        "logical_gib_per_s": logical_bytes / (1024**3) / duration if duration else None,
        "put_latency_p50_ms": (
            percentile(latencies, 0.50) * 1000 if latencies else None
        ),
        "put_latency_p95_ms": (
            percentile(latencies, 0.95) * 1000 if latencies else None
        ),
        "put_latency_p99_ms": (
            percentile(latencies, 0.99) * 1000 if latencies else None
        ),
    }


def _base_key(row_id: int) -> str:
    return f"{BASE_KEY_PREFIX}{row_id:012d}"


def _producer_key(producer_id: int, local_index: int) -> str:
    return f"{PRODUCER_KEY_PREFIX}{producer_id:04d}-{local_index:012d}"


def _producer_row_id(producer_id: int, local_index: int) -> int:
    return PRODUCER_ROW_BASE + producer_id * PRODUCER_ROW_STRIDE + local_index


def _row_id_from_key(key: str) -> int:
    if key.startswith(BASE_KEY_PREFIX):
        return int(key[len(BASE_KEY_PREFIX) :])
    if key.startswith(PRODUCER_KEY_PREFIX):
        producer_id, local_index = key[len(PRODUCER_KEY_PREFIX) :].split("-", 1)
        return _producer_row_id(int(producer_id), int(local_index))
    raise ValueError(f"unrecognized benchmark key: {key!r}")


def _make_payload(
    row_ids: list[int],
    config: BenchmarkConfig,
) -> tuple[Any, list[dict[str, int | float]], int]:
    """Build a deterministic jagged TensorDict and per-row tags."""
    import torch
    from tensordict import TensorDict

    lengths = [
        sequence_length(
            row_id,
            config.min_seq_len,
            config.max_seq_len,
            config.seed,
        )
        for row_id in row_ids
    ]

    def jagged_rows(kind: str, dtype: Any) -> Any:
        rows = []
        for row_id, length in zip(row_ids, lengths, strict=True):
            positions = torch.arange(length, dtype=torch.int64)
            if kind == "tokens":
                row = ((positions + row_id * 31) % 50_000 + 1).to(dtype)
            elif kind == "logprobs":
                row = (
                    -2.0 + ((positions + row_id * 17) % 1024).to(torch.float32) / 1024.0
                ).to(dtype)
            elif kind == "prev_logprobs":
                row = (
                    -2.25
                    + ((positions + row_id * 19) % 1024).to(torch.float32) / 1024.0
                ).to(dtype)
            elif kind == "reference_logprobs":
                row = (
                    -2.5 + ((positions + row_id * 23) % 1024).to(torch.float32) / 1024.0
                ).to(dtype)
            elif kind == "advantages":
                row = (
                    ((positions + row_id * 29) % 257).to(torch.float32) / 128.0 - 1.0
                ).to(dtype)
            elif kind == "mask":
                row = torch.ones(length, dtype=dtype)
            else:
                raise ValueError(f"unknown payload kind: {kind}")
            rows.append(row)
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    fields: dict[str, Any] = {
        "input_ids": jagged_rows("tokens", torch.int64),
        "input_lengths": torch.tensor(lengths, dtype=torch.int64),
        "generation_logprobs": jagged_rows("logprobs", torch.bfloat16),
    }
    if config.payload_profile == "train-ready":
        fields.update(
            {
                "prev_logprobs": jagged_rows("prev_logprobs", torch.bfloat16),
                "reference_policy_logprobs": jagged_rows(
                    "reference_logprobs", torch.bfloat16
                ),
                "advantages": jagged_rows("advantages", torch.bfloat16),
            }
        )
    fields.update(
        {
            "token_mask": jagged_rows("mask", torch.int32),
            "sample_mask": torch.ones(len(row_ids), dtype=torch.int32),
        }
    )

    tags: list[dict[str, int | float]] = []
    for row_id in row_ids:
        tags.append(
            {
                "prompt_id": row_id // config.group_size,
                "generation_index": row_id % config.group_size,
                "weight_version": config.weight_version,
                "total_reward": ((row_id * 37) % 2001) / 1000.0 - 1.0,
            }
        )

    tensor_bytes = sum(
        logical_tensor_bytes_for_length(length, config.payload_profile)
        for length in lengths
    )
    return TensorDict(fields, batch_size=[len(row_ids)]), tags, tensor_bytes


def _tq_config(num_storage_units: int) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": None,
                    "num_data_storage_units": num_storage_units,
                },
            },
        }
    )


def _init_tq(config: BenchmarkConfig) -> tuple[Any, Any]:
    import ray
    import torch
    import transfer_queue as tq

    torch.set_num_threads(config.torch_num_threads)
    if not ray.is_initialized():
        ray_kwargs: dict[str, Any] = {
            "namespace": f"tq-checkpoint-bench-{Path(config.run_dir).name}",
            "include_dashboard": False,
        }
        if config.ray_address:
            ray_kwargs["address"] = config.ray_address
        ray.init(**ray_kwargs)
    try:
        ray.get_actor("TransferQueueController", namespace="transfer_queue")
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "an existing TransferQueueController is already running in this "
            "Ray cluster. Run the benchmark in an isolated allocation/cluster; "
            "attaching would ignore this benchmark's storage topology and could "
            "modify another job's data."
        )
    tq.init(conf=_tq_config(config.num_storage_units))
    return ray, tq


def _close_tq(ray: Any, tq: Any) -> None:
    try:
        tq.close()
    finally:
        if ray.is_initialized():
            ray.shutdown()


def _directory_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _system_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "path": str(path),
        "process_rss_bytes": _process_rss_bytes(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["system_memory_total_bytes"] = int(memory.total)
        info["system_memory_available_bytes"] = int(memory.available)
    except (ImportError, OSError):
        pass
    if shutil.which("findmnt"):
        result = subprocess.run(
            ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE,SOURCE"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["filesystem"] = result.stdout.strip()
    try:
        from importlib.metadata import version

        info["transfer_queue_version"] = version("TransferQueue")
        info["torch_version"] = version("torch")
        info["ray_version"] = version("ray")
    except Exception:
        pass
    return info


class _LocalProducer:
    def __init__(self, config: BenchmarkConfig, producer_id: int) -> None:
        self.config = config
        self.producer_id = producer_id
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.records: list[dict[str, float | int]] = []
        self.completed_rows = 0
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"tq-producer-{producer_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            import transfer_queue as tq

            while not self.stop_event.is_set():
                with self.lock:
                    start_index = self.completed_rows
                if (
                    self.config.producer_max_rows > 0
                    and start_index >= self.config.producer_max_rows
                ):
                    return
                batch_rows = self.config.producer_batch_rows
                if self.config.producer_max_rows > 0:
                    batch_rows = min(
                        batch_rows,
                        self.config.producer_max_rows - start_index,
                    )
                local_indices = list(range(start_index, start_index + batch_rows))
                row_ids = [
                    _producer_row_id(self.producer_id, index) for index in local_indices
                ]
                keys = [
                    _producer_key(self.producer_id, index) for index in local_indices
                ]

                build_started = time.perf_counter()
                fields, tags, tensor_bytes = _make_payload(row_ids, self.config)
                build_duration = time.perf_counter() - build_started
                put_started = time.perf_counter()
                tq.kv_batch_put(
                    keys=keys,
                    partition_id=PARTITION_ID,
                    fields=fields,
                    tags=tags,
                )
                put_duration = time.perf_counter() - put_started
                completed_at = time.time()
                with self.lock:
                    self.completed_rows += batch_rows
                    self.records.append(
                        {
                            "completed_at": completed_at,
                            "rows": batch_rows,
                            "logical_tensor_bytes": tensor_bytes,
                            "build_duration_s": build_duration,
                            "put_duration_s": put_duration,
                        }
                    )
                if self.config.producer_sleep_ms:
                    self.stop_event.wait(self.config.producer_sleep_ms / 1000.0)
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "producer_id": self.producer_id,
                "completed_rows": self.completed_rows,
                "records": [dict(record) for record in self.records],
                "error": self.error,
            }

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError(f"producer thread {self.producer_id} did not stop")
        return self.snapshot()


class _RayProducer:
    """Ray actor wrapper with a background writer thread.

    Actor RPCs remain available for snapshots while the background thread is
    blocked in a synchronous TQ put.
    """

    def __init__(self, config_dict: dict[str, Any], producer_id: int) -> None:
        self.config = BenchmarkConfig(**config_dict)
        self.producer = _LocalProducer(self.config, producer_id)
        import torch
        import transfer_queue as tq

        torch.set_num_threads(self.config.torch_num_threads)
        tq.init()

    def start(self) -> None:
        self.producer.start()

    def snapshot(self) -> dict[str, Any]:
        return self.producer.snapshot()

    def stop(self) -> dict[str, Any]:
        # Do not call tq.close(): an attaching process would kill the shared
        # controller. The coordinator owns full TQ shutdown.
        return self.producer.stop()


def _start_producers(
    config: BenchmarkConfig,
    ray: Any,
) -> tuple[list[Any], list[Any]]:
    local: list[_LocalProducer] = []
    actors: list[Any] = []
    if config.producer_mode == "thread":
        local = [_LocalProducer(config, index) for index in range(config.num_producers)]
        for producer in local:
            producer.start()
    elif config.producer_mode == "process":
        actor_type = ray.remote(num_cpus=1)(_RayProducer)
        actors = [
            actor_type.remote(asdict(config), index)
            for index in range(config.num_producers)
        ]
        ray.get([actor.start.remote() for actor in actors])
    return local, actors


def _snapshot_producers(
    local: list[Any],
    actors: list[Any],
    ray: Any,
) -> list[dict[str, Any]]:
    if local:
        return [producer.snapshot() for producer in local]
    if actors:
        return ray.get([actor.snapshot.remote() for actor in actors])
    return []


def _stop_producers(
    local: list[Any],
    actors: list[Any],
    ray: Any,
) -> list[dict[str, Any]]:
    if local:
        return [producer.stop() for producer in local]
    if actors:
        snapshots = ray.get([actor.stop.remote() for actor in actors])
        for actor in actors:
            ray.kill(actor)
        return snapshots
    return []


def _raise_producer_errors(snapshots: list[dict[str, Any]]) -> None:
    errors = [
        f"producer {snapshot['producer_id']}: {snapshot['error']}"
        for snapshot in snapshots
        if snapshot.get("error")
    ]
    if errors:
        raise RuntimeError("concurrent producer failed: " + "; ".join(errors))


def _fill_base_rows(config: BenchmarkConfig, tq: Any) -> dict[str, Any]:
    wall_started = time.perf_counter()
    build_duration = 0.0
    put_duration = 0.0
    tensor_bytes = 0
    next_progress = max(config.batch_rows, config.num_rows // 10)

    for start in range(0, config.num_rows, config.batch_rows):
        stop = min(config.num_rows, start + config.batch_rows)
        row_ids = list(range(start, stop))
        keys = [_base_key(row_id) for row_id in row_ids]

        build_started = time.perf_counter()
        fields, tags, batch_bytes = _make_payload(row_ids, config)
        build_duration += time.perf_counter() - build_started

        put_started = time.perf_counter()
        tq.kv_batch_put(
            keys=keys,
            partition_id=PARTITION_ID,
            fields=fields,
            tags=tags,
        )
        put_duration += time.perf_counter() - put_started
        tensor_bytes += batch_bytes
        if stop >= next_progress or stop == config.num_rows:
            print(
                f"[save] filled {stop:,}/{config.num_rows:,} base rows",
                flush=True,
            )
            next_progress += max(config.batch_rows, config.num_rows // 10)

    wall_duration = time.perf_counter() - wall_started
    return {
        "rows": config.num_rows,
        "logical_tensor_bytes": tensor_bytes,
        "payload_build_s": build_duration,
        "put_s": put_duration,
        "wall_s": wall_duration,
        "put_logical_gib_per_s": (
            tensor_bytes / (1024**3) / put_duration if put_duration else None
        ),
        "wall_logical_gib_per_s": (
            tensor_bytes / (1024**3) / wall_duration if wall_duration else None
        ),
    }


def _save_phase(config: BenchmarkConfig) -> None:
    ray, tq = _init_tq(config)
    local: list[Any] = []
    actors: list[Any] = []
    try:
        print(
            f"[save] TQ initialized with {config.num_storage_units} "
            "SimpleStorage units",
            flush=True,
        )
        base_fill = _fill_base_rows(config, tq)
        producer_started_at = time.time()
        local, actors = _start_producers(config, ray)
        if local or actors:
            print(
                f"[save] started {config.num_producers} "
                f"{config.producer_mode} producers",
                flush=True,
            )
            time.sleep(config.producer_warmup_s)

        pre_checkpoint = _snapshot_producers(local, actors, ray)
        _raise_producer_errors(pre_checkpoint)
        producers_exhausted_before_checkpoint = bool(pre_checkpoint) and all(
            config.producer_max_rows > 0
            and snapshot["completed_rows"] >= config.producer_max_rows
            for snapshot in pre_checkpoint
        )
        if producers_exhausted_before_checkpoint:
            print(
                "[save] WARNING: all producers reached --producer-max-rows "
                "before checkpointing; increase the limit or reduce warmup to "
                "measure write/checkpoint overlap",
                flush=True,
            )
        checkpoint_started_at = time.time()
        rss_before = _process_rss_bytes()
        print(f"[save] checkpointing to {config.checkpoint_dir}", flush=True)
        save_started = time.perf_counter()
        tq.save_checkpoint(
            config.checkpoint_dir,
            metadata={
                "benchmark_run": Path(config.run_dir).name,
                "payload_profile": config.payload_profile,
                "base_rows": config.num_rows,
                "producer_mode": config.producer_mode,
                "producer_rows_acknowledged_before_save": {
                    str(snapshot["producer_id"]): snapshot["completed_rows"]
                    for snapshot in pre_checkpoint
                },
            },
        )
        save_duration = time.perf_counter() - save_started
        checkpoint_finished_at = time.time()
        rss_after = _process_rss_bytes()
        print(f"[save] checkpoint completed in {save_duration:.3f}s", flush=True)

        post_checkpoint = _snapshot_producers(local, actors, ray)
        _raise_producer_errors(post_checkpoint)
        if local or actors:
            time.sleep(config.producer_cooldown_s)
        final_producers = _stop_producers(local, actors, ray)
        local, actors = [], []
        _raise_producer_errors(final_producers)
        producer_stopped_at = time.time()

        records = [
            record for snapshot in final_producers for record in snapshot["records"]
        ]
        producer_metrics = {
            "started_at": producer_started_at,
            "checkpoint_started_at": checkpoint_started_at,
            "checkpoint_finished_at": checkpoint_finished_at,
            "stopped_at": producer_stopped_at,
            "acknowledged_before_checkpoint": {
                str(snapshot["producer_id"]): snapshot["completed_rows"]
                for snapshot in pre_checkpoint
            },
            "acknowledged_by_checkpoint_return": {
                str(snapshot["producer_id"]): snapshot["completed_rows"]
                for snapshot in post_checkpoint
            },
            "final_acknowledged": {
                str(snapshot["producer_id"]): snapshot["completed_rows"]
                for snapshot in final_producers
            },
            "before": summarize_records(
                records,
                window_start=producer_started_at,
                window_end=checkpoint_started_at,
            ),
            "during": summarize_records(
                records,
                window_start=checkpoint_started_at,
                window_end=checkpoint_finished_at,
            ),
            "after": summarize_records(
                records,
                window_start=checkpoint_finished_at,
                window_end=producer_stopped_at,
            ),
            "all_exhausted_before_checkpoint": (producers_exhausted_before_checkpoint),
        }
        producer_metrics["overlap_observed"] = (
            int(producer_metrics["during"]["rows"]) > 0
        )
        result = {
            "phase": "save",
            "config": asdict(config),
            "system": _system_info(Path(config.checkpoint_root)),
            "base_fill": base_fill,
            "checkpoint": {
                "duration_s": save_duration,
                "disk_bytes": _directory_size(config.checkpoint_dir),
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
            },
            "producers": producer_metrics,
        }
        _write_json(Path(config.run_dir) / "save_result.json", result)
    finally:
        if local or actors:
            try:
                _stop_producers(local, actors, ray)
            except Exception:
                pass
        _close_tq(ray, tq)


def _sample_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [0]
    return sorted({round(index * (size - 1) / (count - 1)) for index in range(count)})


def _assert_tensor_equal(actual: Any, expected: Any, field: str) -> None:
    import torch

    actual_rows = list(actual.unbind())
    expected_rows = list(expected.unbind())
    if len(actual_rows) != len(expected_rows):
        raise AssertionError(
            f"{field}: row count {len(actual_rows)} != {len(expected_rows)}"
        )
    for index, (actual_row, expected_row) in enumerate(
        zip(actual_rows, expected_rows, strict=True)
    ):
        if not torch.equal(actual_row, expected_row):
            raise AssertionError(f"{field}: restored row {index} differs")


def _verification_keys(
    restored_keys: list[str],
    guaranteed_producer_counts: dict[str, int],
    config: BenchmarkConfig,
) -> list[str]:
    if config.verify_mode == "none":
        return []
    if config.verify_mode == "all":
        return sorted(restored_keys)

    selected = [
        _base_key(index)
        for index in _sample_indices(config.num_rows, config.verify_samples)
    ]
    per_producer = max(1, config.verify_samples // max(1, config.num_producers))
    for producer_id_text, count in guaranteed_producer_counts.items():
        producer_id = int(producer_id_text)
        selected.extend(
            _producer_key(producer_id, index)
            for index in _sample_indices(count, per_producer)
        )
    return sorted(set(selected))


def _verify_restored_data(
    tq: Any,
    listing: dict[str, Any],
    keys: list[str],
    config: BenchmarkConfig,
) -> None:
    import torch

    torch.set_num_threads(config.torch_num_threads)
    for start in range(0, len(keys), config.verify_batch_rows):
        batch_keys = keys[start : start + config.verify_batch_rows]
        row_ids = [_row_id_from_key(key) for key in batch_keys]
        expected, expected_tags, _ = _make_payload(row_ids, config)
        actual = tq.kv_batch_get(
            keys=batch_keys,
            partition_id=PARTITION_ID,
            select_fields=payload_field_names(config.payload_profile),
        )
        for field in payload_field_names(config.payload_profile):
            _assert_tensor_equal(actual[field], expected[field], field)
        for key, expected_tag in zip(batch_keys, expected_tags, strict=True):
            actual_tag = listing[key]
            for tag_name, expected_value in expected_tag.items():
                actual_value = actual_tag[tag_name]
                if isinstance(expected_value, float):
                    if not math.isclose(
                        float(actual_value),
                        expected_value,
                        rel_tol=0,
                        abs_tol=1e-12,
                    ):
                        raise AssertionError(
                            f"{key} tag {tag_name}: "
                            f"{actual_value!r} != {expected_value!r}"
                        )
                elif actual_value != expected_value:
                    raise AssertionError(
                        f"{key} tag {tag_name}: {actual_value!r} != {expected_value!r}"
                    )
        print(
            f"[load] verified {min(start + len(batch_keys), len(keys)):,}/"
            f"{len(keys):,} selected rows",
            flush=True,
        )


def _load_phase(config: BenchmarkConfig) -> None:
    save_result = _read_json(Path(config.run_dir) / "save_result.json")
    guaranteed_counts = {
        str(key): int(value)
        for key, value in save_result["producers"][
            "acknowledged_before_checkpoint"
        ].items()
    }
    acknowledged_by_return = {
        str(key): int(value)
        for key, value in save_result["producers"][
            "acknowledged_by_checkpoint_return"
        ].items()
    }
    ray, tq = _init_tq(config)
    try:
        print("[load] fresh TQ process initialized", flush=True)
        rss_before = _process_rss_bytes()
        load_started = time.perf_counter()
        tq.load_checkpoint(config.checkpoint_dir)
        load_duration = time.perf_counter() - load_started
        rss_after = _process_rss_bytes()
        print(f"[load] checkpoint loaded in {load_duration:.3f}s", flush=True)

        listing_all = tq.kv_list(partition_id=PARTITION_ID)
        listing = listing_all.get(PARTITION_ID, {})
        restored_keys = list(listing.keys())
        restored_set = set(restored_keys)

        expected_base = {_base_key(index) for index in range(config.num_rows)}
        missing_base = sorted(expected_base - restored_set)
        guaranteed_producer_keys = {
            _producer_key(int(producer_id), index)
            for producer_id, count in guaranteed_counts.items()
            for index in range(count)
        }
        missing_guaranteed = sorted(guaranteed_producer_keys - restored_set)
        malformed: list[str] = []
        unexpected: list[str] = []
        restored_row_ids: list[int] = []
        for key in restored_keys:
            try:
                row_id = _row_id_from_key(key)
                if key.startswith(BASE_KEY_PREFIX):
                    base_index = int(key[len(BASE_KEY_PREFIX) :])
                    if not 0 <= base_index < config.num_rows:
                        unexpected.append(key)
                        continue
                else:
                    producer_id, local_index = key[len(PRODUCER_KEY_PREFIX) :].split(
                        "-", 1
                    )
                    upper_bound = acknowledged_by_return.get(producer_id)
                    if upper_bound is None or not 0 <= int(local_index) < upper_bound:
                        unexpected.append(key)
                        continue
                restored_row_ids.append(row_id)
            except (ValueError, IndexError):
                malformed.append(key)
        if missing_base or missing_guaranteed or malformed or unexpected:
            raise AssertionError(
                "checkpoint key validation failed: "
                f"missing_base={missing_base[:10]}, "
                f"missing_guaranteed={missing_guaranteed[:10]}, "
                f"malformed={malformed[:10]}, "
                f"unexpected={unexpected[:10]}"
            )

        verify_keys = _verification_keys(
            restored_keys,
            guaranteed_counts,
            config,
        )
        verification_started = time.perf_counter()
        _verify_restored_data(tq, listing, verify_keys, config)
        verification_duration = time.perf_counter() - verification_started

        restored_producer_counts: dict[str, int] = {}
        for key in restored_keys:
            if key.startswith(PRODUCER_KEY_PREFIX):
                producer_id = key[len(PRODUCER_KEY_PREFIX) :].split("-", 1)[0]
                restored_producer_counts[producer_id] = (
                    restored_producer_counts.get(producer_id, 0) + 1
                )

        lengths = [
            sequence_length(
                row_id,
                config.min_seq_len,
                config.max_seq_len,
                config.seed,
            )
            for row_id in restored_row_ids
        ]
        restored_tensor_bytes = sum(
            logical_tensor_bytes_for_length(length, config.payload_profile)
            for length in lengths
        )
        result = {
            "phase": "load",
            "system": _system_info(Path(config.checkpoint_root)),
            "checkpoint": {
                "load_duration_s": load_duration,
                "verification_duration_s": verification_duration,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
            },
            "restored": {
                "total_rows": len(restored_keys),
                "base_rows": len(expected_base),
                "producer_rows": restored_producer_counts,
                "logical_tensor_bytes": restored_tensor_bytes,
                "lengths": summarize_lengths(lengths),
            },
            "verification": {
                "mode": config.verify_mode,
                "verified_rows": len(verify_keys),
                "missing_base_rows": len(missing_base),
                "missing_guaranteed_producer_rows": len(missing_guaranteed),
                "malformed_keys": len(malformed),
                "unexpected_keys": len(unexpected),
                "status": "pass",
            },
        }
        _write_json(Path(config.run_dir) / "load_result.json", result)
    finally:
        _close_tq(ray, tq)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_config(config: BenchmarkConfig) -> None:
    positive_ints = {
        "num_rows": config.num_rows,
        "min_seq_len": config.min_seq_len,
        "max_seq_len": config.max_seq_len,
        "batch_rows": config.batch_rows,
        "num_storage_units": config.num_storage_units,
        "verify_batch_rows": config.verify_batch_rows,
        "group_size": config.group_size,
        "torch_num_threads": config.torch_num_threads,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if config.min_seq_len > config.max_seq_len:
        raise ValueError("min_seq_len must be <= max_seq_len")
    if config.producer_mode != "quiescent":
        if config.num_producers <= 0:
            raise ValueError("num_producers must be > 0 for concurrent modes")
        if config.producer_batch_rows <= 0:
            raise ValueError("producer_batch_rows must be > 0")
    if config.producer_max_rows < 0:
        raise ValueError("producer_max_rows must be >= 0 (0 means unlimited)")
    nonnegative_floats = {
        "producer_warmup_s": config.producer_warmup_s,
        "producer_cooldown_s": config.producer_cooldown_s,
        "producer_sleep_ms": config.producer_sleep_ms,
    }
    for name, value in nonnegative_floats.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if config.verify_samples < 0:
        raise ValueError("verify_samples must be >= 0")
    if config.phase_timeout_s <= 0:
        raise ValueError("phase_timeout_s must be > 0")


def _estimate_base_tensor_bytes(config: BenchmarkConfig) -> int:
    return logical_tensor_bytes(
        list(range(config.num_rows)),
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        profile=config.payload_profile,
    )


def _preflight(config: BenchmarkConfig, skip_space_check: bool) -> None:
    checkpoint_root = Path(config.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    estimated_bytes = _estimate_base_tensor_bytes(config)
    free_bytes = shutil.disk_usage(checkpoint_root).free
    print(
        "Planned base payload: "
        f"{config.num_rows:,} rows, ~{estimated_bytes / 1024**3:.2f} GiB "
        f"of unpadded tensors; free disk={free_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    if not skip_space_check and estimated_bytes * 1.5 > free_bytes:
        raise RuntimeError(
            "estimated base tensor bytes need more than 2/3 of available disk. "
            "Choose a larger filesystem or pass --skip-space-check after "
            "validating capacity manually."
        )


def _run_child(phase: str, config_path: Path, timeout_s: float) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_phase",
        phase,
        "--_config",
        str(config_path),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            f"{phase} phase exceeded {timeout_s}s and was terminated. "
            "TQ v0.1.9 can hang when checkpoint I/O fails; inspect filesystem "
            "permissions, capacity, and the child logs."
        ) from None
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _print_summary(result: dict[str, Any]) -> None:
    save = result["save"]
    load = result["load"]
    save_seconds = float(save["checkpoint"]["duration_s"])
    load_seconds = float(load["checkpoint"]["load_duration_s"])
    disk_bytes = int(save["checkpoint"]["disk_bytes"])
    logical_bytes = int(load["restored"]["logical_tensor_bytes"])
    print("\nTQ checkpoint benchmark PASS")
    print(f"  result:             {result['run_dir']}/result.json")
    print(f"  restored rows:      {load['restored']['total_rows']:,}")
    print(
        f"  valid tokens:       {load['restored']['lengths']['total_valid_tokens']:,}"
    )
    print(f"  logical tensors:    {logical_bytes / 1024**3:.3f} GiB")
    print(f"  checkpoint on disk: {disk_bytes / 1024**3:.3f} GiB")
    print(f"  save:               {save_seconds:.3f}s")
    print(f"  load:               {load_seconds:.3f}s")
    print(f"  effective save:     {logical_bytes / 1024**3 / save_seconds:.3f} GiB/s")
    print(f"  effective load:     {logical_bytes / 1024**3 / load_seconds:.3f} GiB/s")
    print(
        f"  disk/logical ratio: {disk_bytes / logical_bytes:.3f}x"
        if logical_bytes
        else "  disk/logical ratio: n/a"
    )
    if save["config"]["producer_mode"] != "quiescent":
        producers = save["producers"]
        print(
            "  producer rows/s:    "
            f"before={producers['before']['rows_per_s'] or 0:.1f}, "
            f"during={producers['during']['rows_per_s'] or 0:.1f}, "
            f"after={producers['after']['rows_per_s'] or 0:.1f}"
        )
        print(
            f"  write/save overlap: {'yes' if producers['overlap_observed'] else 'NO'}"
        )


def _parent_phase(args: argparse.Namespace) -> None:
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    run_name = args.run_name or (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    run_dir = checkpoint_root / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose another --run-name"
        )
    run_dir.mkdir(parents=True)
    config = BenchmarkConfig(
        checkpoint_root=str(checkpoint_root),
        run_dir=str(run_dir),
        num_rows=args.num_rows,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        payload_profile=args.payload_profile,
        batch_rows=args.batch_rows,
        num_storage_units=args.num_storage_units,
        producer_mode=args.producer_mode,
        num_producers=args.num_producers,
        producer_batch_rows=args.producer_batch_rows,
        producer_max_rows=args.producer_max_rows,
        producer_warmup_s=args.producer_warmup_s,
        producer_cooldown_s=args.producer_cooldown_s,
        producer_sleep_ms=args.producer_sleep_ms,
        verify_mode=args.verify_mode,
        verify_samples=args.verify_samples,
        verify_batch_rows=args.verify_batch_rows,
        group_size=args.group_size,
        weight_version=args.weight_version,
        seed=args.seed,
        ray_address=args.ray_address,
        phase_timeout_s=args.phase_timeout_s,
        torch_num_threads=args.torch_num_threads,
    )
    _validate_config(config)
    _preflight(config, args.skip_space_check)
    config_path = run_dir / "config.json"
    _write_json(config_path, asdict(config))

    _run_child("save", config_path, config.phase_timeout_s)
    _run_child("load", config_path, config.phase_timeout_s)

    result = {
        "run_dir": str(run_dir),
        "save": _read_json(run_dir / "save_result.json"),
        "load": _read_json(run_dir / "load_result.json"),
    }
    logical_bytes = int(result["load"]["restored"]["logical_tensor_bytes"])
    save_seconds = float(result["save"]["checkpoint"]["duration_s"])
    load_seconds = float(result["load"]["checkpoint"]["load_duration_s"])
    disk_bytes = int(result["save"]["checkpoint"]["disk_bytes"])
    result["summary"] = {
        "effective_save_gib_per_s": logical_bytes / 1024**3 / save_seconds,
        "effective_load_gib_per_s": logical_bytes / 1024**3 / load_seconds,
        "checkpoint_to_logical_ratio": (
            disk_bytes / logical_bytes if logical_bytes else None
        ),
    }
    _write_json(run_dir / "result.json", result)
    _print_summary(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        help="Shared filesystem directory under which a unique run is created.",
    )
    parser.add_argument("--run-name", help="Optional unique run directory name.")
    parser.add_argument("--num-rows", type=int, default=1024)
    parser.add_argument("--min-seq-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--payload-profile",
        choices=("generation", "train-ready"),
        default="generation",
    )
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--num-storage-units", type=int, default=2)
    parser.add_argument(
        "--producer-mode",
        choices=("quiescent", "thread", "process"),
        default="quiescent",
    )
    parser.add_argument("--num-producers", type=int, default=4)
    parser.add_argument("--producer-batch-rows", type=int, default=32)
    parser.add_argument(
        "--producer-max-rows",
        type=int,
        default=4096,
        help="Maximum rows per producer; 0 means unlimited.",
    )
    parser.add_argument("--producer-warmup-s", type=float, default=1.0)
    parser.add_argument("--producer-cooldown-s", type=float, default=0.25)
    parser.add_argument("--producer-sleep-ms", type=float, default=0.0)
    parser.add_argument(
        "--verify-mode",
        choices=("none", "sample", "all"),
        default="sample",
    )
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--verify-batch-rows", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--weight-version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ray-address",
        default="",
        help="Optional existing Ray address. Empty starts an isolated local Ray.",
    )
    parser.add_argument("--phase-timeout-s", type=float, default=1800)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip the conservative free-disk preflight.",
    )
    parser.add_argument(
        "--_phase",
        choices=("save", "load"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_config", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args._phase:
        if not args._config:
            parser.error("--_config is required for an internal phase")
        config = BenchmarkConfig(**_read_json(Path(args._config)))
        _validate_config(config)
        if args._phase == "save":
            _save_phase(config)
        else:
            _load_phase(config)
        return
    if not args.checkpoint_root:
        parser.error("--checkpoint-root is required")
    _parent_phase(args)


if __name__ == "__main__":
    main()
