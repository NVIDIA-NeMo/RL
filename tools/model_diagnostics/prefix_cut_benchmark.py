# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise production prefix-cut admission without a trainer or reward models.

Four physical nodes: driver/load clients, two TP=4 generation engines (configurable
cut slots), and configurable real SimpleStorage units on the remaining node. The load
clients emulate Gym model participants' wire contract, not environment rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import yaml
from nemo_gym._checkpoint.generation_cut import (
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutReceipt,
)

from nemo_rl.models.generation.vllm.config import (
    CompactOutputConfig,
    GenerationCutPipelineConfig,
)
from nemo_rl.utils.checkpoint_latency import CheckpointLatencyConfig
from nemo_rl.utils.outdated_config_checks import check_outdated_config


def worker_pipeline_config(
    generation_config: dict[str, Any], *, enabled: bool, write_concurrency: int
) -> GenerationCutPipelineConfig:
    """Preserve inherited payload bounds while applying benchmark CLI controls."""
    inherited = (generation_config.get("generation_cut_control") or {}).get(
        "pipeline"
    ) or {}
    return GenerationCutPipelineConfig.model_validate(
        {
            **inherited,
            "enabled": enabled,
            "write_concurrency": write_concurrency,
        }
    )


G_WORKER_FQN = (
    "tools.model_diagnostics.prefix_cut_benchmark_worker.PrefixCutBenchmarkWorker"
)
G_WORKER_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"


@dataclass(frozen=True)
class Call:
    rollout_id: str
    model_call_id: str
    admitted_at: float


def inventories_for_calls(
    calls: list[Call], *, checkpoint_id: str, count: int
) -> list[GenerationCutInventory]:
    """Partition calls into disjoint, nonempty Gym-participant inventories."""
    if count < 1 or len(calls) < count:
        raise ValueError(f"Need at least {count} active calls; got {len(calls)}")
    if len({call.model_call_id for call in calls}) != len(calls):
        raise ValueError("Active model-call IDs must be unique")
    groups: list[list[GenerationCutPrefix]] = [[] for _ in range(count)]
    for index, call in enumerate(calls):
        groups[index % count].append(
            GenerationCutPrefix(
                ticket_id=call.model_call_id,
                rollout_id=call.rollout_id,
                attempt=0,
                model_call_id=call.model_call_id,
                admitted_at=call.admitted_at,
            )
        )
    return [
        GenerationCutInventory.build(
            checkpoint_id=checkpoint_id,
            server_name=f"benchmark-model-participant-{index}",
            active_prefixes=prefixes,
        )
        for index, prefixes in enumerate(groups)
    ]


class StorageService:
    """Bootstrap and save real TQ storage on the dedicated storage node."""

    def __init__(self, config: dict[str, Any], node_id: str) -> None:
        # Backend dependencies are optional outside the cluster benchmark.
        from nemo_rl.data_plane import build_data_plane_client

        self.client = build_data_plane_client(
            config,
            checkpointing=True,
            storage_unit_node_ids=[node_id] * config["simple"]["num_storage_units"],
        )

    def ready(self) -> bool:
        return True

    def save(self, path: str) -> float:
        started = time.monotonic()
        self.client.save_checkpoint(
            path, metadata={"benchmark": "prefix_cut_without_training"}
        )
        metadata = json.loads((Path(path) / "metadata.json").read_text())
        if metadata.get("storage_saved") is not True:
            raise RuntimeError("TQ did not commit a complete storage snapshot")
        return time.monotonic() - started

    def verify_keys(self, keys: list[str], partition: str) -> int:
        present = set(self.client.list_sample_ids(partition_id=partition))
        missing = sorted(set(keys) - present)
        if missing:
            raise RuntimeError(
                f"Durable receipts name {len(missing)} absent TQ rows: {missing[:5]}"
            )
        return len(set(keys))


class Load:
    """Keep a bounded number of ordinary captured HTTP calls outstanding."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str,
        name: str,
    ) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = name
        self.allow_new = asyncio.Event()
        self.allow_new.set()
        self.active: dict[str, Call] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.next_id = 0
        self.completed = 0
        self.errors = 0
        self.first_errors: list[str] = []
        self.output_tokens = 0

    def resize(self, outstanding: int) -> None:
        for _ in range(len(self.tasks), outstanding):
            self.tasks.append(asyncio.create_task(self._run()))

    async def _run(self) -> None:
        while True:
            await self.allow_new.wait()
            self.next_id += 1
            index = self.next_id
            identifier = f"{self.name}-call-{index}"
            call = Call(identifier, identifier, time.time())
            self.active[identifier] = call
            # Mix short completions with long-lived calls so final formatting
            # and cuts contend naturally, including while the fence is closed.
            max_tokens = (256, 1024, 8192, 16384, 32768)[index % 5]
            prompt = (
                f"Problem {index}: Derive and explain mathematical identities, "
                "checking each step carefully. Continue with numbered examples.\n"
                + "Consider algebra, probability, geometry, and number theory. "
                * 80
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 0,
                "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "truncate_history_thinking": False,
                },
                "ng_capture": {
                    "rollout_id": call.rollout_id,
                    "model_call_id": call.model_call_id,
                    "mode": "text",
                },
            }
            try:
                async with self.session.post(
                    self.base_url + "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=7200, connect=60),
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: {(await response.text())[:300]}"
                        )
                    body = await response.json()
                    if "ng_commit_coords" not in body:
                        raise RuntimeError(
                            "Captured generation response has no staging coordinates"
                        )
                    self.output_tokens += body.get("usage", {}).get(
                        "completion_tokens", 0
                    )
                    self.completed += 1
            except (aiohttp.ClientError, TimeoutError, RuntimeError) as error:
                self.errors += 1
                if len(self.first_errors) < 5:
                    self.first_errors.append(str(error))
                await asyncio.sleep(1)
            finally:
                self.active.pop(identifier, None)

    def summary(self) -> dict[str, Any]:
        return {
            "active_http_calls": len(self.active),
            "completed": self.completed,
            "errors": self.errors,
            "output_tokens": self.output_tokens,
            "first_errors": self.first_errors,
        }

    def check_healthy(self) -> None:
        for task in self.tasks:
            if task.done():
                task.result()
                raise RuntimeError("A generation load task exited unexpectedly")
        if self.errors > max(100, self.completed):
            raise RuntimeError(f"Generation load is failing: {self.summary()}")

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def worker_call(generation: Any, method: str) -> Any:
    """Call a diagnostic on the single TP leader in this generation group."""
    # Ray is optional for CPU-only protocol tests.
    import ray

    results = ray.get(
        generation.worker_group.run_all_workers_single_data(
            method, run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]
        ),
        timeout=60,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one engine leader, got {len(results)}")
    return results[0]


async def collect_heap_summaries(
    generations: list[Any], output: Path, *, label: str
) -> None:
    """Run the intrusive census between cuts, with separately recorded cost."""
    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            asyncio.to_thread(worker_call, generation, "benchmark_heap_summary")
            for generation in generations
        ),
        return_exceptions=True,
    )
    workers = []
    for result in results:
        if isinstance(result, BaseException):
            # A diagnostic failure must remain visible without losing a
            # successfully committed snapshot or leaking generation requests.
            workers.append({"error": f"{type(result).__name__}: {result}"})
        else:
            workers.append(result)
    artifact = {
        "label": label,
        "wall_s": time.perf_counter() - started,
        "workers": workers,
    }
    (output / f"heap-{label}.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps({"event": "heap_census", **artifact}), flush=True)


async def request_cut(
    session: aiohttp.ClientSession,
    base_url: str,
    inventory: GenerationCutInventory,
    *,
    auth_token: str,
    deadline_ts: float,
) -> dict[str, Any]:
    started = time.monotonic()
    endpoint = base_url.rstrip("/").removesuffix("/v1") + GENERATION_CUT_ROUTE
    try:
        async with session.post(
            endpoint,
            json=inventory.model_dump(mode="json"),
            headers={
                "Authorization": f"Bearer {auth_token}",
                "x-nemo-gym-cut-deadline": str(deadline_ts),
            },
            timeout=aiohttp.ClientTimeout(
                total=max(1, deadline_ts - time.time()) + 10, connect=30
            ),
        ) as response:
            if response.status != 200:
                return {
                    "http_status": response.status,
                    "elapsed_s": time.monotonic() - started,
                    "calls": len(inventory.active_prefixes),
                }
            receipt = GenerationCutReceipt.model_validate(await response.json())
            receipt.validate_for(inventory)
    except (aiohttp.ClientError, TimeoutError) as error:
        return {
            "transport_error": str(error),
            "elapsed_s": time.monotonic() - started,
            "calls": len(inventory.active_prefixes),
        }
    return {
        "http_status": 200,
        "elapsed_s": time.monotonic() - started,
        "calls": len(inventory.active_prefixes),
        "durable_prefixes": sum(
            ack.disposition == "durable_prefix" for ack in receipt.prefixes
        ),
        "receipt": receipt.model_dump(mode="json"),
    }


async def run_snapshot(
    generations: list[Any],
    loads: list[Load],
    storage: Any,
    session: aiohttp.ClientSession,
    *,
    auth_token: str,
    round_index: int,
    inventory_count: int,
    output: Path,
    cut_budget: float,
    staging_partition: str = "rollout_staging",
) -> dict[str, Any]:
    # Ray is optional outside the cluster benchmark.
    import ray

    checkpoint_id = f"benchmark-round-{round_index}"
    started = time.monotonic()
    timings: dict[str, float] = {}
    for load in loads:
        load.allow_new.clear()
    try:
        await asyncio.gather(
            *(
                asyncio.to_thread(worker_call, generation, "benchmark_begin_cut_window")
                for generation in generations
            )
        )
        fence_started = time.monotonic()
        fence_results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    generation.begin_token_capture_snapshot_fence,
                    timeout_s=180,
                    checkpoint_id=checkpoint_id,
                )
                for generation in generations
            ),
            return_exceptions=True,
        )
        for result in fence_results:
            if isinstance(result, BaseException):
                raise RuntimeError(
                    "Could not acquire every generation fence"
                ) from result
        timings["fence_acquire_s"] = time.monotonic() - fence_started
        calls_by_worker = await asyncio.gather(
            *(
                asyncio.to_thread(worker_call, generation, "benchmark_active_calls")
                for generation in generations
            )
        )
        all_inventories = []
        for calls, load in zip(calls_by_worker, loads, strict=True):
            records = [
                load.active[call_id] for call_id in calls if call_id in load.active
            ]
            all_inventories.append(
                inventories_for_calls(
                    records, checkpoint_id=checkpoint_id, count=inventory_count
                )
            )
        cut_started = time.monotonic()
        deadline = time.time() + cut_budget
        results = await asyncio.gather(
            *(
                asyncio.gather(
                    *(
                        request_cut(
                            session,
                            load.base_url,
                            inventory,
                            auth_token=auth_token,
                            deadline_ts=deadline,
                        )
                        for inventory in inventories
                    )
                )
                for load, inventories in zip(loads, all_inventories, strict=True)
            )
        )
        timings["cut_http_wait_s"] = time.monotonic() - cut_started
        (output / f"cut-receipts-{round_index:03d}.json").write_text(
            json.dumps(results)
        )
        # Expired HTTP requests can leave native writes running. Never take a
        # storage snapshot or clear rows while those writes still own slots.
        drain_started = time.monotonic()
        while True:
            statuses = await asyncio.gather(
                *(
                    asyncio.to_thread(worker_call, generation, "benchmark_status")
                    for generation in generations
                )
            )
            if all(
                not status["active_inventories"]
                and not status["pending_inventories"]
                and not status.get("inventory_grouping", {}).get("pending_members", 0)
                and not status.get("inventory_grouping", {}).get("active_groups", 0)
                and not status.get("cut_pipeline", {}).get("pending_inventories", 0)
                and not status.get("cut_pipeline", {}).get("active_writes", 0)
                for status in statuses
            ):
                break
            if time.monotonic() - drain_started > 240:
                raise TimeoutError(
                    "Native cut writers did not quiesce before the storage snapshot"
                )
            await asyncio.sleep(1)
        timings["native_writer_drain_s"] = time.monotonic() - drain_started
        for load, receipts in zip(loads, results, strict=True):
            if not any(receipt.get("durable_prefixes", 0) for receipt in receipts):
                raise RuntimeError(
                    f"{load.name} produced no durable prefixes; refusing to call this a successful prefix-cut smoke"
                )
        keys = [
            key
            for receipts in results
            for receipt in receipts
            for ack in receipt.get("receipt", {}).get("prefixes", [])
            if ack["disposition"] == "durable_prefix"
            for key in ack["staging_keys"]
        ]
        verified_rows = await asyncio.to_thread(
            ray.get, storage.verify_keys.remote(keys, staging_partition)
        )
        snapshot_dir = output / f"snapshot-{round_index:03d}"
        snapshot_dir.mkdir()
        save_started = time.monotonic()
        tq_seconds = await asyncio.to_thread(
            ray.get, storage.save.remote(str(snapshot_dir / "data_plane"))
        )
        timings["tq_save_s"] = time.monotonic() - save_started
        timings["tq_storage_save_s"] = tq_seconds
        (snapshot_dir / "receipts.json").write_text(json.dumps(results))
        (snapshot_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": "generation_only_prefix_cut_benchmark",
                    "checkpoint_id": checkpoint_id,
                    "committed": True,
                    "verified_prefix_rows": verified_rows,
                },
                indent=2,
            )
        )
    finally:
        release_started = time.monotonic()
        try:
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        generation.end_token_capture_snapshot_fence, timeout_s=180
                    )
                    for generation in generations
                )
            )
        finally:
            timings["fence_release_s"] = time.monotonic() - release_started
            for load in loads:
                load.allow_new.set()
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        worker_call, generation, "benchmark_end_cut_window"
                    )
                    for generation in generations
                )
            )
    timings["total_s"] = time.monotonic() - started
    compact = []
    for load, inventories, receipts, status in zip(
        loads, all_inventories, results, statuses, strict=True
    ):
        compact.append(
            {
                "name": load.name,
                "slots": status["slots"],
                "inventories": len(inventories),
                "calls": sum(
                    len(inventory.active_prefixes) for inventory in inventories
                ),
                "durable_prefixes": sum(
                    receipt.get("durable_prefixes", 0) for receipt in receipts
                ),
                "http_successes": sum(
                    receipt.get("http_status") == 200 for receipt in receipts
                ),
                "inventory_http_latency_s": [
                    receipt["elapsed_s"] for receipt in receipts
                ],
                "diagnostics": status,
                "load": load.summary(),
            }
        )
    result = {
        "round": round_index,
        "checkpoint_id": checkpoint_id,
        "timings": timings,
        "workers": compact,
    }
    (output / f"round-{round_index:03d}.json").write_text(json.dumps(result, indent=2))
    return result


async def exercise(
    generations: list[Any],
    storage: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
    auth_token: str,
    wandb_run: Any,
) -> None:
    connector = aiohttp.TCPConnector(
        limit=16384, limit_per_host=8192, ttl_dns_cache=300
    )
    output = Path(args.output)
    async with aiohttp.ClientSession(connector=connector) as session:
        loads = [
            Load(
                session,
                generation.dp_openai_server_base_urls[0],
                config["policy"]["model_name"],
                generation.http_server_api_key,
                f"engine-{index}-slots-{slots}",
            )
            for index, (generation, slots) in enumerate(
                zip(generations, args.slots, strict=True)
            )
        ]
        try:
            for load in loads:
                load.resize(args.outstanding)
            for second in range(args.warmup_seconds):
                if second % 30 == 0:
                    for load in loads:
                        load.check_healthy()
                    print(
                        json.dumps(
                            {
                                "event": "load_warmup",
                                "elapsed_s": second,
                                "loads": [load.summary() for load in loads],
                            }
                        ),
                        flush=True,
                    )
                await asyncio.sleep(1)
            if args.heap_census:
                await collect_heap_summaries(generations, output, label="warmup")
            else:
                statuses = await asyncio.gather(
                    *(
                        asyncio.to_thread(worker_call, generation, "benchmark_status")
                        for generation in generations
                    )
                )
                (output / "status-warmup.json").write_text(
                    json.dumps({"label": "warmup", "workers": statuses}, indent=2)
                )
            for index in range(args.rounds):
                for load in loads:
                    load.check_healthy()
                if any(load.errors > max(10, load.completed) for load in loads):
                    raise RuntimeError(
                        f"Generation load is failing: {[load.summary() for load in loads]}"
                    )
                result = await run_snapshot(
                    generations,
                    loads,
                    storage,
                    session,
                    auth_token=auth_token,
                    round_index=index,
                    inventory_count=args.inventories,
                    output=output,
                    cut_budget=args.cut_budget,
                    staging_partition=config["token_capture"]["staging_partition"],
                )
                print(
                    json.dumps(
                        {
                            "event": "snapshot_committed",
                            "round": index,
                            "timings": result["timings"],
                            "workers": [
                                {
                                    k: v
                                    for k, v in worker.items()
                                    if k
                                    not in ["diagnostics", "inventory_http_latency_s"]
                                }
                                for worker in result["workers"]
                            ],
                        }
                    ),
                    flush=True,
                )
                log_values = {
                    f"benchmark/{key}": value
                    for key, value in result["timings"].items()
                }
                for worker in result["workers"]:
                    name = worker["name"]
                    log_values[f"benchmark/{name}/durable_prefixes"] = worker[
                        "durable_prefixes"
                    ]
                    log_values[f"benchmark/{name}/calls"] = worker["calls"]
                    for phase, values in worker["diagnostics"]["phases"].items():
                        log_values[f"benchmark/{name}/{phase}/max_s"] = values["max_s"]
                    for key, value in (
                        worker["diagnostics"].get("write_batching", {}).items()
                    ):
                        log_values[f"benchmark/{name}/write_batching/{key}"] = value
                    for key, value in (
                        worker["diagnostics"].get("decoder_copy", {}).items()
                    ):
                        log_values[f"benchmark/{name}/decoder_copy/{key}"] = value
                    for group in (
                        "logprob_detachment",
                        "compact_output",
                        "frontend_timings",
                        "numeric_capture",
                        "inventory_grouping",
                        "cut_pipeline",
                    ):
                        for key, value in worker["diagnostics"].get(group, {}).items():
                            log_values[f"benchmark/{name}/{group}/{key}"] = value
                wandb_run.log(log_values, step=index)
                # run_snapshot has released both fences and stopped cut timing.
                if args.heap_census:
                    await collect_heap_summaries(
                        generations, output, label=f"after-{index:03d}"
                    )
                if index + 1 < args.rounds:
                    await asyncio.sleep(args.interval_seconds)
        finally:
            await asyncio.gather(*(load.close() for load in loads))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--slots",
        type=int,
        nargs=2,
        choices=range(1, 5),
        default=[2, 4],
        help="Admission slots on each of the two generation engines",
    )
    parser.add_argument("--outstanding", type=int, default=2048)
    parser.add_argument("--inventories", type=int, default=64)
    parser.add_argument(
        "--storage-units",
        type=int,
        default=8,
        help="Positive number of SimpleStorage units on the dedicated storage node",
    )
    parser.add_argument("--warmup-seconds", type=int, default=360)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--cut-budget", type=float, default=150)
    parser.add_argument(
        "--gil-switch-interval-ms",
        type=float,
        default=None,
        help="Benchmark generation actors only; unset keeps Python's GIL interval",
    )
    parser.add_argument(
        "--output-chunk-size",
        type=int,
        default=32,
        help="vLLM output-processing chunk size",
    )
    parser.add_argument(
        "--batch-inventory-writes",
        action="store_true",
        help="Combine prepared prefix writes across admitted inventories",
    )
    parser.add_argument(
        "--group-pending-inventories",
        action="store_true",
        help="Group pending inventories before admission (eight inventories/256 calls)",
    )
    parser.add_argument(
        "--direct-prefix-batch-encoding",
        action="store_true",
        help="Build text prefix batches directly without per-record tensors",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--copy-decoded-arrays",
        action="store_true",
        help="Own NumPy arrays during decoding before outputs enter the queue",
    )
    parser.add_argument("--check-runtime-imports", action="store_true")
    parser.add_argument(
        "--compact-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Queue encoded engine payloads and batch numeric captured logprobs",
    )
    parser.add_argument(
        "--packed-capture-logprobs",
        action="store_true",
        help="Keep selected-token histories in packed numeric buffers with either queue mode",
    )
    parser.add_argument("--frontend-timings", action="store_true")
    parser.add_argument(
        "--checkpoint-latency-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fence, pipeline, native TQ and GC timing probes; otherwise inherit config",
    )
    parser.add_argument(
        "--cut-pipeline",
        action="store_true",
        help="Stream all inventories through byte-bounded preparation and concurrent writes",
    )
    parser.add_argument(
        "--compact-queue-max-bytes", type=int, default=2 * 1024 * 1024 * 1024
    )
    parser.add_argument(
        "--heap-census",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan tracked objects between snapshots (default: enabled)",
    )
    parser.add_argument(
        "--router-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect router arrays for prefix checkpoints (default: enabled)",
    )
    args = parser.parse_args()
    if args.gil_switch_interval_ms is not None and (
        not math.isfinite(args.gil_switch_interval_ms / 1000)
        or args.gil_switch_interval_ms / 1000 <= 0
    ):
        parser.error("GIL switch interval must be finite and positive")
    if args.outstanding < args.inventories or args.rounds < 1 or args.cut_budget <= 0:
        parser.error(
            "Need outstanding >= inventories >= 1, rounds >= 1, and positive cut budget"
        )
    if args.inventories < 1 or min(args.warmup_seconds, args.interval_seconds) < 0:
        parser.error("Inventories must be positive and waits nonnegative")
    if args.storage_units < 1:
        parser.error("Storage units must be positive")
    if args.output_chunk_size < 1:
        parser.error("Output chunk size must be positive")
    if args.compact_queue_max_bytes < 1:
        parser.error("Compact queue byte limit must be positive")
    if args.compact_output and args.copy_decoded_arrays:
        parser.error("Compact output and decoder copying are mutually exclusive")
    if args.cut_pipeline and (
        args.group_pending_inventories or args.batch_inventory_writes
    ):
        parser.error(
            "Cut pipeline replaces --group-pending-inventories and --batch-inventory-writes"
        )
    config = yaml.safe_load(Path(args.reference_config).read_text())
    check_outdated_config(config)
    generation_config = copy.deepcopy(config["policy"]["generation"])
    control_config = generation_config.get("generation_cut_control")
    raw_latency_config = (
        control_config.get("latency_diagnostics")
        if control_config is not None
        else None
    )
    latency_config = CheckpointLatencyConfig.model_validate(raw_latency_config or {})
    if args.checkpoint_latency_diagnostics is not None:
        latency_config = latency_config.model_copy(
            update={"enabled": args.checkpoint_latency_diagnostics}
        )
        latency_config = type(latency_config).model_validate(
            latency_config.model_dump()
        )

    if args.gil_switch_interval_ms is not None:
        generation_config["_benchmark_gil_switch_interval_ms"] = (
            args.gil_switch_interval_ms
        )
    generation_config["vllm_cfg"]["load_format"] = "auto"
    # This benchmark controls copying through its instrumented worker hook.
    # Prevent an inherited production setting from enabling a second hook.
    generation_config["vllm_cfg"]["copy_decoded_arrays"] = False
    inherited_compact = CompactOutputConfig.model_validate(
        generation_config["vllm_cfg"].get("compact_output", {})
    )
    if inherited_compact.detach_queued_logprobs and (
        args.compact_output or args.copy_decoded_arrays
    ):
        parser.error(
            "Queued logprob detachment requires the native queue without full decoder copying"
        )
    generation_config["vllm_cfg"]["compact_output"] = {
        "enabled": args.compact_output,
        "packed_capture_logprobs": args.packed_capture_logprobs,
        "detach_queued_logprobs": inherited_compact.detach_queued_logprobs,
        "max_queued_bytes": args.compact_queue_max_bytes,
        "max_queued_messages": 32768,
        "frontend_timings": args.frontend_timings,
    }
    generation_config["_mtp_weights_from_refit"] = False
    generation_config["_draft_weights_from_refit"] = False
    generation_config["worker_extension_cls_fqn"] = G_WORKER_FQN
    generation_config["colocated"]["enabled"] = False
    generation_config["colocated"]["resources"]["num_nodes"] = 1
    config["policy"]["router_replay"]["enabled"] = args.router_replay
    generation_config["vllm_kwargs"]["enable_return_routed_experts"] = (
        args.router_replay
    )
    generation_config["vllm_cfg"]["enable_return_routed_experts"] = args.router_replay
    assert generation_config["vllm_cfg"]["tensor_parallel_size"] == 4
    assert generation_config["vllm_cfg"]["flat_logprobs"]
    dp_config = copy.deepcopy(config["data_plane"])
    if dp_config["backend"] != "simple":
        raise ValueError("This benchmark pins real SimpleStorage units to one node")
    dp_config["simple"]["num_storage_units"] = args.storage_units
    dp_config["simple"]["storage_unit_placement"] = None
    plan = {
        "nodes": 4,
        "generation_nodes": 2,
        "trainer_nodes": 0,
        "slots": args.slots,
        "outstanding_per_engine": args.outstanding,
        "inventories_per_engine": args.inventories,
        "storage_units": args.storage_units,
        "router_replay": args.router_replay,
        "return_routed_experts": generation_config["vllm_kwargs"][
            "enable_return_routed_experts"
        ],
        "load_format": "auto",
        "reference_config": args.reference_config,
        "model": config["policy"]["model_name"],
        "warmup_seconds": args.warmup_seconds,
        "rounds": args.rounds,
        "interval_seconds": args.interval_seconds,
        "output_chunk_size": args.output_chunk_size,
        "gil_switch_interval_ms": args.gil_switch_interval_ms,
        "batch_inventory_writes": args.batch_inventory_writes,
        "group_pending_inventories": args.group_pending_inventories,
        "cut_pipeline": args.cut_pipeline,
        "pipeline": worker_pipeline_config(
            generation_config,
            enabled=args.cut_pipeline,
            write_concurrency=args.slots[0],
        ).model_dump(),
        "pipeline_per_worker": [
            worker_pipeline_config(
                generation_config, enabled=args.cut_pipeline, write_concurrency=slots
            ).model_dump()
            for slots in args.slots
        ],
        "checkpoint_latency_diagnostics": latency_config.model_dump(),
        "direct_prefix_batch_encoding": args.direct_prefix_batch_encoding,
        "copy_decoded_arrays": args.copy_decoded_arrays,
        "production_decoder_copy": generation_config["vllm_cfg"]["copy_decoded_arrays"],
        "heap_census": args.heap_census,
        "compact_output": args.compact_output,
        "frontend_timings": args.frontend_timings,
        "compact_queue_max_bytes": args.compact_queue_max_bytes,
        "packed_capture_logprobs": args.packed_capture_logprobs,
        "detach_queued_logprobs": inherited_compact.detach_queued_logprobs,
    }
    print(json.dumps({"event": "benchmark_plan", **plan}), flush=True)
    if args.dry_run:
        return
    # GPU/Ray/backend dependencies are optional for protocol tests and dry runs.
    import ray
    import wandb
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from nemo_rl.data_plane.background_prefix_cleanup import (
        GenerationPrefixCleanupConfig,
    )
    from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
    from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry

    if args.check_runtime_imports:
        print("PASS: driver runtime imports", flush=True)
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    os.environ["PREFIX_CUT_BENCH_OUTPUT"] = str(output / "worker_diagnostics")
    os.environ["VLLM_V1_OUTPUT_PROC_CHUNK_SIZE"] = str(args.output_chunk_size)
    os.environ["PREFIX_CUT_BENCH_COPY_DECODED_ARRAYS"] = (
        "1" if args.copy_decoded_arrays else "0"
    )
    telemetry_config = SimpleNamespace(**config)
    telemetry_config.telemetry = SimpleNamespace(**config["telemetry"])
    init_telemetry_driver(telemetry_config, algorithm="prefix_cut_benchmark")
    maybe_configure_data_plane_env(dp_config)
    init_ray()
    ACTOR_ENVIRONMENT_REGISTRY[G_WORKER_FQN] = G_WORKER_PYTHON
    gpu_nodes = sorted(
        [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 4
        ],
        key=lambda node: node["NodeManagerAddress"],
    )
    if len(gpu_nodes) != 4:
        raise RuntimeError(f"Expected four GPU nodes, found {len(gpu_nodes)}")
    driver_ip = ray.util.get_node_ip_address()
    remaining = [node for node in gpu_nodes if node["NodeManagerAddress"] != driver_ip]
    if len(remaining) != 3:
        raise RuntimeError(
            "Could not isolate driver, two engines and storage onto four physical nodes"
        )
    generation_nodes, storage_node = remaining[:2], remaining[2]
    (output / "node_roles.json").write_text(
        json.dumps(
            {
                "driver": driver_ip,
                "generation": [node["NodeManagerAddress"] for node in generation_nodes],
                "storage": storage_node["NodeManagerAddress"],
            },
            indent=2,
        )
    )
    storage = (
        ray.remote(num_cpus=1)(StorageService)
        .options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                storage_node["NodeID"], soft=False
            )
        )
        .remote(dp_config, storage_node["NodeID"])
    )
    ray.get(storage.ready.remote(), timeout=180)
    auth_token = secrets.token_hex(32)

    def create_generation(item: tuple[int, tuple[dict[str, Any], int]]) -> Any:
        index, (node, slots) = item
        per_worker = copy.deepcopy(generation_config)
        per_worker["generation_cut_control"] = {
            "max_workers": slots,
            "latency_diagnostics": latency_config.model_dump(),
            "pipeline": worker_pipeline_config(
                generation_config, enabled=args.cut_pipeline, write_concurrency=slots
            ).model_dump(),
            "batch_inventory_writes": args.batch_inventory_writes,
            "group_pending_inventories": args.group_pending_inventories,
            "direct_prefix_batch_encoding": args.direct_prefix_batch_encoding,
        }
        cluster = RayVirtualCluster(
            [4],
            num_gpus_per_node=4,
            name=f"prefix_bench_engine_{index}_slots_{slots}",
            node_resource_constraints=[{f"node:{node['NodeManagerAddress']}": 0.001}],
        )
        generation = VllmGeneration(
            cluster,
            per_worker,
            name_prefix=f"prefix_bench_engine_{index}_slots_{slots}",
        )
        generation.setup_token_capture(
            dp_config,
            config["token_capture"]["staging_partition"],
            generation_prefix_cuts_enabled=True,
            generation_cut_control_token=auth_token,
            generation_cut_control_timeout_s=120,
            generation_prefix_batch_size=config["rollout_recovery"][
                "generation_prefix_batch_size"
            ],
            generation_prefix_batch_max_tokens=config["rollout_recovery"][
                "generation_prefix_batch_max_tokens"
            ],
            generation_prefix_cleanup=GenerationPrefixCleanupConfig.model_validate(
                config["rollout_recovery"]["generation_prefix_cleanup"]
            ),
        )
        return generation

    generations = []
    completed = False
    run = wandb.init(
        entity="nvidia",
        project="nano35-bf16-prefix-router-replay",
        name=output.name,
        job_type="generation-only-prefix-cut-benchmark",
        config=plan,
        dir=str(output),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            generations = list(
                pool.map(
                    create_generation,
                    enumerate(zip(generation_nodes, args.slots, strict=True)),
                )
            )
        asyncio.run(exercise(generations, storage, config, args, auth_token, run))
        (output / "COMPLETE").write_text(
            "All requested prefix-cut snapshots committed.\n"
        )
        completed = True
    finally:
        for generation in generations:
            generation.shutdown()
        run.finish(exit_code=0 if completed else 1)
        shutdown_telemetry()
        ray.shutdown()


if __name__ == "__main__":
    main()
