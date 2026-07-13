"""Native-vLLM NCCL baseline for a TP rollout (TP2 by default).

This deliberately uses vLLM's PyNccl implementation on both sides, avoiding the
metadata/communicator mismatch between NeMo's ``nccl.core`` package and vLLM.

Workflow:

1. Start ``sender`` in a one-GPU pod using the same vLLM image as the rollout.
2. Run ``controller`` where it can reach the rollout pod's DYN_SYSTEM_PORT.
3. Controller initializes a 1+TP-rank group (sender rank 0, receiver ranks
   1..TP), then starts packed send/receive concurrently and reports update wall
   time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "refit-stage-v1"
SOURCE_LAYOUT_MODES = (
    "preconsolidated_transport_only",
    "consolidated_e2e",
)
STAGE_NAMES = (
    "control_discovery",
    "source_preparation",
    "setup_registration",
    "transfer_planning",
    "wire_transfer",
    "receive_sync",
    "transformation",
    "installation",
    "post_install",
    "rollout_readiness",
)


def _pctl(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(round((percentile / 100.0) * (len(ordered) - 1))),
        ),
    )
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _pctl(values, 95),
        "max": max(values) if values else None,
    }


def _version_path(base: str | Path, version: int) -> Path:
    """Return the distinct handshake path for one weight version."""
    return Path(f"{base}.v{version}")


def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        from safetensors.torch import load_file

        weights: dict[str, torch.Tensor] = {}
        for shard in sorted(checkpoint.glob("*.safetensors")):
            weights.update(load_file(str(shard), device="cpu"))
        if not weights:
            raise RuntimeError(f"no safetensors found under {checkpoint}")
        return weights
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    return payload.get("hf_weights", payload)


def _wait_for_path(path: Path, timeout: float) -> float:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"handshake file did not appear: {path}")
        time.sleep(0.05)
    return time.perf_counter() - started


def _aggregate_cycles(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate measured records; callers exclude warmups before passing them."""
    stages: dict[str, dict[str, Any]] = {}
    for name in STAGE_NAMES:
        entries = [record["stages"][name] for record in records]
        values = [
            float(entry["seconds"])
            for entry in entries
            if entry.get("seconds") is not None
        ]
        statuses = sorted({str(entry.get("status", "unavailable")) for entry in entries})
        stages[name] = {
            "status": statuses[0] if len(statuses) == 1 else "mixed",
            "statuses": statuses,
            **_stats(values),
        }
    return {
        "cycles": len(records),
        "total_seconds": _stats([float(record["total_seconds"]) for record in records]),
        "unattributed_seconds": _stats(
            [float(record["unattributed_seconds"]) for record in records]
        ),
        "gbps": _stats(
            [float(record["gbps"]) for record in records if record.get("gbps") is not None]
        ),
        "stages": stages,
    }


def _measured_cycles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if not record.get("excluded", False)]


def _source_layout_metadata(args: argparse.Namespace) -> dict[str, Any]:
    preconsolidated = args.source_layout == "preconsolidated_transport_only"
    return {
        "mode": args.source_layout,
        "declared_source_layout": f"EP{args.source_ep_size}",
        "destination_layout": f"TP{args.destination_tp_size}",
        "actual_source_processes": 1 if preconsolidated else 0,
        "true_ep_topology_match": False,
        "consolidation_requested": not preconsolidated,
        "consolidation_included": False,
        "detail": (
            "One rank already owns the complete HF payload; EP shard consolidation "
            "is excluded. This is transport-only source-layout semantics, not a "
            f"true EP{args.source_ep_size} source topology."
            if preconsolidated
            else "Megatron-independent EP shard consolidation is not implemented."
        ),
    }


def _unsupported_source_layout_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "nccl",
        "role": args.role,
        "status": "unsupported",
        "reason_code": "ep_shard_consolidation_not_implemented",
        "source_layout": _source_layout_metadata(args),
        "stages": {
            name: _stage(
                "unsupported" if name == "source_preparation" else "not_run",
                detail=(
                    "An explicit EP shard consolidation implementation requires "
                    "checkpoint-specific/Megatron layout metadata."
                    if name == "source_preparation"
                    else None
                ),
            )
            for name in STAGE_NAMES
        },
        "bytes": None,
        "total_seconds": None,
        "gbps": None,
    }


def _write_unsupported_source_layout(args: argparse.Namespace) -> int:
    result = _unsupported_source_layout_result(args)
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    print("NCCL_SOURCE_LAYOUT_UNSUPPORTED", json.dumps(result), flush=True)
    return 0


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _stage(
    status: str = "unavailable",
    seconds: float | None = None,
    *,
    source: str | None = None,
    combined_group: str | None = None,
    combined_with: list[str] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status, "seconds": seconds}
    if source:
        value["source"] = source
    if combined_group:
        value["combined_group"] = combined_group
    if combined_with:
        value["combined_with"] = combined_with
    if detail:
        value["detail"] = detail
    return value


def _cache_metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Read cache behavior only when the server explicitly reports it."""
    containers = [response]
    for key in ("metadata", "timing", "timings", "refit_timing"):
        value = response.get(key)
        if isinstance(value, dict):
            containers.append(value)

    cache_keys = (
        "cache_reset",
        "cache_flushed",
        "prefix_cache_reset",
        "prefix_cache_flushed",
    )
    for container in containers:
        for key in cache_keys:
            if key in container:
                return {
                    "reported": True,
                    "behavior": key,
                    "value": container[key],
                }
        cache = container.get("cache")
        if isinstance(cache, dict):
            return {"reported": True, **cache}
    return {"reported": False, "behavior": "unavailable"}


def _timing_seconds(response: dict[str, Any], *names: str) -> float | None:
    containers = [response]
    for key in ("metadata", "timing", "timings", "refit_timing"):
        value = response.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, dict):
                seconds = value.get("seconds", value.get("duration_s"))
                if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                    return float(seconds)
                milliseconds = value.get("duration_ms")
                if isinstance(milliseconds, (int, float)) and not isinstance(
                    milliseconds, bool
                ):
                    return float(milliseconds) / 1000.0
    return None


def _response_stage(response: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Convert one canonical Dynamo route stage to the benchmark schema."""
    timing = response.get("timing")
    if not isinstance(timing, dict):
        return None
    stages = timing.get("stages")
    if not isinstance(stages, dict):
        return None
    source = stages.get(name)
    if not isinstance(source, dict):
        return None

    status = str(source.get("status", "unavailable"))
    seconds = _timing_seconds({"timing": {name: source}}, name)
    if status == "measured":
        status = "available"
    detail_parts = []
    if route_phases := source.get("route_phases"):
        detail_parts.append(f"route phases: {', '.join(map(str, route_phases))}")
    elif route_phase := source.get("route_phase"):
        detail_parts.append(f"route phase: {route_phase}")
    if reason := source.get("reason"):
        detail_parts.append(str(reason))
    return _stage(
        status,
        seconds,
        source="Dynamo vLLM receiver timing",
        combined_group=(
            "receiver_update" if status == "combined" else None
        ),
        combined_with=(
            [str(value) for value in source.get("combined_with", [])]
            if status == "combined"
            else None
        ),
        detail="; ".join(detail_parts) or None,
    )


def _result_schema(
    *,
    role: str,
    stages: dict[str, dict[str, Any]],
    total_seconds: float,
    byte_count: int,
    rate_seconds: float | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = {name: stages.get(name, _stage()) for name in STAGE_NAMES}
    measured = 0.0
    combined_group_seconds: dict[str, float] = {}
    for stage in normalized.values():
        if (
            stage.get("seconds") is None
            or stage.get("status") not in ("available", "combined")
        ):
            continue
        combined_group = stage.get("combined_group")
        if combined_group:
            group = str(combined_group)
            combined_group_seconds[group] = max(
                combined_group_seconds.get(group, 0.0),
                float(stage["seconds"]),
            )
        else:
            measured += float(stage["seconds"])
    measured += sum(combined_group_seconds.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "nccl",
        "role": role,
        "stages": normalized,
        "total_seconds": total_seconds,
        "unattributed_seconds": max(0.0, total_seconds - measured),
        "bytes": byte_count,
        "gbps": (
            byte_count * 8 / rate_seconds / 1e9
            if rate_seconds is not None and rate_seconds > 0
            else None
        ),
        "metadata": metadata or {},
    }


def _ordered(weights: dict[str, torch.Tensor]):
    import re

    def key(item):
        name = item[0]
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match:
            return (1, int(match.group(1)), name)
        if "embed_tokens" in name:
            return (0, 0, name)
        return (2, 0, name)

    return sorted(weights.items(), key=key)


def sender(args) -> int:
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferEngine,
    )

    total_start = time.perf_counter()
    checkpoint_start = time.perf_counter()
    weights = _load_checkpoint(args.checkpoint)
    checkpoint_seconds = time.perf_counter() - checkpoint_start
    ordered = _ordered(weights)
    manifest = {
        "names": [name for name, _ in ordered],
        "dtype_names": [str(tensor.dtype).removeprefix("torch.") for _, tensor in ordered],
        "shapes": [list(tensor.shape) for _, tensor in ordered],
        "bytes": sum(tensor.numel() * tensor.element_size() for _, tensor in ordered),
    }
    Path(args.manifest).write_text(json.dumps(manifest))

    gpu_ordered = None
    preload_seconds = 0.0
    if args.preload_gpu:
        preload_start = time.perf_counter()
        gpu_ordered = [
            (name, tensor.to("cuda", non_blocking=False))
            for name, tensor in ordered
        ]
        torch.cuda.synchronize()
        preload_seconds = time.perf_counter() - preload_start

    init_start = time.perf_counter()
    group = NCCLWeightTransferEngine.trainer_init(
        {
            "master_address": args.master_address,
            "master_port": args.master_port,
            "world_size": 1 + args.destination_tp_size,
        }
    )
    init_seconds = time.perf_counter() - init_start
    print(f"NCCL_SENDER_READY init={init_seconds:.3f}s bytes={manifest['bytes']}", flush=True)

    def gpu_weights():
        if gpu_ordered is not None:
            yield from gpu_ordered
        else:
            for name, tensor in ordered:
                yield name, tensor.to("cuda", non_blocking=False)

    preparation_seconds = checkpoint_seconds + preload_seconds
    preparation_status = "combined" if args.preload_gpu else "available"
    one_time_stages = {
        "source_preparation": _stage(
            preparation_status,
            preparation_seconds,
            source="sender_checkpoint",
            combined_group="checkpoint_load_and_gpu_preload" if args.preload_gpu else None,
            combined_with=(
                ["checkpoint_deserialization", "gpu_preload"]
                if args.preload_gpu
                else None
            ),
        ),
        "setup_registration": _stage(
            "available", init_seconds, source="NCCLWeightTransferEngine.trainer_init"
        ),
    }
    ack_base = args.ack or f"{args.trigger}.ack"
    raw_cycles: list[dict[str, Any]] = []
    total_cycles = args.warmup_cycles + args.cycles
    for index in range(total_cycles):
        version = args.start_version + index
        cycle_start = time.perf_counter()
        trigger_path = _version_path(args.trigger, version)
        trigger_seconds = _wait_for_path(trigger_path, args.timeout)

        send_start = time.perf_counter()
        NCCLWeightTransferEngine.trainer_send_weights(
            gpu_weights(),
            {
                "group": group,
                "packed": True,
                "packed_buffer_size_bytes": args.buffer_bytes,
                "packed_num_buffers": args.num_buffers,
            },
        )
        torch.cuda.synchronize()
        send_seconds = time.perf_counter() - send_start
        ack_path = _version_path(ack_base, version)
        ack_path.touch()
        stages = {
            "control_discovery": _stage(
                "available",
                trigger_seconds,
                source="controller trigger wait",
            ),
            "source_preparation": _stage(
                "not_applicable", detail="one-time timing is in one_time_stages"
            ),
            "setup_registration": _stage(
                "not_applicable", detail="one-time timing is in one_time_stages"
            ),
            "transfer_planning": _stage(
                "unavailable",
                detail="vLLM does not expose packed manifest planning timing",
            ),
            "wire_transfer": _stage(
                "available",
                send_seconds,
                source="trainer_send_weights packed completion",
            ),
            "receive_sync": _stage("not_applicable"),
            "transformation": _stage("not_applicable"),
            "installation": _stage("not_applicable"),
            "post_install": _stage("not_applicable"),
            "rollout_readiness": _stage("not_applicable"),
        }
        timing = _result_schema(
            role="sender",
            stages=stages,
            total_seconds=time.perf_counter() - cycle_start,
            byte_count=manifest["bytes"],
            rate_seconds=send_seconds,
            metadata={
                "cycle": index - args.warmup_cycles,
                "version": version,
                "excluded": index < args.warmup_cycles,
                "trigger_path": str(trigger_path),
                "ack_path": str(ack_path),
                "packed": True,
                "source_layout": _source_layout_metadata(args),
            },
        )
        timing.update(
            {
                "cycle": index - args.warmup_cycles,
                "version": version,
                "excluded": index < args.warmup_cycles,
                "send_seconds": send_seconds,
            }
        )
        raw_cycles.append(timing)
        label = "warmup" if index < args.warmup_cycles else "cycle"
        print(
            f"NCCL_SENDER_{label.upper()} version={version} "
            f"send={send_seconds:.3f}s excluded={index < args.warmup_cycles}",
            flush=True,
        )

    measured = _measured_cycles(raw_cycles)
    last_timing = measured[-1] if measured else raw_cycles[-1]
    send_stats = _stats([float(record["send_seconds"]) for record in measured])
    result = {
        **last_timing,
        "role": "sender",
        "bytes": manifest["bytes"],
        "cycles": args.cycles,
        "warmup_cycles": args.warmup_cycles,
        "raw_cycles": raw_cycles,
        "aggregate": _aggregate_cycles(measured),
        "one_time_stages": one_time_stages,
        "init_seconds": init_seconds,
        "checkpoint_load_seconds": checkpoint_seconds,
        "preload_seconds": preload_seconds,
        "send_seconds": send_stats["median"],
        "effective_gbps": (
            manifest["bytes"] * 8 / float(send_stats["median"]) / 1e9
            if send_stats["median"]
            else None
        ),
        "process_seconds": time.perf_counter() - total_start,
        "source_layout": _source_layout_metadata(args),
        "refit_timing": last_timing,
    }
    Path(args.result).write_text(json.dumps(result, indent=2))
    print("NCCL_SENDER_RESULT", json.dumps(result), flush=True)
    return 0


def controller(args) -> int:
    manifest_path = Path(args.manifest)
    deadline = time.monotonic() + args.timeout
    while not manifest_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"manifest did not appear: {manifest_path}")
        time.sleep(0.1)
    manifest = json.loads(manifest_path.read_text())

    total_start = time.perf_counter()
    init_start = time.perf_counter()
    init = _post(
        f"{args.system_url}/engine/init_weights_update_group",
        {
            "master_address": args.master_address,
            "master_port": args.master_port,
            "rank_offset": 1,
            "world_size": 1 + args.destination_tp_size,
        },
        args.timeout,
    )
    init_seconds = time.perf_counter() - init_start
    if init.get("status") not in ("ok", "success"):
        raise RuntimeError(f"NCCL init failed: {init}")

    update_group = [
        "wire_transfer",
        "receive_sync",
        "transformation",
        "installation",
    ]
    one_time_stages = {
        "source_preparation": _stage(
            "unavailable", detail="measured by the sender result"
        ),
        "setup_registration": _stage(
            "available",
            init_seconds,
            source="init_weights_update_group HTTP route",
            detail=(
                "server collective: "
                f"{server_init_stage['seconds']:.6f}s"
                if (
                    (server_init_stage := _response_stage(init, "setup_registration"))
                    and server_init_stage.get("seconds") is not None
                )
                else None
            ),
        ),
    }
    ack_base = args.ack or f"{args.trigger}.ack"
    raw_cycles: list[dict[str, Any]] = []
    total_cycles = args.warmup_cycles + args.cycles
    for index in range(total_cycles):
        version = args.start_version + index
        cycle_start = time.perf_counter()
        pause_start = time.perf_counter()
        paused = _post(f"{args.system_url}/engine/pause_generation", {}, 30)
        pause_seconds = time.perf_counter() - pause_start
        if paused.get("status") not in ("ok", "success"):
            raise RuntimeError(f"pause failed: {paused}")

        trigger_path = _version_path(args.trigger, version)
        trigger_path.touch()
        update: dict[str, Any] = {}
        resume: dict[str, Any] = {}
        ack_wait_seconds = 0.0
        update_seconds = 0.0
        update_start = time.perf_counter()
        try:
            update_body = {
                "packed": True,
                "packed_buffer_size_bytes": args.buffer_bytes,
                "packed_num_buffers": args.num_buffers,
                "weight_version": f"nccl-baseline-{version}",
            }
            if args.manifest_path_on_worker:
                update_body["manifest_path"] = args.manifest_path_on_worker
            else:
                update_body.update(
                    {
                        "names": manifest["names"],
                        "dtype_names": manifest["dtype_names"],
                        "shapes": manifest["shapes"],
                    }
                )
            update = _post(
                f"{args.system_url}/engine/update_weights_from_distributed",
                update_body,
                args.timeout,
            )
            update_seconds = time.perf_counter() - update_start
            ack_path = _version_path(ack_base, version)
            ack_wait_seconds = _wait_for_path(ack_path, args.timeout)
        finally:
            if update_seconds == 0.0:
                update_seconds = time.perf_counter() - update_start
            resume_start = time.perf_counter()
            resume = _post(f"{args.system_url}/engine/resume_generation", {}, 30)
            resume_seconds = time.perf_counter() - resume_start
        if update.get("status") not in ("ok", "success"):
            raise RuntimeError(f"NCCL update failed: {update}")
        if resume.get("status") not in ("ok", "success"):
            raise RuntimeError(f"resume failed: {resume}")

        cache = _cache_metadata(update)
        cache_seconds = _timing_seconds(
            update, "cache_reset_seconds", "cache_flush_seconds", "cache_reset"
        )
        receiver_stages = {
            name: _response_stage(update, name)
            for name in (
                "transfer_planning",
                "wire_transfer",
                "receive_sync",
                "transformation",
                "installation",
                "post_install",
            )
        }
        stages = {
            "control_discovery": _stage(
                "available", pause_seconds, source="pause_generation HTTP route"
            ),
            "source_preparation": _stage(
                "not_applicable", detail="one-time timing is in one_time_stages"
            ),
            "setup_registration": _stage(
                "not_applicable", detail="one-time timing is in one_time_stages"
            ),
            "transfer_planning": _stage(
                "unavailable",
                detail="receiver planning is internal to vLLM",
            ),
            "wire_transfer": _stage(
                "combined",
                update_seconds,
                source="update_weights_from_distributed HTTP route",
                combined_group="receiver_update",
                combined_with=update_group,
                detail="owner of the combined receiver update duration",
            ),
            "receive_sync": _stage(
                "combined",
                combined_group="receiver_update",
                combined_with=update_group,
            ),
            "transformation": _stage(
                "combined",
                combined_group="receiver_update",
                combined_with=update_group,
            ),
            "installation": _stage(
                "combined",
                combined_group="receiver_update",
                combined_with=update_group,
            ),
            "post_install": (
                _stage(
                    "combined",
                    source="Dynamo update response metadata",
                    combined_group="receiver_update",
                    combined_with=["wire_transfer", "installation"],
                    detail=(
                        "server reported cache behavior inside the combined update"
                        + (
                            f" ({cache_seconds:.6f}s)"
                            if cache_seconds is not None
                            else ""
                        )
                    ),
                )
                if cache["reported"]
                else _stage(
                    "unavailable",
                    detail="Dynamo response did not report cache behavior",
                )
            ),
            "rollout_readiness": _stage(
                "available",
                ack_wait_seconds + resume_seconds,
                source="sender completion ack plus resume_generation",
            ),
        }
        for name, server_stage in receiver_stages.items():
            if server_stage is not None and server_stage["status"] != "unavailable":
                stages[name] = server_stage
        timing = _result_schema(
            role="controller",
            stages=stages,
            total_seconds=time.perf_counter() - cycle_start,
            byte_count=manifest["bytes"],
            rate_seconds=(
                stages["wire_transfer"].get("seconds") or update_seconds
            ),
            metadata={
                "cycle": index - args.warmup_cycles,
                "version": version,
                "excluded": index < args.warmup_cycles,
                "trigger_path": str(trigger_path),
                "ack_path": str(ack_path),
                "workers": args.destination_tp_size,
                "packed": True,
                "source_layout": _source_layout_metadata(args),
                "cache": cache,
                "cache_seconds_from_response": cache_seconds,
                "controller_boundary_seconds": {
                    "pause": pause_seconds,
                    "update": update_seconds,
                    "sender_ack_wait": ack_wait_seconds,
                    "resume": resume_seconds,
                },
            },
        )
        timing.update(
            {
                "cycle": index - args.warmup_cycles,
                "version": version,
                "excluded": index < args.warmup_cycles,
                "pause_seconds": pause_seconds,
                "update_seconds": update_seconds,
                "resume_seconds": resume_seconds,
                "cache": cache,
                "response_timing": {
                    key: update[key]
                    for key in ("metadata", "timing", "timings", "refit_timing")
                    if key in update
                },
            }
        )
        raw_cycles.append(timing)
        label = "warmup" if index < args.warmup_cycles else "cycle"
        print(
            f"NCCL_CONTROLLER_{label.upper()} version={version} "
            f"update={update_seconds:.3f}s excluded={index < args.warmup_cycles}",
            flush=True,
        )

    measured = _measured_cycles(raw_cycles)
    last_timing = measured[-1] if measured else raw_cycles[-1]
    pause_stats = _stats([float(record["pause_seconds"]) for record in measured])
    update_stats = _stats([float(record["update_seconds"]) for record in measured])
    resume_stats = _stats([float(record["resume_seconds"]) for record in measured])
    result = {
        **last_timing,
        "role": "controller",
        "bytes_per_tp_rank": manifest["bytes"],
        "cycles": args.cycles,
        "warmup_cycles": args.warmup_cycles,
        "raw_cycles": raw_cycles,
        "aggregate": _aggregate_cycles(measured),
        "one_time_stages": one_time_stages,
        "init_seconds": init_seconds,
        "pause_seconds": pause_stats["median"],
        "update_seconds": update_stats["median"],
        "resume_seconds": resume_stats["median"],
        "effective_gbps_per_rank": (
            manifest["bytes"] * 8 / float(update_stats["median"]) / 1e9
            if update_stats["median"]
            else None
        ),
        "workers": args.destination_tp_size,
        "process_seconds": time.perf_counter() - total_start,
        "source_layout": _source_layout_metadata(args),
        "refit_timing": last_timing,
    }
    Path(args.result).write_text(json.dumps(result, indent=2))
    print("NCCL_CONTROLLER_RESULT", json.dumps(result), flush=True)
    return 0


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("sender", "controller"))
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, default=29600)
    parser.add_argument("--checkpoint")
    parser.add_argument("--system-url")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--manifest-path-on-worker",
        default="",
        help=(
            "shared filesystem path visible to the rollout; avoids sending large "
            "FP8 manifests through the Dynamo HTTP body"
        ),
    )
    parser.add_argument("--trigger")
    parser.add_argument("--ack")
    parser.add_argument("--result", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--buffer-bytes", type=int, default=1024**3)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--preload-gpu", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--source-layout",
        choices=SOURCE_LAYOUT_MODES,
        default="preconsolidated_transport_only",
        help=(
            "preconsolidated_transport_only: one sender already owns full HF "
            "weights; consolidated_e2e: emit an explicit unsupported result"
        ),
    )
    parser.add_argument(
        "--source-ep-size",
        type=int,
        default=1,
        help="declared original EP shard count; metadata only in preconsolidated mode",
    )
    parser.add_argument(
        "--destination-tp-size",
        type=int,
        default=2,
        help="number of rollout receiver ranks (default 2 preserves existing behavior)",
    )
    args = parser.parse_args(argv)
    if (
        args.source_layout == "preconsolidated_transport_only"
        and args.role == "sender"
        and not args.checkpoint
    ):
        parser.error("sender requires --checkpoint")
    if (
        args.source_layout == "preconsolidated_transport_only"
        and args.role == "controller"
        and not args.system_url
    ):
        parser.error("controller requires --system-url")
    if args.source_layout == "preconsolidated_transport_only" and not args.manifest:
        parser.error("preconsolidated_transport_only requires --manifest")
    if args.source_layout == "preconsolidated_transport_only" and not args.trigger:
        parser.error("preconsolidated_transport_only requires --trigger")
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles cannot be negative")
    if args.source_ep_size < 1:
        parser.error("--source-ep-size must be at least 1")
    if args.destination_tp_size < 1:
        parser.error("--destination-tp-size must be at least 1")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.source_layout == "consolidated_e2e":
        raise SystemExit(_write_unsupported_source_layout(parsed))
    raise SystemExit(sender(parsed) if parsed.role == "sender" else controller(parsed))
