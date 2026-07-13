"""One-GPU Hugging Face checkpoint publisher for MX refit benchmarks.

The publisher preloads one checkpoint's ``hf_weights`` onto a single GPU, then
publishes a distinct READY version for one excluded warmup plus N measured
cycles. It coordinates with ``mx_vs_nccl_refit_bench.py`` through versioned
trigger and acknowledgement files.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "refit-stage-v1"
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


def _version_path(base: str | Path, version: int) -> Path:
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


def _stage(
    status: str,
    seconds: float | None = None,
    *,
    source: str | None = None,
    detail: str | None = None,
    combined_with: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "seconds": seconds}
    if source:
        result["source"] = source
    if detail:
        result["detail"] = detail
    if combined_with:
        result["combined_with"] = combined_with
    return result


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


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for name in STAGE_NAMES:
        entries = [record["stages"][name] for record in records]
        values = [
            float(entry["seconds"])
            for entry in entries
            if entry.get("seconds") is not None
        ]
        statuses = sorted({str(entry["status"]) for entry in entries})
        stages[name] = {
            "status": statuses[0] if len(statuses) == 1 else "mixed",
            "statuses": statuses,
            **_stats(values),
        }
    return {
        "cycles": len(records),
        "total_seconds": _stats([float(record["total_seconds"]) for record in records]),
        "stages": stages,
    }


def publisher(args: argparse.Namespace) -> int:
    from modelexpress.nemo_rl_v2 import (
        MxV2TrainingPublisher,
        TrainerWorldLayout,
    )

    torch.cuda.set_device(args.device)
    process_start = time.perf_counter()
    checkpoint_start = time.perf_counter()
    weights = _load_checkpoint(args.checkpoint)
    checkpoint_seconds = time.perf_counter() - checkpoint_start

    preload_start = time.perf_counter()
    gpu_weights = {
        name: tensor.to(f"cuda:{args.device}", non_blocking=False).contiguous()
        for name, tensor in weights.items()
    }
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - preload_start
    byte_count = sum(tensor.numel() * tensor.element_size() for tensor in gpu_weights.values())
    dtype = (
        str(next(iter(gpu_weights.values())).dtype).removeprefix("torch.")
        if gpu_weights
        else "unknown"
    )

    setup_start = time.perf_counter()
    pub = MxV2TrainingPublisher(
        agent_name=args.agent_name,
        device_id=args.device,
        mx_server_url=args.mx_server_url,
        worker_rank=0,
        world_layout=TrainerWorldLayout(),
        heartbeat=False,
    )
    pub.initialize(model_name=args.model, dtype=dtype)
    for name, tensor in gpu_weights.items():
        pub.add_tensor(name=name, tensor=tensor)
    setup_seconds = time.perf_counter() - setup_start

    ack_base = args.ack or f"{args.trigger}.ready"
    raw_cycles: list[dict[str, Any]] = []
    try:
        for index in range(args.warmup_cycles + args.cycles):
            version = args.start_version + index
            cycle_start = time.perf_counter()
            trigger_path = _version_path(args.trigger, version)
            trigger_wait = _wait_for_path(trigger_path, args.timeout)

            publish_start = time.perf_counter()
            source_id = pub.publish(version=version)
            publish_seconds = time.perf_counter() - publish_start
            ready_start = time.perf_counter()
            if not pub.mark_ready():
                raise RuntimeError(f"MX server rejected READY for version {version}")
            ready_seconds = time.perf_counter() - ready_start
            ack_path = _version_path(ack_base, version)
            ack_path.touch()

            excluded = index < args.warmup_cycles
            stages = {
                "control_discovery": _stage(
                    "available", trigger_wait, source="receiver trigger wait"
                ),
                "source_preparation": _stage(
                    "not_applicable", detail="one-time timing is in one_time_stages"
                ),
                "setup_registration": _stage(
                    "not_applicable", detail="one-time timing is in one_time_stages"
                ),
                "transfer_planning": _stage(
                    "combined",
                    publish_seconds,
                    source="MxTrainingPublisher.publish_weights",
                    detail=(
                        "first call combines NIXL tensor registration and MX metadata "
                        "publication; later calls reuse registration"
                    ),
                    combined_with=["setup_registration", "transfer_planning"],
                ),
                "wire_transfer": _stage(
                    "unavailable", detail="wire time is measured by the receiver"
                ),
                "receive_sync": _stage("not_applicable"),
                "transformation": _stage("not_applicable"),
                "installation": _stage("not_applicable"),
                "post_install": _stage("not_applicable"),
                "rollout_readiness": _stage(
                    "available",
                    ready_seconds,
                    source="MxTrainingPublisher.mark_ready",
                ),
            }
            total_seconds = time.perf_counter() - cycle_start
            measured = sum(
                float(stage["seconds"])
                for stage in stages.values()
                if stage.get("seconds") is not None
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "backend": "mx",
                "role": "publisher",
                "cycle": index - args.warmup_cycles,
                "version": version,
                "excluded": excluded,
                "stages": stages,
                "total_seconds": total_seconds,
                "unattributed_seconds": max(0.0, total_seconds - measured),
                "bytes": byte_count,
                "gbps": None,
                "source_id": source_id,
                "trigger_path": str(trigger_path),
                "ack_path": str(ack_path),
            }
            raw_cycles.append(record)
            print("MX_HF_PUBLISHER_CYCLE", json.dumps(record), flush=True)

            if args.done:
                _wait_for_path(_version_path(args.done, version), args.timeout)

        if not args.done and args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
    finally:
        pub.shutdown()

    measured_records = [record for record in raw_cycles if not record["excluded"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": "mx",
        "role": "publisher",
        "cycles": args.cycles,
        "warmup_cycles": args.warmup_cycles,
        "bytes": byte_count,
        "raw_cycles": raw_cycles,
        "aggregate": _aggregate(measured_records),
        "one_time_stages": {
            "source_preparation": _stage(
                "combined",
                checkpoint_seconds + preload_seconds,
                source="checkpoint mmap load plus GPU preload",
                combined_with=["checkpoint_load", "gpu_preload"],
            ),
            "setup_registration": _stage(
                "available",
                setup_seconds,
                source="MxTrainingPublisher.initialize",
            ),
        },
        "checkpoint_load_seconds": checkpoint_seconds,
        "gpu_preload_seconds": preload_seconds,
        "setup_seconds": setup_seconds,
        "process_seconds": time.perf_counter() - process_start,
    }
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MX_HF_PUBLISHER_RESULT", json.dumps(result), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mx-server-url",
        default=os.environ.get("MODEL_EXPRESS_URL", "localhost:8001"),
    )
    parser.add_argument("--agent-name", default="nemo-rl-bench-hf-publisher")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--ack")
    parser.add_argument("--done")
    parser.add_argument("--result", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=300.0,
        help="keep the final source alive when --done is not supplied",
    )
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles cannot be negative")
    return args


if __name__ == "__main__":
    raise SystemExit(publisher(parse_args()))
