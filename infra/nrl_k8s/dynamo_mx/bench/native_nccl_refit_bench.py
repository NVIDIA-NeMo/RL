"""Native-vLLM NCCL baseline for a TP2 rollout.

This deliberately uses vLLM's PyNccl implementation on both sides, avoiding the
metadata/communicator mismatch between NeMo's ``nccl.core`` package and vLLM.

Workflow:

1. Start ``sender`` in a one-GPU pod using the same vLLM image as the rollout.
2. Run ``controller`` where it can reach the rollout pod's DYN_SYSTEM_PORT.
3. Controller initializes a 3-rank group (sender rank 0, TP ranks 1–2), then
   starts packed send/receive concurrently and reports update wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import torch


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


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

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    weights = payload.get("hf_weights", payload)
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
            "world_size": 3,
        }
    )
    init_seconds = time.perf_counter() - init_start
    print(f"NCCL_SENDER_READY init={init_seconds:.3f}s bytes={manifest['bytes']}", flush=True)

    trigger = Path(args.trigger)
    while not trigger.exists():
        time.sleep(0.05)

    def gpu_weights():
        if gpu_ordered is not None:
            yield from gpu_ordered
        else:
            for name, tensor in ordered:
                yield name, tensor.to("cuda", non_blocking=False)

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
    result = {
        "role": "sender",
        "bytes": manifest["bytes"],
        "init_seconds": init_seconds,
        "preload_seconds": preload_seconds,
        "send_seconds": send_seconds,
        "effective_gbps": manifest["bytes"] * 8 / send_seconds / 1e9,
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

    init_start = time.perf_counter()
    init = _post(
        f"{args.system_url}/engine/init_weights_update_group",
        {
            "master_address": args.master_address,
            "master_port": args.master_port,
            "rank_offset": 1,
            "world_size": 3,
        },
        args.timeout,
    )
    init_seconds = time.perf_counter() - init_start
    if init.get("status") not in ("ok", "success"):
        raise RuntimeError(f"NCCL init failed: {init}")

    paused = _post(f"{args.system_url}/engine/pause_generation", {}, 30)
    if paused.get("status") not in ("ok", "success"):
        raise RuntimeError(f"pause failed: {paused}")

    Path(args.trigger).touch()
    update_start = time.perf_counter()
    try:
        update = _post(
            f"{args.system_url}/engine/update_weights_from_distributed",
            {
                "names": manifest["names"],
                "dtype_names": manifest["dtype_names"],
                "shapes": manifest["shapes"],
                "packed": True,
                "packed_buffer_size_bytes": args.buffer_bytes,
                "packed_num_buffers": args.num_buffers,
                "weight_version": "nccl-baseline",
            },
            args.timeout,
        )
    finally:
        _post(f"{args.system_url}/engine/resume_generation", {}, 30)
    update_seconds = time.perf_counter() - update_start
    if update.get("status") not in ("ok", "success"):
        raise RuntimeError(f"NCCL update failed: {update}")

    result = {
        "role": "controller",
        "bytes_per_tp_rank": manifest["bytes"],
        "init_seconds": init_seconds,
        "update_seconds": update_seconds,
        "effective_gbps_per_rank": manifest["bytes"] * 8 / update_seconds / 1e9,
        "workers": 2,
    }
    Path(args.result).write_text(json.dumps(result, indent=2))
    print("NCCL_CONTROLLER_RESULT", json.dumps(result), flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("sender", "controller"))
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, default=29600)
    parser.add_argument("--checkpoint")
    parser.add_argument("--system-url")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--buffer-bytes", type=int, default=1024**3)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--preload-gpu", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.role == "sender" and not args.checkpoint:
        parser.error("sender requires --checkpoint")
    if args.role == "controller" and not args.system_url:
        parser.error("controller requires --system-url")
    return args


if __name__ == "__main__":
    raise SystemExit(sender(parse_args()) if os.sys.argv[1] == "sender" else controller(parse_args()))
