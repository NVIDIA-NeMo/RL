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

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nemo_rl.utils.checkpoint_engines.nixl import NIXLCheckpointEngine

_GIB = 1024**3


@dataclass(frozen=True)
class Variant:
    name: str
    transfer_mode: str
    direct_stripe_count: int
    metadata_batch_size: int
    buffer_count: int


def _json(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def _parse_variants(value: str) -> list[Variant]:
    variants = []
    for item in value.split(","):
        fields = item.split(":")
        if len(fields) not in (4, 5):
            raise ValueError(
                "variants must be name:transfer_mode:stripe:metadata_batch[:buffers]"
            )
        name, transfer_mode, stripe, metadata_batch, *rest = fields
        variants.append(
            Variant(
                name=name,
                transfer_mode=transfer_mode,
                direct_stripe_count=int(stripe),
                metadata_batch_size=int(metadata_batch),
                buffer_count=int(rest[0]) if rest else 3,
            )
        )
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def _parse_key_value(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        parsed[key] = value
    return parsed


def _rank() -> int:
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))


def _world_size() -> int:
    return int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))


def _wait_for_file(path: Path, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pickle.dump(value, handle)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _barrier(path: Path, rank: int, world_size: int, timeout_s: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{rank}.ready").write_text("ready")
    for peer_rank in range(world_size):
        _wait_for_file(path / f"{peer_rank}.ready", timeout_s)


def _make_engine(args: argparse.Namespace, variant: Variant, device: str):
    return NIXLCheckpointEngine(
        bucket_size=args.bucket_mb * 1024 * 1024,
        device=device,
        backend_name=args.backend_name,
        backend_init_params=args.backend_init_params,
        cleanup_after_load=True,
        topology="paired",
        transfer_mode=variant.transfer_mode,
        buffer_count=variant.buffer_count,
        direct_stripe_count=variant.direct_stripe_count,
        metadata_batch_size=variant.metadata_batch_size,
    )


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


async def _run_sender(args: argparse.Namespace, variant: Variant, device: str) -> dict:
    engine = _make_engine(args, variant, device)
    metadata_dir = args.rendezvous_dir / variant.name
    metadata = engine.prepare()
    _write_pickle(metadata_dir / "metadata_0.pkl", metadata)
    _wait_for_file(metadata_dir / "metadata_1.pkl", args.timeout_s)
    peer_metadata = [
        _read_pickle(metadata_dir / "metadata_0.pkl"),
        _read_pickle(metadata_dir / "metadata_1.pkl"),
    ]
    engine.init_policy_process_group(
        worker_rank=0,
        train_world_size=1,
        rollout_world_size=1,
        metadata=peer_metadata,
    )
    _barrier(metadata_dir / "start", 0, 2, args.timeout_s)
    tensor = torch.empty(args.size_bytes, dtype=torch.uint8, device=device)
    tensor[0] = 17
    tensor[-1] = 91
    _sync(device)
    start = time.perf_counter()
    await engine.send_weights(iter([("bench.weight", tensor)]))
    _sync(device)
    elapsed = time.perf_counter() - start
    engine.close()
    return {
        "role": "sender",
        "elapsed_s": elapsed,
        "logical_gib_s": args.size_bytes / elapsed / _GIB,
    }


async def _run_receiver(
    args: argparse.Namespace, variant: Variant, device: str
) -> dict:
    engine = _make_engine(args, variant, device)
    metadata_dir = args.rendezvous_dir / variant.name
    metadata = engine.prepare()
    _write_pickle(metadata_dir / "metadata_1.pkl", metadata)
    _wait_for_file(metadata_dir / "metadata_0.pkl", args.timeout_s)
    peer_metadata = [
        _read_pickle(metadata_dir / "metadata_0.pkl"),
        _read_pickle(metadata_dir / "metadata_1.pkl"),
    ]
    engine.init_rollout_process_group(
        rollout_rank=0,
        train_world_size=1,
        rollout_world_size=1,
        metadata=peer_metadata,
    )
    _barrier(metadata_dir / "start", 1, 2, args.timeout_s)
    received_bytes = 0
    first_value = None
    last_value = None
    _sync(device)
    start = time.perf_counter()
    async for batch in engine.receive_weight_batches():
        for _name, weight in batch:
            flat = weight.reshape(-1)
            if first_value is None and flat.numel() > 0:
                first_value = int(flat[0].item())
            if flat.numel() > 0:
                last_value = int(flat[-1].item())
            received_bytes += weight.nbytes
        if hasattr(batch, "record_cuda_load_complete"):
            batch.record_cuda_load_complete()
        if received_bytes >= args.size_bytes:
            break
    _sync(device)
    elapsed = time.perf_counter() - start
    engine.close()
    return {
        "role": "receiver",
        "elapsed_s": elapsed,
        "logical_gib_s": received_bytes / elapsed / _GIB,
        "received_bytes": received_bytes,
        "first_value": first_value,
        "last_value": last_value,
    }


async def _run(args: argparse.Namespace) -> None:
    rank = _rank()
    world_size = _world_size()
    if world_size != 2:
        raise RuntimeError(f"standalone benchmark requires 2 tasks, got {world_size}")
    if args.device == "cuda":
        torch.cuda.set_device(0)
    results = []
    for variant in args.variants:
        _json(
            "STANDALONE_REFIT_VARIANT_START",
            rank=rank,
            variant=variant.__dict__,
            device=args.device,
            size_bytes=args.size_bytes,
        )
        if rank == 0:
            result = await _run_sender(args, variant, args.device)
        else:
            result = await _run_receiver(args, variant, args.device)
        result.update({"rank": rank, "variant": variant.name, "device": args.device})
        results.append(result)
        _json("STANDALONE_REFIT_RUN", **result)
        _barrier(args.rendezvous_dir / variant.name / "done", rank, 2, args.timeout_s)
    if rank == 0:
        _json(
            "STANDALONE_REFIT_COMPLETED",
            rank=rank,
            variants=[variant.name for variant in args.variants],
            median_sender_elapsed_s=statistics.median(
                result["elapsed_s"] for result in results
            ),
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendezvous-dir", type=Path, required=True)
    parser.add_argument("--backend-name", default="UCX")
    parser.add_argument("--backend-init-param", action="append", default=[])
    parser.add_argument("--bucket-mb", type=int, default=1024)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument(
        "--variants",
        default="direct:direct:1:1:3,direct_s4_b4:direct:4:4:5",
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    args.variants = _parse_variants(args.variants)
    args.backend_init_params = _parse_key_value(args.backend_init_param)
    args.size_bytes = int(args.size_gib * _GIB)
    return args


if __name__ == "__main__":
    asyncio.run(_run(_args()))
