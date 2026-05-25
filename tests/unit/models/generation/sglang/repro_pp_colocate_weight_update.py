#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone colocate/CUDA-IPC repro for SGLang weight update.

This is the colocate companion to ``repro_pp_weight_update.py``. It avoids
Megatron and the nemo-rl policy worker, but it keeps the production colocate
transport:

1. A Ray actor hosts an ``sglang.srt.entrypoints.engine.Engine``.
2. The main process is a mock trainer on the same visible GPU.
3. The trainer loads the sliced Qwen checkpoint, builds HF weight buckets, and
   sends them through ``send_hf_buckets_via_ipc_actor_impl``.
4. SGLang snapshots its initial weights, randomizes them, receives the IPC
   refit, post-processes weights, and compares back to the snapshot.

Usage inside the nemo-rl container:

    CUDA_VISIBLE_DEVICES=0 python \
      tests/unit/models/generation/sglang/repro_pp_colocate_weight_update.py \
      --pp 1 --tp 1 --dp 1 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from safetensors import safe_open

_DTYPE_FROM_STR = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

_SAFETENSORS_TO_TORCH_NAME = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def _ckpt_path() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    path = Path(hf_home) / "hub" / "qwen3-30b-a3b-sliced-4"
    if not path.is_dir():
        sys.exit(
            f"sliced ckpt not found at {path}; produce it first via "
            "tests/unit/models/generation/sglang/_qwen3_slicer.py"
        )
    return path


def _collect_specs(ckpt: Path) -> tuple[list[tuple[str, str, list[int]]], dict[str, str]]:
    """Return (ordered list of (name, dtype_str, shape), name -> shard file)."""
    index_path = ckpt / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        single = next(ckpt.glob("*.safetensors")).name
        with safe_open(ckpt / single, framework="pt") as reader:
            weight_map = {name: single for name in reader.keys()}

    per_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, []).append(name)

    specs: list[tuple[str, str, list[int]]] = []
    for shard in sorted(per_shard):
        with safe_open(ckpt / shard, framework="pt") as reader:
            for name in sorted(per_shard[shard]):
                tensor_slice = reader.get_slice(name)
                st_dtype = tensor_slice.get_dtype()
                if st_dtype not in _SAFETENSORS_TO_TORCH_NAME:
                    raise RuntimeError(f"unmapped safetensors dtype {st_dtype!r} for {name}")
                specs.append(
                    (
                        name,
                        _SAFETENSORS_TO_TORCH_NAME[st_dtype],
                        list(tensor_slice.get_shape()),
                    )
                )

    specs.sort(key=lambda x: x[0])
    return specs, weight_map


def _result_failed(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0] is False, str(result[1])
    if isinstance(result, Mapping):
        success = result.get("success", True)
        message = (
            result.get("error_message")
            or result.get("error")
            or result.get("message")
            or "unknown error"
        )
        return success is False, str(message)
    if hasattr(result, "success"):
        message = getattr(result, "error_message", None) or getattr(
            result, "message", "unknown error"
        )
        return result.success is False, str(message)
    return False, ""


def _raise_if_failed(result: Any, action: str) -> Any:
    failed, message = _result_failed(result)
    if failed:
        raise RuntimeError(f"{action} failed: {message}")
    return result


def _text(output: Any) -> str:
    if isinstance(output, list):
        output = output[0]
    if isinstance(output, Mapping):
        return str(output.get("text", repr(output)))
    return repr(output)


@ray.remote(num_gpus=1)
class _SGLangEngineActor:
    """Ray actor facade matching SGLangGenerationWorker's weight-update API."""

    def __init__(
        self,
        *,
        ckpt: str,
        pp: int,
        tp: int,
        dp: int,
        dtype: str,
        mem_fraction_static: float,
        moe_runner_backend: str | None,
    ):
        os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"

        try:
            from nemo_rl.models.generation.sglang.sglang_worker import (
                _apply_sglang_compat_patches,
            )

            _apply_sglang_compat_patches()
        except Exception as exc:  # noqa: BLE001
            print(f"[actor] WARN: failed to apply nemo-rl SGLang patches: {exc}", flush=True)

        from sglang.srt.entrypoints.engine import Engine

        kwargs: dict[str, Any] = {
            "model_path": ckpt,
            "tp_size": tp,
            "pp_size": pp,
            "dp_size": dp,
            "dtype": dtype,
            "mem_fraction_static": mem_fraction_static,
            "log_level": "info",
            "random_seed": 42,
            "disable_cuda_graph": True,
            "enable_memory_saver": False,
        }
        if moe_runner_backend is not None:
            kwargs["moe_runner_backend"] = moe_runner_backend

        print(
            f"[actor] starting Engine pp={pp} tp={tp} dp={dp} dtype={dtype} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )
        self.engine = Engine(**kwargs)
        print("[actor] engine up", flush=True)

    def generate(self, prompt: str, sampling_params: dict[str, Any]) -> Any:
        return self.engine.generate(prompt, sampling_params=sampling_params)

    def check_weights(self, action: str) -> Any:
        from sglang.srt.managers.io_struct import CheckWeightsReqInput

        result = self.engine.loop.run_until_complete(
            self.engine.tokenizer_manager.check_weights(
                CheckWeightsReqInput(action=action)
            )
        )
        return _raise_if_failed(result, f"check_weights({action!r})")

    def post_process_weights(
        self,
        *,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = True,
    ) -> Any:
        from sglang.srt.managers.io_struct import PostProcessWeightsReqInput

        result = self.engine.loop.run_until_complete(
            self.engine.tokenizer_manager.post_process_weights(
                PostProcessWeightsReqInput(
                    restore_weights_before_load=restore_weights_before_load,
                    post_process_quantization=post_process_quantization,
                )
            )
        )
        return _raise_if_failed(result, "post_process_weights")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ) -> Any:
        del weight_version
        result = self.engine.update_weights_from_tensor(
            named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return _raise_if_failed(result, "update_weights_from_tensor")

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


class _MockTrainer:
    """Load HF checkpoint tensors on one GPU and yield byte-bounded buckets."""

    def __init__(
        self,
        *,
        ckpt: Path,
        device: str,
        target_dtype: torch.dtype,
        specs: list[tuple[str, str, list[int]]],
        weight_map: dict[str, str],
    ):
        self._tensors: list[tuple[str, torch.Tensor]] = []
        for idx, (name, dtype_str, _shape) in enumerate(specs):
            with safe_open(ckpt / weight_map[name], framework="pt") as reader:
                tensor = reader.get_tensor(name)
            dtype = target_dtype if tensor.is_floating_point() else _DTYPE_FROM_STR[dtype_str]
            self._tensors.append((name, tensor.to(device=device, dtype=dtype).contiguous()))
            if idx % 25 == 0 or idx == len(specs) - 1:
                print(f"[trainer] loaded {idx + 1}/{len(specs)} {name}", flush=True)

    def iter_buckets(self, buffer_size_bytes: int) -> Iterable[list[tuple[str, torch.Tensor]]]:
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_size = 0
        for name, tensor in self._tensors:
            tensor_size = tensor.numel() * tensor.element_size()
            if bucket and bucket_size + tensor_size > buffer_size_bytes:
                yield bucket
                bucket = []
                bucket_size = 0
            bucket.append((name, tensor))
            bucket_size += tensor_size
        if bucket:
            yield bucket


def _init_single_rank_gloo(master_port: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    return dist.new_group(ranks=[0], backend="gloo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--buffer-size-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--master-port", type=int, default=29565)
    parser.add_argument("--mem-fraction-static", type=float, default=0.3)
    parser.add_argument(
        "--moe-runner-backend",
        type=str,
        default=None,
        help="Optional pass-through to SGLang Engine, e.g. flashinfer_trtllm_routed.",
    )
    args = parser.parse_args()

    n_engine = args.pp * args.tp * args.dp
    n_avail = torch.cuda.device_count()
    if n_avail < n_engine:
        sys.exit(f"need at least {n_engine} visible GPU(s), have {n_avail}")
    if args.tp != 1:
        sys.exit("this standalone single-trainer colocate repro supports --tp 1 only")

    torch.cuda.set_device(0)
    ckpt = _ckpt_path()
    specs, weight_map = _collect_specs(ckpt)
    target_dtype = _DTYPE_FROM_STR[args.dtype]

    print(f"[main] ckpt={ckpt}", flush=True)
    print(
        f"[main] pp={args.pp} tp={args.tp} dp={args.dp} dtype={args.dtype} "
        f"n_engine={n_engine}",
        flush=True,
    )
    print(f"[main] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"[main] {len(specs)} weight tensors to round-trip", flush=True)

    gather_group = None
    engine = None
    try:
        gather_group = _init_single_rank_gloo(args.master_port)

        from nemo_rl.models.policy.torch_reductions_utils import (
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()

        ray.init(
            num_gpus=n_avail,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=True,
        )
        engine = _SGLangEngineActor.options(num_gpus=n_engine).remote(
            ckpt=str(ckpt),
            pp=args.pp,
            tp=args.tp,
            dp=args.dp,
            dtype=args.dtype,
            mem_fraction_static=args.mem_fraction_static,
            moe_runner_backend=args.moe_runner_backend,
        )

        prompt = "The capital of France is"
        sampling = {"max_new_tokens": 16, "temperature": 0.0}

        out_pre = ray.get(engine.generate.remote(prompt, sampling))
        print(f"[main] pre-update text={_text(out_pre)!r}", flush=True)

        print("[main] snapshot...", flush=True)
        ray.get(engine.check_weights.remote("snapshot"))

        print("[main] reset_tensors (randomize sglang weights)...", flush=True)
        ray.get(engine.check_weights.remote("reset_tensors"))
        out_random = ray.get(engine.generate.remote(prompt, sampling))
        print(f"[main] post-reset text={_text(out_random)!r}", flush=True)
        if _text(out_random) == _text(out_pre):
            print(
                "[main] WARN: post-reset output matches pre; proceeding with weight compare",
                flush=True,
            )

        print("[main] loading trainer tensors...", flush=True)
        trainer = _MockTrainer(
            ckpt=ckpt,
            device="cuda:0",
            target_dtype=target_dtype,
            specs=specs,
            weight_map=weight_map,
        )

        worker_state: dict[str, Any] = {
            "_ipc_gather_group": gather_group,
            "_ipc_gather_src": 0,
            "_ipc_engine_index": 0,
            "_ipc_layout_key": ((1,), (0,)),
            "_ipc_monkey_patched": True,
            "weight_version": 0,
        }

        from nemo_rl.models.policy.utils import send_hf_buckets_via_ipc_actor_impl

        print("[main] send_hf_buckets_via_ipc_actor_impl...", flush=True)
        t0 = time.time()
        send_hf_buckets_via_ipc_actor_impl(
            bucket_iterator=trainer.iter_buckets(args.buffer_size_bytes),
            rollout_engines=[engine],
            worker_state=worker_state,
            weight_version=1,
        )
        print(f"[main] send_hf_buckets done in {time.time() - t0:.1f}s", flush=True)

        print("[main] post_process_weights...", flush=True)
        ray.get(
            engine.post_process_weights.remote(
                restore_weights_before_load=False,
                post_process_quantization=True,
            )
        )

        print("[main] compare against snapshot...", flush=True)
        ray.get(engine.check_weights.remote("compare"))
        print("[main] compare passed.", flush=True)

        out_post = ray.get(engine.generate.remote(prompt, sampling))
        print(f"[main] post-update text={_text(out_post)!r}", flush=True)

        if _text(out_pre) != _text(out_post):
            print("[main] FAIL: pre/post generation outputs differ", flush=True)
            print(f"  pre : {_text(out_pre)!r}", flush=True)
            print(f"  post: {_text(out_post)!r}", flush=True)
            sys.exit(1)

        print("[main] PASS: snapshot+reset+refit+compare colocate roundtrip", flush=True)
    finally:
        if engine is not None:
            try:
                ray.get(engine.shutdown.remote(), timeout=30)
            except Exception as exc:  # noqa: BLE001
                print(f"[main] WARN: actor shutdown failed: {exc}", flush=True)
            try:
                ray.kill(engine, no_restart=True)
            except Exception:
                pass
        if dist.is_initialized():
            if gather_group is not None:
                dist.destroy_process_group(gather_group)
            dist.destroy_process_group()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
