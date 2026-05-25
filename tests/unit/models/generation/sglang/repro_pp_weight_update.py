#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone repro for the sglang weight-update failure with ``pp_size > 1``.

Hypothesis under test
---------------------
``model_runner.init_weights_update_group`` (sglang) computes
``rank = rank_offset + self.tp_rank``, ignoring ``self.pp_rank`` /
``self.dp_rank``.  When sglang runs with ``pp_size > 1``, multiple engine
workers pick the same NCCL rank and the rendezvous either deadlocks or
the subsequent broadcast lands on the wrong device.

Test plan
---------
Stage 1 (sanity)  : ``--pp 1 --tp 1 --dp 1`` — 1 trainer + 1 engine rank.
                    Pre/post ``generate`` must produce identical text.
Stage 2 (bug)     : ``--pp 2 --tp 1 --dp 1`` — 1 trainer + 2 engine ranks.
                    Expected to hang at ``init_weights_update_group`` or
                    fail mid-broadcast with CUDA invalid-argument.

Usage (inside the sglang-nemorl-e2e-zhw container)::

    python tests/unit/models/generation/sglang/repro_pp_weight_update.py --pp 1
    python tests/unit/models/generation/sglang/repro_pp_weight_update.py --pp 2

We deliberately do NOT use Megatron / nemo-rl plumbing — only the bare
sglang Engine API and ``torch.distributed`` from a hand-rolled trainer
process.  Weights come from the cached sliced Qwen3-30B-A3B checkpoint
(produced by ``_qwen3_slicer.ensure_sliced_model``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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

# safetensors uses its own dtype tag strings; map to torch dtype names.
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
    p = Path(hf_home) / "hub" / "qwen3-30b-a3b-sliced-4"
    if not p.is_dir():
        sys.exit(
            f"sliced ckpt not found at {p}; produce it first via "
            "tests/unit/models/generation/sglang/_qwen3_slicer.py"
        )
    return p


def _collect_specs(ckpt: Path) -> tuple[list[tuple[str, str, list[int]]], dict[str, str]]:
    """Return (ordered list of (name, dtype_str, shape), name -> shard filename).

    Order is deterministic (sorted by name) so the trainer side and the
    engine side can iterate in lockstep without any explicit handshake.
    """
    index_path = ckpt / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        single = next(ckpt.glob("*.safetensors")).name
        with safe_open(ckpt / single, framework="pt") as r:
            weight_map = {k: single for k in r.keys()}

    # Read header (shape/dtype) for each tensor without materialising the
    # storage. ``safe_open`` exposes ``get_slice`` which is metadata-only.
    per_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, []).append(name)

    specs: list[tuple[str, str, list[int]]] = []
    for shard in sorted(per_shard.keys()):
        with safe_open(ckpt / shard, framework="pt") as r:
            for name in sorted(per_shard[shard]):
                sl = r.get_slice(name)
                st_dtype = sl.get_dtype()  # safetensors tag, e.g. "BF16"
                if st_dtype not in _SAFETENSORS_TO_TORCH_NAME:
                    raise RuntimeError(f"unmapped safetensors dtype {st_dtype!r} for {name}")
                dtype_str = _SAFETENSORS_TO_TORCH_NAME[st_dtype]
                shape = list(sl.get_shape())
                specs.append((name, dtype_str, shape))
    specs.sort(key=lambda x: x[0])
    return specs, weight_map


def _trainer_proc(
    *,
    trainer_gpu: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ckpt_str: str,
    specs: list,
    weight_map: dict,
    group_name: str,
    rank: int = 0,
):
    """Run as a separate process; broadcasts each tensor to the engine."""
    try:
        # Trainer is launched by mp.spawn with CUDA_VISIBLE_DEVICES already
        # restricted by the parent — ``trainer_gpu`` is a *visible* index.
        torch.cuda.set_device(trainer_gpu)

        from nemo_rl.models.policy.utils import init_process_group

        print(
            f"[trainer rank={rank}] init_process_group "
            f"world={world_size} master={master_addr}:{master_port} group={group_name}",
            flush=True,
        )
        pg = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(f"[trainer rank={rank}] joined group; about to broadcast {len(specs)} tensors", flush=True)

        ckpt = Path(ckpt_str)
        device = f"cuda:{trainer_gpu}"
        for i, (name, dtype_str, shape) in enumerate(specs):
            shard = ckpt / weight_map[name]
            with safe_open(shard, framework="pt") as r:
                t = r.get_tensor(name)
            t = t.to(device=device, dtype=_DTYPE_FROM_STR[dtype_str]).contiguous()
            dist.broadcast(t, src=0, group=pg)
            del t
            if i % 25 == 0 or i == len(specs) - 1:
                print(
                    f"[trainer rank={rank}] broadcast {i+1}/{len(specs)} {name} shape={shape}",
                    flush=True,
                )
        print(f"[trainer rank={rank}] all broadcasts complete", flush=True)
        dist.destroy_process_group(pg)
    except Exception:  # noqa: BLE001
        # Crash visibly — parent watches exit code.
        traceback.print_exc()
        os._exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29555)
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="If > 0, only round-trip the first N tensors (faster smoke test).",
    )
    parser.add_argument(
        "--moe-runner-backend",
        type=str,
        default=None,
        help="Pass through to sglang Engine (e.g. flashinfer_trtllm_routed, triton, auto).",
    )
    args = parser.parse_args()

    n_engine = args.pp * args.tp * args.dp
    n_trainer = 1
    world_size = n_engine + n_trainer
    rank_offset = n_trainer
    group_name = "weight_update_group"

    n_avail = torch.cuda.device_count()
    if n_avail < n_engine + 1:
        sys.exit(f"need at least {n_engine + 1} GPUs visible, have {n_avail}")

    ckpt = _ckpt_path()
    print(f"[main] ckpt={ckpt}", flush=True)
    print(
        f"[main] pp={args.pp} tp={args.tp} dp={args.dp} "
        f"n_engine={n_engine} n_trainer={n_trainer} world={world_size}",
        flush=True,
    )

    specs, weight_map = _collect_specs(ckpt)
    if args.max_tensors > 0:
        specs = specs[: args.max_tensors]
    print(f"[main] {len(specs)} weight tensors to round-trip", flush=True)

    # GPU layout: caller controls absolute device ids via CUDA_VISIBLE_DEVICES.
    # Within that view, engine takes visible [0, n_engine) and trainer takes n_engine.
    print(f"[main] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)

    from sglang.srt.entrypoints.engine import Engine

    engine_kwargs = {}
    if args.moe_runner_backend is not None:
        engine_kwargs["moe_runner_backend"] = args.moe_runner_backend

    engine = Engine(
        model_path=str(ckpt),
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        mem_fraction_static=0.5,
        log_level="info",
        random_seed=42,
        disable_cuda_graph=True,
        **engine_kwargs,
    )
    print("[main] engine up", flush=True)

    # Pre-update generation (greedy so we can compare deterministically).
    sampling = {"max_new_tokens": 16, "temperature": 0.0}
    prompt = "The capital of France is"

    def _text(x):
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, dict):
            return x.get("text", repr(x))
        return repr(x)

    out_pre = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] pre-update text={_text(out_pre)!r}", flush=True)

    # Snapshot the freshly-loaded weights, then randomize them so the
    # subsequent refit MUST overwrite them. Without this step the
    # post-update equality check trivially passes regardless of whether
    # the broadcast loop actually delivered anything.
    from sglang.srt.managers.io_struct import (
        CheckWeightsReqInput,
        PostProcessWeightsReqInput,
    )

    def _check_weights(action: str):
        res = engine.loop.run_until_complete(
            engine.tokenizer_manager.check_weights(CheckWeightsReqInput(action=action))
        )
        # tokenizer_manager.check_weights returns _Communicator.merge_results,
        # which is a ``(all_success: bool, joined_message: str)`` tuple — NOT
        # a CheckWeightsReqOutput. The HTTP layer would convert
        # ``success=False`` to a 400; this in-process path does not, so we
        # have to raise ourselves to avoid a failed compare masquerading as
        # success.
        success, message = res
        if not success:
            raise RuntimeError(f"check_weights({action!r}) failed: {message}")
        return res

    def _post_process_weights():
        """Re-run sglang's ``process_weights_after_loading`` hook on every module
        after the broadcast/load loop. Without this, flashinfer trtllm MoE
        layers keep canonical-shape weights from the broadcast and never get
        re-packed into the block layout that the kernel expects — so post-
        update inference crashes / produces garbage. nemo-rl's production
        refit dispatch path always calls this (see
        ``megatron_policy_worker.py:1870/1958``); the bare sglang
        ``update_weights_from_distributed`` API does NOT do it implicitly.
        """
        res = engine.loop.run_until_complete(
            engine.tokenizer_manager.post_process_weights(
                PostProcessWeightsReqInput(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                )
            )
        )
        success, message = res
        if not success:
            raise RuntimeError(f"post_process_weights failed: {message}")
        return res

    print("[main] snapshot...", flush=True)
    _check_weights("snapshot")
    print("[main] reset_tensors (randomize sglang weights)...", flush=True)
    _check_weights("reset_tensors")
    out_random = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] post-reset text={_text(out_random)!r}", flush=True)
    if _text(out_random) == _text(out_pre):
        print(
            "[main] WARN: post-reset output matches pre — reset_tensors may "
            "have been a no-op for this prompt; proceeding anyway",
            flush=True,
        )

    # Spawn the trainer FIRST so it is waiting on the rendezvous when the
    # engine workers join.  Trainer is on the GPU just past the engine block.
    trainer_gpu = n_engine
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_trainer_proc,
        kwargs=dict(
            trainer_gpu=trainer_gpu,
            world_size=world_size,
            master_addr="127.0.0.1",
            master_port=args.master_port,
            ckpt_str=str(ckpt),
            specs=specs,
            weight_map=weight_map,
            group_name=group_name,
            rank=0,
        ),
    )
    p.start()

    print(
        f"[main] engine.init_weights_update_group(rank_offset={rank_offset}, world={world_size}) ...",
        flush=True,
    )
    t0 = time.time()
    success, msg = engine.init_weights_update_group(
        master_address="127.0.0.1",
        master_port=args.master_port,
        rank_offset=rank_offset,
        world_size=world_size,
        group_name=group_name,
        backend="nccl",
    )
    print(
        f"[main] init_weights_update_group => success={success} msg={msg} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    if not success:
        if p.is_alive():
            p.terminate()
        sys.exit(f"init_weights_update_group failed: {msg}")

    print("[main] starting weight update loop", flush=True)
    t0 = time.time()
    for i, (name, dtype_str, shape) in enumerate(specs):
        engine.update_weights_from_distributed(
            names=[name],
            dtypes=[dtype_str],
            shapes=[shape],
            group_name=group_name,
            flush_cache=False,
        )
        if i % 25 == 0 or i == len(specs) - 1:
            print(
                f"[main] update {i+1}/{len(specs)} {name}  (elapsed {time.time()-t0:.1f}s)",
                flush=True,
            )
    print(f"[main] weight updates complete in {time.time()-t0:.1f}s", flush=True)

    # Tear down the engine side of the weight-update NCCL group so the
    # trainer's dist.destroy_process_group(pg) can pair with it. Without
    # this the trainer's destroy hangs (engine never matches it) and the
    # script aborts before reaching the snapshot compare below.
    print("[main] destroy_weights_update_group...", flush=True)
    engine.destroy_weights_update_group(group_name=group_name)

    # Re-run ``process_weights_after_loading`` on every module so the
    # flashinfer trtllm MoE layers re-pack the just-broadcast canonical
    # weights into the 4-D block layout the kernel expects. nemo-rl's
    # production refit dispatch does this via
    # ``policy_generation.post_process_weights()``; bare sglang's
    # ``update_weights_from_distributed`` does NOT do it implicitly.
    print("[main] post_process_weights...", flush=True)
    _post_process_weights()
    print("[main] post_process_weights done.", flush=True)

    p.join(timeout=120)
    if p.is_alive():
        print("[main] trainer still alive after timeout; terminating", flush=True)
        p.terminate()
        p.join()
    if p.exitcode not in (0, None):
        sys.exit(f"trainer exited with code {p.exitcode}")

    # Compare current weights against the snapshot — strongest correctness
    # signal: raises if any tensor differs from the freshly-loaded original.
    print("[main] compare against snapshot...", flush=True)
    _check_weights("compare")
    print("[main] compare passed.", flush=True)

    out_post = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] post-update text={_text(out_post)!r}", flush=True)

    pre_text = _text(out_pre)
    post_text = _text(out_post)
    if pre_text == post_text:
        print("[main] PASS: snapshot+reset+broadcast+compare roundtrip", flush=True)
    else:
        print("[main] FAIL: pre/post generation outputs differ", flush=True)
        print(f"  pre : {pre_text!r}")
        print(f"  post: {post_text!r}")
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
