#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone repro for the sglang colocate (CUDA-IPC) weight-update path.

Companion to ``repro_pp_weight_update.py`` (which exercises the disag /
NCCL-broadcast path). This script exercises ``send_hf_buckets_via_ipc_actor_impl``
end-to-end **without Megatron / nemo-rl plumbing**: a mock trainer loads
every weight from the sliced Qwen3-30B-A3B-Instruct-2507 checkpoint into
GPU memory, builds a dummy ``(name, tensor)`` bucket iterator, and feeds
it to the production helper. Pre/post ``generate`` outputs must match
byte-for-byte.

The function under test expects an ``ipc_engine`` Ray actor handle whose
``update_weights_from_tensor.remote(...)`` accepts the
``serialized_named_tensors`` / ``load_format`` / ``weight_version``
kwargs (i.e. the SGLangGenerationWorker shape). To avoid the full Ray
cluster + HTTP-server overhead in this isolated script we wrap an
in-process ``sglang.srt.entrypoints.engine.Engine`` in a tiny fake-Ray
shim (see :class:`_FakeRayActorHandle` below) and monkey-patch
``ray.get`` to recognise the fake refs. Production paths that go through
the real Ray actor are unaffected.

Usage (inside the sglang-nemorl-e2e-zhw container)::

    python tests/unit/models/generation/sglang/repro_pp_colocate_weight_update.py --pp 1
    python tests/unit/models/generation/sglang/repro_pp_colocate_weight_update.py --pp 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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
    p = Path(hf_home) / "hub" / "qwen3-30b-a3b-sliced-4"
    if not p.is_dir():
        sys.exit(
            f"sliced ckpt not found at {p}; produce it first via "
            "tests/unit/models/generation/sglang/_qwen3_slicer.py"
        )
    return p


def _collect_specs(ckpt: Path):
    """Return ``(name, dtype_str, shape, shard_filename)`` records sorted by name."""
    index_path = ckpt / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        single = next(ckpt.glob("*.safetensors")).name
        with safe_open(ckpt / single, framework="pt") as r:
            weight_map = {k: single for k in r.keys()}

    per_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, []).append(name)

    specs: list[tuple[str, str, list[int]]] = []
    for shard in sorted(per_shard.keys()):
        with safe_open(ckpt / shard, framework="pt") as r:
            for name in sorted(per_shard[shard]):
                sl = r.get_slice(name)
                st_dtype = sl.get_dtype()
                if st_dtype not in _SAFETENSORS_TO_TORCH_NAME:
                    raise RuntimeError(f"unmapped safetensors dtype {st_dtype!r} for {name}")
                specs.append((name, _SAFETENSORS_TO_TORCH_NAME[st_dtype], list(sl.get_shape())))
    specs.sort(key=lambda x: x[0])
    return specs, weight_map


# ---------------------------------------------------------------------------
# Fake Ray actor wrapper around an in-process sglang Engine.
# ---------------------------------------------------------------------------
class _FakeRef:
    """Stand-in for ``ray.ObjectRef`` produced by the fake actor's ``.remote()``."""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value


class _FakeRemoteCallable:
    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return _FakeRef(self._fn(*args, **kwargs))


class _FakeRayActorHandle:
    """Mimics the production SGLangGenerationWorker handle used by
    ``send_hf_buckets_via_ipc_actor_impl``: ``handle.update_weights_from_tensor.remote(...)``
    must return something that ``ray.get`` can resolve.

    Translates the production kwargs (``serialized_named_tensors`` /
    ``weight_version``) onto sglang's in-process ``Engine.update_weights_from_tensor``
    (``named_tensors``).
    """

    def __init__(self, engine):
        self._engine = engine

    @property
    def update_weights_from_tensor(self):
        def _call(
            serialized_named_tensors,
            load_format=None,
            flush_cache=False,
            weight_version=None,
        ):
            return self._engine.update_weights_from_tensor(
                named_tensors=serialized_named_tensors,
                load_format=load_format,
                flush_cache=flush_cache,
            )

        return _FakeRemoteCallable(_call)


def _patch_ray_get_for_fake_refs() -> None:
    """Make ``ray.get`` pass-through for our fake refs. Real refs still work."""
    import ray

    if getattr(ray.get, "__fake_ray_patched__", False):
        return
    _real_get = ray.get

    def _patched_get(refs, *args, **kwargs):
        if isinstance(refs, _FakeRef):
            return refs._value
        if isinstance(refs, list) and refs and all(isinstance(r, _FakeRef) for r in refs):
            return [r._value for r in refs]
        return _real_get(refs, *args, **kwargs)

    _patched_get.__fake_ray_patched__ = True  # type: ignore[attr-defined]
    ray.get = _patched_get  # type: ignore[assignment]


def _check_weights(engine, action: str):
    """Drive sglang's WeightChecker (``snapshot`` / ``reset_tensors`` / ``compare``)
    against an in-process Engine. Production exposes this via the
    ``/weights_checker`` HTTP endpoint; we go straight through tokenizer_manager.

    ``tokenizer_manager.check_weights`` returns ``_Communicator.merge_results``,
    a ``(all_success: bool, joined_message: str)`` tuple — NOT a
    ``CheckWeightsReqOutput``. The HTTP layer converts ``success=False`` to a
    400; this in-process path does not, so raise here to avoid a failed
    compare masquerading as success.
    """
    from sglang.srt.managers.io_struct import CheckWeightsReqInput

    res = engine.loop.run_until_complete(
        engine.tokenizer_manager.check_weights(CheckWeightsReqInput(action=action))
    )
    success, message = res
    if not success:
        raise RuntimeError(f"check_weights({action!r}) failed: {message}")
    return res


def _post_process_weights(engine):
    """Re-run sglang's ``process_weights_after_loading`` hook on every module.

    Required after any weight refit path (NCCL broadcast or colocate IPC) when
    the model uses a quant_method whose runtime layout differs from the load-
    time layout — e.g. flashinfer trtllm BF16 / FP8 / MXFP8 MoE, which pack
    weights into a block layout that the kernel expects but
    ``update_weights_from_tensor`` / ``update_weights_from_distributed`` only
    write back canonical bytes. Without this, post-refit inference crashes or
    silently produces garbage. nemo-rl's production refit calls this via
    ``policy_generation.post_process_weights()`` (megatron_policy_worker.py).
    """
    from sglang.srt.managers.io_struct import PostProcessWeightsReqInput

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


# ---------------------------------------------------------------------------
# Mock trainer: load all weights on one GPU, yield buckets bounded by bytes.
# ---------------------------------------------------------------------------
class _MockTrainer:
    """Loads every safetensors tensor onto ``device`` once, then iterates
    them in deterministic byte-bounded buckets — mirroring the contract
    of ``MegatronSGLangHfWeightIterator.iter_hf_weight_buckets`` without
    pulling in Megatron / AutoBridge.
    """

    def __init__(self, ckpt: Path, device: str, specs, weight_map):
        self.device = device
        self._tensors: list[tuple[str, torch.Tensor]] = []
        for name, dtype_str, _shape in specs:
            with safe_open(ckpt / weight_map[name], framework="pt") as r:
                t = r.get_tensor(name)
            self._tensors.append(
                (name, t.to(device=device, dtype=_DTYPE_FROM_STR[dtype_str]).contiguous())
            )

    def iter_buckets(self, buffer_size_bytes: int) -> Iterable[list[tuple[str, torch.Tensor]]]:
        bucket: list[tuple[str, torch.Tensor]] = []
        size = 0
        for name, tensor in self._tensors:
            tsize = tensor.numel() * tensor.element_size()
            if bucket and size + tsize > buffer_size_bytes:
                yield bucket
                bucket = []
                size = 0
            bucket.append((name, tensor))
            size += tsize
        if bucket:
            yield bucket


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument(
        "--buffer-size-bytes",
        type=int,
        default=512 * 1024 * 1024,  # 512 MiB
        help="Per-chunk byte budget for the dummy bucket iterator.",
    )
    parser.add_argument("--master-port", type=int, default=29565)
    args = parser.parse_args()

    n_engine = args.pp * args.tp * args.dp
    n_avail = torch.cuda.device_count()
    if n_avail < n_engine:
        sys.exit(f"need at least {n_engine} GPUs visible, have {n_avail}")

    ckpt = _ckpt_path()
    print(f"[main] ckpt={ckpt}", flush=True)
    print(
        f"[main] pp={args.pp} tp={args.tp} dp={args.dp} n_engine={n_engine}",
        flush=True,
    )
    print(f"[main] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)

    specs, weight_map = _collect_specs(ckpt)
    print(f"[main] {len(specs)} weight tensors to round-trip", flush=True)

    # Trivial single-rank gather group. ``send_hf_buckets_via_ipc_actor_impl``
    # needs torch.distributed initialised so it can call ``dist.gather_object``;
    # with world_size=1 the gather is a no-op but the call is still required.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    gather_group = dist.new_group(ranks=[0], backend="gloo")

    # ipc_engine handle path expects a Ray actor; wire up our fake.
    _patch_ray_get_for_fake_refs()

    # Apply the same monkey-patch the production trainer uses so that
    # IPC handles encode device-by-UUID — the engine's CUDA_VISIBLE_DEVICES
    # view may differ from ours.
    from nemo_rl.models.policy.torch_reductions_utils import (
        monkey_patch_torch_reductions,
    )

    monkey_patch_torch_reductions()

    # Stand up the engine in-process (skip the Ray actor / HTTP server).
    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=str(ckpt),
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        mem_fraction_static=0.3,
        log_level="info",
        random_seed=42,
        disable_cuda_graph=True,
        # Required by the colocate IPC path: the engine must keep param
        # storage alive across update_weights_from_tensor calls (no offload).
        enable_memory_saver=False,
    )
    print("[main] engine up", flush=True)

    sampling = {"max_new_tokens": 16, "temperature": 0.0}
    prompt = "The capital of France is"

    def _text(x):
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, dict):
            return x.get("text", repr(x))
        return repr(x)

    # Step 1: capture a baseline generation with the freshly-loaded weights.
    out_pre = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] pre-update text={_text(out_pre)!r}", flush=True)

    # Step 2: snapshot the in-engine weights so we can compare bit-exactly later.
    print("[main] snapshot...", flush=True)
    _check_weights(engine, "snapshot")

    # Step 3: randomize the in-engine weights so a no-op refit would visibly
    # change generation. If we skipped this, pre/post equality would trivially
    # pass even if send_hf_buckets_via_ipc_actor_impl never copied anything.
    print("[main] reset_tensors (randomize sglang weights)...", flush=True)
    _check_weights(engine, "reset_tensors")
    out_random = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] post-reset text={_text(out_random)!r}", flush=True)
    if _text(out_random) == _text(out_pre):
        # Reset is meant to be visible. If it isn't, the test below isn't
        # really testing anything.
        print(
            "[main] WARN: post-reset output matches pre — reset_tensors may "
            "have been a no-op for this prompt; proceeding anyway",
            flush=True,
        )

    fake_engine_handle = _FakeRayActorHandle(engine)
    trainer = _MockTrainer(ckpt, device="cuda:0", specs=specs, weight_map=weight_map)
    worker_state: dict[str, Any] = {
        "_ipc_gather_group": gather_group,
        "_ipc_gather_src": 0,
        "_ipc_engine_index": 0,
        "_ipc_layout_key": ((1,), (0,)),
        "_ipc_monkey_patched": True,
        "weight_version": 0,
    }

    from nemo_rl.models.policy.utils import send_hf_buckets_via_ipc_actor_impl

    # Step 4: refit via the production helper — must overwrite the random
    # weights with the trainer's authentic copy.
    print("[main] send_hf_buckets_via_ipc_actor_impl...", flush=True)
    t0 = time.time()
    send_hf_buckets_via_ipc_actor_impl(
        bucket_iterator=trainer.iter_buckets(args.buffer_size_bytes),
        rollout_engines=[fake_engine_handle],
        worker_state=worker_state,
        weight_version=1,
    )
    print(f"[main] send_hf_buckets done in {time.time() - t0:.1f}s", flush=True)

    # Step 4.5: re-run process_weights_after_loading so flashinfer trtllm MoE
    # layers re-pack the canonical bytes that send_hf_buckets_via_ipc_actor_impl
    # just wrote. nemo-rl's production refit path always calls this; the bare
    # update_weights_from_tensor API does NOT do it implicitly.
    print("[main] post_process_weights...", flush=True)
    _post_process_weights(engine)
    print("[main] post_process_weights done.", flush=True)

    # Step 5: compare current weights against the snapshot. Raises if any
    # tensor differs — the strongest correctness signal we have.
    print("[main] compare against snapshot...", flush=True)
    _check_weights(engine, "compare")
    print("[main] compare passed.", flush=True)

    # Step 6: generation should now match the pre-snapshot output.
    out_post = engine.generate(prompt, sampling_params=sampling)
    print(f"[main] post-update text={_text(out_post)!r}", flush=True)

    pre_text = _text(out_pre)
    post_text = _text(out_post)
    if pre_text == post_text:
        print("[main] PASS: snapshot+reset+refit+compare roundtrip", flush=True)
    else:
        print("[main] FAIL: pre/post generation outputs differ", flush=True)
        print(f"  pre : {pre_text!r}")
        print(f"  post: {post_text!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
