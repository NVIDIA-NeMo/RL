"""Two-node EP8 consolidation followed by native NCCL TP2 refit.

The EP8 phase loads the real Megatron model on eight ranks, packs each rank's
disjoint expert tensors, and gathers all expert shards onto rank 0 over NCCL.
Replicated/non-expert tensors are counted once from rank 0. The resulting timing
artifact is paired with ``native_nccl_refit_bench.py`` using the equivalent full
HF checkpoint.

The gathered native shards are not used as the vLLM payload because translating
Megatron's fused native layout into HF names is a separate receiver operation in
the current integration. The result is therefore a staged, topology-matched
consolidation-inclusive NCCL baseline, with consolidation and transfer reported
as separate measured phases.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DEVICE = LOCAL_RANK
MODEL_ID = os.environ.get(
    "MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
WARMUPS = int(os.environ.get("CONSOLIDATION_WARMUPS", "1"))
CYCLES = int(os.environ.get("CONSOLIDATION_CYCLES", "3"))
RESULT = Path(
    os.environ.get(
        "CONSOLIDATION_RESULT",
        "/mnt/rl-workspace/kavink/ep8_nccl_consolidation.json",
    )
)


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": ordered[-1],
        "max": max(values),
    }


def main() -> int:
    if WORLD != 8:
        raise RuntimeError(f"EP8 consolidation requires world_size=8, got {WORLD}")

    torch.cuda.set_device(DEVICE)
    dist.init_process_group("nccl")

    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=WORLD,
    )
    ep_rank = parallel_state.get_expert_model_parallel_rank()

    load_start = time.perf_counter()
    bridge = AutoBridge.from_hf_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = WORLD
    provider.expert_tensor_parallel_size = 1
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.sequence_parallel = False
    provider.finalize()
    model_list = provider.provide_distributed_model(wrap_with_ddp=False)
    model = model_list[0] if isinstance(model_list, list) else model_list
    model_load_seconds = time.perf_counter() - load_start

    from nemo_rl.distributed.mx_megatron_helpers import (
        collect_megatron_publish_set,
    )

    tcfg = getattr(bridge, "transformer_config", None) or provider
    num_heads = getattr(tcfg, "num_attention_heads", None)
    kv_groups = getattr(tcfg, "num_query_groups", None) or num_heads
    hidden = getattr(tcfg, "hidden_size", None)
    kv_channels = getattr(tcfg, "kv_channels", None) or (
        hidden // num_heads if num_heads else None
    )
    num_experts = (
        getattr(tcfg, "num_moe_experts", None)
        or getattr(tcfg, "num_experts", None)
    )
    num_local_experts = int(num_experts) // WORLD

    entries = list(
        collect_megatron_publish_set(
            model,
            tp_size=1,
            pp_size=1,
            pp_rank=0,
            ep_size=WORLD,
            ep_rank=ep_rank,
            tp_rank=0,
            num_local_experts=num_local_experts,
            num_attention_heads=num_heads,
            num_kv_heads=kv_groups,
            head_dim=kv_channels,
            target_dtype=torch.bfloat16,
        )
    )
    expert_tensors = [
        local.contiguous()
        for _name, local, spec, _extras in entries
        if spec.is_expert
    ]
    nonexpert_bytes = sum(
        local.numel() * local.element_size()
        for _name, local, spec, _extras in entries
        if not spec.is_expert
    )
    expert_numel = sum(tensor.numel() for tensor in expert_tensors)
    expert_bytes = expert_numel * torch.tensor(
        [], dtype=torch.bfloat16
    ).element_size()

    metadata = {
        "expert_tensors": len(expert_tensors),
        "expert_numel": expert_numel,
        "expert_bytes": expert_bytes,
        "nonexpert_bytes": nonexpert_bytes,
    }
    gathered_metadata: list[dict] = [None] * WORLD  # type: ignore[list-item]
    dist.all_gather_object(gathered_metadata, metadata)
    if len({item["expert_numel"] for item in gathered_metadata}) != 1:
        raise RuntimeError(f"EP ranks have inconsistent expert payloads: {gathered_metadata}")

    receive_buffers = (
        [
            torch.empty(
                expert_numel,
                dtype=torch.bfloat16,
                device=f"cuda:{DEVICE}",
            )
            for _ in range(WORLD)
        ]
        if RANK == 0
        else []
    )

    durations: list[float] = []
    total = WARMUPS + CYCLES
    for cycle in range(total):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        packed = torch.cat([tensor.reshape(-1) for tensor in expert_tensors])

        if RANK == 0:
            receive_buffers[0].copy_(packed)
            ops = [
                dist.P2POp(dist.irecv, receive_buffers[source], source)
                for source in range(1, WORLD)
            ]
        else:
            ops = [dist.P2POp(dist.isend, packed, 0)]
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()
        torch.cuda.synchronize()

        elapsed = torch.tensor(
            time.perf_counter() - started,
            dtype=torch.float64,
            device=f"cuda:{DEVICE}",
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds = float(elapsed.item())
        if cycle >= WARMUPS:
            durations.append(seconds)
        if RANK == 0:
            label = "warmup" if cycle < WARMUPS else "measured"
            print(
                f"EP8_CONSOLIDATION {label}={cycle} seconds={seconds:.6f}",
                flush=True,
            )
        del packed

    consolidated_bytes = (
        gathered_metadata[0]["nonexpert_bytes"]
        + sum(item["expert_bytes"] for item in gathered_metadata)
    )
    if RANK == 0:
        RESULT.write_text(
            json.dumps(
                {
                    "schema_version": "ep8-nccl-consolidation-v1",
                    "status": "ok",
                    "model": MODEL_ID,
                    "world_size": WORLD,
                    "nodes": 2,
                    "gpus_per_node": 4,
                    "trainer_parallelism": "EP8/TP1/PP1/DP1",
                    "destination_parallelism": "TP2",
                    "model_load_seconds": model_load_seconds,
                    "expert_tensors_per_rank": len(expert_tensors),
                    "expert_bytes_per_rank": expert_bytes,
                    "replicated_nonexpert_bytes": gathered_metadata[0][
                        "nonexpert_bytes"
                    ],
                    "consolidated_bytes": consolidated_bytes,
                    "warmup_cycles": WARMUPS,
                    "measured_cycles": CYCLES,
                    "consolidation_seconds": _stats(durations),
                    "method": (
                        "pack each EP rank's native expert tensors and gather "
                        "all eight GPU buffers to trainer rank 0 via NCCL"
                    ),
                    "payload_handoff": (
                        "timed native shard gather followed by equivalent full "
                        "HF checkpoint NCCL refit"
                    ),
                },
                indent=2,
            )
        )

    dist.barrier()
    del receive_buffers, expert_tensors, entries, model, model_list, provider, bridge
    gc.collect()
    torch.cuda.empty_cache()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
