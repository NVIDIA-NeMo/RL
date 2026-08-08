#!/usr/bin/env python3

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    max_tokens = int(os.environ["PROBE_MAX_TOKENS"])
    probe_mode = os.environ.get("PROBE_MODE", "construct")

    visible_device_count = torch.cuda.device_count()
    device_index = 0 if visible_device_count == 1 else local_rank
    torch.cuda.set_device(device_index)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    from deep_ep import HybridEPBuffer

    if rank == 0:
        print(
            "probe-start "
            f"host={socket.gethostname()} world_size={world_size} "
            f"visible_devices={visible_device_count} device_index={device_index} "
            f"max_tokens={max_tokens} mode={probe_mode} combine_chunk="
            f"{os.environ.get('NUM_OF_TOKENS_PER_CHUNK_COMBINE_API')} "
            f"deepep={__import__('deep_ep').__file__}",
            flush=True,
        )

    hidden_dim = 1024 if probe_mode == "dispatch" else 4096
    num_local_experts = 4 if probe_mode == "dispatch" else 16
    buffer = HybridEPBuffer(
        group=dist.group.WORLD,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=max_tokens,
        num_local_experts=num_local_experts,
        use_fp8=False,
        num_sms_dispatch_api=32,
        num_sms_combine_api=32,
    )
    dist.barrier()
    if rank == 0:
        print("probe-buffer-created", flush=True)

    if probe_mode == "dispatch":
        num_tokens = int(os.environ.get("PROBE_NUM_TOKENS", min(max_tokens, 128)))
        with_probs = os.environ.get("PROBE_WITH_PROBS", "1") == "1"
        if not 0 < num_tokens <= max_tokens:
            raise ValueError(
                f"PROBE_NUM_TOKENS must be in [1, {max_tokens}], got {num_tokens}"
            )
        num_experts = world_size * num_local_experts
        hidden = torch.full(
            (num_tokens, hidden_dim),
            rank + 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk_idx = torch.full(
            (num_tokens, 1),
            ((rank + 1) % world_size) * num_local_experts,
            dtype=torch.int64,
            device="cuda",
        )
        topk_weights = torch.ones(
            (num_tokens, 1), dtype=torch.float32, device="cuda"
        )
        if rank == 0:
            print(
                "probe-dispatch-start "
                f"tokens={num_tokens} hidden_dim={hidden_dim} "
                f"num_experts={num_experts} with_probs={with_probs}",
                flush=True,
            )
        dispatched, _, _, handle = buffer.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights if with_probs else None,
            num_of_experts=num_experts,
        )
        torch.cuda.synchronize()
        if rank == 0:
            print(f"probe-dispatch-pass shape={tuple(dispatched.shape)}", flush=True)
        combined, _ = buffer.combine(hidden=dispatched, handle=handle)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"probe-combine-pass shape={tuple(combined.shape)}", flush=True)

    del buffer
    dist.barrier()
    if rank == 0:
        print("probe-pass", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
