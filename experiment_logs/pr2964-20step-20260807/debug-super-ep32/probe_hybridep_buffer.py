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

    torch.cuda.set_device(local_rank)
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
            f"max_tokens={max_tokens} combine_chunk="
            f"{os.environ.get('NUM_OF_TOKENS_PER_CHUNK_COMBINE_API')} "
            f"deepep={__import__('deep_ep').__file__}",
            flush=True,
        )

    buffer = HybridEPBuffer(
        group=dist.group.WORLD,
        hidden_dim=4096,
        max_num_of_tokens_per_rank=max_tokens,
        num_local_experts=16,
        use_fp8=False,
        num_sms_dispatch_api=32,
        num_sms_combine_api=32,
    )
    dist.barrier()
    if rank == 0:
        print("probe-buffer-created", flush=True)
    del buffer
    dist.barrier()
    if rank == 0:
        print("probe-pass", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
