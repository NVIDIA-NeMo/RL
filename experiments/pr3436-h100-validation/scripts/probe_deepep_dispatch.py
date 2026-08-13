from __future__ import annotations

import os
from datetime import timedelta

import deep_ep
import torch
import torch.distributed as dist


def _required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return int(value)


def main() -> None:
    rank = _required_int("SLURM_PROCID")
    local_rank = _required_int("SLURM_LOCALID")
    world_size = _required_int("SLURM_NTASKS")
    if world_size != 16:
        raise RuntimeError(f"Expected 16 ranks, got {world_size}")

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )

    buffer: deep_ep.Buffer | None = None
    try:
        buffer = deep_ep.Buffer(
            dist.group.WORLD,
            int(2e9),
            int(1e9),
            num_qps_per_rank=24,
            explicitly_destroy=True,
        )
        if rank == 0:
            print("DEEPEP_BUFFER_INIT_PASS", flush=True)

        num_tokens = 256
        hidden_size = 7168
        num_experts = world_size * 2
        node_base_rank = (rank // 8) * 8
        remote_rank = (rank + 8) % world_size
        neighboring_rank = node_base_rank + ((local_rank + 1) % 8)

        x = torch.full(
            (num_tokens, hidden_size),
            float(rank + 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk_idx = torch.empty(
            (num_tokens, 2),
            dtype=torch.int64,
            device="cuda",
        )
        topk_idx[:, 0] = remote_rank * 2
        topk_idx[:, 1] = neighboring_rank * 2 + 1
        topk_weights = torch.full(
            (num_tokens, 2),
            0.5,
            dtype=torch.float32,
            device="cuda",
        )

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)

        (
            recv_x,
            _,
            recv_topk_weights,
            _,
            handle,
            _,
        ) = buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        torch.cuda.synchronize()
        if recv_topk_weights is None:
            raise RuntimeError("DeepEP dispatch did not return top-k weights")

        combined_x, _, _ = buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=recv_topk_weights,
        )
        torch.cuda.synchronize()

        expected = x * 2
        if not torch.equal(combined_x, expected):
            max_error = (combined_x.float() - expected.float()).abs().max().item()
            raise RuntimeError(f"DeepEP round-trip mismatch: max_error={max_error}")

        dist.barrier()
        if rank == 0:
            print("DEEPEP_INTER_NODE_DISPATCH_PASS", flush=True)
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
