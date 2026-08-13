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
    if world_size != 8:
        raise RuntimeError(f"Expected 8 ranks, got {world_size}")

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

    buffer: deep_ep.HybridEPBuffer | None = None
    try:
        num_tokens = 256
        hidden_size = 7168
        num_local_experts = 2
        num_experts = world_size * num_local_experts
        buffer = deep_ep.HybridEPBuffer(
            group=dist.group.WORLD,
            hidden_dim=hidden_size,
            max_num_of_tokens_per_rank=num_tokens,
            num_local_experts=num_local_experts,
            use_fp8=False,
        )
        if rank == 0:
            print("HYBRIDEP_BUFFER_INIT_PASS", flush=True)

        neighboring_rank = (rank + 1) % world_size
        opposite_rank = (rank + 4) % world_size

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
        topk_idx[:, 0] = neighboring_rank * 2
        topk_idx[:, 1] = opposite_rank * 2 + 1
        topk_weights = torch.full(
            (num_tokens, 2),
            1.0,
            dtype=torch.float32,
            device="cuda",
        )

        (
            recv_x,
            recv_topk_weights,
            _,
            handle,
        ) = buffer.dispatch(
            hidden=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_of_experts=num_experts,
        )
        torch.cuda.synchronize()
        if recv_topk_weights is None:
            raise RuntimeError("HybridEP dispatch did not return top-k weights")

        num_dispatched_tokens = int(handle[3].item())
        local_expert_routing_map = handle[4][:num_dispatched_tokens]
        copy_times = local_expert_routing_map.sum(dim=1)
        hidden_to_combine = recv_x * copy_times.unsqueeze(1)

        combined_x, _ = buffer.combine(
            hidden=hidden_to_combine,
            handle=handle,
            probs=recv_topk_weights,
        )
        torch.cuda.synchronize()

        expected = x * 2
        if not torch.equal(combined_x, expected):
            max_error = (combined_x.float() - expected.float()).abs().max().item()
            raise RuntimeError(f"DeepEP round-trip mismatch: max_error={max_error}")

        dist.barrier()
        if rank == 0:
            print("HYBRIDEP_INTRANODE_DISPATCH_PASS", flush=True)
    finally:
        del buffer
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
