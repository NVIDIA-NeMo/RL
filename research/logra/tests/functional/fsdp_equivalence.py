"""Run with torch.distributed.run --nproc-per-node=2; compare to a dense reference."""

import copy

import torch
import torch.distributed as dist
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(19)
    reference = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).cuda()
    sharded = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    fully_shard(sharded, mesh=mesh)
    config = LoGRAConfig(rank=3, target_modules=["0"], optimizer="sgd")
    optimizer = LoGRAOptimizer(
        sharded,
        torch.optim.AdamW(sharded.parameters(), lr=0.01, weight_decay=0),
        config,
    )
    baseline = torch.optim.AdamW(
        [p for name, p in reference.named_parameters() if name != "0.weight"],
        lr=0.01,
        weight_decay=0,
    )
    torch.manual_seed(22)
    x = torch.randn(8, 7, device="cuda", requires_grad=True)
    reference(x).square().sum().div(8).backward()
    sharded(x.chunk(world)[rank]).square().sum().mul(world / 8).backward()
    a = optimizer.layers[0].projection
    projected = reference[0].weight.grad @ a.T @ a
    expected_norm = (
        projected.square().sum()
        + sum(
            p.grad.square().sum()
            for name, p in reference.named_parameters()
            if name != "0.weight"
        )
    ).sqrt()
    norm = optimizer.synchronize_and_clip(dist.group.WORLD, 0.3)
    torch.testing.assert_close(
        torch.tensor(norm, device="cuda"), expected_norm, rtol=1e-5, atol=1e-5
    )
    scale = min(1.0, 0.3 / (expected_norm.item() + 1e-6))
    with torch.no_grad():
        reference[0].weight.add_(projected, alpha=-0.01 * scale)
        for name, p in reference.named_parameters():
            if name != "0.weight":
                p.grad.mul_(scale)
    baseline.step()
    optimizer.step()
    for (name, actual), expected in zip(
        sharded.named_parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual.full_tensor(), expected, rtol=1e-5, atol=1e-6, msg=name
        )
    assert sharded[0].weight.grad is None
    if rank == 0:
        print("FSDP_EQUIVALENCE_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
