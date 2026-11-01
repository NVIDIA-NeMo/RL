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

"""Validate native AutoModel EP sketches against uncompressed expert gradients.

Run with torch.distributed.run --nproc-per-node=4. Includes unequal local batch
sizes, an expert rank receiving no tokens, and activation recomputation.
"""

import copy

import torch
import torch.distributed as dist
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint


def check_equivalence(
    *, recompute: bool, leave_rank_empty: bool, dtype: torch.dtype
) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(17)
    dim = 32 if dtype == torch.bfloat16 else 7
    inter_dim = 32 if dtype == torch.bfloat16 else 5
    rtol, atol = (0.02, 3e-4) if dtype == torch.bfloat16 else (2e-5, 1e-6)
    reference = GroupedExperts(
        MoEConfig(
            n_routed_experts=world * 2,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0,
            aux_loss_coeff=0,
            score_func="softmax",
            route_scale=1,
            dim=dim,
            inter_dim=inter_dim,
            moe_inter_dim=inter_dim,
            norm_topk_prob=True,
            expert_activation="relu2",
            dtype=dtype,
        ),
        backend=BackendConfig(experts="torch_mm") if dtype == torch.bfloat16 else None,
    ).cuda()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.2)
    sharded = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    for name, parameter in list(sharded.named_parameters()):
        sharded.register_parameter(
            name, nn.Parameter(distribute_tensor(parameter, mesh, [Shard(0)]))
        )
    optimizer = LoGRAOptimizer(
        sharded,
        torch.optim.AdamW(sharded.parameters(), lr=0.01, weight_decay=0),
        LoGRAConfig(
            rank=3,
            optimizer="sgd",
            target_modules=[],
            target_parameters=["gate_and_up_projs", "down_projs"],
        ),
    )
    lengths = list(range(1, world + 1))
    total = sum(lengths)
    x = torch.randn(total, dim, device="cuda", dtype=dtype, requires_grad=True)
    active_experts = (world - int(leave_rank_empty)) * 2
    indices = torch.arange(total * 2, device="cuda").reshape(total, 2) % active_experts
    weights = torch.full((total, 2), 0.5, device="cuda")
    mask = torch.ones(total, dtype=torch.bool, device="cuda")
    reference(x, mask, weights, indices).square().sum().div(total).backward()
    start = sum(lengths[:rank])
    stop = start + lengths[rank]
    local_x = x[start:stop].detach().clone().requires_grad_(True)
    args = (local_x, mask[start:stop], weights[start:stop], indices[start:stop])
    output = (
        checkpoint(sharded, *args, use_reentrant=False) if recompute else sharded(*args)
    )
    output.square().sum().mul(world / total).backward()
    expected_updates = {}
    expected_norm2 = torch.zeros((), device="cuda")
    for layer in optimizer.layers:
        parameter = dict(reference.named_parameters())[layer.weight_name]
        gradient = parameter.grad.float().reshape(-1, parameter.shape[-1])
        update = gradient @ layer.projection.T @ layer.projection
        expected_updates[layer.weight_name] = update.view_as(parameter)
        expected_norm2 += update.square().sum()
    norm = optimizer.synchronize_and_clip(dist.group.WORLD, None)
    torch.testing.assert_close(
        torch.tensor(norm, device="cuda"), expected_norm2.sqrt(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        local_x.grad / world, x.grad[start:stop], rtol=rtol, atol=atol
    )
    optimizer.step()
    for name, parameter in sharded.named_parameters():
        expected = (
            dict(reference.named_parameters())[name].float()
            - 0.01 * expected_updates[name]
        )
        torch.testing.assert_close(
            parameter.full_tensor(), expected.to(dtype), rtol=rtol, atol=atol
        )
        assert parameter.grad is None
    if rank == 0:
        print(
            f"EXPERT_EQUIVALENCE_OK dtype={dtype} recompute={recompute} empty_rank={leave_rank_empty}",
            flush=True,
        )


def main() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    if dist.get_world_size() < 2:
        raise ValueError("This check requires at least two GPUs")
    for dtype in (torch.float32, torch.bfloat16):
        for recompute in (False, True):
            for leave_rank_empty in (False, True):
                check_equivalence(
                    recompute=recompute, leave_rank_empty=leave_rank_empty, dtype=dtype
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
