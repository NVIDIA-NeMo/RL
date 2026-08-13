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

"""GPU oracle tests for native Qwen3-MoE expert parallelism.

Run this file under ``torchrun --nproc-per-node=4`` for EP=4 or under
``torchrun --nproc-per-node=8`` for EP=4 with an EP-shard dimension of two.
Ordinary pytest runs skip it because the differentiable routing-weight gather
only exists for EP > 1.
"""

from __future__ import annotations

import copy
import os
from datetime import timedelta
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = [
    pytest.mark.automodel,
    pytest.mark.skipif(
        _WORLD_SIZE not in {4, 8} or "LOCAL_RANK" not in os.environ,
        reason="requires torchrun with four or eight ranks",
    ),
]


def _config(*, aux_loss_coeff: float, n_routed_experts: int = 4):
    from nemo_automodel.components.moe.config import MoEConfig

    return MoEConfig(
        n_routed_experts=n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=aux_loss_coeff,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=6,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )


def _make_gate(config, device):
    from nemo_automodel.components.moe.layers import Gate

    gate = Gate(config, gate_precision=torch.float32).to(device).train()
    with torch.no_grad():
        if config.n_routed_experts == 4:
            value = torch.tensor(
                [
                    [1.20, -0.25, 0.10, 0.05],
                    [-0.35, 1.10, 0.20, -0.15],
                    [0.15, -0.30, 1.05, 0.25],
                    [-0.20, 0.15, -0.40, 1.25],
                ],
                device=device,
            )
        else:
            indices = torch.arange(
                config.n_routed_experts * config.dim,
                device=device,
                dtype=torch.float32,
            ).reshape(config.n_routed_experts, config.dim)
            value = 0.7 * torch.sin(indices * 0.37 + 0.2) + 0.4 * torch.cos(
                indices * 0.13 - 0.1
            )
        gate.weight.copy_(value)
    return gate


def _make_experts(config, device):
    from nemo_automodel.components.moe.experts import GroupedExperts

    experts = GroupedExperts(config).to(device).train()
    # This tiny numerical oracle validates EP/FSDP/autograd semantics rather
    # than Inductor.  The pinned implementation wraps this four-op activation
    # in several separately compiled custom-autograd kernels; eight ranks each
    # launching a 32-worker max-autotune pool obscures collective failures and
    # can exceed the short-job limit before the first EP forward completes.
    # Keep production's compiled activation unchanged and use the exact eager
    # forward math here; ordinary autograd supplies the same derivatives.
    experts.expert_activation_grouped = _eager_weighted_swiglu
    with torch.no_grad():
        for parameter_index, parameter in enumerate(experts.parameters()):
            values = torch.arange(
                parameter.numel(), device=device, dtype=torch.float32
            ).reshape(parameter.shape)
            parameter.copy_(0.07 * torch.sin(values * 0.17 + parameter_index * 0.31))
    return experts


def _eager_weighted_swiglu(
    value: torch.Tensor,
    weights: torch.Tensor,
    fp8_input_store: bool = False,
) -> torch.Tensor:
    assert not fp8_input_store
    gate, up = torch.chunk(value, 2, dim=-1)
    return (F.silu(gate) * up * weights).to(value.dtype)


def _global_inputs(device):
    # Unequal per-rank token counts exercise the variable-length EP gather.
    local_lengths = [2, 3, 4, 5]
    values = torch.arange(sum(local_lengths) * 4, device=device, dtype=torch.float32)
    x = (0.65 * torch.sin(values * 0.23) + 0.35 * torch.cos(values * 0.11)).reshape(
        -1, 4
    )
    targets = torch.cos(values.reshape(-1, 4) * 0.19 + 0.4)
    return x, targets, local_lengths


def _global_inputs_ep_shard(device):
    # Both EP-shard replicas receive distinct data, and every EP group still
    # contains unequal local token counts.
    local_lengths = [2, 3, 4, 5, 3, 2, 5, 4]
    values = torch.arange(sum(local_lengths) * 4, device=device, dtype=torch.float32)
    x = (0.55 * torch.sin(values * 0.21) + 0.45 * torch.cos(values * 0.09)).reshape(
        -1, 4
    )
    targets = torch.sin(values.reshape(-1, 4) * 0.17 + 0.35)
    return x, targets, local_lengths


def _local_dtensor(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full_dtensor(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _install_logical_router(gate):
    from nemo_rl.models.automodel.shared_prefix_moe import _logical_router_forward

    gate._nemo_shared_prefix_original_forward = gate.forward
    gate.forward = MethodType(_logical_router_forward, gate)


def _production_w8_meshes():
    """Construct exactly the DP and overlapping EP views used by NeMo-RL."""
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.mesh_utils import (
        create_device_mesh,
        get_submesh,
    )

    device_mesh, moe_mesh = create_device_mesh(
        FSDP2Config(backend="nccl"),
        dp_replicate_size=1,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ep_size=4,
        world_size=8,
    )
    assert device_mesh is not None
    assert moe_mesh is not None
    # ``MeshContext._dp_axis_names()`` passes only ``dp_shard_cp`` to the
    # native MoE parallelizer for this topology.  The flattened ``dp`` view is
    # a convenient logical data mesh, but it is not the root FSDP mesh used by
    # production when EP introduces an overlapping ``ep_shard`` dimension.
    return get_submesh(device_mesh, ("dp_shard_cp",)), moe_mesh


class _TinyMoE(torch.nn.Module):
    def __init__(self, gate, experts):
        super().__init__()
        self.gate = gate
        self.experts = experts

    def forward(self, x, token_mask):
        weights, indices, aux_loss = self.gate(x, token_mask, None)
        output = self.experts(x, token_mask, weights, indices)
        return output, indices, aux_loss


@pytest.mark.skipif(_WORLD_SIZE != 4, reason="requires exactly four ranks")
def test_ep4_matches_dense_main_and_logical_router_gradients():
    pytest.importorskip("nemo_automodel")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires four visible CUDA devices")

    from nemo_automodel.components.moe.megatron.moe_utils import (
        MoEAuxLossAutoScaler,
    )
    from nemo_automodel.components.moe.parallelizer import ExpertParallel
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_rl.models.automodel.shared_prefix_moe import (
        collect_shared_prefix_moe_statistics,
        enable_qwen3_moe_ep_router_gradients,
        shared_prefix_moe_context,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=2))

    original_aux_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 4
        enable_qwen3_moe_ep_router_gradients()
        ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))

        # Main-loss oracle: all ranks independently evaluate the full dense MoE,
        # while EP owns one expert per rank and receives a different token count.
        main_config = _config(aux_loss_coeff=0.0)
        dense_gate = _make_gate(main_config, device)
        ep_gate = copy.deepcopy(dense_gate)
        dense_experts = _make_experts(main_config, device)
        ep_experts = copy.deepcopy(dense_experts)
        parallelize_module(ep_experts, ep_mesh, ExpertParallel())

        global_x_values, global_targets, local_lengths = _global_inputs(device)
        start = sum(local_lengths[:rank])
        stop = start + local_lengths[rank]
        local_x = global_x_values[start:stop].clone().requires_grad_(True)
        dense_x = global_x_values.clone().requires_grad_(True)
        local_mask = torch.ones(local_x.shape[0], device=device, dtype=torch.bool)
        dense_mask = torch.ones(dense_x.shape[0], device=device, dtype=torch.bool)

        ep_weights, ep_indices, _ = ep_gate(local_x, local_mask, None)
        ep_output = ep_experts(local_x, local_mask, ep_weights, ep_indices)
        dense_weights, dense_indices, _ = dense_gate(dense_x, dense_mask, None)
        dense_output = dense_experts(dense_x, dense_mask, dense_weights, dense_indices)

        assert torch.equal(ep_indices, dense_indices[start:stop])
        torch.testing.assert_close(
            ep_weights, dense_weights[start:stop], rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            ep_output, dense_output[start:stop], rtol=2e-5, atol=2e-6
        )

        (ep_output * global_targets[start:stop]).sum().backward()
        (dense_output * global_targets).sum().backward()

        # Gate parameters are replicated in this focused harness. Summing their
        # rank-local gradients is the DDP/FSDP reduction used by a real policy.
        assert ep_gate.weight.grad is not None
        dist.all_reduce(ep_gate.weight.grad, op=dist.ReduceOp.SUM)
        assert ep_gate.weight.grad.norm() > 0
        torch.testing.assert_close(
            ep_gate.weight.grad, dense_gate.weight.grad, rtol=3e-5, atol=3e-6
        )
        torch.testing.assert_close(
            local_x.grad, dense_x.grad[start:stop], rtol=4e-5, atol=4e-6
        )

        experts_per_rank = main_config.n_routed_experts // world_size
        expert_start = rank * experts_per_rank
        expert_stop = expert_start + experts_per_rank
        for (ep_name, ep_parameter), (dense_name, dense_parameter) in zip(
            ep_experts.named_parameters(), dense_experts.named_parameters()
        ):
            assert ep_name == dense_name
            assert ep_parameter.grad is not None
            torch.testing.assert_close(
                _local_dtensor(ep_parameter.grad),
                dense_parameter.grad[expert_start:expert_stop],
                rtol=4e-5,
                atol=4e-6,
            )

        # Logical-router oracle: repeat compact tokens according to their
        # multiplicity, then compare that ordinary dense aux loss with the
        # ZoRRo adapter. The zero downstream loss ensures any Gate gradient is
        # exclusively injected by the router auxiliary loss. Sending the
        # compact weights through EP also verifies that the routing-weight
        # all-gather preserves this autograd edge.
        aux_config = _config(aux_loss_coeff=0.125)
        logical_gate = _make_gate(aux_config, device)
        dense_aux_gate = copy.deepcopy(logical_gate)
        _install_logical_router(logical_gate)
        logical_gate._track_load_balance = True
        dense_aux_gate._track_load_balance = True

        compact_x = local_x.detach().clone().requires_grad_(True)
        multiplicity = (
            torch.arange(compact_x.shape[0], device=device, dtype=torch.long) % 3
            + rank
            + 1
        )
        layout = SimpleNamespace(logical_token_weights=multiplicity)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            0.625, device=device
        )
        with shared_prefix_moe_context(logical_gate, layout, valid=True) as execution:
            logical_weights, logical_indices, logical_aux = logical_gate(
                compact_x, local_mask, None
            )
            execution.commit()
        logical_output = ep_experts(
            compact_x, local_mask, logical_weights, logical_indices
        )

        expanded_x = (
            compact_x.detach()
            .repeat_interleave(multiplicity, dim=0)
            .requires_grad_(True)
        )
        expanded_mask = torch.ones(expanded_x.shape[0], device=device, dtype=torch.bool)
        oracle_weights, oracle_indices, oracle_aux = dense_aux_gate(
            expanded_x, expanded_mask, None
        )

        assert torch.equal(
            logical_indices.repeat_interleave(multiplicity, dim=0), oracle_indices
        )
        torch.testing.assert_close(
            logical_weights.repeat_interleave(multiplicity, dim=0),
            oracle_weights,
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(logical_aux, oracle_aux, rtol=2e-6, atol=2e-6)
        assert torch.equal(
            logical_gate._last_expert_load, dense_aux_gate._last_expert_load
        )

        (logical_output.sum() * 0.0).backward()
        (oracle_weights.sum() * 0.0).backward()
        assert logical_gate.weight.grad is not None
        dist.all_reduce(logical_gate.weight.grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(dense_aux_gate.weight.grad, op=dist.ReduceOp.SUM)
        assert logical_gate.weight.grad.norm() > 0
        torch.testing.assert_close(
            logical_gate.weight.grad,
            dense_aux_gate.weight.grad,
            rtol=4e-5,
            atol=4e-6,
        )

        # DP statistics collectives must remain aligned when packing gives one
        # rank only dummy microbatches. Simulate that by retaining accumulators
        # on even ranks only and compare against an explicit global reduction.
        expected_load = logical_gate._nemo_shared_prefix_logical_expert_load.clone()
        expected_aux = logical_gate._nemo_shared_prefix_aux_loss_sum.clone()
        expected_aux_count = logical_gate._nemo_shared_prefix_aux_loss_count
        if rank % 2:
            expected_load.zero_()
            expected_aux.zero_()
            expected_aux_count = 0
            logical_gate._nemo_shared_prefix_logical_expert_load = None
            logical_gate._nemo_shared_prefix_aux_loss_sum = None
            logical_gate._nemo_shared_prefix_aux_loss_count = 0
        dist.all_reduce(expected_load)
        expected_aux_pair = torch.stack(
            (
                expected_aux,
                expected_aux.new_tensor(float(expected_aux_count)),
            )
        )
        dist.all_reduce(expected_aux_pair)

        stats_model = torch.nn.Module()
        stats_model.add_module("gate", logical_gate)
        stats_model._nemo_shared_prefix_custom_qwen3_moe = True
        stats = collect_shared_prefix_moe_statistics(stats_model, dist.group.WORLD)
        assert stats["logical_token_layer_events"] == pytest.approx(
            (expected_load.sum() / aux_config.n_activated_experts).item()
        )
        assert stats["logical_router_aux_loss_mean"] == pytest.approx(
            (expected_aux_pair[0] / expected_aux_pair[1]).item()
        )

        # Dummy microbatches still execute collectives on all EP ranks, but
        # they must attach no aux gradient and commit no logical statistics.
        dummy_gate = _make_gate(aux_config, device)
        _install_logical_router(dummy_gate)
        dummy_x = local_x.detach().clone().requires_grad_(True)
        with shared_prefix_moe_context(dummy_gate, layout, valid=False) as execution:
            dummy_weights, dummy_indices, dummy_aux = dummy_gate(
                dummy_x, local_mask, None
            )
            execution.commit()
        dummy_output = ep_experts(dummy_x, local_mask, dummy_weights, dummy_indices)
        (dummy_output.sum() * 0.0).backward()

        assert dummy_aux is None
        assert execution.pending == {}
        assert dummy_gate._last_expert_load is None
        assert (
            getattr(dummy_gate, "_nemo_shared_prefix_logical_expert_load", None) is None
        )
        assert dummy_gate.weight.grad is not None
        torch.testing.assert_close(
            dummy_gate.weight.grad, torch.zeros_like(dummy_gate.weight.grad)
        )
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = original_aux_scale
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(_WORLD_SIZE != 8, reason="requires exactly eight ranks")
def test_ep4_ep_shard2_matches_dense_post_scaled_main_and_aux_gradients():
    """Exercise the production W=8, EP=4, EP-shard=2 gradient topology."""
    pytest.importorskip("nemo_automodel")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("requires eight visible CUDA devices")

    from nemo_automodel.components.moe.megatron.moe_utils import (
        MoEAuxLossAutoScaler,
    )
    from nemo_automodel.components.moe.parallelizer import ExpertParallel
    from nemo_automodel.components.training.utils import (
        scale_grads_and_clip_grad_norm,
    )
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_rl.models.automodel.setup import (
        _prewarm_native_shared_prefix_moe_world,
    )
    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_ep_router_gradients,
        shared_prefix_moe_context,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=2))

    original_aux_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 8
        _prewarm_native_shared_prefix_moe_world(world_size)
        enable_qwen3_moe_ep_router_gradients()

        # This is the same overlapping interpretation used by NeMo-RL:
        # all eight ranks are DP data ranks, while the native experts view them
        # as two EP replicas, each containing four expert-parallel ranks.
        dp_mesh, moe_mesh = _production_w8_meshes()

        config = _config(aux_loss_coeff=0.125, n_routed_experts=8)
        dense_model = _TinyMoE(
            _make_gate(config, device), _make_experts(config, device)
        ).train()
        ep_model = copy.deepcopy(dense_model).train()
        _install_logical_router(dense_model.gate)
        _install_logical_router(ep_model.gate)

        # Production order: shard expert dimension across EP first; FSDP-shard
        # each local expert's hidden dimension across replica groups second;
        # finally FSDP the shared Gate over the complete DP mesh while ignoring
        # the already-managed expert parameters.
        parallelize_module(ep_model.experts, moe_mesh["ep"], ExpertParallel())
        fully_shard(
            ep_model.experts,
            mesh=moe_mesh["ep_shard"],
            shard_placement_fn=lambda _: Shard(1),
            reshard_after_forward=False,
        )
        fully_shard(
            ep_model,
            mesh=dp_mesh,
            ignored_params=set(ep_model.experts.parameters()),
            reshard_after_forward=False,
        )

        global_x, global_targets, local_lengths = _global_inputs_ep_shard(device)
        offsets = [0]
        for local_length in local_lengths:
            offsets.append(offsets[-1] + local_length)

        # The logical aux objective is normalized per rank/microbatch and is
        # nonlinear, so the EP=1 oracle accumulates the eight local objectives
        # rather than computing one aux loss from concatenated tokens.
        dense_loss = global_x.new_zeros(())
        dense_inputs = []
        dense_outputs = []
        dense_aux_losses = []
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=device)
        for data_rank, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            dense_x = global_x[start:stop].clone().requires_grad_(True)
            dense_mask = torch.ones(dense_x.shape[0], device=device, dtype=torch.bool)
            multiplicity = (
                torch.arange(dense_x.shape[0], device=device, dtype=torch.long) % 3
                + data_rank
                + 1
            )
            layout = SimpleNamespace(logical_token_weights=multiplicity)
            with shared_prefix_moe_context(
                dense_model, layout, valid=True
            ) as execution:
                dense_output, _, dense_aux = dense_model(dense_x, dense_mask)
                execution.commit()
            assert dense_aux is not None
            dense_inputs.append(dense_x)
            dense_outputs.append(dense_output.detach())
            dense_aux_losses.append(dense_aux.detach())
            dense_loss = dense_loss + (dense_output * global_targets[start:stop]).sum()
        dense_loss.backward()

        start, stop = offsets[rank], offsets[rank + 1]
        local_x = global_x[start:stop].clone().requires_grad_(True)
        local_mask = torch.ones(local_x.shape[0], device=device, dtype=torch.bool)
        multiplicity = (
            torch.arange(local_x.shape[0], device=device, dtype=torch.long) % 3
            + rank
            + 1
        )
        layout = SimpleNamespace(logical_token_weights=multiplicity)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            float(world_size), device=device
        )
        with shared_prefix_moe_context(ep_model, layout, valid=True) as execution:
            ep_output, _, ep_aux = ep_model(local_x, local_mask)
            execution.commit()
        assert ep_aux is not None
        torch.testing.assert_close(ep_aux, dense_aux_losses[rank])
        torch.testing.assert_close(ep_output, dense_outputs[rank], rtol=3e-5, atol=3e-6)

        # Policy.train multiplies the already globally normalized local main
        # loss by W before backward. The logical aux autoscaler uses the same W.
        # FSDP averages shared gradients over W and expert gradients over the
        # two EP-shard replicas; the existing post-scale divides expert grads by
        # W / EP-shard = 4, leaving the same global sum as the dense oracle.
        ((ep_output * global_targets[start:stop]).sum() * world_size).backward()
        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[ep_model],
            pp_enabled=False,
            moe_mesh=moe_mesh,
            ep_axis_name="ep",
            dp_group_size=world_size,
        )

        assert local_x.grad is not None
        torch.testing.assert_close(
            local_x.grad / world_size,
            dense_inputs[rank].grad,
            rtol=5e-5,
            atol=5e-6,
        )

        dense_parameters = dict(dense_model.named_parameters())
        ep_parameters = dict(ep_model.named_parameters())
        assert ep_parameters.keys() == dense_parameters.keys()
        for name in sorted(ep_parameters):
            ep_parameter = ep_parameters[name]
            dense_parameter = dense_parameters[name]
            assert ep_parameter.grad is not None, name
            assert dense_parameter.grad is not None, name
            if name.startswith("experts."):
                assert isinstance(ep_parameter, DTensor), name
                assert "ep" in ep_parameter.device_mesh.mesh_dim_names, name
                assert "ep_shard" in ep_parameter.device_mesh.mesh_dim_names, name
            ep_grad = _full_dtensor(ep_parameter.grad)
            torch.testing.assert_close(
                ep_grad,
                dense_parameter.grad,
                rtol=6e-5,
                atol=6e-6,
                msg=lambda message, parameter_name=name: (
                    f"post-scaled gradient mismatch for {parameter_name}: {message}"
                ),
            )
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = original_aux_scale
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
