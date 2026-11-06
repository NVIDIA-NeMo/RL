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

"""Lifecycle helpers for Automodel's flex MoE token-dispatch managers."""

from typing import Any

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
)
from torch import nn

# Per-dispatch tensors that Automodel's flex token-dispatch managers (DeepEP,
# HybridEP, UCCL-EP) keep as attributes. Each dispatch reassigns them, so none
# is needed across steps.
# TODO(https://github.com/NVIDIA-NeMo/Automodel/issues/4401): drop this workaround
# once Automodel releases this state itself.
_MOE_DISPATCH_STATE_ATTRS = (
    "token_probs",
    "token_indices",
    "routing_map",
    "handle",
    "tokens_per_expert",
    "num_permuted_tokens",
    "dispatched_indices",
    "dispatched_probs",
    "dispatched_routing_map",
    "reversed_mapping_for_combine",
)
# HybridEP pads each dispatch to the EP group's maximum token count rounded up
# to this multiple (``_HYBRIDEP_TOKEN_ALIGNMENT`` in Automodel's token dispatcher).
_HYBRIDEP_TOKEN_ALIGNMENT = 4


def _dispatch_managers(model: nn.Module) -> list[tuple[nn.Module, Any]]:
    """Return ``(experts module, manager)`` for each distinct flex dispatch manager."""
    managers = {}
    for module in model.modules():
        # MoEFlexTokenDispatcher is a plain object held by the experts module,
        # not a submodule, so model.modules() does not yield it.
        dispatcher = getattr(module, "token_dispatcher", None)
        if isinstance(dispatcher, MoEFlexTokenDispatcher):
            manager = getattr(dispatcher, "_comm_manager", None)
            if manager is not None:
                managers.setdefault(id(manager), (module, manager))
    return list(managers.values())


def release_moe_dispatcher_state(model: nn.Module) -> int:
    """Drop the per-dispatch tensors cached on the model's MoE dispatch managers.

    The managers keep the last dispatch's routing tensors as attributes;
    ``combine`` clears only ``handle`` (and HybridEP's ``num_permuted_tokens``).
    After a backward with activation checkpointing, a manager's last dispatch
    is a recompute, so ``token_probs`` and ``dispatched_probs`` carry a
    ``grad_fn`` that keeps that recompute graph and its saved activations
    alive. The live blocks pin cached allocator segments, so
    ``torch.cuda.empty_cache()`` cannot return them before colocated
    generation wakes up. Ported from NeMo RL #4756 (r0.8.0).

    Args:
        model: Model whose ``token_dispatcher`` attributes are scanned. A
            manager shared across layers is cleared once.

    Returns:
        The number of attributes set to ``None``.
    """
    cleared = 0
    for _, manager in _dispatch_managers(model):
        for attr in _MOE_DISPATCH_STATE_ATTRS:
            if getattr(manager, attr, None) is not None:
                setattr(manager, attr, None)
                cleared += 1
    return cleared


def presize_hybrid_ep_buffer(model: nn.Module, max_tokens_per_rank: int) -> bool:
    """Allocate the HybridEP buffer for the largest micro-batch before the first dispatch.

    Automodel sizes the shared HybridEP buffer from the first dispatch and
    reallocates it when a later micro-batch is larger. Reallocation is a
    collective that rebuilds the communication buffers mid-step; across nodes it
    also recreates the RDMA queue pairs, which has failed with
    ``Failed to create ...th QP``. Allocating once for the configured maximum
    keeps every later dispatch within the buffer.

    Must be called on every rank of the expert-parallel group, after the model's
    token dispatchers are initialized and before its first forward.

    Args:
        model: Model whose MoE layers use the HybridEP dispatcher.
        max_tokens_per_rank: Largest number of tokens one rank dispatches in a
            single forward (the micro-batch token budget).

    Returns:
        True if the buffer was allocated here, False if the model has no
        HybridEP dispatcher.
    """
    if max_tokens_per_rank <= 0:
        raise ValueError(
            f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}"
        )
    from nemo_automodel.components.moe.megatron import fused_a2a
    from nemo_automodel.components.moe.megatron.token_dispatcher import (
        _HybridEPManager,
    )

    hybrid = [
        (experts, manager)
        for experts, manager in _dispatch_managers(model)
        if isinstance(manager, _HybridEPManager)
    ]
    if not hybrid:
        return False
    if fused_a2a._hybrid_ep_buffer is not None:
        raise RuntimeError("The HybridEP buffer is already allocated")
    # All HybridEP managers share Automodel's single module-level buffer, so
    # their buffer-shaping arguments must agree.
    shapes = {
        (
            manager.group,
            experts.config.expert_dim,
            manager.num_local_experts,
            manager.moe_hybridep_num_sms,
        )
        for experts, manager in hybrid
    }
    if len(shapes) != 1:
        raise ValueError(f"HybridEP managers disagree on the buffer shape: {shapes}")
    group, hidden_dim, num_local_experts, num_sms = shapes.pop()
    num_tokens = (
        -(-max_tokens_per_rank // _HYBRIDEP_TOKEN_ALIGNMENT) * _HYBRIDEP_TOKEN_ALIGNMENT
    )
    fused_a2a.init_hybrid_ep_buffer(
        group,
        hidden_dim,
        num_tokens,
        num_local_experts,
        num_sms,
        num_sms,
        False,
    )
    return True
