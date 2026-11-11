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
"""Bound temporary memory while restoring precision-aware Adam checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from megatron.core.optimizer import MegatronOptimizer


def _compatible_state(
    optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]
) -> dict[torch.Tensor, dict[str, torch.Tensor]] | None:
    groups = state_dict.get("param_groups")
    states = state_dict.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        return None
    if len(groups) != len(optimizer.param_groups):
        return None
    restored = {}
    seen = set()
    for saved, current in zip(groups, optimizer.param_groups):
        if not isinstance(saved, dict):
            return None
        ids = saved.get("params")
        if not isinstance(ids, list) or len(ids) != len(current["params"]):
            return None
        for identity, param in zip(ids, current["params"]):
            if not isinstance(identity, int) or identity in seen:
                return None
            seen.add(identity)
            values = states.get(identity)
            if not isinstance(values, dict) or set(values) != {
                "exp_avg",
                "exp_avg_sq",
                "master_param",
            }:
                return None
            for name, value in values.items():
                expected = torch.int16 if name == "master_param" else torch.float32
                if type(value) is not torch.Tensor or value.dtype != expected:
                    return None
                if (
                    value.device != param.device
                    or value.shape != param.shape
                    or not value.is_contiguous()
                    or value.requires_grad
                ):
                    return None
            restored[param] = values
    if set(states) != seen:
        return None
    return restored


@contextmanager
def precision_aware_state_loading(
    optimizer: MegatronOptimizer | None,
) -> Iterator[None]:
    """Reuse compatible FP32 moments and BF16 remainder buffers during resume.

    Transformer Engine first casts a complete state dictionary to parameter dtype,
    then replaces it with separately allocated state. For plain BF16 parameters
    with FP32 moments and integer master remainders, the incoming tensors already
    have the final representation. Preserve them while PyTorch restores and
    validates parameter-group metadata. Other representations retain TE's loader.

    Eligible Megatron wrappers expose non-FSDP DDP configuration. The load consumes
    tensor storage by reference: containers are not mutated, but subsequent updates
    change tensors also referenced by the in-memory input. Callers must not retain
    that input as an immutable checkpoint copy.
    """
    if optimizer is None or getattr(optimizer, "is_stub_optimizer", False):
        yield
        return
    # TE is only available in the Megatron worker environment.
    from transformer_engine.pytorch.optimizers import FusedAdam

    handles = []
    try:
        for distributed in getattr(optimizer, "chained_optimizers", [optimizer]):
            if getattr(distributed, "is_stub_optimizer", False):
                continue
            config = distributed.config
            inner = distributed.optimizer
            if not config.use_precision_aware_optimizer or config.optimizer_cpu_offload:
                continue
            ddp_config = getattr(distributed, "ddp_config", None)
            if (
                type(inner) is not FusedAdam
                or ddp_config is None
                or ddp_config.use_megatron_fsdp
            ):
                continue
            if not inner.master_weights or not inner.store_param_remainders:
                continue
            if inner.name_to_dtype_map != dict.fromkeys(
                ("exp_avg", "exp_avg_sq", "master_param"), torch.float32
            ):
                continue
            if (
                inner._optimizer_load_state_dict_pre_hooks
                or inner._optimizer_load_state_dict_post_hooks
            ):
                continue
            params = [
                param for group in inner.param_groups for param in group["params"]
            ]
            if not params or any(
                type(param) not in (torch.Tensor, torch.nn.Parameter)
                or param.dtype != torch.bfloat16
                for param in params
            ):
                continue
            if "load_state_dict" in inner.__dict__:
                continue
            original = inner.load_state_dict

            def load_state(
                target: torch.optim.Optimizer,
                state_dict: dict[str, Any],
                *,
                fallback: Callable[[dict[str, Any]], None] = original,
            ) -> None:
                restored = _compatible_state(target, state_dict)
                if (
                    restored is None
                    or target._optimizer_load_state_dict_pre_hooks
                    or target._optimizer_load_state_dict_post_hooks
                ):
                    fallback(state_dict)
                    return
                # Use a new dictionary: the checkpoint's tensors and metadata
                # must remain intact for later distributed-checkpoint consumers.
                torch.optim.Optimizer.load_state_dict(
                    target, {**state_dict, "state": {}}
                )
                target.state.update(
                    {param: dict(values) for param, values in restored.items()}
                )
                logger.info(
                    "Reused %d precision-aware Adam state buffers while loading",
                    len(restored),
                )

            handles.append(inner)
            inner.load_state_dict = MethodType(load_state, inner)
        yield
    finally:
        for inner in reversed(handles):
            del inner.load_state_dict
