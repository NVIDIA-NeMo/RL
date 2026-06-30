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

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch


class VllmExpertParamLayout(TypedDict):
    tp_rank: int
    tp_size: int
    local_expert_ids: list[int] | None


VllmWeightLayout = dict[str, VllmExpertParamLayout]


@dataclass(frozen=True)
class HfExpertWeight:
    parameter_name: str
    expert_id: int
    shard_id: Literal["w1", "w2", "w3"]
    tp_shard_dim: int


_HF_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\."
    r"(?P<expert_id>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def parse_hf_expert_weight(name: str) -> HfExpertWeight | None:
    match = _HF_EXPERT_WEIGHT_RE.match(name)
    if match is None:
        return None

    projection = match.group("projection")
    shard_id: Literal["w1", "w2", "w3"] = {
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
    }[projection]
    parameter_leaf = "w2_weight" if shard_id == "w2" else "w13_weight"
    return HfExpertWeight(
        parameter_name=f"{match.group('prefix')}.{parameter_leaf}",
        expert_id=int(match.group("expert_id")),
        shard_id=shard_id,
        tp_shard_dim=1 if shard_id == "w2" else 0,
    )


def select_hf_weight_for_vllm_target(
    name: str,
    tensor: torch.Tensor,
    *,
    target_layout: VllmWeightLayout,
) -> torch.Tensor | None:
    """Return the destination-local expert weight, or ``None`` if not owned.

    vLLM tensor-parallel MoE layers shard every expert tensor. Expert-parallel
    layers instead place complete experts on selected ranks, so their live
    ownership map must be used rather than applying another TP slice.
    """
    expert_weight = parse_hf_expert_weight(name)
    if expert_weight is None:
        return tensor

    param_layout = target_layout.get(expert_weight.parameter_name)
    if param_layout is None:
        # A missing expert parameter belongs to another pipeline stage.
        return None

    local_expert_ids = param_layout["local_expert_ids"]
    if local_expert_ids is not None and expert_weight.expert_id not in local_expert_ids:
        return None

    tp_rank = param_layout["tp_rank"]
    tp_size = param_layout["tp_size"]

    shard_dim = expert_weight.tp_shard_dim
    if tp_size == 1:
        return tensor
    if tensor.shape[shard_dim] % tp_size != 0:
        raise ValueError(
            f"Cannot shard {name} dimension {shard_dim} of size "
            f"{tensor.shape[shard_dim]} across vLLM TP size {tp_size}."
        )

    shard_size = tensor.shape[shard_dim] // tp_size
    return tensor.narrow(shard_dim, tp_rank * shard_size, shard_size).contiguous()
