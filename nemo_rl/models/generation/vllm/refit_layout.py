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


class VllmWeightLayout(TypedDict):
    expert_params: dict[str, VllmExpertParamLayout]
    missing_weight_prefixes: list[str]


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
_HF_PROJECTION_SHARDS: dict[str, Literal["w1", "w2", "w3"]] = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def parse_hf_expert_weight(name: str) -> HfExpertWeight | None:
    match = _HF_EXPERT_WEIGHT_RE.match(name)
    if match is None:
        return None

    projection = match.group("projection")
    shard_id = _HF_PROJECTION_SHARDS[projection]
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
    """Return the destination-local weight, or ``None`` if not owned.

    Pipeline stages omit complete parameter prefixes. Within an owned stage,
    tensor-parallel MoE layers shard every expert tensor while expert-parallel
    layers place complete experts on selected ranks.
    """
    if any(
        name.startswith(prefix) for prefix in target_layout["missing_weight_prefixes"]
    ):
        return None

    expert_weight = parse_hf_expert_weight(name)
    if expert_weight is None:
        return tensor

    param_layout = target_layout["expert_params"].get(expert_weight.parameter_name)
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
