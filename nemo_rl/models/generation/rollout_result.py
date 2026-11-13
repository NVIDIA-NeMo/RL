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

"""One rollout's result and its JSON wire form, for every generation backend.

An engine served through Dynamo sends the result to the driver in
``engine_data``; the vLLM worker builds it with ``request_pipeline.rollout_result``.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
)


@dataclass
class RolloutResult:
    """One rollout's outcome, independent of where its output is assembled."""

    token_ids: list[int]
    # The sampled token's log probability at each position; 0.0 where vLLM
    # reported none.
    logprobs: list[float]
    # The engine stopped at max_tokens.
    truncated: bool
    # [prompt + generated, num_moe_layers, topk], aligned for router replay.
    routed_experts: Optional[torch.Tensor]
    # missing/expected/actual route counts; set when router replay is on.
    r3_stats: Optional[dict[str, int]]

    @classmethod
    def nothing_generated(cls) -> "RolloutResult":
        """The result when no new token fits: the prompt alone, untruncated."""
        return cls([], [], False, None, None)


def rollout_result_to_wire(result: RolloutResult) -> dict[str, Any]:
    """JSON-safe form of ``result``, for engines that answer over HTTP."""
    return {
        "token_ids": result.token_ids,
        # Strict JSON has no NaN or infinity, and Dynamo's frontend turns them
        # into null; send them as the strings float() reads back.
        "logprobs": [lp if math.isfinite(lp) else str(lp) for lp in result.logprobs],
        "truncated": result.truncated,
        "routed_experts": (
            None
            if result.routed_experts is None
            else encode_routed_experts(result.routed_experts)
        ),
        "r3_stats": result.r3_stats,
    }


def rollout_result_from_wire(wire: dict[str, Any]) -> RolloutResult:
    """Inverse of :func:`rollout_result_to_wire`; routes keep their own dtype."""
    routed_experts = wire["routed_experts"]
    if routed_experts is not None:
        routed_experts = decode_routed_experts(routed_experts, dtype=None)
    return RolloutResult(
        token_ids=list(wire["token_ids"]),
        logprobs=[float(lp) for lp in wire["logprobs"]],
        truncated=bool(wire["truncated"]),
        routed_experts=routed_experts,
        r3_stats=wire["r3_stats"],
    )
