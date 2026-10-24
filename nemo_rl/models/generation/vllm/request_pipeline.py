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

"""Turn a finished vLLM rollout into NeMo RL's per-sample generation output.

Every path that serves a rollout from a NeMo RL vLLM engine goes through here:
the worker's own ``generate_async``, and Dynamo serving (the engine builds the
result, the driver assembles the output). A rollout feature added here, such
as router replay (R3), therefore behaves the same on every path.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationOutputSpec
from nemo_rl.models.generation.vllm.utils import pad_and_align_routed_expert_indices
from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
    routed_experts_envelope_dtype,
)

LOGGER = logging.getLogger(__name__)


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


def selected_logprobs(completion: Any) -> list[float]:
    """The sampled token's log probability at each position of a completion."""
    by_position = completion.logprobs or ()
    logprobs = []
    for idx, token_id in enumerate(completion.token_ids):
        candidates = by_position[idx] if idx < len(by_position) else None
        entry = candidates.get(token_id) if candidates else None
        logprobs.append(0.0 if entry is None else float(entry.logprob))
    return logprobs


def rollout_result(
    final_output: Any,
    *,
    input_length: int,
    return_routed_experts: bool,
    routed_experts_dtype: torch.dtype,
    device: torch.device,
    sample_idx: int = 0,
) -> RolloutResult:
    """Build the result of a finished rollout from vLLM's final RequestOutput.

    Args:
        final_output: The request's last (cumulative) ``RequestOutput``.
        input_length: Prompt length in tokens.
        return_routed_experts: Router replay is on, so routes are required.
        routed_experts_dtype: The dtype router replay consumes.
        device: Where to place the routed-experts tensor.
        sample_idx: For log messages only.
    """
    completion = final_output.outputs[0]
    token_ids = list(completion.token_ids)
    total_length = input_length + len(token_ids)
    # return_stats=True returns the (routes, stats) pair.
    routed_experts, r3_stats = cast(
        tuple[Optional[torch.Tensor], dict[str, int]],
        pad_and_align_routed_expert_indices(
            final_output,
            completion,
            valid_length=total_length,
            padded_length=total_length,
            device=device,
            require_complete_routed_experts=return_routed_experts,
            return_stats=True,
            routed_experts_dtype=routed_experts_dtype,
        ),
    )
    if return_routed_experts and routed_experts is None:
        raise RuntimeError(
            "vLLM was asked to return routed experts but the generation output "
            "did not include routed_experts."
        )
    if return_routed_experts and r3_stats["missing_routes"] > 0:
        LOGGER.warning(
            "R3 router replay fallback: vLLM returned incomplete "
            "routed_experts for sample_idx=%d, missing_token_routes=%d, "
            "actual_routes=%d, expected_routes=%d. Megatron will use its "
            "own router for those missing token routes.",
            sample_idx,
            r3_stats["missing_routes"],
            r3_stats["actual_routes"],
            r3_stats["expected_routes"],
        )
    return RolloutResult(
        token_ids=token_ids,
        logprobs=selected_logprobs(completion),
        truncated=completion.finish_reason == "length",
        routed_experts=routed_experts,
        r3_stats=r3_stats if return_routed_experts else None,
    )


def rollout_output(
    result: RolloutResult,
    *,
    input_ids: torch.Tensor,
    input_length: int,
    pad_token_id: int,
) -> BatchedDataDict[GenerationOutputSpec]:
    """The single-sample output ``generate_async`` yields for ``result``.

    Args:
        result: The rollout's result.
        input_ids: The sample's (right-padded) prompt row.
        input_length: Prompt length in tokens.
        pad_token_id: The tokenizer's pad token, as ``generate_async`` uses it.
    """
    device = input_ids.device
    generated = result.token_ids
    output_length = input_length + len(generated)
    output_ids = torch.full(
        (output_length,), pad_token_id, dtype=input_ids.dtype, device=device
    )
    output_ids[:input_length] = input_ids[:input_length]
    output_ids[input_length:] = torch.tensor(
        generated, dtype=input_ids.dtype, device=device
    )
    logprobs = torch.zeros((1, output_length), dtype=torch.float32, device=device)
    logprobs[0, input_length:] = torch.tensor(
        result.logprobs, dtype=torch.float32, device=device
    )
    output: dict[str, Any] = {
        "output_ids": output_ids.unsqueeze(0),
        "logprobs": logprobs,
        "generation_lengths": torch.tensor(
            [len(generated)], dtype=torch.long, device=device
        ),
        "unpadded_sequence_lengths": torch.tensor(
            [output_length], dtype=torch.long, device=device
        ),
        "truncated": torch.tensor([result.truncated], dtype=torch.bool, device=device),
    }
    if result.r3_stats is not None:
        for key in ("missing_routes", "expected_routes", "actual_routes"):
            output[f"r3_routed_experts_{key}"] = torch.tensor(
                [result.r3_stats[key]], dtype=torch.long, device=device
            )
    if result.routed_experts is not None:
        output["routed_experts"] = result.routed_experts.to(device).unsqueeze(0)
    return BatchedDataDict[GenerationOutputSpec](output)


def rollout_result_to_wire(result: RolloutResult) -> dict[str, Any]:
    """JSON-safe form of ``result``, for engines that answer over HTTP."""
    return {
        "token_ids": result.token_ids,
        "logprobs": result.logprobs,
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
        routed_experts = decode_routed_experts(
            routed_experts, routed_experts_envelope_dtype(routed_experts)
        )
    if len(wire["logprobs"]) != len(wire["token_ids"]):
        raise ValueError(
            f"rollout result has {len(wire['logprobs'])} logprobs for "
            f"{len(wire['token_ids'])} tokens"
        )
    return RolloutResult(
        token_ids=list(wire["token_ids"]),
        logprobs=list(wire["logprobs"]),
        truncated=bool(wire["truncated"]),
        routed_experts=routed_experts,
        r3_stats=wire["r3_stats"],
    )
