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
"""The rollout result pipeline behind generate_async."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm.request_pipeline import (
    RolloutResult,
    rollout_output,
    rollout_result,
    rollout_result_from_wire,
    rollout_result_to_wire,
)
from nemo_rl.models.generation.vllm.utils import R3_MISSING_ROUTE_SENTINEL

LAYERS, TOPK = 2, 2


def _logprob(value: float):
    return SimpleNamespace(logprob=value)


def _final_output(
    generated: list[int],
    *,
    finish_reason: str = "stop",
    prompt_routes: int | None = None,
    generated_routes: int | None = None,
):
    def routes(count):
        if count is None:
            return None
        return np.arange(count * LAYERS * TOPK, dtype=np.int16).reshape(
            count, LAYERS, TOPK
        )

    return SimpleNamespace(
        prompt_routed_experts=routes(prompt_routes),
        num_cached_tokens=0,
        outputs=[
            SimpleNamespace(
                token_ids=generated,
                # The second token has no reported logprob.
                logprobs=[{generated[0]: _logprob(-0.5)}, {}]
                + [{t: _logprob(-1.0)} for t in generated[2:]],
                finish_reason=finish_reason,
                routed_experts=routes(generated_routes),
            )
        ],
    )


def _result(final_output, *, input_length=3, r3=False) -> RolloutResult:
    return rollout_result(
        final_output,
        input_length=input_length,
        return_routed_experts=r3,
        routed_experts_dtype=torch.int16,
        device=torch.device("cpu"),
    )


def test_output_layout_matches_generate_async() -> None:
    result = _result(_final_output([7, 8, 9], finish_reason="length"))
    output = rollout_output(
        result, input_ids=torch.tensor([1, 2, 3, 0, 0]), input_length=3, pad_token_id=0
    )

    assert output["output_ids"].tolist() == [[1, 2, 3, 7, 8, 9]]
    # A position without a reported logprob stays 0.0, aligned with its token.
    assert output["logprobs"].tolist() == [[0.0, 0.0, 0.0, -0.5, 0.0, -1.0]]
    assert output["generation_lengths"].tolist() == [3]
    assert output["unpadded_sequence_lengths"].tolist() == [6]
    assert output["truncated"].tolist() == [True]
    assert "routed_experts" not in output


def test_nothing_generated_returns_the_prompt_untruncated() -> None:
    output = rollout_output(
        RolloutResult.nothing_generated(),
        input_ids=torch.tensor([1, 2, 3, 0]),
        input_length=3,
        pad_token_id=0,
    )

    assert output["output_ids"].tolist() == [[1, 2, 3]]
    assert output["logprobs"].tolist() == [[0.0, 0.0, 0.0]]
    assert output["generation_lengths"].tolist() == [0]
    assert output["truncated"].tolist() == [False]


def test_router_replay_routes_are_aligned_and_counted() -> None:
    # 3 prompt + 3 generated tokens need 5 routes; vLLM reported 3 + 1.
    result = _result(
        _final_output([7, 8, 9], prompt_routes=3, generated_routes=1), r3=True
    )
    output = rollout_output(
        result, input_ids=torch.tensor([1, 2, 3]), input_length=3, pad_token_id=0
    )

    routed = output["routed_experts"]
    assert routed.shape == (1, 6, LAYERS, TOPK) and routed.dtype == torch.int16
    assert (routed[0, 4:] == R3_MISSING_ROUTE_SENTINEL).all()
    assert output["r3_routed_experts_expected_routes"].tolist() == [5]
    assert output["r3_routed_experts_actual_routes"].tolist() == [4]
    assert output["r3_routed_experts_missing_routes"].tolist() == [1]


def test_router_replay_requires_routes() -> None:
    with pytest.raises(RuntimeError, match="did not include routed_experts"):
        _result(_final_output([7, 8, 9]), r3=True)


def test_wire_round_trip_preserves_the_output() -> None:
    result = _result(
        _final_output([7, 8, 9], prompt_routes=3, generated_routes=3), r3=True
    )
    wire = rollout_result_to_wire(result)
    input_ids = torch.tensor([1, 2, 3])

    direct = rollout_output(result, input_ids=input_ids, input_length=3, pad_token_id=0)
    shipped = rollout_output(
        rollout_result_from_wire(wire),
        input_ids=input_ids,
        input_length=3,
        pad_token_id=0,
    )

    assert direct.keys() == shipped.keys()
    for key in direct:
        assert torch.equal(direct[key], shipped[key]), key
