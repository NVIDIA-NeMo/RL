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

"""Cross-repository context-compaction contract tests.

These tests deliberately cross the real Gym response schema, JSON transport,
NeMo-RL postprocessor, trace builder, and serialized-bundle validator. The
generation server is mocked; prompt correctness against a real chat template
is covered by the separate live Nemotron Omni validation.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request, Response
import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.visual_history import (
    CompactionScheduleConfig,
    HistoryPolicyConfig,
    RecencyHistoryPolicyConfig,
    VisualHistoryConfig,
)
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.environments.nemo_gym_trace import validate_rollout_trace_bundle
from responses_api_agents.scripted_multimodal_agent.app import (
    ScriptedMultimodalAgent,
    ScriptedMultimodalAgentConfig,
    ScriptedMultimodalAgentRunRequest,
    scripted_observations,
)

pytestmark = pytest.mark.asyncio


class _Tokenizer:
    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        return [" ".join(map(str, token_ids)) for token_ids in batch]


class _MockNemoGymActor:
    cfg: dict[str, Any] = {}


def _http_response(payload: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.ok = True
    response.cookies = {}
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def _model_response(
    turn_index: int,
    *,
    prompt_token_ids: list[int],
    generation_token_id: int,
) -> dict[str, Any]:
    return {
        "id": f"response-{turn_index}",
        "created_at": 1.0,
        "model": "dummy-model",
        "object": "response",
        "output": [
            {
                "id": f"assistant-{turn_index}",
                "content": [
                    {
                        "annotations": [],
                        "text": f"assistant turn {turn_index}",
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": [generation_token_id],
                "generation_log_probs": [-0.1 - turn_index],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _prefix_consistent_model_calls(num_turns: int, *, compact: bool):
    turn_index = 0
    previous_call_tokens: list[int] = []

    def respond(*, url_path: str, json: Any, **_: Any) -> MagicMock:
        nonlocal previous_call_tokens, turn_index
        assert url_path == "/v1/responses"
        assert turn_index < num_turns
        required_prefix = list(json.required_prefix_token_ids or [])
        if compact:
            prompt_prefix = required_prefix
        else:
            # In the legacy append-only route the model server tokenizes the
            # complete accumulated conversation on every call. Simulate that
            # cumulative prompt even though Gym does not send an explicit
            # required-prefix constraint in shadow mode.
            prompt_prefix = previous_call_tokens
        generation_token_id = 2000 + turn_index
        prompt_token_ids = [*prompt_prefix, 1000 + turn_index]
        response = _http_response(
            _model_response(
                turn_index,
                prompt_token_ids=prompt_token_ids,
                generation_token_id=generation_token_id,
            )
        )
        previous_call_tokens = [*prompt_token_ids, generation_token_id]
        turn_index += 1
        return response

    return respond


def _agent_config(
    *,
    compact: bool,
    reverse_ordered_pair: bool = False,
) -> ScriptedMultimodalAgentConfig:
    config = ScriptedMultimodalAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="scripted_multimodal_agent",
        model_server=ModelServerRef(
            type="responses_api_models",
            name="model",
        ),
        fixture="media_contract",
        reverse_ordered_pair=reverse_ordered_pair,
    )
    if compact:
        config.visual_history = VisualHistoryConfig(
            enabled=True,
            shadow_only=False,
            policy=HistoryPolicyConfig(
                type="recency",
                config=RecencyHistoryPolicyConfig(
                    keep_last_image_groups=1,
                ),
            ),
            schedule=CompactionScheduleConfig(
                type="turn_chunked_recency",
                actions_per_chunk=2,
            ),
        )
    else:
        config.visual_history = VisualHistoryConfig(
            enabled=True,
            shadow_only=True,
            policy=HistoryPolicyConfig(type="identity"),
            schedule=CompactionScheduleConfig(
                type="rolling_recency",
                actions_per_chunk=1,
            ),
        )
    return config


async def _serialized_gym_result(
    *,
    compact: bool,
    rollout_index: int = 0,
    reverse_ordered_pair: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _agent_config(
        compact=compact,
        reverse_ordered_pair=reverse_ordered_pair,
    )
    group_id = "bridge-group"
    task_id = f"bridge-task-{rollout_index}"
    rollout_id = f"{group_id}:rollout-{rollout_index:06d}:attempt-000000"

    model_client = MagicMock(spec=ServerClient)
    model_client.post.side_effect = _prefix_consistent_model_calls(
        config.num_turns,
        compact=compact,
    )
    model_facing_agent = ScriptedMultimodalAgent(
        config=config,
        server_client=model_client,
    )
    model_request = MagicMock(spec=Request)
    model_request.cookies = {"session": rollout_id}
    response = await model_facing_agent.responses(
        request=model_request,
        response=Response(),
        body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
    )

    run_client = MagicMock(spec=ServerClient)
    run_client.post.return_value = _http_response(response.model_dump(mode="json"))
    run_agent = ScriptedMultimodalAgent(
        config=config,
        server_client=run_client,
    )
    run_request = MagicMock(spec=Request)
    run_request.cookies = {}
    verify_response = await run_agent.run(
        request=run_request,
        body=ScriptedMultimodalAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input="initial text"
            ),
            context_compaction_rollout_id=rollout_id,
            context_compaction_group_id=group_id,
            context_compaction_task_id=task_id,
            context_compaction_rollout_index=rollout_index,
            context_compaction_attempt_index=0,
        ),
    )

    # Exercise the actual HTTP representation rather than passing Pydantic
    # instances directly across the repository boundary.
    serialized_result = json.loads(verify_response.model_dump_json())
    row = {
        "_rowidx": rollout_index,
        "context_compaction_contract_version": 2,
        "context_compaction_group_id": group_id,
        "context_compaction_task_id": task_id,
        "context_compaction_rollout_index": rollout_index,
        "context_compaction_attempt_index": 0,
        "context_compaction_rollout_id": rollout_id,
    }
    return row, serialized_result


def _postprocess(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    generation_only: bool,
) -> dict[str, Any]:
    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    return postprocess(
        _MockNemoGymActor(),
        row,
        result,
        _Tokenizer(),
        generation_only=generation_only,
    )


@pytest.mark.parametrize(
    ("compact", "expected_turns"),
    [
        (False, [[1, 2, 3, 4, 5]]),
        (True, [[1, 2], [3, 4], [5]]),
    ],
)
async def test_actual_gym_json_crosses_actual_nemo_rl_postprocessor(
    compact: bool,
    expected_turns: list[list[int]],
) -> None:
    row, gym_result = await _serialized_gym_result(compact=compact)
    media_assets = gym_result["response"]["media_assets"]

    normalized = _postprocess(
        row,
        gym_result,
        generation_only=compact,
    )
    bundle = normalized["rollout_trace_bundle"]

    assert bundle["schema_version"] == 3
    assert [
        trace["source_turn_ids"] for trace in bundle["physical_traces"]
    ] == expected_turns
    assert len(normalized["physical_message_logs"]) == len(expected_turns)
    assert validate_rollout_trace_bundle(
        json.loads(json.dumps(bundle)),
        media_assets=json.loads(json.dumps(media_assets)),
        strict=compact,
    ) == {
        "model_call_count": 5,
        "physical_trace_count": len(expected_turns),
        "sampled_token_count": 5,
        "eligible_trainable_token_count": 5,
        "media_occurrence_count": sum(
            len(call["media_ids"]) for call in bundle["model_calls"]
        ),
    }
    if compact:
        full_result = normalized["full_result"]
        assert full_result["nemo_rl_trace_bundle"] == bundle
        assert (
            full_result["context_compaction_gym_http_bytes"]
            > (full_result["context_compaction_ray_env_extras_bytes"])
        )
        assert set(full_result["response"]).isdisjoint(
            {
                "agent_input",
                "seed_obs",
                "media_assets",
                "completion_evidence",
                "final_policy_decision",
                "lineage_deltas",
            }
        )
        assert not any(
            value.startswith("data:image") for value in _walk_strings(full_result)
        )


async def test_actual_bridge_rejects_output_evidence_mismatch() -> None:
    row, gym_result = await _serialized_gym_result(compact=True)
    first_output = next(
        item
        for item in gym_result["response"]["output"]
        if "generation_token_ids" in item
    )
    first_output["generation_token_ids"] = [999999]

    with pytest.raises(
        ValueError,
        match="model-call metadata digest does not match",
    ):
        _postprocess(
            row,
            gym_result,
            generation_only=True,
        )


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


async def test_actual_bridge_preserves_reversed_same_shape_media() -> None:
    row, gym_result = await _serialized_gym_result(
        compact=True,
        reverse_ordered_pair=True,
    )
    expected_turn_three_urls = [
        part["image_url"]
        for part in scripted_observations(
            reverse_ordered_pair=True,
        )[2].model_dump()["content"]
        if part["type"] == "input_image"
    ]
    media_assets = gym_result["response"]["media_assets"]

    normalized = _postprocess(
        row,
        gym_result,
        generation_only=True,
    )
    turn_three = normalized["rollout_trace_bundle"]["model_calls"][2]
    observed_urls = [
        media_assets[media_id]["source_part"]["image_url"]
        for media_id in turn_three["media_ids"][-2:]
    ]

    assert len(expected_turn_three_urls) == 2
    assert observed_urls == expected_turn_three_urls


async def test_actual_bridge_keeps_two_rollouts_isolated() -> None:
    normalized_results = []
    for rollout_index in (0, 1):
        row, gym_result = await _serialized_gym_result(
            compact=True,
            rollout_index=rollout_index,
        )
        normalized_results.append(
            _postprocess(
                row,
                gym_result,
                generation_only=True,
            )
        )

    bundles = [result["rollout_trace_bundle"] for result in normalized_results]
    assert bundles[0]["rollout_id"] != bundles[1]["rollout_id"]
    assert bundles[0]["source_row_index"] == 0
    assert bundles[1]["source_row_index"] == 1
    assert [
        [trace["source_turn_ids"] for trace in bundle["physical_traces"]]
        for bundle in bundles
    ] == [
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5]],
    ]
