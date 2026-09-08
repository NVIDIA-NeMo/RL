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

import pytest

from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _warn_on_missing_routed_experts,
)


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _routes(num_tokens: int) -> list[list[list[int]]]:
    return [[[token_idx, token_idx + 100]] for token_idx in range(num_tokens)]


def test_nemo_gym_postprocess_slices_routed_experts():
    first_turn_routes = _routes(3)
    first_turn_routes[-1] = [[0, 1]]
    second_turn_routes = _routes(7)
    second_turn_routes[2] = [[30, 31]]
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": first_turn_routes,
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                    "routed_experts": second_turn_routes,
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    message_log = result["message_log"]
    assert message_log[0]["token_ids"].tolist() == [1, 2]
    assert message_log[0]["routed_experts"].tolist() == first_turn_routes[:2]
    assert message_log[1]["token_ids"].tolist() == [3]
    assert message_log[1]["routed_experts"].tolist() == second_turn_routes[2:3]
    assert message_log[2]["token_ids"].tolist() == [4, 5]
    assert message_log[2]["routed_experts"].tolist() == second_turn_routes[3:5]
    assert message_log[3]["token_ids"].tolist() == [6, 7]
    assert message_log[3]["routed_experts"].tolist() == second_turn_routes[5:7]


def test_missing_routed_experts_warning_is_nonfatal_and_aggregates_rollout(capsys):
    output_items = [
        {
            "generation_token_ids": [1],
            "generation_log_probs": [-0.1],
            "routed_experts": _routes(1),
        },
        {"type": "function_call_output", "output": "tool result"},
        {
            "generation_token_ids": [2],
            "generation_log_probs": [-0.2],
        },
        {
            "generation_token_ids": [],
            "generation_log_probs": [],
        },
        {
            "generation_token_ids": [3],
            "generation_log_probs": [-0.3],
            "routed_experts": None,
        },
    ]
    row = {
        "_rowidx": 17,
        "_ng_task_index": 42,
        "_ng_rollout_index": 3,
        "_ng_attempt_index": 1,
        "_ng_target_weight_version": 9,
        "agent_ref": {"name": "test_agent"},
        "dataset": "test_dataset",
        "metadata": {"uuid": "test-uuid"},
    }

    coverage = _warn_on_missing_routed_experts(
        output_items,
        row,
        required=True,
    )

    assert coverage == {
        "response_item_count": 5,
        "trainable_turn_count": 3,
        "missing_routed_experts_turn_count": 2,
        "missing_routed_experts_turn_indices": [1, 2],
        "missing_routed_experts_response_output_indices": [2, 4],
    }
    trace = capsys.readouterr().out
    assert '"event":"actor_routed_experts_invariant_violation"' in trace
    assert '"level":"warning"' in trace
    assert '"task_index":42' in trace
    assert '"rollout_index":3' in trace
    assert '"target_weight_version":9' in trace
    assert '"agent_name":"test_agent"' in trace
    assert '"missing_routed_experts_turn_indices":[1,2]' in trace
    assert '"missing_routed_experts_response_output_indices":[2,4]' in trace


def test_missing_routed_experts_warning_is_disabled_without_router_replay(capsys):
    coverage = _warn_on_missing_routed_experts(
        [
            {
                "generation_token_ids": [1],
                "generation_log_probs": [-0.1],
            }
        ],
        {},
        required=False,
    )

    assert coverage is None
    assert capsys.readouterr().out == ""


def test_nemo_gym_postprocess_requires_routed_experts_when_configured(capsys):
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    nemo_gym_row = {
        "_ng_task_index": 42,
        "_ng_rollout_index": 3,
        "agent_ref": {"name": "test_agent"},
    }
    with pytest.raises(
        ValueError,
        match=(
            "requires NeMo Gym output items.*response_output_index=0, "
            "trainable_turn_index=0"
        ),
    ):
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_row, nemo_gym_result, _Tokenizer()
        )

    trace = capsys.readouterr().out
    assert '"event":"actor_routed_experts_invariant_violation"' in trace
    assert '"task_index":42' in trace
    assert '"missing_routed_experts_turn_count":1' in trace
    assert '"missing_routed_experts_turn_indices":[0]' in trace


def test_nemo_gym_postprocess_casts_routed_experts_to_configured_dtype():
    import torch

    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": _routes(3),
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True, "routed_experts_dtype": "int8"}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    for message in result["message_log"]:
        if "routed_experts" in message:
            assert message["routed_experts"].dtype == torch.int8
