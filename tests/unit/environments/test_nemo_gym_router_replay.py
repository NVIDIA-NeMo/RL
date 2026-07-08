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

from nemo_rl.environments.nemo_gym import NemoGym


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _routes(num_tokens: int) -> list[list[list[int]]]:
    return [[[token_idx, token_idx + 100]] for token_idx in range(num_tokens)]


def test_nemo_gym_postprocess_slices_routed_experts():
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": _routes(3),
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                    "routed_experts": _routes(7),
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_result, _Tokenizer()
        )
    )

    message_log = result["message_log"]
    assert message_log[0]["token_ids"].tolist() == [1, 2]
    assert message_log[0]["routed_experts"].tolist() == _routes(2)
    assert message_log[1]["token_ids"].tolist() == [3]
    assert message_log[1]["routed_experts"].tolist() == _routes(3)[2:3]
    assert message_log[2]["token_ids"].tolist() == [4, 5]
    assert message_log[2]["routed_experts"].tolist() == _routes(7)[3:5]
    assert message_log[3]["token_ids"].tolist() == [6, 7]
    assert message_log[3]["routed_experts"].tolist() == _routes(7)[5:7]


def test_nemo_gym_postprocess_requires_routed_experts_when_configured():
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

    with pytest.raises(ValueError, match="requires NeMo Gym output items"):
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_result, _Tokenizer()
        )


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
            _MockSelf(), nemo_gym_result, _Tokenizer()
        )
    )

    for message in result["message_log"]:
        if "routed_experts" in message:
            assert message["routed_experts"].dtype == torch.int8


def _routes_from(num_tokens: int, base: int) -> list[list[list[int]]]:
    return [
        [[base + token_idx, base + token_idx + 100]] for token_idx in range(num_tokens)
    ]


def _two_turn_result() -> dict:
    return {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": _routes_from(3, 0),
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                    "routed_experts": _routes_from(7, 1000),
                },
            ]
        },
        "responses_create_params": {"input": []},
    }


class _R3MockSelf:
    cfg = {"require_routed_experts": True}


def _postprocess(result):
    return (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _R3MockSelf(), result, _Tokenizer()
        )
    )


def test_prev_turn_final_token_route_spliced_from_next_turn_prefill():
    message_log = _postprocess(_two_turn_result())["message_log"]

    turn1_assistant = message_log[1]
    # Turn-1 assistant covers global position 2 only (token id 3). Its route row
    # was the placeholder from turn 1; the splice must replace it with turn 2's
    # real prefill row at global position 2.
    assert turn1_assistant["routed_experts"].tolist() == _routes_from(7, 1000)[2:3]
    # Turn-2 assistant (final turn) keeps its own rows untouched — the last
    # position placeholder is unavoidable for the final turn.
    turn2_assistant = message_log[3]
    assert turn2_assistant["routed_experts"].tolist() == _routes_from(7, 1000)[5:7]


def test_single_turn_final_token_route_unchanged():
    result = _two_turn_result()
    result["response"]["output"] = result["response"]["output"][:1]
    message_log = _postprocess(result)["message_log"]
    assert message_log[1]["routed_experts"].tolist() == _routes_from(3, 0)[2:3]


def test_surplus_final_token_route_is_accepted():
    result = _two_turn_result()
    # expected tokens for turn 2 = 7; one surplus row is the legal final-token route
    result["response"]["output"][1]["routed_experts"] = _routes_from(8, 1000)
    _postprocess(result)


def test_more_than_one_surplus_route_row_raises():
    result = _two_turn_result()
    result["response"]["output"][1]["routed_experts"] = _routes_from(9, 1000)
    with pytest.raises(ValueError, match="too many routed_experts rows"):
        _postprocess(result)
