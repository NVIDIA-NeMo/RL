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
