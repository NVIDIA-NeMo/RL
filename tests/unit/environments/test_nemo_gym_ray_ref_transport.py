# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.interfaces import (
    NEMO_GYM_ROLLOUT_INDEX_KEY,
    NEMO_GYM_TARGET_WEIGHT_VERSION_KEY,
    NEMO_GYM_TASK_INDEX_KEY,
)
from nemo_rl.experience.rollouts import _prepare_nemo_gym_rows
from nemo_rl.models.generation.interfaces import GenerationSamplingParams
from nemo_rl.utils.routed_experts_ref import (
    ROUTED_EXPERTS_REF_DTYPE,
    ROUTED_EXPERTS_REF_KEY,
    ROUTED_EXPERTS_REF_SCHEMA,
)


def _full_ref() -> dict:
    return {
        "schema": ROUTED_EXPERTS_REF_SCHEMA,
        "store": "store-a",
        "store_instance_id": "instance-a",
        "request_id": "request-a",
        "key": ROUTED_EXPERTS_REF_KEY,
        "task_index": 7,
        "rollout_index": 2,
        "target_weight_version": 4,
        "offset": 0,
        "length": 3,
        "shape": [3, 2, 2],
        "dtype": ROUTED_EXPERTS_REF_DTYPE,
    }


def test_prepare_nemo_gym_rows_stamps_rollout_and_target_identity():
    rows = [
        {
            NEMO_GYM_TASK_INDEX_KEY: 11,
            "responses_create_params": {},
        },
        {
            NEMO_GYM_TASK_INDEX_KEY: 11,
            "responses_create_params": {},
        },
    ]

    _prepare_nemo_gym_rows(
        rows,
        {
            "max_new_tokens": 16,
        },
        GenerationSamplingParams(temperature=0.7, top_p=0.9, top_k=None),
        target_weight_version=27,
    )

    assert [row[NEMO_GYM_ROLLOUT_INDEX_KEY] for row in rows] == [0, 1]
    assert [row[NEMO_GYM_TARGET_WEIGHT_VERSION_KEY] for row in rows] == [27, 27]


def test_nemo_gym_postprocess_keeps_two_views_of_one_ray_object():
    full_ref = _full_ref()
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": full_ref,
                }
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _Tokenizer:
        def batch_decode(self, batch):
            return ["decoded"] * len(batch)

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            {},
            nemo_gym_result,
            _Tokenizer(),
        )
    )

    prompt_ref = result["message_log"][0]["routed_experts"]
    generation_ref = result["message_log"][1]["routed_experts"]
    assert prompt_ref == full_ref | {"offset": 0, "length": 2}
    assert generation_ref == full_ref | {"offset": 2, "length": 1}


def test_nemo_gym_postprocess_splices_ray_routes_across_multiple_turns():
    first_ref = _full_ref() | {
        "request_id": "request-a",
        "length": 4,
        "shape": [4, 2, 2],
    }
    second_ref = first_ref | {
        "request_id": "request-b",
        "length": 7,
        "shape": [7, 2, 2],
    }
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                    "routed_experts": first_ref,
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.3, -0.4],
                    "routed_experts": second_ref,
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _Tokenizer:
        def batch_decode(self, batch):
            return ["decoded"] * len(batch)

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    message_log = result["message_log"]
    assert message_log[0]["routed_experts"] == first_ref | {
        "offset": 0,
        "length": 2,
    }
    # Token 4's padded decode route is replaced by its real route from the
    # second request's prefill without resolving either Ray object.
    assert message_log[1]["routed_experts"] == [
        first_ref | {"offset": 2, "length": 1},
        second_ref | {"offset": 3, "length": 1},
    ]
    assert message_log[2]["routed_experts"] == second_ref | {
        "offset": 4,
        "length": 1,
    }
    assert message_log[3]["routed_experts"] == second_ref | {
        "offset": 5,
        "length": 2,
    }
