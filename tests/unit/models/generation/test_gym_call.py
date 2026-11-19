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
"""NeMo-Gym's Dynamo dialect packs and lifts the fields NeMo RL's call models define."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.nemo_gym
pytest.importorskip("nemo_gym.token_id_capture.external_capture")

from nemo_gym.openai_utils import (  # noqa: E402
    TOKEN_METADATA_FIELDS,
    NeMoGymAsyncOpenAI,
)
from nemo_gym.token_id_capture import (  # noqa: E402
    NG_CAPTURE_FIELD,
    NG_COMMIT_COORDS_FIELD,
)
from nemo_gym.token_id_capture.external_capture import (  # noqa: E402
    DYNAMO_GYM_EXTRAS_ANNOTATION_PREFIX,
    add_dynamo_gym_extras,
    pop_dynamo_call_result,
)
from responses_api_models.vllm_model.app import VLLMModel  # noqa: E402
from responses_api_models.vllm_model.tests import (  # noqa: E402
    test_framework_context as gym_harness,
)

from nemo_rl.models.generation.gym_call import (  # noqa: E402
    GYM_EXTRAS_ANNOTATION_PREFIX,
    GYM_RESULT_NVEXT_KEY,
    GymCallExtras,
    GymCallResult,
)


def test_gyms_extras_are_the_ones_nemo_rl_reads() -> None:
    assert DYNAMO_GYM_EXTRAS_ANNOTATION_PREFIX == GYM_EXTRAS_ANNOTATION_PREFIX
    # Gym's own request preparation: the last echoed turn becomes the prefix.
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "a",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
                "generation_log_probs": [-0.5],
            },
        ],
        "nvext": {"annotations": ["other"]},
    }
    VLLMModel._prepare_dynamo_request(
        SimpleNamespace(_external_capture_handler=None), body
    )
    # The capture handler's admission and the measure flag ride in the same object.
    add_dynamo_gym_extras(
        body, {NG_CAPTURE_FIELD: {"capture_key": "r"}, "ng_measure": True}
    )

    # Unknown names are refused, so a field Gym renames fails here.
    extras = GymCallExtras.from_annotations(body["nvext"]["annotations"])
    assert extras == GymCallExtras(
        ng_capture={"capture_key": "r"},
        required_prefix_token_ids=[1, 2, 3],
        ng_measure=True,
    )
    assert body["nvext"]["annotations"][0] == "other"
    assert "prompt_token_ids" not in body["messages"][1]


def test_gym_lifts_the_call_result_nemo_rl_returns() -> None:
    result = GymCallResult(
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.5],
        routed_experts="nrlre1:int16:3x1x1:AAABAAIA",
        ng_commit_coords={"staging_key": "r/c1"},
    ).model_dump(exclude_none=True)

    assert pop_dynamo_call_result({"nvext": {GYM_RESULT_NVEXT_KEY: result}}) == result
    # The fields Gym copies onto the message, and its capture acknowledgement.
    assert TOKEN_METADATA_FIELDS | {NG_COMMIT_COORDS_FIELD} <= result.keys()


def test_gyms_measure_call_is_the_one_nemo_rl_answers(monkeypatch, tmp_path) -> None:
    # Gym's own /context/measure, in its Dynamo dialect.
    harness = gym_harness.make_capture_harness(monkeypatch, tmp_path, dynamo=True)
    first = gym_harness.assert_clean(harness.post("attempt", gym_harness.HISTORY))
    seen = []

    async def frontend(client, *, extra_headers=None, **body):
        # Unknown names are refused, so a measure flag Gym renames fails here.
        seen.append(GymCallExtras.from_annotations(body["nvext"]["annotations"]))
        result = GymCallResult(prompt_token_count=19, ng_context_measured=True)
        return {
            "id": "chatcmpl-measure",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "nvext": {GYM_RESULT_NVEXT_KEY: result.model_dump(exclude_none=True)},
        }

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", frontend)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/context/measure",
        json={
            "input": gym_harness.HISTORY
            + first["output"]
            + [{"role": "user", "content": "next"}]
        },
    )

    # Gym read the result back by NeMo RL's names; a renamed one answers 502.
    assert response.status_code == 200, response.text
    assert response.json() == {"prompt_token_count": 19}
    (extras,) = seen
    assert extras.ng_measure is True
