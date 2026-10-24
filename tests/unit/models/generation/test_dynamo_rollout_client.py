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
import json

import pytest

from nemo_rl.models.generation.dynamo.rollout_client import (
    ROLLOUT_ANNOTATION,
    parse_rollout_response,
    rollout_payload,
)


def test_rollout_payload_leaves_sampling_and_stopping_to_the_engine() -> None:
    payload = rollout_payload(
        model_name="model",
        prompt_token_ids=[5, 6, 7],
        max_model_len=10,
        greedy=True,
        stop_strings=["</answer>"],
    )

    assert payload["prompt"] == [5, 6, 7]
    # The frontend's own stop rules must never end the stream first.
    assert payload["ignore_eos"] is True
    assert payload["max_tokens"] == 7
    assert "stop" not in payload and "temperature" not in payload
    (annotation,) = payload["nvext"]["annotations"]
    assert annotation.startswith(ROLLOUT_ANNOTATION)
    assert json.loads(annotation[len(ROLLOUT_ANNOTATION) :]) == {
        "greedy": True,
        "stop_strings": ["</answer>"],
    }
    assert payload["nvext"]["extra_fields"] == ["engine_data"]


@pytest.mark.parametrize(
    "response, match",
    [
        ({"status": "error", "http_status": 400, "raw": "bad"}, "HTTP 400"),
        ({"nvext": {}}, "no rollout engine_data"),
        ({"nvext": {"engine_data": {"prompt_logprobs": []}}}, "no rollout engine_data"),
    ],
)
def test_parse_rollout_response_rejects_unusable_responses(response, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        parse_rollout_response(response, url="http://frontend/v1/completions")


def test_parse_rollout_response_returns_engine_data() -> None:
    engine_data = {"rollout": {"token_ids": []}, "gen_leader_worker_idx": 1}

    assert (
        parse_rollout_response({"nvext": {"engine_data": engine_data}}, url="u")
        == engine_data
    )
