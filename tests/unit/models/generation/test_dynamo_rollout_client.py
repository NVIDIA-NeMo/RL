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
import asyncio

import pytest

from nemo_rl.models.generation.dynamo import rollout_client
from nemo_rl.models.generation.dynamo.rollout_client import (
    parse_rollout_response,
    post_rollout,
)


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


_ENGINE_DATA = {"rollout": {"token_ids": [8]}, "gen_leader_worker_idx": 1}


@pytest.mark.parametrize(
    ("responses", "error"),
    [
        # A transient failure is retried, and the rollout still succeeds.
        (
            [
                {"status": "error", "http_status": 503},
                {"nvext": {"engine_data": _ENGINE_DATA}},
            ],
            None,
        ),
        # A request the frontend refuses is final.
        ([{"status": "error", "http_status": 400}], "HTTP 400"),
        # A timeout is final, and names the setting that bounds it.
        (
            [{"status": "error", "timeout_error": "no response within 1.0s"}],
            "no response.*dynamo_cfg.request_timeout_s",
        ),
        # Retries are bounded.
        (
            [{"status": "error", "http_status": 503}]
            * rollout_client._HTTP_MAX_ATTEMPTS,
            "HTTP 503",
        ),
    ],
)
def test_post_rollout_retries_only_transient_failures(
    monkeypatch, responses, error
) -> None:
    calls = []

    async def fake_post(url, payload, timeout_s):
        calls.append(payload)
        return responses[len(calls) - 1]

    monkeypatch.setattr(rollout_client, "async_http_post_json", fake_post)
    monkeypatch.setattr(rollout_client, "_HTTP_RETRY_DELAY_S", 0.0)
    rollout = post_rollout("http://frontend/v1/completions", {}, timeout_s=1.0)

    if error is None:
        assert asyncio.run(rollout) == _ENGINE_DATA
    else:
        with pytest.raises(RuntimeError, match=error):
            asyncio.run(rollout)
    assert len(calls) == len(responses)
