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

"""Native rollouts through the Dynamo frontend for Megatron engines."""

import asyncio
from typing import Any

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron import dynamo_rollouts as rollouts

_URL = "http://10.0.0.1:3000/v1"


def _cfg(**overrides) -> dict[str, Any]:
    cfg = {
        "model_name": "Qwen/Qwen3-0.6B",
        "max_new_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "stop_token_ids": [2],
        "stop_strings": ["stop"],
        "_pad_token_id": 0,
        "mcore_generation_config": {"max_model_len": 5},
    }
    cfg.update(overrides)
    return cfg


def _data(stop_strings: Any = None) -> BatchedDataDict:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "input_lengths": torch.tensor([3]),
        }
    )
    if stop_strings is not None:
        data["stop_strings"] = [stop_strings]
    return data


def _completion_response(token_ids: list[int]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.5] * len(token_ids)},
            }
        ],
        "nvext": {"completion_token_ids": list(token_ids)},
    }


def test_generation_posts_token_ids_and_rebuilds_the_sample(monkeypatch) -> None:
    requests: list[tuple[str, dict[str, Any], Any]] = []

    async def fake_post(url, payload, timeout_s):
        requests.append((url, payload, timeout_s))
        return _completion_response([8, 9])

    monkeypatch.setattr(rollouts, "async_http_post_json", fake_post)
    output = asyncio.run(
        rollouts.generate_through_frontend(
            _cfg(),
            frontend_url=_URL,
            data=_data(stop_strings=["done"]),
            greedy=False,
            request_timeout_s=30.0,
        )
    )

    url, payload, timeout_s = requests[0]
    assert url == f"{_URL}/completions" and timeout_s == 30.0
    assert payload["prompt"] == [1, 2, 3]
    # The budget is the context left after the prompt, not max_new_tokens.
    assert payload["max_tokens"] == 2
    assert payload["stop"] == ["done", "stop"]
    assert payload["stop_token_ids"] == [2]
    assert payload["nvext"] == {"extra_fields": ["completion_token_ids"]}
    assert payload["top_k"] == -1 and payload["temperature"] == 0.7
    assert output["output_ids"].tolist() == [[1, 2, 3, 8, 9]]
    assert output["logprobs"].tolist() == [[0.0, 0.0, 0.0, -0.5, -0.5]]
    assert output["generation_lengths"].tolist() == [2]
    assert output["unpadded_sequence_lengths"].tolist() == [5]
    assert output["truncated"].tolist() == [False]


def test_greedy_payload_pins_sampling() -> None:
    payload = rollouts.completion_payload(
        _cfg(top_k=40),
        prompt_token_ids=[1],
        greedy=True,
        stop_strings=None,
        max_new_tokens=4,
    )
    assert payload["temperature"] == 0.0 and payload["top_k"] == 1
    assert "stop" not in payload


@pytest.mark.parametrize("prompt_key", ["vllm_content", "vllm_multi_modal_data"])
def test_multimodal_prompt_columns_are_rejected(monkeypatch, prompt_key) -> None:
    async def fake_post(*args):
        raise AssertionError("Multimodal prompts must not reach HTTP")

    monkeypatch.setattr(rollouts, "async_http_post_json", fake_post)
    data = _data()
    data[prompt_key] = ["prompt" if prompt_key == "vllm_content" else {"image": "img"}]
    with pytest.raises(NotImplementedError, match="token-ID prompts only"):
        asyncio.run(
            rollouts.generate_through_frontend(
                _cfg(),
                frontend_url=_URL,
                data=data,
                greedy=False,
                request_timeout_s=None,
            )
        )


def test_prompt_at_context_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt length 5 must be less than"):
        rollouts.allowed_new_tokens(_cfg(), 5)


def test_merged_stop_strings_enforce_the_frontend_limit() -> None:
    configured = [f"configured-{index}" for index in range(16)]
    merged = rollouts.merge_stop_strings(
        configured, [f"request-{index}" for index in range(16)]
    )
    assert merged is not None and len(merged) == 32
    assert rollouts.merge_stop_strings(None, None) is None
    assert rollouts.merge_stop_strings(None, "one") == ["one"]
    with pytest.raises(ValueError, match="at most 32 stop strings"):
        rollouts.merge_stop_strings(
            configured, [f"request-{index}" for index in range(17)]
        )


def test_completion_parser_rejects_misaligned_logprobs() -> None:
    response = _completion_response([8, 9])
    response["choices"][0]["logprobs"]["token_logprobs"] = [-0.1]
    with pytest.raises(RuntimeError, match="1 token logprobs for 2"):
        rollouts.parse_completion_response(response, request_url=_URL)


def test_completion_parser_accepts_an_empty_completion() -> None:
    response = _completion_response([])
    response["choices"][0]["finish_reason"] = "length"
    assert rollouts.parse_completion_response(response, request_url=_URL) == (
        [],
        [],
        True,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"http_status": 408}, True),
        ({"http_status": 429}, True),
        ({"http_status": 503}, True),
        ({"transport_error": "refused"}, True),
        ({"json_decode_error": True}, True),
        ({"http_status": 400}, False),
        ({"http_status": 404}, False),
        ("not a dict", False),
    ],
)
def test_completion_retry_predicate(response, expected) -> None:
    assert rollouts.is_retryable(response) is expected


@pytest.mark.parametrize(
    ("statuses", "token_ids", "error", "attempts"),
    [
        pytest.param([503, None], [8], None, 2, id="recovers"),
        pytest.param([400], None, "HTTP 400", 1, id="nonretryable"),
        pytest.param([503, 503, 503], None, "HTTP 503", 3, id="exhausted"),
    ],
)
def test_completion_retries_transient_failures(
    monkeypatch, statuses, token_ids, error, attempts
) -> None:
    responses = iter(
        _completion_response([8])
        if status is None
        else {"status": "error", "http_status": status}
        for status in statuses
    )
    calls: list[Any] = []

    async def fake_post(*args):
        calls.append(args)
        return next(responses)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(rollouts, "async_http_post_json", fake_post)
    monkeypatch.setattr(rollouts.asyncio, "sleep", no_sleep)
    if error is None:
        assert asyncio.run(rollouts.post_completion(_URL, {}, None))[0] == token_ids
    else:
        with pytest.raises(RuntimeError, match=error):
            asyncio.run(rollouts.post_completion(_URL, {}, None))
    assert len(calls) == attempts
