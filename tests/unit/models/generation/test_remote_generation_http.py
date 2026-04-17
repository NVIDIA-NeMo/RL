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
"""Unit tests for the surviving (json-only) code path in RemoteGeneration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.remote_generation import RemoteGeneration


def _make_rg(
    shards=("http://shard-0:8000/v1", "http://shard-1:8001/v1"),
    model_name="test-model",
    max_new_tokens=4,
    max_model_len=16,
):
    """Build a RemoteGeneration with both HTTP round-trips short-circuited.

    We mock `_fetch_remote_config` and `_fetch_shard_urls` so the constructor
    doesn't hit a real server. The returned instance has the surviving json
    path fully wired (cfg, shard round-robin, dp_openai_server_base_urls).
    """
    fake_cfg = {
        "vllm_cfg": {"max_model_len": max_model_len},
        "max_new_tokens": max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "model_name": model_name,
    }
    with (
        patch.object(RemoteGeneration, "_fetch_remote_config", return_value=fake_cfg),
        patch.object(RemoteGeneration, "_fetch_shard_urls", return_value=list(shards)),
    ):
        return RemoteGeneration(
            generation=None,
            server_url="http://control:8089",
            config={},
        )


class _MockResp:
    """Minimal async context manager mimicking aiohttp.ClientResponse."""

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _MockSession:
    """Stand-in for aiohttp.ClientSession that records posts and yields canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.posts: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, *, json):
        self.posts.append((url, json))
        return _MockResp(self._responses.pop(0))


def _completion_response(token_ids, logprobs=None, finish_reason="stop"):
    tokens = [f"token_id:{tid}" for tid in token_ids]
    if logprobs is None:
        logprobs = [-0.1] * len(token_ids)
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "logprobs": {"tokens": tokens, "token_logprobs": logprobs},
            }
        ]
    }


def test_constructor_exposes_router_and_shards():
    rg = _make_rg()
    assert rg._http_mode is True
    assert rg._shard_urls == ["http://shard-0:8000/v1", "http://shard-1:8001/v1"]
    # The /v1 on the control URL is what NemoGym calls.
    assert rg.dp_openai_server_base_urls == ["http://control:8089/v1"]
    # Round-robin starts at 0.
    assert rg._shard_rr_idx == 0


def test_generate_json_completions_parses_tokens_and_advances_round_robin():
    rg = _make_rg(max_new_tokens=3, max_model_len=16)
    # Two-sample batch: input_lengths 2 and 3, gen 3 tokens each.
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 0, 0], [20, 21, 22, 0]]),
            "input_lengths": torch.tensor([2, 3]),
        }
    )
    # First sample: normal tokens + logprobs. Second sample: truncated, one None logprob.
    responses = [
        _completion_response([100, 101, 102], logprobs=[-0.1, -0.2, -0.3]),
        _completion_response([200, 201, 202], logprobs=[-1.0, None, -0.5], finish_reason="length"),
    ]
    mock_session = _MockSession(responses)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        out = asyncio.run(rg._generate_json_completions(data, greedy=False))

    # Both requests hit the first shard (round-robin advances once per batch call).
    urls = [u for u, _ in mock_session.posts]
    assert urls == ["http://shard-0:8000/v1/completions"] * 2
    # Shard idx incremented exactly once.
    assert rg._shard_rr_idx == 1

    # Output shape: 2 samples, prompt length 2/3 + gen 3 = 5/6, pad to 6.
    assert out["output_ids"].shape == (2, 6)
    # Sample 0: prompt [10,11] + gen [100,101,102] + pad 0.
    assert out["output_ids"][0].tolist() == [10, 11, 100, 101, 102, 0]
    # Sample 1: prompt [20,21,22] + gen [200,201,202].
    assert out["output_ids"][1].tolist() == [20, 21, 22, 200, 201, 202]
    # generation_lengths reflect produced tokens only.
    assert out["generation_lengths"].tolist() == [3, 3]
    # unpadded = input + gen.
    assert out["unpadded_sequence_lengths"].tolist() == [5, 6]
    # Truncated only on sample 1 (finish_reason="length").
    assert out["truncated"].tolist() == [False, True]
    # logprobs: zeros over prompt tokens, then the vLLM values (None→0.0), then pad zero.
    sample0_lp = out["logprobs"][0].tolist()
    assert sample0_lp[:2] == [0.0, 0.0]
    assert sample0_lp[2:5] == pytest.approx([-0.1, -0.2, -0.3])
    assert sample0_lp[5] == 0.0
    sample1_lp = out["logprobs"][1].tolist()
    assert sample1_lp[:3] == [0.0, 0.0, 0.0]
    assert sample1_lp[3:6] == pytest.approx([-1.0, 0.0, -0.5])


def test_generate_json_completions_rotates_shards_across_calls():
    rg = _make_rg()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )
    responses_first = [_completion_response([99])]
    responses_second = [_completion_response([98])]

    with patch("aiohttp.ClientSession", return_value=_MockSession(responses_first)) as s1:
        asyncio.run(rg._generate_json_completions(data, greedy=False))
    with patch("aiohttp.ClientSession", return_value=_MockSession(responses_second)) as s2:
        asyncio.run(rg._generate_json_completions(data, greedy=False))

    # First call hits shard 0, second call hits shard 1.
    first_url = s1.return_value.posts[0][0]
    second_url = s2.return_value.posts[0][0]
    assert first_url.startswith("http://shard-0:8000")
    assert second_url.startswith("http://shard-1:8001")


def test_greedy_overrides_sampling_params():
    rg = _make_rg()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[7, 8]]),
            "input_lengths": torch.tensor([2]),
        }
    )
    mock_session = _MockSession([_completion_response([9])])
    with patch("aiohttp.ClientSession", return_value=mock_session):
        asyncio.run(rg._generate_json_completions(data, greedy=True))

    _, body = mock_session.posts[0]
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["prompt"] == [7, 8]
    assert body["model"] == "test-model"
