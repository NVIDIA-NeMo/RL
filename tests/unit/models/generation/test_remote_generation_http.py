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
    server_url="http://router:8089",
    model_name="test-model",
    max_new_tokens=4,
    max_model_len=16,
):
    """Build a RemoteGeneration with both HTTP round-trips short-circuited.

    We mock `_fetch_remote_config` and `_fetch_shard_urls` so the constructor
    doesn't hit a real server. The returned instance has the surviving json
    path fully wired (cfg, dp_openai_server_base_urls).
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
            server_url=server_url,
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


def test_constructor_exposes_router_url_for_data_plane():
    """In HTTP mode the unified router URL is the single ingress for both
    /v1/completions (data plane) and /init_collective (control plane).
    ``dp_openai_server_base_urls`` exposes the router so gym round-robin
    just hits one URL — cordon + replay is invisible to gym."""
    rg = _make_rg(server_url="http://router:8089")
    assert rg._http_mode is True
    assert rg.server_url == "http://router:8089"
    assert rg.dp_openai_server_base_urls == ["http://router:8089/v1"]
    # Shard URLs are still pulled once for diagnostics.
    assert rg._shard_urls == ["http://shard-0:8000/v1", "http://shard-1:8001/v1"]


def test_constructor_normalizes_trailing_slash_on_server_url():
    """Trailing slash on server_url is stripped so ``f"{server_url}/v1"``
    doesn't produce a double-slash URL."""
    fake_cfg = {
        "vllm_cfg": {"max_model_len": 16},
        "max_new_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "model_name": "test-model",
    }
    with (
        patch.object(RemoteGeneration, "_fetch_remote_config", return_value=fake_cfg),
        patch.object(
            RemoteGeneration, "_fetch_shard_urls", return_value=["http://s/v1"]
        ),
    ):
        rg = RemoteGeneration(
            generation=None,
            server_url="http://router:8089/",
            config={},
        )
    assert rg.server_url == "http://router:8089"
    assert rg.dp_openai_server_base_urls == ["http://router:8089/v1"]


def test_generate_json_completions_targets_router_data_plane():
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

    # All requests target the unified router, not individual shards.
    urls = [u for u, _ in mock_session.posts]
    assert urls == ["http://router:8089/v1/completions"] * 2

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


def test_generate_json_completions_keeps_targeting_router_across_calls():
    """Both calls go to the same router URL — the router internally fans
    out across shards. RemoteGeneration no longer round-robins itself."""
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

    first_url = s1.return_value.posts[0][0]
    second_url = s2.return_value.posts[0][0]
    assert first_url == "http://router:8089/v1/completions"
    assert second_url == "http://router:8089/v1/completions"


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


# =====================================================================
# get_step_metrics_snapshot — disagg HTTP-only path
# =====================================================================


_SAMPLE_REMOTE_SNAPSHOT = {
    "vllm_logger_metrics": {
        "inflight_batch_sizes": {0: [4, 5, 6]},
        "num_pending_samples": {0: [0, 0, 0]},
        "kv_cache_usage_perc": {0: [0.41, 0.42, 0.43]},
        "generation_tokens": {0: [10, 20, 30]},
    },
    "spec_decode_metrics": {
        "vllm/spec_num_drafts": 12,
        "vllm/spec_num_accepted_tokens": 100,
        "vllm/spec_acceptance_rate": 0.83,
    },
    "router_metrics": {
        "num_ready_shards": 3,
        "num_total_shards": 4,
        "num_cordoned_shards": 1,
        "num_joining_shards": 0,
        "num_draining_shards": 0,
        "total_shards_at_bootstrap": 4,
        "cumulative_shards_removed": 1,
        "cumulative_shards_added": 0,
        "per_shard_world_size": 2,
        "current_gen_world_size": 6,
        "nccl_reinit_in_progress": False,
        "last_fault_event": {
            "kind": "remove",
            "shard_id": "dp-3",
            "reason": "test",
            "monotonic_ts": 1000.0,
        },
        "per_shard": [
            {"shard_id": "dp-0", "status": "ready", "inflight": 2,
             "consecutive_failures": 0, "last_health_ok_age_s": 0.5},
        ],
    },
}


def _mock_requests_get_for_snapshot(payload):
    """Patch requests.get so the snapshot endpoint returns ``payload``."""
    class _Resp:
        def __init__(self, body):
            self._body = body
        def raise_for_status(self):
            return None
        def json(self):
            return self._body

    def _get(url, *args, **kwargs):
        if url.endswith("/step_metrics_snapshot"):
            return _Resp(payload)
        raise AssertionError(f"unexpected URL: {url}")

    return _get


def test_get_step_metrics_snapshot_http_round_trip():
    rg = _make_rg()
    with patch(
        "nemo_rl.models.generation.remote_generation.requests.get",
        side_effect=_mock_requests_get_for_snapshot(_SAMPLE_REMOTE_SNAPSHOT),
    ):
        snap = rg.get_step_metrics_snapshot()
    # The payload round-trips verbatim: HTTP mode just JSON-decodes.
    assert snap == _SAMPLE_REMOTE_SNAPSHOT
    # The three top-level keys must be present so the train side can
    # blindly index into them without is-None checks.
    assert set(snap) >= {"vllm_logger_metrics", "spec_decode_metrics", "router_metrics"}


def test_get_step_metrics_snapshot_per_step_cache():
    rg = _make_rg()
    call_count = {"n": 0}
    payload_a = {
        "vllm_logger_metrics": {"inflight_batch_sizes": {0: [1]}},
        "spec_decode_metrics": {},
        "router_metrics": {},
    }
    payload_b = {
        "vllm_logger_metrics": {"inflight_batch_sizes": {0: [2]}},
        "spec_decode_metrics": {},
        "router_metrics": {},
    }

    class _Resp:
        def __init__(self, body):
            self._body = body
        def raise_for_status(self):
            return None
        def json(self):
            return self._body

    def _get(url, *args, **kwargs):
        call_count["n"] += 1
        # Return a different payload each call so we can prove the cache
        # surfaced the first one even when the network would have given
        # us something newer.
        return _Resp(payload_a if call_count["n"] == 1 else payload_b)

    with patch(
        "nemo_rl.models.generation.remote_generation.requests.get",
        side_effect=_get,
    ):
        first = rg.get_step_metrics_snapshot(step=10)
        second = rg.get_step_metrics_snapshot(step=10)  # same step → cached
        third = rg.get_step_metrics_snapshot(step=11)  # new step → re-fetch

    assert first == payload_a
    assert second == payload_a
    assert third == payload_b
    # Two HTTP fetches: one for step 10 (then cached), one for step 11.
    assert call_count["n"] == 2


def test_get_step_metrics_snapshot_transport_failure_returns_empty_shape():
    """A network blip must not break wandb logging — return the empty shape."""
    import requests as _requests

    rg = _make_rg()
    with patch(
        "nemo_rl.models.generation.remote_generation.requests.get",
        side_effect=_requests.ConnectionError("boom"),
    ):
        snap = rg.get_step_metrics_snapshot()
    assert snap == {
        "vllm_logger_metrics": {},
        "spec_decode_metrics": {},
        "router_metrics": {},
    }


def test_get_step_metrics_snapshot_colocated_uses_inner_directly():
    """Co-located mode synthesizes the snapshot from the wrapped inner
    generation object — no HTTP, even though the URL would be available."""
    inner = MagicMock()
    inner.cfg = {"vllm_cfg": {"max_model_len": 16}, "max_new_tokens": 4,
                 "model_name": "x"}
    inner.get_logger_metrics.return_value = {
        "inflight_batch_sizes": {0: [7, 8]},
    }

    # Co-located path: pass the inner generation. The constructor must NOT
    # try to fetch config (we have one) or hit the wire for shard URLs.
    rg = RemoteGeneration(generation=inner, server_url="http://router:8089",
                          config={})
    # Should never hit requests.get in colocated mode.
    with patch(
        "nemo_rl.models.generation.remote_generation.requests.get",
        side_effect=AssertionError("must not hit HTTP in colocated mode"),
    ):
        snap = rg.get_step_metrics_snapshot()
    assert snap["vllm_logger_metrics"] == {"inflight_batch_sizes": {0: [7, 8]}}
    # spec_decode_metrics is intentionally empty — it's consumed by grpo's
    # existing get_step_metrics() path, not the snapshot endpoint, to avoid
    # double-consuming the destructive delta-since-snapshot baseline.
    assert snap["spec_decode_metrics"] == {}
    inner.get_step_metrics.assert_not_called()
    # Co-located mode has no router exposed through this surface — train
    # already shares a Ray cluster with gen, so router metrics aren't
    # the point. Empty dict matches the HTTP-mode contract.
    assert snap["router_metrics"] == {}


# =====================================================================
# Unified router /step_metrics_snapshot endpoint shape
# =====================================================================


def test_unified_router_step_metrics_snapshot_endpoint_shape():
    """/step_metrics_snapshot on the unified router returns the consolidated
    dict of vllm logger metrics + spec decode + router metrics. Verifies
    the endpoint shape matches what RemoteGeneration.get_step_metrics_snapshot()
    expects."""
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.generation_router import GenerationRouter

    fake_gen = MagicMock()
    fake_gen.cfg = {"vllm_cfg": {}}
    fake_gen.get_logger_metrics.return_value = {
        "inflight_batch_sizes": {0: [1, 2, 3]},
    }

    router = GenerationRouter(port=0, generation=fake_gen)
    router.register_shards(
        [("dp-0", "http://shard-0/v1"), ("dp-1", "http://shard-1/v1")],
        per_shard_world_size=1,
        generation=fake_gen,
    )
    with TestClient(router.get_app()) as client:
        resp = client.get("/step_metrics_snapshot")
    assert resp.status_code == 200
    body = resp.json()
    # Three top-level keys exactly match what RemoteGeneration expects.
    assert set(body) >= {"vllm_logger_metrics", "spec_decode_metrics", "router_metrics"}
    assert body["vllm_logger_metrics"] == {"inflight_batch_sizes": {"0": [1, 2, 3]}}
    # The snapshot endpoint must not call the destructive get_step_metrics().
    assert body["spec_decode_metrics"] == {}
    fake_gen.get_step_metrics.assert_not_called()
    # Router metrics now come from the same router instance.
    assert body["router_metrics"]["num_ready_shards"] == 2


def test_unified_router_step_metrics_snapshot_no_generation():
    """Without a wired generation the endpoint 503s — there's nothing to
    snapshot. (Cordon-only test setups exercise this path.)"""
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.generation_router import GenerationRouter

    router = GenerationRouter(port=0, generation=None)
    with TestClient(router.get_app()) as client:
        resp = client.get("/step_metrics_snapshot")
    assert resp.status_code == 503


# =====================================================================
# flatten_router_metrics_for_wandb — train-side flattening
# =====================================================================


def test_flatten_router_metrics_marks_fault_event_only_on_transition():
    """The fault marker is 1 only on the step where the latest fault event
    timestamp differs from the prior step's snapshot."""
    from nemo_rl.algorithms.utils import flatten_router_metrics_for_wandb

    snap_no_fault = {
        "num_ready_shards": 3,
        "num_total_shards": 3,
        "cumulative_shards_removed": 0,
        "cumulative_shards_added": 0,
        "last_fault_event": None,
    }
    flat, prev = flatten_router_metrics_for_wandb(snap_no_fault, prev_fault_ts=None)
    assert flat["fault_event_this_step"] == 0.0
    assert prev is None

    # New fault appears.
    snap_fault = dict(snap_no_fault, last_fault_event={
        "kind": "remove", "shard_id": "dp-1", "reason": "x",
        "monotonic_ts": 42.0,
    })
    flat, prev = flatten_router_metrics_for_wandb(snap_fault, prev_fault_ts=None)
    assert flat["fault_event_this_step"] == 1.0
    assert prev == 42.0

    # Same fault, next step → marker resets to 0.
    flat, prev = flatten_router_metrics_for_wandb(snap_fault, prev_fault_ts=42.0)
    assert flat["fault_event_this_step"] == 0.0
    assert prev == 42.0

    # Newer fault.
    snap_fault2 = dict(snap_fault, last_fault_event={
        "kind": "remove", "shard_id": "dp-2", "reason": "y",
        "monotonic_ts": 99.0,
    })
    flat, prev = flatten_router_metrics_for_wandb(snap_fault2, prev_fault_ts=42.0)
    assert flat["fault_event_this_step"] == 1.0
    assert prev == 99.0


def test_flatten_router_metrics_empty_input_returns_empty():
    """Co-located mode passes {} → no wandb keys emitted."""
    from nemo_rl.algorithms.utils import flatten_router_metrics_for_wandb

    flat, prev = flatten_router_metrics_for_wandb({}, prev_fault_ts=None)
    assert flat == {}
    assert prev is None


def test_flatten_router_metrics_max_health_age():
    """We log the max age across shards as a single canary metric."""
    from nemo_rl.algorithms.utils import flatten_router_metrics_for_wandb

    snap = {
        "num_ready_shards": 2,
        "num_total_shards": 2,
        "cumulative_shards_removed": 0,
        "cumulative_shards_added": 0,
        "per_shard": [
            {"shard_id": "a", "last_health_ok_age_s": 0.1},
            {"shard_id": "b", "last_health_ok_age_s": 4.7},
            {"shard_id": "c", "last_health_ok_age_s": None},
        ],
    }
    flat, _ = flatten_router_metrics_for_wandb(snap, prev_fault_ts=None)
    assert flat["max_last_health_ok_age_s"] == 4.7
