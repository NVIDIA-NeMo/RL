# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pure-mock unit tests for the Dynamo URL-forwarder backend.

These do not exercise a real DynamoGraphDeployment — they only verify the
class's contract: K8s detection, URL derivation, lifecycle no-ops, direct
generation HTTP payloads, and that unsupported methods fail loudly. End-to-end
coverage is provided separately in the Phase 2 integration tests once nrl-k8s
can stand up a DGD.
"""

import asyncio
import pickle

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.dynamo import DynamoConfig, DynamoGeneration
from nemo_rl.models.generation.dynamo import dynamo_generation as _dynmod
from nemo_rl.utils import k8s as k8s_utils


def _base_config(**dynamo_cfg_overrides) -> DynamoConfig:
    cfg: DynamoConfig = {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "dynamo_cfg": {
            "dgd_name": "my-dgd",
            "request_timeout_s": 30.0,
            **dynamo_cfg_overrides,
        },
    }
    return cfg


def _generation_data(
    input_ids: list[list[int]],
    input_lengths: list[int],
    stop_strings: list[list[str] | None] | None = None,
) -> BatchedDataDict:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        }
    )
    if stop_strings is not None:
        data["stop_strings"] = stop_strings
    return data


@pytest.fixture
def in_k8s(monkeypatch):
    """Pretend the test process is running inside a pod."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    return monkeypatch


@pytest.fixture
def stub_namespace(monkeypatch, tmp_path):
    """Redirect the namespace-projection file to a tmp file we control."""
    ns_file = tmp_path / "namespace"
    ns_file.write_text("test-ns")
    monkeypatch.setattr(k8s_utils, "POD_NAMESPACE_FILE", str(ns_file), raising=True)
    return ns_file


def test_assert_inside_k8s_for_dgd_name_path(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    with pytest.raises(RuntimeError, match="dgd_name requires running inside"):
        DynamoGeneration(cluster=None, config=_base_config())


def test_missing_dgd_name_and_frontend_url(in_k8s):
    cfg = _base_config()
    cfg["dynamo_cfg"] = {}  # type: ignore[typeddict-item]
    with pytest.raises(RuntimeError, match="dgd_name.*frontend_url"):
        DynamoGeneration(cluster=None, config=cfg)


def test_explicit_frontend_url_skips_k8s_check(monkeypatch):
    """frontend_url opts out of the in-pod check — works on slurm, laptop, etc."""
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {"frontend_url": "http://my-dgd.example.com:8000/v1"}  # type: ignore[typeddict-item]
    g = DynamoGeneration(cluster=None, config=cfg)
    assert g.dp_openai_server_base_urls == ["http://my-dgd.example.com:8000/v1"]


def test_explicit_frontend_url_overrides_dgd_name(in_k8s):
    cfg = _base_config(frontend_url="http://override.example.com:9000/v1")
    g = DynamoGeneration(cluster=None, config=cfg)
    assert g.dp_openai_server_base_urls == ["http://override.example.com:9000/v1"]


def test_empty_frontend_url_raises(in_k8s):
    cfg = _base_config(frontend_url="")
    with pytest.raises(RuntimeError, match="frontend_url is set but empty"):
        DynamoGeneration(cluster=None, config=cfg)


def test_url_derivation_explicit_namespace(in_k8s):
    g = DynamoGeneration(
        cluster=None,
        config=_base_config(namespace="bar", frontend_port=9000),
    )
    assert g.dp_openai_server_base_urls == [
        "http://my-dgd-frontend.bar.svc.cluster.local:9000/v1"
    ]


def test_url_derivation_default_port(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config())
    assert g.dp_openai_server_base_urls == [
        "http://my-dgd-frontend.test-ns.svc.cluster.local:8000/v1"
    ]


def test_namespace_from_pod_serviceaccount_file(in_k8s, stub_namespace):
    # Config does not set namespace — class must read it from the projected file.
    g = DynamoGeneration(cluster=None, config=_base_config(frontend_port=8001))
    assert "test-ns.svc.cluster.local" in g.dp_openai_server_base_urls[0]


def test_namespace_fallback_warns(in_k8s, monkeypatch, tmp_path):
    # No namespace in config and no projected file — class warns and falls back to "default".
    missing = tmp_path / "missing"
    monkeypatch.setattr(k8s_utils, "POD_NAMESPACE_FILE", str(missing), raising=True)

    with pytest.warns(UserWarning, match="namespace"):
        g = DynamoGeneration(cluster=None, config=_base_config())
    assert "default.svc.cluster.local" in g.dp_openai_server_base_urls[0]


def test_lifecycle_noops(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config())
    assert g.prepare_for_generation() is True
    assert g.finish_generation() is True
    assert g.shutdown() is True


def test_unsupported_methods_raise(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config())
    with pytest.raises(NotImplementedError):
        g.init_collective(ip="127.0.0.1", port=1, world_size=1)
    with pytest.raises(NotImplementedError):
        g.update_weights_via_ipc_zmq()
    with pytest.raises(NotImplementedError):
        g.update_weights_from_collective()


def test_prepare_refit_info_is_noop(in_k8s, stub_namespace):
    # Receiver-side MX polling architecture: nothing to forward trainer-side, so
    # prepare_refit_info is a deliberate no-op (not NotImplementedError).
    g = DynamoGeneration(cluster=None, config=_base_config())
    assert g.prepare_refit_info({}) is None


def test_pickle_roundtrip(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config(frontend_port=8123))
    expected_url = g.dp_openai_server_base_urls[0]
    restored = pickle.loads(pickle.dumps(g))
    assert restored.dp_openai_server_base_urls == [expected_url]
    assert restored.cfg["dynamo_cfg"]["dgd_name"] == "my-dgd"


# ---------------------------------------------------------------------------
# Direct generation — OpenAI completions payload + vLLM-parity tensors.
# ---------------------------------------------------------------------------


def test_generate_requires_request_timeout_s(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {"frontend_url": "http://my-dgd.example.com:8000/v1"}  # type: ignore[typeddict-item]
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="request_timeout_s"):
        g.generate(_generation_data([[1, 2, 0]], [2]))


def test_generate_builds_non_greedy_payload_and_vllm_sync_tensors(
    monkeypatch,
):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(
        frontend_url="http://my-dgd.example.com:8000/v1",
        request_timeout_s=42.0,
    )
    cfg["temperature"] = 0.7
    cfg["top_p"] = 0.9
    cfg["top_k"] = 50
    cfg["stop_token_ids"] = [128001]
    cfg["stop_strings"] = ["global-stop"]
    calls = []

    def fake_post_json(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        if payload["prompt"] == [1, 2, 3]:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ],
                "nvext": {"completion_token_ids": [10, 11]},
            }
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "logprobs": {"token_logprobs": [-0.3]},
                }
            ],
            "nvext": {"completion_token_ids": [12]},
        }

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)
    out = g.generate(
        _generation_data(
            [[1, 2, 3, 0], [4, 5, 0, 0]],
            [3, 2],
            stop_strings=[["sample-a"], ["sample-b"]],
        )
    )

    assert [call[0] for call in calls] == [
        "http://my-dgd.example.com:8000/v1/completions",
        "http://my-dgd.example.com:8000/v1/completions",
    ]
    assert [call[2] for call in calls] == [42.0, 42.0]
    for _, payload, _ in calls:
        assert payload["model"] == "Qwen/Qwen3-0.6B"
        assert payload["max_tokens"] == 16
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 50
        assert payload["logprobs"] == 0
        assert payload["n"] == 1
        assert payload["return_tokens_as_token_ids"] is True
        assert payload["include_stop_str_in_output"] is True
        assert payload["stop_token_ids"] == [128001]
        assert set(payload["stop"]) == {"global-stop", "sample-a", "sample-b"}
        assert payload["nvext"] == {"extra_fields": ["completion_token_ids"]}
        assert "greed_sampling" not in payload["nvext"]

    assert out["output_ids"].tolist() == [
        [1, 2, 3, 10, 11, 0],
        [4, 5, 12, 0, 0, 0],
    ]
    assert out["generation_lengths"].tolist() == [2, 1]
    assert out["unpadded_sequence_lengths"].tolist() == [5, 3]
    assert out["truncated"].tolist() == [False, True]
    assert torch.allclose(
        out["logprobs"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.1, -0.2, 0.0],
                [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_generate_builds_greedy_payload_without_dynamo_greed_sampling(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg["temperature"] = 1.2
    cfg["top_p"] = 0.75
    cfg["top_k"] = None
    calls = []

    def fake_post_json(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return {
            "choices": [{"finish_reason": "stop", "logprobs": None}],
            "nvext": {"completion_token_ids": [9]},
        }

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)
    out = g.generate(_generation_data([[7, 8]], [2]), greedy=True)

    payload = calls[0][1]
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["top_p"] == 0.75
    assert payload["nvext"] == {"extra_fields": ["completion_token_ids"]}
    assert "greed_sampling" not in payload["nvext"]
    assert out["output_ids"].tolist() == [[7, 8, 9]]
    assert out["logprobs"].tolist() == [[0.0, 0.0, 0.0]]


def test_generate_raises_on_http_error(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")

    def fake_post_json(url, payload, timeout_s):
        return {"status": "error", "http_status": 500, "raw": "boom"}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="HTTP 500: boom"):
        g.generate(_generation_data([[1]], [1]))


def test_generate_raises_on_missing_completion_token_ids(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")

    def fake_post_json(url, payload, timeout_s):
        return {"choices": [{"finish_reason": "stop"}]}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="did not include nvext"):
        g.generate(_generation_data([[1]], [1]))


def test_generate_async_caps_context_and_yields_compact_result(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg["max_new_tokens"] = 10
    cfg["vllm_cfg"] = {"max_model_len": 5}  # type: ignore[typeddict-item]
    calls = []

    def fake_post_json(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.4]},
                }
            ],
            "nvext": {"completion_token_ids": [8]},
        }

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    async def collect():
        return [
            item
            async for item in g.generate_async(
                _generation_data([[1, 2, 3, 0]], [3], stop_strings=[["sample-stop"]])
            )
        ]

    results = asyncio.run(collect())

    assert len(results) == 1
    original_idx, out = results[0]
    assert original_idx == 0
    assert calls[0][1]["max_tokens"] == 2
    assert calls[0][1]["prompt"] == [1, 2, 3]
    assert calls[0][1]["stop"] == ["sample-stop"]
    assert out["output_ids"].tolist() == [[1, 2, 3, 8]]
    assert out["generation_lengths"].tolist() == [1]
    assert out["unpadded_sequence_lengths"].tolist() == [4]
    assert out["truncated"].tolist() == [False]
    assert torch.allclose(out["logprobs"], torch.tensor([[0.0, 0.0, 0.0, -0.4]]))


def test_generate_async_zero_budget_skips_http(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg["vllm_cfg"] = {"max_model_len": 3}  # type: ignore[typeddict-item]

    def fake_post_json(url, payload, timeout_s):
        raise AssertionError("HTTP should not be called when no context remains")

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    async def collect():
        return [
            item
            async for item in g.generate_async(_generation_data([[1, 2, 3, 0]], [3]))
        ]

    results = asyncio.run(collect())
    original_idx, out = results[0]

    assert original_idx == 0
    assert out["output_ids"].tolist() == [[1, 2, 3]]
    assert out["generation_lengths"].tolist() == [0]
    assert out["unpadded_sequence_lengths"].tolist() == [3]
    assert out["truncated"].tolist() == [False]
    assert out["logprobs"].tolist() == [[0.0, 0.0, 0.0]]


def test_generate_async_rejects_multi_sample_batch(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg["vllm_cfg"] = {"max_model_len": 10}  # type: ignore[typeddict-item]
    g = DynamoGeneration(cluster=None, config=cfg)

    async def collect():
        return [
            item
            async for item in g.generate_async(_generation_data([[1], [2]], [1, 1]))
        ]

    with pytest.raises(AssertionError, match="single samples"):
        asyncio.run(collect())


# ---------------------------------------------------------------------------
# Engine-telemetry sampler — Prometheus parsing + get/clear contract.
# ---------------------------------------------------------------------------


def test_parse_prometheus_basic_and_colon_sanitization():
    text = (
        "# HELP vllm:num_requests_running running\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model="x",worker_id="abc"} 3.0\n'
        'vllm:num_requests_waiting{model="x"} 0.0\n'
        'vllm:kv_cache_usage_perc{model="x"} 8.34e-05\n'
        'dynamo_component_kv_cache_hit_rate{dp_rank="0"} 0.42\n'
    )
    got = _dynmod._parse_prometheus_metrics(text)
    assert got["vllm_num_requests_running"] == 3.0  # ':' sanitized to '_'
    assert got["vllm_num_requests_waiting"] == 0.0
    assert abs(got["vllm_kv_cache_usage_perc"] - 8.34e-05) < 1e-12  # sci-notation
    assert got["dynamo_component_kv_cache_hit_rate"] == 0.42
    assert all(":" not in k for k in got)


def test_parse_prometheus_skips_buckets_created_keeps_sum_count_sums_labels():
    text = (
        'vllm:ttft_seconds_bucket{le="0.1"} 5\n'
        'vllm:ttft_seconds_bucket{le="+Inf"} 10\n'
        "vllm:ttft_seconds_sum 1.5\n"
        "vllm:ttft_seconds_count 10\n"
        "vllm:generation_tokens_created 1.749e9\n"
        'vllm:request_success_total{reason="stop"} 5\n'
        'vllm:request_success_total{reason="length"} 3\n'
    )
    got = _dynmod._parse_prometheus_metrics(text)
    assert "vllm_ttft_seconds_bucket" not in got  # histogram buckets skipped
    assert "vllm_generation_tokens_created" not in got  # creation timestamps skipped
    assert got["vllm_ttft_seconds_sum"] == 1.5  # scalar _sum/_count kept
    assert got["vllm_ttft_seconds_count"] == 10.0
    assert got["vllm_request_success_total"] == 8.0  # summed across label sets


def test_parse_prometheus_include_exclude_filters():
    text = (
        "vllm:foo 1\n"
        'python_gc_objects_collected_total{generation="0"} 100\n'
        "process_cpu_seconds_total 1.5\n"
        "dynamo_component_bar 2\n"
    )
    # Default drops interpreter/process noise, keeps engine families generically.
    got = _dynmod._parse_prometheus_metrics(text)
    assert "python_gc_objects_collected_total" not in got
    assert "process_cpu_seconds_total" not in got
    assert got["vllm_foo"] == 1.0 and got["dynamo_component_bar"] == 2.0
    # include_prefixes matches the RAW name (before ':' sanitization).
    only_vllm = _dynmod._parse_prometheus_metrics(text, include_prefixes=("vllm:",))
    assert set(only_vllm) == {"vllm_foo"}


def test_parse_prometheus_never_raises_on_junk():
    assert _dynmod._parse_prometheus_metrics("") == {}
    # comment / 1-token / non-numeric lines are skipped, not raised
    out = _dynmod._parse_prometheus_metrics("garbage\n# c\nname_only\n{bad} x\nok 1\n")
    assert out == {"ok": 1.0}


def test_http_get_text_returns_none_on_transport_error(monkeypatch):
    import urllib.error

    def _boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert _dynmod._http_get_text("http://w:9090/metrics", 1.0) is None


def test_logger_metrics_disabled_by_default(in_k8s, stub_namespace):
    # No vllm_cfg -> sampler never starts; get/clear are inert no-ops.
    g = DynamoGeneration(cluster=None, config=_base_config())
    assert g.get_logger_metrics() == {}
    g.clear_logger_metrics()
    assert g.get_logger_metrics() == {}


def test_logger_metrics_enabled_shape_deepcopy_and_clear(
    in_k8s, stub_namespace, monkeypatch
):
    # Exercise the get/clear contract without spawning the real sampler thread.
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config()
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    g = DynamoGeneration(cluster=None, config=cfg)
    assert g._metrics_enabled is True
    # Nothing scraped yet: no raw metric keys (canonical aliases covered separately).
    assert "vllm_num_requests_running" not in g.get_logger_metrics()

    with g._dyn_metrics_lock:
        g._dyn_logger_metrics = {"custom_metric": {0: [1.0, 2.0], 1: [3.0]}}
    out = g.get_logger_metrics()
    assert out["custom_metric"] == {0: [1.0, 2.0], 1: [3.0]}
    # Returned timelines are copies — mutating them must not corrupt the sampler.
    out["custom_metric"][0].append(999.0)
    assert g.get_logger_metrics()["custom_metric"][0] == [1.0, 2.0]

    g.clear_logger_metrics()
    assert "custom_metric" not in g.get_logger_metrics()  # cleared


def test_pickle_roundtrip_with_metrics_enabled(in_k8s, stub_namespace, monkeypatch):
    # __getstate__ omits the lock/thread, so a metrics-enabled object still
    # pickles; the restored (actor-side) copy never samples and its get/clear/
    # shutdown stay safe via the getattr guards.
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config()
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    g = DynamoGeneration(cluster=None, config=cfg)
    restored = pickle.loads(pickle.dumps(g))
    assert restored.dp_openai_server_base_urls == g.dp_openai_server_base_urls
    assert restored.get_logger_metrics() == {}
    restored.clear_logger_metrics()
    assert restored.shutdown() is True


def test_logger_metrics_always_has_canonical_vllm_keys(
    in_k8s, stub_namespace, monkeypatch
):
    # print_performance_metrics (algorithms/utils.py) hard-asserts these canonical
    # names exist and are dicts; get_logger_metrics must always supply them,
    # mapped from the generic scrape (or empty), so the async loop never crashes.
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config()
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    g = DynamoGeneration(cluster=None, config=cfg)
    canon = (
        "inflight_batch_sizes",
        "num_pending_samples",
        "kv_cache_usage_perc",
        "generation_tokens",
    )

    # (a) empty scrape -> canonical keys still present as dicts (assert-safe).
    out = g.get_logger_metrics()
    for k in canon:
        assert k in out and isinstance(out[k], dict), k

    # (b) populated scrape -> canonical keys mapped from the raw vllm_* names,
    #     and raw alias sources are dropped to avoid duplicate timelines.
    with g._dyn_metrics_lock:
        g._dyn_logger_metrics = {
            "vllm_num_requests_running": {0: [3.0, 4.0]},
            "vllm_num_requests_waiting": {0: [1.0]},
            "vllm_kv_cache_usage_perc": {0: [0.5]},
            "vllm_generation_tokens_total": {0: [100.0]},
        }
    out = g.get_logger_metrics()
    assert out["inflight_batch_sizes"] == {0: [3.0, 4.0]}
    assert out["num_pending_samples"] == {0: [1.0]}
    assert out["kv_cache_usage_perc"] == {0: [0.5]}
    assert out["generation_tokens"] == {0: [100.0]}
    assert "vllm_num_requests_running" not in out
