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
import json
import pickle
import urllib.error

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.dynamo import DynamoCfg, DynamoConfig, DynamoGeneration
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
            "engine_world_size": 1,
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
    cfg["dynamo_cfg"] = {"engine_world_size": 1}  # type: ignore[typeddict-item]
    with pytest.raises(RuntimeError, match="dgd_name.*frontend_url"):
        DynamoGeneration(cluster=None, config=cfg)


def test_dynamo_cfg_validates_defaults_and_preserves_extra_fields():
    cfg = DynamoCfg.model_validate(
        {"engine_world_size": 2, "future_dynamo_option": "enabled"}
    )

    assert cfg.frontend_port == 8000
    assert cfg.dyn_system_port == 9090
    assert cfg.deployment == "external"
    assert cfg.worker_args.tool_call_parser is None
    assert cfg.worker_args.reasoning_parser is None
    assert cfg.model_extra == {"future_dynamo_option": "enabled"}


def test_ray_managed_dynamo_requires_inference_cluster(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "deployment": "ray",
        "engine_world_size": 1,
        "frontend_port": 0,
    }
    with pytest.raises(RuntimeError, match="requires an inference RayVirtualCluster"):
        DynamoGeneration(cluster=None, config=cfg)


def test_ray_managed_dynamo_owns_runtime_and_refit_workers(monkeypatch):
    from nemo_rl.models.generation.dynamo import managed_runtime as runtime_module

    calls = []

    class FakeManagedRuntime:
        frontend_url = "http://10.1.2.3:8123/v1"

        def __init__(self, *, cluster, config, dynamo_cfg):
            calls.append(("init", cluster, dynamo_cfg.deployment))

        def refit_workers(self):
            return [
                {
                    "instance_id": "worker-0",
                    "system_url": "http://10.2.3.4:29000",
                }
            ]

        def validate_workers(self, expected):
            calls.append(("validate", expected))
            return expected

        def shutdown(self):
            calls.append(("shutdown",))

    monkeypatch.setattr(runtime_module, "ManagedDynamoRuntime", FakeManagedRuntime)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "deployment": "ray",
        "engine_world_size": 1,
        "frontend_port": 0,
    }
    cluster = object()
    generation = DynamoGeneration(cluster=cluster, config=cfg)

    assert generation.dp_openai_server_base_urls == ["http://10.1.2.3:8123/v1"]
    assert generation.get_inference_world_size() == 1
    assert generation._validate_refit_workers() == [
        {"instance_id": "worker-0", "system_url": "http://10.2.3.4:29000"}
    ]
    assert generation.shutdown() is True
    assert calls[0] == ("init", cluster, "ray")
    assert calls[-1] == ("shutdown",)


def test_dynamo_cfg_rejects_nonpositive_engine_world_size():
    with pytest.raises(ValidationError, match="engine_world_size"):
        DynamoCfg.model_validate({"engine_world_size": 0})


def test_explicit_frontend_url_skips_k8s_check(monkeypatch):
    """frontend_url opts out of the in-pod check — works on slurm, laptop, etc."""
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "engine_world_size": 1,
        "frontend_url": "http://my-dgd.example.com:8000/v1",
    }
    g = DynamoGeneration(cluster=None, config=cfg)
    assert g.dp_openai_server_base_urls == ["http://my-dgd.example.com:8000/v1"]


def test_explicit_frontend_url_overrides_dgd_name(in_k8s):
    cfg = _base_config(frontend_url="http://override.example.com:9000/v1")
    g = DynamoGeneration(cluster=None, config=cfg)
    assert g.dp_openai_server_base_urls == ["http://override.example.com:9000/v1"]


def test_expose_http_server_routes_gym_rollouts_through_token_wrapper(monkeypatch):
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "engine_world_size": 1,
        "frontend_url": "http://dynamo.example.com:8000/v1",
        "request_timeout_s": 30.0,
    }
    cfg["vllm_cfg"] = {"expose_http_server": True}  # type: ignore[typeddict-item]
    created_wrappers = []

    class FakeTokenWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_called = False
            created_wrappers.append(self)

        def start(self):
            return "http://wrapper.example.com:9000/v1"

        def shutdown(self):
            self.shutdown_called = True

    tokenizer = object()
    monkeypatch.setattr(_dynmod, "DynamoTokenWrapperServer", FakeTokenWrapper)

    g = DynamoGeneration(
        cluster=None,
        config=cfg,
        tokenizer=tokenizer,
        tokenizer_config={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert g.dp_openai_server_base_urls == ["http://wrapper.example.com:9000/v1"]
    assert g._completion_url() == "http://dynamo.example.com:8000/v1/completions"
    assert created_wrappers[0].kwargs == {
        "dynamo_frontend_base_url": "http://dynamo.example.com:8000/v1",
        "tokenizer": tokenizer,
        "tokenizer_chat_template_kwargs": {"enable_thinking": False},
        "request_timeout_s": 30.0,
    }

    assert g.shutdown() is True
    assert created_wrappers[0].shutdown_called is True


def test_expose_http_server_requires_tokenizer():
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "engine_world_size": 1,
        "frontend_url": "http://dynamo.example.com:8000/v1",
    }
    cfg["vllm_cfg"] = {"expose_http_server": True}  # type: ignore[typeddict-item]

    with pytest.raises(RuntimeError, match="requires a tokenizer"):
        DynamoGeneration(cluster=None, config=cfg)


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


def test_ipc_weight_update_is_unsupported(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config())
    with pytest.raises(NotImplementedError):
        g.update_weights_via_ipc_zmq()


def test_prepare_refit_info_serializes_vllm_metadata(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config())
    g.prepare_refit_info(
        {
            "model.embed.weight": (torch.Size([4, 8]), torch.bfloat16),
            "model.norm.weight": (torch.Size([8]), torch.float32),
        }
    )
    assert g._refit_update_info == {
        "names": ["model.embed.weight", "model.norm.weight"],
        "dtype_names": ["bfloat16", "float32"],
        "shapes": [[4, 8], [8]],
        "packed": True,
    }


def test_discover_worker_instances_ignores_non_rl_endpoint(monkeypatch):
    payload = {
        "instances": [
            {
                "namespace": "test-ns-my-dgd",
                "component": "backend",
                "endpoint": "update_weights",
                "instance_id": "worker-0",
                "transport": {"tcp": "10.0.0.5:1234/channel/update_weights"},
            },
            {
                "namespace": "test-ns-my-dgd",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": "worker-0",
                "transport": {"tcp": "10.0.0.5:1234/channel/generate"},
            },
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(url, timeout):
        assert url == "http://my-dgd-frontend.test-ns.svc.cluster.local:8000/health"
        assert timeout == 15.0
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    instances = _dynmod._discover_worker_instances(
        frontend_host="my-dgd-frontend.test-ns.svc.cluster.local",
        frontend_port=8000,
        dyn_namespaces={"test-ns-my-dgd"},
        dyn_system_port=9090,
    )

    assert instances == []


def test_discover_worker_instances_accepts_enable_rl_endpoint(monkeypatch):
    payload = {
        "instances": [
            {
                "namespace": "test-ns-my-dgd",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-0",
                "transport": {"tcp": "10.0.0.8:5555/channel/rl"},
            },
            {
                "namespace": "test-ns-my-dgd",
                "component": "Planner",
                "endpoint": "rl",
                "instance_id": "planner",
                "transport": {"tcp": "10.0.0.9:5555/channel/rl"},
            },
            {
                "namespace": "other-ns-other-dgd",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-other",
                "transport": {"tcp": "10.0.0.10:5555/channel/rl"},
            },
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    instances = _dynmod._discover_worker_instances(
        frontend_host="my-dgd-frontend.test-ns.svc.cluster.local",
        frontend_port=8000,
        dyn_namespaces={"test-ns-my-dgd"},
        dyn_system_port=9090,
    )

    assert instances == [
        {
            "instance_id": "worker-0",
            "system_url": "http://10.0.0.8:9090",
        }
    ]


@pytest.mark.parametrize(
    ("transport_url", "expected_system_url"),
    [
        ("10.0.0.8:5555/channel/rl", "http://10.0.0.8:9090"),
        ("tcp://10.0.0.8:5555/channel/rl", "http://10.0.0.8:9090"),
        ("tcp://[fd00::8]:5555/channel/rl", "http://[fd00::8]:9090"),
    ],
)
def test_discover_worker_instances_parses_transport_hosts(
    monkeypatch, transport_url, expected_system_url
):
    payload = {
        "instances": [
            {
                "namespace": "test-ns-my-dgd",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-0",
                "transport": {"tcp": transport_url},
            }
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    assert _dynmod._discover_worker_instances(
        frontend_host="my-dgd-frontend.test-ns.svc.cluster.local",
        frontend_port=8000,
        dyn_namespaces={"test-ns-my-dgd"},
        dyn_system_port=9090,
    ) == [{"instance_id": "worker-0", "system_url": expected_system_url}]


def test_discover_worker_instances_retries_transient_failure(monkeypatch):
    payload = {"instances": []}
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.URLError("temporary DNS failure")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _delay: None)

    assert (
        _dynmod._discover_worker_instances(
            frontend_host="my-dgd-frontend.test-ns.svc.cluster.local",
            frontend_port=8000,
            dyn_namespaces={"test-ns-my-dgd"},
            dyn_system_port=9090,
        )
        == []
    )
    assert calls == 3


def test_discover_worker_instances_reports_exhausted_fetch_not_membership(
    monkeypatch,
):
    calls = 0

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("frontend busy")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _delay: None)

    with pytest.raises(
        _dynmod._WorkerDiscoveryError,
        match="failed after 3 attempts",
    ) as exc_info:
        _dynmod._discover_worker_instances(
            frontend_host="my-dgd-frontend.test-ns.svc.cluster.local",
            frontend_port=8000,
            dyn_namespaces={"test-ns-my-dgd"},
            dyn_system_port=9090,
        )

    assert "membership changed" not in str(exc_info.value)
    assert calls == 3


def test_init_collective_assigns_rank_offsets_per_worker(
    in_k8s, stub_namespace, monkeypatch
):
    workers = [
        {"instance_id": "worker-a", "system_url": "http://10.0.0.1:9090"},
        {"instance_id": "worker-b", "system_url": "http://10.0.0.2:9090"},
    ]
    monkeypatch.setattr(_dynmod, "_discover_worker_instances", lambda **kwargs: workers)
    calls = []

    def fake_remote(**kwargs):
        calls.append(kwargs)
        return f"ref-{len(calls)}"

    monkeypatch.setattr(_dynmod._post_dynamo_worker_route_remote, "remote", fake_remote)

    g = DynamoGeneration(cluster=None, config=_base_config(engine_world_size=2))
    refs = g.init_collective(
        ip="10.1.0.1",
        port=23456,
        world_size=7,
        train_world_size=3,
    )

    assert refs == ["ref-1", "ref-2"]
    assert [call["payload"]["init_info"]["rank_offset"] for call in calls] == [
        3,
        5,
    ]
    assert all(
        call["payload"]["engine_rpc"] == "init_weight_transfer_engine"
        and call["payload"]["init_info"]["world_size"] == 7
        for call in calls
    )


def test_update_weights_uses_fixed_worker_snapshot(in_k8s, stub_namespace, monkeypatch):
    workers = [
        {"instance_id": "worker-a", "system_url": "http://10.0.0.1:9090"},
        {"instance_id": "worker-b", "system_url": "http://10.0.0.2:9090"},
    ]
    monkeypatch.setattr(_dynmod, "_discover_worker_instances", lambda **kwargs: workers)
    calls = []

    def fake_remote(**kwargs):
        calls.append(kwargs)
        return f"ref-{len(calls)}"

    monkeypatch.setattr(
        _dynmod._update_dynamo_worker_weights_remote, "remote", fake_remote
    )

    g = DynamoGeneration(cluster=None, config=_base_config())
    assert g.get_inference_world_size() == 2
    g.prepare_refit_info({"weight": (torch.Size([2, 2]), torch.bfloat16)})
    assert g.update_weights_from_collective() == ["ref-1", "ref-2"]
    assert [call["system_url"] for call in calls] == [
        "http://10.0.0.1:9090",
        "http://10.0.0.2:9090",
    ]
    assert calls[0]["update_info"] == {
        "names": ["weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "packed": True,
    }

    monkeypatch.setattr(
        _dynmod,
        "_discover_worker_instances",
        lambda **kwargs: workers[:1],
    )
    with pytest.raises(RuntimeError, match="membership changed"):
        g.update_weights_from_collective()


def test_native_vllm_update_transaction_uses_async_rl_controls(monkeypatch):
    calls = []

    def fake_post(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post)
    result = _dynmod._update_dynamo_worker_weights_remote._function(
        system_url="http://10.0.0.1:9090",
        update_info={"names": ["weight"], "packed": True},
        timeout_s=30.0,
    )

    assert result is True
    assert [payload["engine_rpc"] for _, payload, _ in calls] == [
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ]
    assert calls[0][1]["is_checkpoint_format"] is True
    assert calls[1][1]["update_info"] == {"names": ["weight"], "packed": True}
    assert all(
        payload["allow_unpaused"] is True and payload["reset_prefix_cache"] is False
        for _, payload, _ in calls
    )


def test_dynamo_kv_scale_sync_is_deferred(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    assert DynamoGeneration(cluster=None, config=cfg).requires_kv_scale_sync is False

    cfg["vllm_cfg"] = {"kv_cache_dtype": "auto"}  # type: ignore[typeddict-item]
    assert DynamoGeneration(cluster=None, config=cfg).requires_kv_scale_sync is False

    cfg["vllm_cfg"] = {"kv_cache_dtype": "fp8"}  # type: ignore[typeddict-item]
    assert DynamoGeneration(cluster=None, config=cfg).requires_kv_scale_sync is False

    cfg["vllm_cfg"] = {"kv_cache_dtype": "fp8_e4m3"}  # type: ignore[typeddict-item]
    assert DynamoGeneration(cluster=None, config=cfg).requires_kv_scale_sync is False


def test_pickle_roundtrip_restores_process_local_refit_state(
    in_k8s, stub_namespace, monkeypatch
):
    g = DynamoGeneration(cluster=None, config=_base_config(frontend_port=8123))
    expected_url = g.dp_openai_server_base_urls[0]
    g._refit_workers = [
        {"instance_id": "stale-worker", "system_url": "http://10.0.0.1:9090"}
    ]
    g._refit_discovery_kwargs = {"frontend_host": "stale-host"}
    g._refit_update_info = {"names": ["stale-weight"]}

    restored = pickle.loads(pickle.dumps(g))
    assert restored.dp_openai_server_base_urls == [expected_url]
    assert restored.cfg["dynamo_cfg"]["dgd_name"] == "my-dgd"
    assert restored._dynamo_cfg.frontend_port == 8123
    assert restored._refit_workers is None
    assert restored._refit_discovery_kwargs is None
    assert restored._refit_update_info is None

    workers = [{"instance_id": "worker-a", "system_url": "http://10.0.0.2:9090"}]
    monkeypatch.setattr(_dynmod, "_discover_worker_instances", lambda **kwargs: workers)
    calls = []

    def fake_remote(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(_dynmod._post_dynamo_worker_route_remote, "remote", fake_remote)
    monkeypatch.setattr(_dynmod.ray, "get", lambda refs: refs)

    assert restored.invalidate_kv_cache() is True
    assert calls == [
        {
            "system_url": "http://10.0.0.2:9090",
            "route": "flush_cache",
            "payload": {},
            "timeout_s": 30.0,
        }
    ]


# ---------------------------------------------------------------------------
# Direct generation — OpenAI completions payload + vLLM-parity tensors.
# ---------------------------------------------------------------------------


def test_generate_requires_request_timeout_s(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config()
    cfg["dynamo_cfg"] = {  # type: ignore[typeddict-item]
        "engine_world_size": 1,
        "frontend_url": "http://my-dgd.example.com:8000/v1",
    }
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
        assert payload["nvext"] == {"extra_fields": ["completion_token_ids"]}
        assert "greed_sampling" not in payload["nvext"]
    assert set(calls[0][1]["stop"]) == {"global-stop", "sample-a"}
    assert set(calls[1][1]["stop"]) == {"global-stop", "sample-b"}

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
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.7]},
                }
            ],
            "nvext": {"completion_token_ids": [9]},
        }

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)
    out = g.generate(_generation_data([[7, 8]], [2]), greedy=True)

    payload = calls[0][1]
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["top_p"] == 0.75
    assert payload["logprobs"] == 0
    assert payload["nvext"] == {"extra_fields": ["completion_token_ids"]}
    assert "greed_sampling" not in payload["nvext"]
    assert out["output_ids"].tolist() == [[7, 8, 9]]
    assert torch.allclose(out["logprobs"], torch.tensor([[0.0, 0.0, -0.7]]))


def test_generate_retries_then_raises_on_transient_http_error(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    calls = 0

    def fake_post_json(url, payload, timeout_s):
        nonlocal calls
        calls += 1
        return {"status": "error", "http_status": 500, "raw": "boom"}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _delay: None)
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="HTTP 500: boom"):
        g.generate(_generation_data([[1]], [1]))
    assert calls == 3


def test_generate_recovers_from_transient_http_error(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    responses = [
        {"status": "error", "transport_error": "temporary reset"},
        {"status": "error", "json_decode_error": True, "raw": "partial"},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.2]},
                }
            ],
            "nvext": {"completion_token_ids": [9]},
        },
    ]
    calls = 0

    def fake_post_json(url, payload, timeout_s):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _delay: None)
    g = DynamoGeneration(cluster=None, config=cfg)

    out = g.generate(_generation_data([[1]], [1]))

    assert calls == 3
    assert out["output_ids"].tolist() == [[1, 9]]


def test_generate_does_not_retry_nontransient_http_error(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    calls = 0

    def fake_post_json(url, payload, timeout_s):
        nonlocal calls
        calls += 1
        return {"status": "error", "http_status": 400, "raw": "bad request"}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="HTTP 400: bad request"):
        g.generate(_generation_data([[1]], [1]))
    assert calls == 1


def test_generate_raises_on_missing_completion_token_ids(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")

    def fake_post_json(url, payload, timeout_s):
        return {"choices": [{"finish_reason": "stop"}]}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    with pytest.raises(RuntimeError, match="did not include nvext"):
        g.generate(_generation_data([[1]], [1]))


@pytest.mark.parametrize(
    ("logprobs", "match"),
    [
        (None, "did not include choice.logprobs"),
        ({}, "token_logprobs"),
        ({"token_logprobs": [-0.1]}, "1 token logprobs for 2 generated tokens"),
        ({"token_logprobs": [-0.1, None]}, "invalid logprob"),
    ],
)
def test_parse_completion_rejects_missing_or_misaligned_logprobs(logprobs, match):
    response = {
        "choices": [{"finish_reason": "stop", "logprobs": logprobs}],
        "nvext": {"completion_token_ids": [8, 9]},
    }

    with pytest.raises(RuntimeError, match=match):
        _dynmod._parse_dynamo_completion_response(
            response,
            request_url="http://dynamo.example.com/v1/completions",
        )


def test_generate_caps_context(monkeypatch):
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

    out = g.generate(_generation_data([[1, 2, 3, 0]], [3]))

    assert calls[0][1]["max_tokens"] == 2
    assert out["generation_lengths"].tolist() == [1]


def test_generate_zero_budget_skips_http(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg["vllm_cfg"] = {"max_model_len": 3}  # type: ignore[typeddict-item]

    def fake_post_json(url, payload, timeout_s):
        raise AssertionError("HTTP should not be called when no context remains")

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    out = g.generate(_generation_data([[1, 2, 3, 0]], [3]))

    assert out["output_ids"].tolist() == [[1, 2, 3, 0]]
    assert out["generation_lengths"].tolist() == [0]
    assert out["unpadded_sequence_lengths"].tolist() == [3]


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
    assert calls[0][1]["logprobs"] == 0
    assert calls[0][1]["prompt"] == [1, 2, 3]
    assert calls[0][1]["stop"] == ["sample-stop"]
    assert out["output_ids"].tolist() == [[1, 2, 3, 8]]
    assert out["generation_lengths"].tolist() == [1]
    assert out["unpadded_sequence_lengths"].tolist() == [4]
    assert out["truncated"].tolist() == [False]
    assert torch.allclose(out["logprobs"], torch.tensor([[0.0, 0.0, 0.0, -0.4]]))


def test_generate_async_recovers_from_transient_http_error(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    responses = [
        {"status": "error", "http_status": 503, "raw": "frontend busy"},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.4]},
                }
            ],
            "nvext": {"completion_token_ids": [8]},
        },
    ]
    calls = 0

    def fake_post_json(url, payload, timeout_s):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _delay: None)
    g = DynamoGeneration(cluster=None, config=cfg)

    async def collect():
        return [item async for item in g.generate_async(_generation_data([[1]], [1]))]

    results = asyncio.run(collect())

    assert calls == 2
    assert results[0][1]["output_ids"].tolist() == [[1, 8]]


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


def test_generate_async_without_max_model_len_uses_configured_budget(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = _base_config(frontend_url="http://my-dgd.example.com:8000/v1")
    cfg.pop("vllm_cfg", None)
    calls = []

    def fake_post_json(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.2]},
                }
            ],
            "nvext": {"completion_token_ids": [4]},
        }

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post_json)
    g = DynamoGeneration(cluster=None, config=cfg)

    async def collect():
        return [
            item async for item in g.generate_async(_generation_data([[1, 2, 0]], [2]))
        ]

    results = asyncio.run(collect())

    assert calls[0][1]["max_tokens"] == 16
    assert calls[0][1]["logprobs"] == 0
    assert results[0][1]["generation_lengths"].tolist() == [1]


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
# Dynamo engine telemetry.
# ---------------------------------------------------------------------------


def test_parse_prometheus_metrics_sanitizes_and_sums_labels():
    text = (
        "# HELP vllm:num_requests_running running\n"
        'vllm:num_requests_running{model="x"} 3.0\n'
        'vllm:request_success_total{reason="stop"} 5\n'
        'vllm:request_success_total{reason="length"} 3\n'
        'dynamo_component_kv_cache_hit_rate{dp_rank="0"} 0.42\n'
    )

    metrics = _dynmod._parse_prometheus_metrics(text)

    assert metrics["vllm_num_requests_running"] == 3.0
    assert metrics["vllm_request_success_total"] == 8.0
    assert metrics["dynamo_component_kv_cache_hit_rate"] == 0.42
    assert all(":" not in name for name in metrics)


def test_parse_prometheus_metrics_filters_non_scalar_samples():
    text = (
        'vllm:ttft_seconds_bucket{le="0.1"} 5\n'
        "vllm:ttft_seconds_sum 1.5\n"
        "vllm:ttft_seconds_count 10\n"
        "vllm:generation_tokens_created 1.749e9\n"
        'python_gc_objects_collected_total{generation="0"} 100\n'
        "process_cpu_seconds_total 1.5\n"
        "dynamo_component_requests_total 2\n"
    )

    metrics = _dynmod._parse_prometheus_metrics(text)

    assert metrics == {
        "vllm_ttft_seconds_sum": 1.5,
        "vllm_ttft_seconds_count": 10.0,
        "dynamo_component_requests_total": 2.0,
    }


def test_parse_prometheus_metrics_include_and_exclude_prefixes():
    text = "vllm:foo 1\ndynamo_component_keep 2\ndynamo_component_drop 3\n"

    metrics = _dynmod._parse_prometheus_metrics(
        text,
        include_prefixes=("dynamo_component_",),
        exclude_prefixes=("dynamo_component_drop",),
    )

    assert metrics == {"dynamo_component_keep": 2.0}


def test_parse_prometheus_metrics_skips_malformed_lines():
    text = "garbage\nname_only\nmissing_brace{value 1\nnot_numeric nope\nok 1\n"
    assert _dynmod._parse_prometheus_metrics(text) == {"ok": 1.0}


def test_http_get_text_returns_none_on_transport_error(monkeypatch):
    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", raise_url_error)

    assert _dynmod._http_get_text("http://worker:9090/metrics", 1.0) is None


def test_logger_metrics_disabled_by_default(in_k8s, stub_namespace):
    generation = DynamoGeneration(cluster=None, config=_base_config())

    assert generation.get_logger_metrics() == {}
    generation.clear_logger_metrics()
    assert generation.get_logger_metrics() == {}


def test_logger_metrics_enabled_shape_copy_and_clear(
    in_k8s, stub_namespace, monkeypatch
):
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config(
        metrics_include_prefixes=[],
        metrics_exclude_prefixes=[],
    )
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }

    generation = DynamoGeneration(cluster=None, config=cfg)

    assert generation._metrics_include_prefixes == ()
    assert generation._metrics_exclude_prefixes == ()
    with generation._metrics_lock:
        generation._dynamo_logger_metrics = {"custom_metric": {0: [1.0, 2.0], 1: [3.0]}}

    metrics = generation.get_logger_metrics()
    assert metrics["custom_metric"] == {0: [1.0, 2.0], 1: [3.0]}
    metrics["custom_metric"][0].append(999.0)
    assert generation.get_logger_metrics()["custom_metric"][0] == [1.0, 2.0]

    generation.clear_logger_metrics()
    assert "custom_metric" not in generation.get_logger_metrics()


def test_logger_metrics_canonical_aliases_drop_raw_sources(
    in_k8s, stub_namespace, monkeypatch
):
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config()
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    generation = DynamoGeneration(cluster=None, config=cfg)

    with generation._metrics_lock:
        generation._dynamo_logger_metrics = {
            "dynamo_component_inflight_requests": {0: [3.0, 4.0]},
            "dynamo_work_handler_queue_depth": {0: [1.0]},
            "dynamo_component_gpu_cache_usage_percent": {0: [0.5]},
            "vllm_generation_tokens_total": {0: [100.0]},
        }

    metrics = generation.get_logger_metrics()

    assert metrics["inflight_batch_sizes"] == {0: [3.0, 4.0]}
    assert metrics["num_pending_samples"] == {0: [1.0]}
    assert metrics["kv_cache_usage_perc"] == {0: [0.5]}
    assert metrics["generation_tokens"] == {0: [100.0]}
    assert "dynamo_component_inflight_requests" not in metrics


def test_pickle_roundtrip_with_metrics_enabled(in_k8s, stub_namespace, monkeypatch):
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    cfg = _base_config()
    cfg["vllm_cfg"] = {  # type: ignore[typeddict-item]
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    generation = DynamoGeneration(cluster=None, config=cfg)

    restored = pickle.loads(pickle.dumps(generation))

    assert restored.get_logger_metrics() == {}
    restored.clear_logger_metrics()
    assert restored.shutdown() is True
