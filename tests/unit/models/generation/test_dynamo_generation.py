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
class's contract: K8s detection, URL derivation, lifecycle no-ops, and that
unsupported methods fail loudly. End-to-end coverage is provided separately
in the Phase 2 integration tests once nrl-k8s can stand up a DGD.
"""

import pickle

import pytest

from nemo_rl.models.generation.dynamo import DynamoConfig, DynamoGeneration
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
        "dynamo_cfg": {"dgd_name": "my-dgd", **dynamo_cfg_overrides},
    }
    return cfg


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
        g.generate(data=None, greedy=False)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        g.init_collective(ip="127.0.0.1", port=1, world_size=1)
    with pytest.raises(NotImplementedError):
        g.prepare_refit_info({})
    with pytest.raises(NotImplementedError):
        g.update_weights_via_ipc_zmq()
    with pytest.raises(NotImplementedError):
        g.update_weights_from_collective()


def test_pickle_roundtrip(in_k8s, stub_namespace):
    g = DynamoGeneration(cluster=None, config=_base_config(frontend_port=8123))
    expected_url = g.dp_openai_server_base_urls[0]
    restored = pickle.loads(pickle.dumps(g))
    assert restored.dp_openai_server_base_urls == [expected_url]
    assert restored.cfg["dynamo_cfg"]["dgd_name"] == "my-dgd"
