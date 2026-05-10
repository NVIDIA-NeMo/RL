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

"""Tests for :mod:`nrl_k8s.dgd` — DynamoGraphDeployment ingestion."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from nrl_k8s import dgd, k8s
from nrl_k8s.schema import DynamoGraphSpec, InfraConfig

# =============================================================================
# Shared fixtures (mirror test_k8s.py)
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_load_kubeconfig_cache():
    k8s.load_kubeconfig.cache_clear()
    yield
    k8s.load_kubeconfig.cache_clear()


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _no_real_kubeconfig(monkeypatch):
    monkeypatch.setattr(k8s.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(k8s.config, "load_kube_config", lambda: None)


@pytest.fixture
def mock_custom_api(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(dgd, "custom_objects_api", lambda: api)
    return api


def _api_exc(status: int) -> ApiException:
    return ApiException(status=status)


def _infra(**overrides) -> InfraConfig:
    return InfraConfig.model_validate(
        {
            "namespace": "test-ns",
            "image": "registry/img:tag",
            "imagePullSecrets": ["pull-secret"],
            **overrides,
        }
    )


# =============================================================================
# build_dgd_manifest
# =============================================================================

_INLINE_SPEC = {
    "services": {
        "Frontend": {
            "componentType": "frontend",
            "replicas": 1,
            "extraPodSpec": {
                "mainContainer": {
                    "image": "registry/upstream:1.0",
                    "command": ["python3", "-m", "dynamo.frontend"],
                },
            },
        },
        "VllmDecodeWorker": {
            "componentType": "worker",
            "replicas": 4,
            "extraPodSpec": {
                "mainContainer": {
                    "command": ["python3", "-m", "dynamo.vllm"],
                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                },
            },
        },
    },
}


class TestBuildDgdManifest:
    def _spec(self, **kw) -> DynamoGraphSpec:
        return DynamoGraphSpec(spec=copy.deepcopy(_INLINE_SPEC), name="my-dgd", **kw)

    def test_envelope_built(self):
        m = dgd.build_dgd_manifest(self._spec(), _infra())
        assert m["apiVersion"] == "nvidia.com/v1alpha1"
        assert m["kind"] == "DynamoGraphDeployment"
        assert m["metadata"]["name"] == "my-dgd"
        assert m["metadata"]["namespace"] == "test-ns"

    def test_spec_deep_copied(self):
        spec = self._spec()
        original_replicas = spec.spec["services"]["VllmDecodeWorker"]["replicas"]
        m = dgd.build_dgd_manifest(spec, _infra())
        m["spec"]["services"]["VllmDecodeWorker"]["replicas"] = 999
        assert spec.spec["services"]["VllmDecodeWorker"]["replicas"] == original_replicas

    def test_image_default_when_unset(self):
        m = dgd.build_dgd_manifest(self._spec(), _infra())
        services = m["spec"]["services"]
        assert services["Frontend"]["extraPodSpec"]["mainContainer"]["image"] == "registry/upstream:1.0"
        assert services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["image"] == "registry/img:tag"

    def test_image_pull_secrets(self):
        m = dgd.build_dgd_manifest(self._spec(), _infra())
        for svc in m["spec"]["services"].values():
            assert svc["extraPodSpec"]["imagePullSecrets"] == [{"name": "pull-secret"}]

    def test_managed_by_label(self):
        m = dgd.build_dgd_manifest(self._spec(), _infra())
        assert m["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_user_labels_merged(self):
        spec = self._spec(labels={"team": "rl"})
        m = dgd.build_dgd_manifest(spec, _infra())
        assert m["metadata"]["labels"]["team"] == "rl"
        assert m["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_user_annotations_merged(self):
        spec = self._spec(annotations={"nvidia.com/kai-scheduler-queue": "backfill"})
        m = dgd.build_dgd_manifest(spec, _infra())
        assert m["metadata"]["annotations"]["nvidia.com/kai-scheduler-queue"] == "backfill"

    def test_owner_ref_attached(self):
        owner = dgd.build_owner_reference(
            api_version="ray.io/v1",
            kind="RayCluster",
            name="rc-train",
            uid="abc-123",
        )
        m = dgd.build_dgd_manifest(self._spec(), _infra(), owner_ref=owner)
        assert len(m["metadata"]["ownerReferences"]) == 1
        assert m["metadata"]["ownerReferences"][0]["name"] == "rc-train"

    def test_no_owner_ref_when_unset(self):
        m = dgd.build_dgd_manifest(self._spec(), _infra())
        assert "ownerReferences" not in m["metadata"]


# =============================================================================
# build_owner_reference
# =============================================================================


class TestBuildOwnerReference:
    def test_default_fields(self):
        ref = dgd.build_owner_reference(
            api_version="ray.io/v1", kind="RayCluster", name="x", uid="u"
        )
        assert ref["controller"] is False
        assert ref["blockOwnerDeletion"] is False

    def test_block_owner_deletion_true(self):
        ref = dgd.build_owner_reference(
            api_version="ray.io/v1",
            kind="RayCluster",
            name="x",
            uid="u",
            block_owner_deletion=True,
        )
        assert ref["blockOwnerDeletion"] is True


# =============================================================================
# resolve_dgd_name
# =============================================================================


class TestResolveDgdName:
    def test_returns_name(self):
        spec = DynamoGraphSpec(spec={"services": {}}, name="my-dgd")
        assert dgd.resolve_dgd_name(spec) == "my-dgd"


# =============================================================================
# apply_dgd / get_dgd / delete_dgd
# =============================================================================


class TestApplyDgd:
    def _manifest(self) -> dict:
        return {
            "apiVersion": dgd.DGD_API_VERSION,
            "kind": dgd.DGD_KIND,
            "metadata": {"name": "x", "namespace": "ns"},
            "spec": {},
        }

    def test_create_happy_path(self, mock_custom_api):
        mock_custom_api.create_namespaced_custom_object.return_value = {"x": 1}
        out = dgd.apply_dgd(self._manifest(), "ns")
        assert out == {"x": 1}
        kwargs = mock_custom_api.create_namespaced_custom_object.call_args.kwargs
        assert kwargs["group"] == "nvidia.com"
        assert kwargs["version"] == "v1alpha1"
        assert kwargs["plural"] == "dynamographdeployments"

    def test_409_falls_back_to_patch(self, mock_custom_api):
        mock_custom_api.create_namespaced_custom_object.side_effect = _api_exc(409)
        mock_custom_api.patch_namespaced_custom_object.return_value = {"y": 2}
        out = dgd.apply_dgd(self._manifest(), "ns")
        assert out == {"y": 2}
        mock_custom_api.patch_namespaced_custom_object.assert_called_once()

    def test_other_error_propagates(self, mock_custom_api):
        mock_custom_api.create_namespaced_custom_object.side_effect = _api_exc(500)
        with pytest.raises(ApiException):
            dgd.apply_dgd(self._manifest(), "ns")


class TestGetDgd:
    def test_returns_object(self, mock_custom_api):
        mock_custom_api.get_namespaced_custom_object.return_value = {"x": 1}
        assert dgd.get_dgd("foo", "ns") == {"x": 1}

    def test_404_returns_none(self, mock_custom_api):
        mock_custom_api.get_namespaced_custom_object.side_effect = _api_exc(404)
        assert dgd.get_dgd("foo", "ns") is None


class TestDeleteDgd:
    def test_ignore_missing(self, mock_custom_api):
        mock_custom_api.delete_namespaced_custom_object.side_effect = _api_exc(404)
        # Should not raise.
        dgd.delete_dgd("foo", "ns")

    def test_other_error_propagates(self, mock_custom_api):
        mock_custom_api.delete_namespaced_custom_object.side_effect = _api_exc(500)
        with pytest.raises(ApiException):
            dgd.delete_dgd("foo", "ns")


class TestWaitForDgdReady:
    def test_returns_when_successful(self, mock_custom_api, monkeypatch):
        monkeypatch.setattr(dgd.time, "sleep", lambda _s: None)
        mock_custom_api.get_namespaced_custom_object.side_effect = [
            {"status": {"state": "pending"}},
            {"status": {"state": "successful"}},
        ]
        dgd.wait_for_dgd_ready("foo", "ns", timeout_s=10, poll_s=0)

    def test_raises_on_failed(self, mock_custom_api, monkeypatch):
        monkeypatch.setattr(dgd.time, "sleep", lambda _s: None)
        mock_custom_api.get_namespaced_custom_object.return_value = {
            "status": {"state": "failed", "conditions": [{"type": "Ready", "status": "False"}]}
        }
        with pytest.raises(RuntimeError, match="state=failed"):
            dgd.wait_for_dgd_ready("foo", "ns", timeout_s=10, poll_s=0)

    def test_raises_on_timeout(self, mock_custom_api, monkeypatch):
        monkeypatch.setattr(dgd.time, "sleep", lambda _s: None)
        # Stay forever at "pending" — the deadline check trips.
        mock_custom_api.get_namespaced_custom_object.return_value = {
            "status": {"state": "pending"}
        }
        # Use a tiny timeout so the wall clock crosses it after one iteration.
        with pytest.raises(TimeoutError, match="never reached state="):
            dgd.wait_for_dgd_ready("foo", "ns", timeout_s=0, poll_s=0)


# =============================================================================
# CRD precondition check
# =============================================================================


class TestIsDgdCrdInstalled:
    def test_returns_true_when_list_succeeds(self, mock_custom_api):
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}
        assert dgd.is_dgd_crd_installed("ns-a") is True

    def test_returns_false_on_404(self, mock_custom_api):
        mock_custom_api.list_namespaced_custom_object.side_effect = _api_exc(404)
        assert dgd.is_dgd_crd_installed("ns-a") is False

    def test_403_treated_as_installed(self, mock_custom_api):
        # CRD registered but the user lacks list-RBAC — don't block; the apply
        # step will surface any remaining permission issues with a better message.
        mock_custom_api.list_namespaced_custom_object.side_effect = _api_exc(403)
        assert dgd.is_dgd_crd_installed("ns-a") is True

    def test_other_error_propagates(self, mock_custom_api):
        mock_custom_api.list_namespaced_custom_object.side_effect = _api_exc(500)
        with pytest.raises(ApiException):
            dgd.is_dgd_crd_installed("ns-a")
