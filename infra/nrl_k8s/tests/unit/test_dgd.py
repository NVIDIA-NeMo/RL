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

import textwrap
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


_DGD_YAML = textwrap.dedent(
    """\
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeployment
    metadata:
      name: agg-from-disk
    spec:
      services:
        Frontend:
          componentType: frontend
          replicas: 1
          extraPodSpec:
            mainContainer:
              image: registry/upstream:1.0
              command: ["python3", "-m", "dynamo.frontend"]
        VllmDecodeWorker:
          componentType: worker
          replicas: 4
          extraPodSpec:
            mainContainer:
              command: ["python3", "-m", "dynamo.vllm"]
              args: ["--model", "Qwen/Qwen3-0.6B"]
          resources:
            limits:
              gpu: "1"
    """
)


# =============================================================================
# load_dgd_manifest
# =============================================================================


class TestLoadDgdManifest:
    def test_happy_path(self, tmp_path):
        f = tmp_path / "dgd.yaml"
        f.write_text(_DGD_YAML)
        doc = dgd.load_dgd_manifest("dgd.yaml", base_dir=tmp_path)
        assert doc["kind"] == "DynamoGraphDeployment"
        assert doc["metadata"]["name"] == "agg-from-disk"

    def test_picks_dgd_doc_in_multidoc(self, tmp_path):
        f = tmp_path / "dgd.yaml"
        # First doc is a benchmark Pod; the DGD comes second.
        f.write_text(
            textwrap.dedent(
                """\
                apiVersion: v1
                kind: Pod
                metadata:
                  name: benchmark
                ---
                """
            )
            + _DGD_YAML
        )
        doc = dgd.load_dgd_manifest("dgd.yaml", base_dir=tmp_path)
        assert doc["kind"] == "DynamoGraphDeployment"

    def test_rejects_no_dgd_doc(self, tmp_path):
        f = tmp_path / "dgd.yaml"
        f.write_text("apiVersion: v1\nkind: Pod\nmetadata: {name: x}\n")
        with pytest.raises(ValueError, match="no document with kind=DynamoGraphDeployment"):
            dgd.load_dgd_manifest("dgd.yaml", base_dir=tmp_path)

    def test_rejects_wrong_apiversion(self, tmp_path):
        f = tmp_path / "dgd.yaml"
        f.write_text(
            "apiVersion: nvidia.com/v1beta1\n"
            "kind: DynamoGraphDeployment\n"
            "metadata: {name: x}\n"
        )
        with pytest.raises(ValueError, match="expected apiVersion"):
            dgd.load_dgd_manifest("dgd.yaml", base_dir=tmp_path)

    def test_resolves_relative_path(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "dgd.yaml").write_text(_DGD_YAML)
        doc = dgd.load_dgd_manifest("sub/dgd.yaml", base_dir=tmp_path)
        assert doc["metadata"]["name"] == "agg-from-disk"

    def test_resolves_absolute_path(self, tmp_path):
        f = tmp_path / "dgd.yaml"
        f.write_text(_DGD_YAML)
        doc = dgd.load_dgd_manifest(str(f), base_dir=tmp_path / "unused")
        assert doc["metadata"]["name"] == "agg-from-disk"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dgd.load_dgd_manifest("nope.yaml", base_dir=tmp_path)


# =============================================================================
# build_dgd_manifest
# =============================================================================


class TestBuildDgdManifest:
    def _write(self, tmp_path, body=_DGD_YAML):
        f = tmp_path / "dgd.yaml"
        f.write_text(body)
        return f

    def test_envelope_preserved(self, tmp_path):
        self._write(tmp_path)
        spec = DynamoGraphSpec(manifest="dgd.yaml")
        m = dgd.build_dgd_manifest(spec, _infra(), tmp_path)
        assert m["apiVersion"] == "nvidia.com/v1alpha1"
        assert m["kind"] == "DynamoGraphDeployment"
        assert m["metadata"]["name"] == "agg-from-disk"
        assert m["metadata"]["namespace"] == "test-ns"

    def test_name_override(self, tmp_path):
        self._write(tmp_path)
        spec = DynamoGraphSpec(manifest="dgd.yaml", name="my-dgd")
        m = dgd.build_dgd_manifest(spec, _infra(), tmp_path)
        assert m["metadata"]["name"] == "my-dgd"

    def test_overrides_deep_merged(self, tmp_path):
        self._write(tmp_path)
        spec = DynamoGraphSpec(
            manifest="dgd.yaml",
            overrides={"services": {"VllmDecodeWorker": {"replicas": 1}}},
        )
        m = dgd.build_dgd_manifest(spec, _infra(), tmp_path)
        services = m["spec"]["services"]
        # Override applied:
        assert services["VllmDecodeWorker"]["replicas"] == 1
        # Sibling field on the same service preserved:
        assert services["VllmDecodeWorker"]["componentType"] == "worker"
        # Other service untouched:
        assert services["Frontend"]["replicas"] == 1

    def test_image_default_when_unset(self, tmp_path):
        # Frontend authors its own image; VllmDecodeWorker doesn't.
        self._write(tmp_path)
        m = dgd.build_dgd_manifest(
            DynamoGraphSpec(manifest="dgd.yaml"), _infra(), tmp_path
        )
        services = m["spec"]["services"]
        # Author-set image survives:
        assert (
            services["Frontend"]["extraPodSpec"]["mainContainer"]["image"]
            == "registry/upstream:1.0"
        )
        # Worker had no image — defaulted to infra.image:
        assert (
            services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["image"]
            == "registry/img:tag"
        )

    def test_image_pull_secrets_propagated(self, tmp_path):
        self._write(tmp_path)
        m = dgd.build_dgd_manifest(
            DynamoGraphSpec(manifest="dgd.yaml"), _infra(), tmp_path
        )
        for svc in m["spec"]["services"].values():
            assert svc["extraPodSpec"]["imagePullSecrets"] == [{"name": "pull-secret"}]

    def test_service_account_when_set(self, tmp_path):
        self._write(tmp_path)
        infra = _infra(serviceAccount="my-sa")
        m = dgd.build_dgd_manifest(
            DynamoGraphSpec(manifest="dgd.yaml"), infra, tmp_path
        )
        for svc in m["spec"]["services"].values():
            assert svc["extraPodSpec"]["serviceAccountName"] == "my-sa"

    def test_managed_by_label(self, tmp_path):
        self._write(tmp_path)
        m = dgd.build_dgd_manifest(
            DynamoGraphSpec(manifest="dgd.yaml"), _infra(), tmp_path
        )
        assert m["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_user_labels_merged(self, tmp_path):
        self._write(tmp_path)
        spec = DynamoGraphSpec(manifest="dgd.yaml", labels={"team": "rl"})
        m = dgd.build_dgd_manifest(spec, _infra(), tmp_path)
        assert m["metadata"]["labels"]["team"] == "rl"
        assert m["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_no_auto_service_field(self, tmp_path):
        # The dynamo operator owns Service creation; the manifest should not
        # carry any additional Service envelope.
        self._write(tmp_path)
        m = dgd.build_dgd_manifest(
            DynamoGraphSpec(manifest="dgd.yaml"), _infra(), tmp_path
        )
        # The output is a DGD manifest, not a multi-doc with a Service appended.
        assert m["kind"] == "DynamoGraphDeployment"
        assert "Service" not in (m.get("spec", {}).get("services") or {})


# =============================================================================
# resolve_dgd_name
# =============================================================================


class TestResolveDgdName:
    def test_returns_explicit_name(self, tmp_path):
        # Manifest doesn't even need to exist when name is set.
        spec = DynamoGraphSpec(manifest="missing.yaml", name="explicit")
        assert dgd.resolve_dgd_name(spec, tmp_path) == "explicit"

    def test_falls_back_to_manifest(self, tmp_path):
        (tmp_path / "dgd.yaml").write_text(_DGD_YAML)
        spec = DynamoGraphSpec(manifest="dgd.yaml")
        assert dgd.resolve_dgd_name(spec, tmp_path) == "agg-from-disk"


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
