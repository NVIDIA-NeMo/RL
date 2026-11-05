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

"""Tests for :mod:`nrl_k8s.k8s` — thin wrapper around the official k8s client.

All tests mock ``kubernetes.client`` and ``kubernetes.config`` so they never
touch a live cluster (no kubeconfig read, no HTTP). We also reset the
``lru_cache`` on :func:`load_kubeconfig` between tests so the in-cluster vs
kubeconfig branch can be exercised independently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from nrl_k8s import k8s

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_load_kubeconfig_cache():
    """Drop the @functools.cache memoisation between tests."""
    k8s.load_kubeconfig.cache_clear()
    yield
    k8s.load_kubeconfig.cache_clear()


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    """Collapse tenacity's backoff so transient-failure tests run fast."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _no_real_kubeconfig(monkeypatch):
    """Stub the config loaders so no test reads a real kubeconfig or /var/run."""
    monkeypatch.setattr(k8s.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(k8s.config, "load_kube_config", lambda: None)


@pytest.fixture
def mock_custom_api(monkeypatch):
    """Stub ``custom_objects_api()`` to return a MagicMock."""
    api = MagicMock()
    monkeypatch.setattr(k8s, "custom_objects_api", lambda: api)
    return api


def _api_exc(status: int) -> ApiException:
    exc = ApiException(status=status)
    return exc


# =============================================================================
# load_kubeconfig — in-cluster vs kubeconfig fallback
# =============================================================================


class TestLoadKubeconfig:
    def test_uses_incluster_when_available(self, monkeypatch) -> None:
        incluster = MagicMock()
        kubeconfig = MagicMock()
        monkeypatch.setattr(k8s.config, "load_incluster_config", incluster)
        monkeypatch.setattr(k8s.config, "load_kube_config", kubeconfig)

        k8s.load_kubeconfig()

        incluster.assert_called_once_with()
        kubeconfig.assert_not_called()

    def test_falls_back_to_kubeconfig(self, monkeypatch) -> None:
        def _fail_incluster() -> None:
            raise k8s.config.ConfigException("no service account")

        kubeconfig = MagicMock()
        monkeypatch.setattr(k8s.config, "load_incluster_config", _fail_incluster)
        monkeypatch.setattr(k8s.config, "load_kube_config", kubeconfig)

        k8s.load_kubeconfig()

        kubeconfig.assert_called_once_with()


# =============================================================================
# apply_raycluster
# =============================================================================


class TestApplyRaycluster:
    _manifest = {"metadata": {"name": "rc-a"}, "spec": {}}

    def test_posts_on_first_call(self, mock_custom_api) -> None:
        mock_custom_api.create_namespaced_custom_object.return_value = {"ok": True}
        got = k8s.apply_raycluster(self._manifest, "ns-a")
        assert got == {"ok": True}
        mock_custom_api.create_namespaced_custom_object.assert_called_once()
        mock_custom_api.patch_namespaced_custom_object.assert_not_called()

    def test_patches_on_409_conflict(self, mock_custom_api) -> None:
        mock_custom_api.create_namespaced_custom_object.side_effect = _api_exc(409)
        mock_custom_api.patch_namespaced_custom_object.return_value = {"patched": True}
        got = k8s.apply_raycluster(self._manifest, "ns-a")
        assert got == {"patched": True}
        mock_custom_api.patch_namespaced_custom_object.assert_called_once()

    def test_non_409_bubbles_up(self, mock_custom_api) -> None:
        mock_custom_api.create_namespaced_custom_object.side_effect = _api_exc(500)
        with pytest.raises(ApiException):
            k8s.apply_raycluster(self._manifest, "ns-a")


# =============================================================================
# delete_raycluster
# =============================================================================


class TestDeleteRaycluster:
    def test_swallows_404_when_ignore_missing(self, mock_custom_api) -> None:
        mock_custom_api.delete_namespaced_custom_object.side_effect = _api_exc(404)
        # Should not raise.
        k8s.delete_raycluster("rc-gone", "ns-a", ignore_missing=True)

    def test_raises_404_when_not_ignoring(self, mock_custom_api) -> None:
        mock_custom_api.delete_namespaced_custom_object.side_effect = _api_exc(404)
        with pytest.raises(ApiException):
            k8s.delete_raycluster("rc-gone", "ns-a", ignore_missing=False)

    def test_non_404_always_raises(self, mock_custom_api) -> None:
        mock_custom_api.delete_namespaced_custom_object.side_effect = _api_exc(500)
        with pytest.raises(ApiException):
            k8s.delete_raycluster("rc", "ns-a", ignore_missing=True)


# =============================================================================
# wait_for_raycluster_ready
# =============================================================================


class TestWaitForReady:
    def test_returns_when_state_ready(self, mock_custom_api, monkeypatch) -> None:
        mock_custom_api.get_namespaced_custom_object.return_value = {
            "status": {"state": "ready"}
        }
        # Suppress sleep so the test is fast.
        monkeypatch.setattr(k8s.time, "sleep", lambda _s: None)
        k8s.wait_for_raycluster_ready("rc-a", "ns-a", timeout_s=5, poll_s=0)

    def test_raises_on_timeout(self, mock_custom_api, monkeypatch) -> None:
        mock_custom_api.get_namespaced_custom_object.return_value = {
            "status": {"state": "provisioning"}
        }
        monkeypatch.setattr(k8s.time, "sleep", lambda _s: None)
        # Patch only the k8s module's ``time.monotonic`` — tenacity calls
        # ``time.monotonic`` through its own ``stop`` helpers and we don't
        # want to interfere. Give the wait loop "now > deadline" on the 2nd
        # tick so it enters once (to prove the poll happens) then exits.
        ticks = iter([0.0, 100.0, 100.0, 100.0])
        monkeypatch.setattr(k8s.time, "monotonic", lambda: next(ticks, 100.0))
        with pytest.raises(TimeoutError):
            k8s.wait_for_raycluster_ready("rc-a", "ns-a", timeout_s=10, poll_s=0)


# =============================================================================
# delete_configmap
# =============================================================================


class TestDeleteConfigmap:
    def test_returns_true_on_success(self, monkeypatch) -> None:
        # Bypass the load_kubeconfig() call inside delete_configmap.
        fake_core = MagicMock()
        fake_core.delete_namespaced_config_map.return_value = {"ok": True}
        monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: fake_core)

        assert k8s.delete_configmap("cm", "ns") is True

    def test_returns_false_on_404_when_ignoring(self, monkeypatch) -> None:
        fake_core = MagicMock()
        fake_core.delete_namespaced_config_map.side_effect = _api_exc(404)
        monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: fake_core)

        assert k8s.delete_configmap("cm", "ns", ignore_missing=True) is False

    def test_raises_on_404_when_not_ignoring(self, monkeypatch) -> None:
        fake_core = MagicMock()
        fake_core.delete_namespaced_config_map.side_effect = _api_exc(404)
        monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: fake_core)

        with pytest.raises(ApiException):
            k8s.delete_configmap("cm", "ns", ignore_missing=False)
