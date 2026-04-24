"""Tests for :mod:`nrl_k8s.inspect` — read-only introspection of clusters."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nrl_k8s import inspect as ins

# =============================================================================
# list_cluster_pods — head vs worker by ``ray.io/node-type`` label
# =============================================================================


def _fake_pod(name: str, node_type: str, phase: str = "Running"):
    """Build an object shaped like a kubernetes V1Pod."""
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels={"ray.io/node-type": node_type}),
        status=SimpleNamespace(phase=phase),
    )


@pytest.fixture
def mock_core(monkeypatch):
    """Stub out the CoreV1Api so no API calls happen."""
    monkeypatch.setattr(ins.k8s, "load_kubeconfig", lambda: None)
    fake = MagicMock()
    monkeypatch.setattr(ins.client, "CoreV1Api", lambda: fake)
    return fake


class TestListClusterPods:
    def test_splits_head_and_workers(self, mock_core) -> None:
        pods_resp = SimpleNamespace(
            items=[
                _fake_pod("rc-a-head-0", "head"),
                _fake_pod("rc-a-wg-0", "worker"),
                _fake_pod("rc-a-wg-1", "worker", phase="Pending"),
            ]
        )
        mock_core.list_namespaced_pod.return_value = pods_resp

        result = ins.list_cluster_pods("rc-a", "ns-a")

        assert result.head_name == "rc-a-head-0"
        assert result.head_phase == "Running"
        assert result.worker_names == ["rc-a-wg-0", "rc-a-wg-1"]
        assert result.worker_phases == ["Running", "Pending"]

    def test_no_pods_returns_defaults(self, mock_core) -> None:
        mock_core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
        result = ins.list_cluster_pods("rc-a", "ns-a")
        assert result.head_name is None
        assert result.head_phase is None
        assert result.worker_names == []
        assert result.worker_phases == []

    def test_uses_correct_label_selector(self, mock_core) -> None:
        mock_core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
        ins.list_cluster_pods("my-rc", "my-ns")
        mock_core.list_namespaced_pod.assert_called_once_with(
            namespace="my-ns", label_selector="ray.io/cluster=my-rc"
        )


# =============================================================================
# _latest_daemon_job — base id + suffixed matches
# =============================================================================


def _fake_job(submission_id: str, start_time: int, status_value: str):
    """Shape of a ``ray.dashboard.modules.job.common.JobDetails``-ish object."""
    return SimpleNamespace(
        submission_id=submission_id,
        start_time=start_time,
        status=SimpleNamespace(value=status_value),
    )


def _patch_dashboard(monkeypatch):
    @contextlib.contextmanager
    def _fake(cluster_name, namespace, **kw):
        yield f"http://{cluster_name}.test:8265"

    monkeypatch.setattr(ins.submit, "dashboard_url", _fake)


class TestLatestDaemonJob:
    def test_prefers_most_recent_suffixed(self, monkeypatch) -> None:
        """When both the base id and a suffixed variant exist, pick the
        most recent by start_time.
        """
        _patch_dashboard(monkeypatch)

        jobs = [
            _fake_job("gym-daemon", start_time=1000, status_value="FAILED"),
            _fake_job("gym-daemon-123", start_time=5000, status_value="RUNNING"),
            _fake_job("unrelated", start_time=9999, status_value="RUNNING"),
        ]
        client = MagicMock()
        client.list_jobs.return_value = jobs

        import ray.job_submission as rjs

        monkeypatch.setattr(rjs, "JobSubmissionClient", lambda _url: client)

        sid, status = ins._latest_daemon_job("rc-gym", "ns-a", "gym-daemon")
        assert sid == "gym-daemon-123"
        assert status == "RUNNING"

    def test_falls_back_to_base_id_when_no_match(self, monkeypatch) -> None:
        _patch_dashboard(monkeypatch)

        client = MagicMock()
        client.list_jobs.return_value = [
            _fake_job("someone-else", 1000, "RUNNING"),
        ]
        import ray.job_submission as rjs

        monkeypatch.setattr(rjs, "JobSubmissionClient", lambda _url: client)

        sid, status = ins._latest_daemon_job("rc-gym", "ns-a", "gym-daemon")
        assert sid == "gym-daemon"
        assert status is None

    def test_any_error_returns_base_and_none(self, monkeypatch) -> None:
        """Dashboard failures must not crash ``status`` — we fall back to a
        'unknown' row so the CLI can keep rendering the other clusters.
        """

        @contextlib.contextmanager
        def _fail(*_a, **_kw):
            raise RuntimeError("no kubectl")
            yield  # unreachable; keeps generator signature valid

        monkeypatch.setattr(ins.submit, "dashboard_url", _fail)

        sid, status = ins._latest_daemon_job("rc-gym", "ns-a", "gym-daemon")
        assert sid == "gym-daemon"
        assert status is None
