"""Tests for :mod:`nrl_k8s.submit` — dashboard access + Ray job submission.

``dashboard_url`` has two branches (in-cluster DNS vs. laptop port-forward)
and both must be exercised without spawning anything real. ``tail_job_logs``
bridges Ray's async iterator to a sync generator via a daemon thread.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from nrl_k8s import submit


# =============================================================================
# is_in_cluster
# =============================================================================

class TestIsInCluster:
    def test_true_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        assert submit.is_in_cluster() is True

    def test_false_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        assert submit.is_in_cluster() is False


# =============================================================================
# dashboard_url — in-cluster vs laptop (kubectl port-forward)
# =============================================================================

class TestDashboardUrlInCluster:
    def test_returns_dns_url_without_spawning(self, monkeypatch) -> None:
        """In-cluster path returns the head-svc DNS URL and must not touch
        ``subprocess.Popen``."""
        monkeypatch.setattr(submit, "is_in_cluster", lambda: True)

        def _boom(*a, **kw):  # pragma: no cover — should never run
            raise AssertionError("Popen must not be called in-cluster")

        monkeypatch.setattr(submit.subprocess, "Popen", _boom)

        with submit.dashboard_url("rc-gen", "ns-a") as url:
            assert url == "http://rc-gen-head-svc.ns-a.svc.cluster.local:8265"


class TestDashboardUrlLaptop:
    def test_spawns_kubectl_and_tears_down(self, monkeypatch) -> None:
        """Laptop path spawns ``kubectl port-forward`` and terminates it on exit."""
        monkeypatch.setattr(submit, "is_in_cluster", lambda: False)
        # Skip the kubectl preflight (it runs real subprocesses).
        monkeypatch.setattr(submit, "_KUBECTL_OK", True)
        # Collapse _wait_for_tcp so we never try to open a real socket.
        monkeypatch.setattr(submit, "_wait_for_tcp", lambda *a, **kw: None)
        monkeypatch.setattr(submit, "_free_port", lambda: 19999)

        proc = MagicMock()
        proc.poll.return_value = None  # "still running"
        proc.stdout = io.StringIO("")

        popen = MagicMock(return_value=proc)
        monkeypatch.setattr(submit.subprocess, "Popen", popen)

        with submit.dashboard_url("rc-gen", "ns-a") as url:
            assert url == "http://127.0.0.1:19999"

        # Popen was called with kubectl port-forward, and we tore it down.
        popen.assert_called_once()
        cmd = popen.call_args[0][0]
        assert cmd[0] == "kubectl"
        assert "port-forward" in cmd
        proc.terminate.assert_called_once()

    def test_missing_kubectl_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(submit, "is_in_cluster", lambda: False)
        monkeypatch.setattr(submit, "_KUBECTL_OK", False)
        monkeypatch.setattr(submit.shutil, "which", lambda _bin: None)
        with pytest.raises(RuntimeError, match="kubectl not found"):
            with submit.dashboard_url("rc-gen", "ns-a"):
                pass


# =============================================================================
# tail_job_logs — bridge async -> sync
# =============================================================================

class TestTailJobLogs:
    def test_bridges_async_iterator_to_generator(self, monkeypatch) -> None:
        """``tail_job_logs`` must yield each line produced by the async Ray
        iterator, then stop cleanly when the iterator is exhausted."""
        fake_lines = ["line one\n", "line two\n", "done\n"]

        class _FakeAsyncIter:
            def __init__(self, lines):
                self._it = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        fake_client = MagicMock()
        fake_client.tail_job_logs = lambda _jid: _FakeAsyncIter(fake_lines)

        # Stub the lazy import inside tail_job_logs.
        import ray.job_submission as rjs

        monkeypatch.setattr(rjs, "JobSubmissionClient", lambda _url: fake_client)

        out = list(submit.tail_job_logs("http://127.0.0.1:8265", "job-xyz"))
        assert out == fake_lines
