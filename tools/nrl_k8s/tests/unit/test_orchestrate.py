"""Tests for :mod:`nrl_k8s.orchestrate` — the bring-up / submit pipeline.

Every external system is stubbed: no Kubernetes API, no Ray dashboard, no
workdir staging. We assert on the *orchestration decisions*: when do we
apply, when do we skip, when do we suffix a submissionId for ``--replace``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nrl_k8s import orchestrate
from nrl_k8s.config import LoadedConfig
from nrl_k8s.schema import InfraConfig


# =============================================================================
# Fake config builders
# =============================================================================

def _infra_payload(
    *,
    gym_entrypoint: str | None = None,
    gym_submission_id: str | None = "gym-daemon",
    training_entrypoint: str | None = "python -m train",
    gym_health_url: str | None = None,
) -> dict:
    """Construct an InfraConfig dict with up to 2 declared clusters."""
    base = {
        "namespace": "ns-a",
        "image": "img:1",
        "launch": {"entrypoint": training_entrypoint},
        "clusters": {
            "training": {
                "name": "rc-train",
                "spec": {"headGroupSpec": {"template": {"spec": {"containers": [{"name": "ray-head"}]}}}},
            },
        },
    }
    if gym_entrypoint is not None:
        base["clusters"]["gym"] = {
            "name": "rc-gym",
            "spec": {"headGroupSpec": {"template": {"spec": {"containers": [{"name": "ray-head"}]}}}},
            "daemon": {
                "entrypoint": gym_entrypoint,
                "submissionId": gym_submission_id,
                "healthCheckUrl": gym_health_url,
            },
        }
    return base


def _loaded(**kwargs) -> LoadedConfig:
    from omegaconf import OmegaConf

    infra = InfraConfig.model_validate(_infra_payload(**kwargs))
    return LoadedConfig(
        recipe=OmegaConf.create({"policy": {"x": 1}}),
        infra=infra,
        source_path=Path("/tmp/recipe.yaml"),
    )


@pytest.fixture
def log():
    """Collect log lines in a list; assert on substrings in tests."""
    lines: list[str] = []
    return lines.append, lines


# =============================================================================
# bring_up_cluster
# =============================================================================

class TestBringUpCluster:
    def test_applies_and_waits(self, monkeypatch, log) -> None:
        """The happy path calls both apply + wait_for_ready."""
        log_fn, lines = log
        apply = MagicMock()
        wait = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", wait)

        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-42")
        name = orchestrate.bring_up_cluster("gym", loaded, log=log_fn)

        assert name == "rc-gym"
        apply.assert_called_once()
        wait.assert_called_once_with("rc-gym", "ns-a", timeout_s=900)

    def test_no_wait_skips_readiness(self, monkeypatch, log) -> None:
        log_fn, _ = log
        apply = MagicMock()
        wait = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", wait)

        orchestrate.bring_up_cluster(
            "training", _loaded(), log=log_fn, wait_ready=False
        )
        apply.assert_called_once()
        wait.assert_not_called()


# =============================================================================
# submit_daemon — status-driven branching
# =============================================================================

def _patch_dashboard(monkeypatch):
    """Force dashboard_url to yield a dummy URL without any port-forward."""
    @contextlib.contextmanager
    def _fake(cluster_name, namespace, **kw):
        yield f"http://{cluster_name}.test:8265"

    monkeypatch.setattr(orchestrate.submit, "dashboard_url", _fake)


def _patch_client(monkeypatch, client):
    """Install ``client`` as the ``JobSubmissionClient`` constructor's return."""
    monkeypatch.setattr(
        orchestrate, "JobSubmissionClient", lambda *_a, **_kw: client
    )


class TestSubmitDaemon:
    def test_skips_when_existing_running(self, monkeypatch, log) -> None:
        """A RUNNING daemon without ``--replace`` returns the existing id."""
        from ray.job_submission import JobStatus

        log_fn, lines = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-1")

        _patch_dashboard(monkeypatch)
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.RUNNING
        _patch_client(monkeypatch, client)

        staged = MagicMock()
        monkeypatch.setattr(orchestrate.workdir, "stage_workdir", staged)
        submit_job = MagicMock()
        monkeypatch.setattr(orchestrate.submit, "submit_ray_job", submit_job)

        out = orchestrate.submit_daemon(
            "gym", loaded, "rc-gym", log=log_fn, repo_root=Path("/tmp")
        )

        assert out == "gym-daemon"
        submit_job.assert_not_called()
        staged.assert_not_called()
        assert any("already RUNNING" in ln for ln in lines)

    def test_skips_when_existing_succeeded(self, monkeypatch, log) -> None:
        from ray.job_submission import JobStatus

        log_fn, lines = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-1")
        _patch_dashboard(monkeypatch)
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.SUCCEEDED
        _patch_client(monkeypatch, client)
        submit_job = MagicMock()
        monkeypatch.setattr(orchestrate.submit, "submit_ray_job", submit_job)

        orchestrate.submit_daemon(
            "gym", loaded, "rc-gym", log=log_fn, repo_root=Path("/tmp")
        )
        submit_job.assert_not_called()

    def test_raises_when_existing_failed_without_replace(self, monkeypatch, log) -> None:
        from ray.job_submission import JobStatus

        log_fn, _ = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-1")
        _patch_dashboard(monkeypatch)
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.FAILED
        _patch_client(monkeypatch, client)

        with pytest.raises(RuntimeError, match="FAILED"):
            orchestrate.submit_daemon(
                "gym", loaded, "rc-gym", log=log_fn, repo_root=Path("/tmp")
            )

    def test_replace_stops_running_and_suffixes_id(self, monkeypatch, log) -> None:
        """``--replace`` on a RUNNING daemon stops it and picks a fresh suffix."""
        from ray.job_submission import JobStatus

        log_fn, lines = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-1")
        _patch_dashboard(monkeypatch)
        client = MagicMock()
        # initial status check says RUNNING, then STOPPED once we stop.
        client.get_job_status.side_effect = [JobStatus.RUNNING, JobStatus.STOPPED]
        _patch_client(monkeypatch, client)

        monkeypatch.setattr(
            orchestrate.workdir, "stage_workdir", lambda *a, **kw: Path("/tmp/wd")
        )
        submit_job = MagicMock(return_value="gym-daemon-123")
        monkeypatch.setattr(orchestrate.submit, "submit_ray_job", submit_job)
        # Collapse sleep in _wait_job_stopped.
        monkeypatch.setattr(orchestrate.time, "sleep", lambda _s: None)

        out = orchestrate.submit_daemon(
            "gym", loaded, "rc-gym", log=log_fn, repo_root=Path("/tmp"),
            replace=True,
        )

        client.stop_job.assert_called_once_with("gym-daemon")
        submit_job.assert_called_once()
        kwargs = submit_job.call_args.kwargs
        # The fresh submission_id must start with the original name + "-".
        assert kwargs["submission_id"].startswith("gym-daemon-")
        assert kwargs["submission_id"] != "gym-daemon"
        assert out == "gym-daemon-123"

    def test_no_daemon_returns_none(self, monkeypatch, log) -> None:
        """A cluster without a daemon is a no-op."""
        log_fn, _ = log
        loaded = _loaded()  # no gym cluster declared
        # Give training cluster a fake daemon=None — just call on training.
        out = orchestrate.submit_daemon(
            "training", loaded, "rc-train", log=log_fn, repo_root=Path("/tmp")
        )
        assert out is None


# =============================================================================
# submit_training
# =============================================================================

class TestSubmitTraining:
    def test_raises_when_entrypoint_unset(self, monkeypatch, log) -> None:
        log_fn, _ = log
        loaded = _loaded(training_entrypoint=None)
        with pytest.raises(ValueError, match="entrypoint"):
            orchestrate.submit_training(loaded, log=log_fn, repo_root=Path("/tmp"))


# =============================================================================
# _infer_disagg_job_id — regex over gym entrypoints
# =============================================================================

class TestInferDisaggJobId:
    def test_parses_equals_form(self) -> None:
        loaded = _loaded(gym_entrypoint="python gym.py --job-id=run-xyz --other")
        assert orchestrate._infer_disagg_job_id(loaded.infra) == "run-xyz"

    def test_parses_space_form(self) -> None:
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-abc --other")
        assert orchestrate._infer_disagg_job_id(loaded.infra) == "run-abc"

    def test_returns_none_when_no_flag(self) -> None:
        loaded = _loaded(gym_entrypoint="python gym.py --cluster x")
        assert orchestrate._infer_disagg_job_id(loaded.infra) is None

    def test_returns_none_when_no_gym(self) -> None:
        loaded = _loaded()  # no gym cluster
        assert orchestrate._infer_disagg_job_id(loaded.infra) is None


# =============================================================================
# _reset_endpoint_registry — deletes ConfigMap named after the inferred id
# =============================================================================

class TestResetEndpointRegistry:
    def test_deletes_cm_named_after_job_id(self, monkeypatch, log) -> None:
        log_fn, lines = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id=run-k")

        delete_cm = MagicMock(return_value=True)
        monkeypatch.setattr(orchestrate.k8s, "delete_configmap", delete_cm)

        orchestrate._reset_endpoint_registry(loaded, log=log_fn)

        delete_cm.assert_called_once_with("nemo-rl-endpoints-run-k", "ns-a")
        assert any("deleted endpoint registry" in ln for ln in lines)

    def test_noop_when_no_gym(self, monkeypatch, log) -> None:
        log_fn, _ = log
        loaded = _loaded()  # no gym cluster
        delete_cm = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "delete_configmap", delete_cm)

        orchestrate._reset_endpoint_registry(loaded, log=log_fn)
        delete_cm.assert_not_called()
