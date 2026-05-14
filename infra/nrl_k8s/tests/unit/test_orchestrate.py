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
        "kuberay": {
            "training": {
                "name": "rc-train",
                "spec": {
                    "headGroupSpec": {
                        "template": {"spec": {"containers": [{"name": "ray-head"}]}}
                    }
                },
            },
        },
    }
    if gym_entrypoint is not None:
        base["kuberay"]["gym"] = {
            "name": "rc-gym",
            "spec": {
                "headGroupSpec": {
                    "template": {"spec": {"containers": [{"name": "ray-head"}]}}
                }
            },
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
    monkeypatch.setattr(orchestrate, "JobSubmissionClient", lambda *_a, **_kw: client)


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

    def test_raises_when_existing_failed_without_replace(
        self, monkeypatch, log
    ) -> None:
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
            "gym",
            loaded,
            "rc-gym",
            log=log_fn,
            repo_root=Path("/tmp"),
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
# ensure_cluster — idempotent path used by `go`
# =============================================================================


class TestEnsureCluster:
    def test_applies_when_absent(self, monkeypatch, log) -> None:
        log_fn, lines = log
        get = MagicMock(return_value=None)
        apply = MagicMock()
        wait = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "get_raycluster", get)
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", wait)

        name = orchestrate.ensure_cluster("training", _loaded(), log=log_fn)

        assert name == "rc-train"
        apply.assert_called_once()
        wait.assert_called_once()

    def test_reuses_when_live_matches(self, monkeypatch, log) -> None:
        log_fn, lines = log
        loaded = _loaded()
        rendered = orchestrate.build_raycluster_manifest(
            loaded.infra.kuberay.training, loaded.infra
        )
        live = {
            "metadata": {
                "name": "rc-train",
                "resourceVersion": "9",
                "uid": "abc",
            },
            "spec": rendered["spec"],
            "status": {"state": "ready"},
        }
        monkeypatch.setattr(
            orchestrate.k8s, "get_raycluster", MagicMock(return_value=live)
        )
        apply = MagicMock()
        wait = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", wait)

        orchestrate.ensure_cluster("training", loaded, log=log_fn)

        apply.assert_not_called()
        wait.assert_called_once()
        assert any("already exists and matches" in ln for ln in lines)

    def test_warns_on_drift_and_reuses(self, monkeypatch, log) -> None:
        log_fn, lines = log
        loaded = _loaded()
        # Start from the rendered spec, mutate one field to simulate drift.
        rendered = orchestrate.build_raycluster_manifest(
            loaded.infra.kuberay.training, loaded.infra
        )
        drifted = {"metadata": {"name": "rc-train"}, "spec": dict(rendered["spec"])}
        drifted["spec"]["rayVersion"] = "drifted"
        monkeypatch.setattr(
            orchestrate.k8s, "get_raycluster", MagicMock(return_value=drifted)
        )
        apply = MagicMock()
        delete = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "delete_raycluster", delete)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", MagicMock())

        orchestrate.ensure_cluster("training", loaded, log=log_fn)

        apply.assert_not_called()
        delete.assert_not_called()
        assert any("drifted" in ln and "reusing" in ln for ln in lines)

    def test_recreate_deletes_and_reapplies_on_drift(self, monkeypatch, log) -> None:
        log_fn, lines = log
        loaded = _loaded()
        rendered = orchestrate.build_raycluster_manifest(
            loaded.infra.kuberay.training, loaded.infra
        )
        drifted = {"metadata": {"name": "rc-train"}, "spec": dict(rendered["spec"])}
        drifted["spec"]["rayVersion"] = "drifted"
        monkeypatch.setattr(
            orchestrate.k8s, "get_raycluster", MagicMock(return_value=drifted)
        )
        apply = MagicMock()
        delete = MagicMock()
        gone = MagicMock()
        monkeypatch.setattr(orchestrate.k8s, "apply_raycluster", apply)
        monkeypatch.setattr(orchestrate.k8s, "delete_raycluster", delete)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_gone", gone)
        monkeypatch.setattr(orchestrate.k8s, "wait_for_raycluster_ready", MagicMock())

        orchestrate.ensure_cluster("training", loaded, log=log_fn, recreate=True)

        delete.assert_called_once_with("rc-train", "ns-a")
        gone.assert_called_once()
        apply.assert_called_once()


# =============================================================================
# run — idempotent bring-up + daemon + training submit
# =============================================================================


class TestRun:
    def test_skip_daemons_bypasses_daemon_submit(self, monkeypatch, log) -> None:
        log_fn, lines = log
        loaded = _loaded(gym_entrypoint="python gym.py --job-id run-q")

        ensure = MagicMock(side_effect=lambda role, *a, **kw: f"rc-{role}")
        daemon = MagicMock()
        train = MagicMock(return_value="TRAIN_RESULT")
        monkeypatch.setattr(orchestrate, "ensure_cluster", ensure)
        monkeypatch.setattr(orchestrate, "submit_daemon", daemon)
        monkeypatch.setattr(orchestrate, "submit_training", train)

        out = orchestrate.run(
            loaded, log=log_fn, repo_root=Path("/tmp"), skip_daemons=True
        )

        assert out == "TRAIN_RESULT"
        # ensure_cluster called once per declared role.
        roles_ensured = {c.args[0] for c in ensure.call_args_list}
        assert roles_ensured == {"gym", "training"}
        # Only training gets a submit_daemon call; gym is skipped.
        roles_daemoned = [c.args[0] for c in daemon.call_args_list]
        assert roles_daemoned == ["training"]
        train.assert_called_once()

    def test_recreate_flag_propagates(self, monkeypatch, log) -> None:
        log_fn, _ = log
        loaded = _loaded()

        ensure = MagicMock(return_value="rc-train")
        monkeypatch.setattr(orchestrate, "ensure_cluster", ensure)
        monkeypatch.setattr(orchestrate, "submit_daemon", MagicMock())
        monkeypatch.setattr(orchestrate, "submit_training", MagicMock())

        orchestrate.run(loaded, log=log_fn, repo_root=Path("/tmp"), recreate=True)

        assert ensure.call_args.kwargs["recreate"] is True


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


# =============================================================================
# Recipe path translation + entrypoint --config rewrite
# =============================================================================


class TestRecipePathInPod:
    def test_upload_returns_staged_filename(self, tmp_path: Path) -> None:
        loaded = _loaded()
        loaded.source_path = tmp_path / "subdir" / "recipe.yaml"
        # repo_root irrelevant in upload mode — recipe is staged into
        # the working_dir as a fixed filename.
        assert (
            orchestrate._recipe_path_in_pod(loaded, tmp_path, upload=True)
            == "nrl_k8s_run.yaml"
        )

    def test_image_lustre_returns_repo_relative_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        recipe = repo / "infra" / "nrl_k8s" / "examples" / "x.yaml"
        recipe.parent.mkdir(parents=True)
        recipe.write_text("policy: {}")
        loaded = _loaded()
        loaded.source_path = recipe

        result = orchestrate._recipe_path_in_pod(loaded, repo, upload=False)
        assert result == "infra/nrl_k8s/examples/x.yaml"

    def test_image_lustre_recipe_outside_repo_returns_none(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "elsewhere" / "recipe.yaml"
        outside.parent.mkdir()
        outside.write_text("policy: {}")
        loaded = _loaded()
        loaded.source_path = outside

        # None makes the rewriter leave the entrypoint alone rather
        # than invent a path the pod can't resolve.
        assert orchestrate._recipe_path_in_pod(loaded, repo, upload=False) is None


class TestRewriteEntrypointRecipe:
    @staticmethod
    def _loaded_for(repo: Path, recipe_relpath: str) -> LoadedConfig:
        recipe = repo / recipe_relpath
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text("policy: {}")
        loaded = _loaded()
        loaded.source_path = recipe
        return loaded

    def test_rewrites_hardcoded_config_to_cli_recipe(self, tmp_path: Path, log) -> None:
        log_fn, lines = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "infra/recipes/x.yaml")
        ep = "cd /opt/nemo-rl && python train.py --config a/b.yaml --foo"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert "--config infra/recipes/x.yaml" in result
        assert "a/b.yaml" not in result
        # Substitution is logged so it shows up in the submission log.
        assert any("rewrote" in ln and "a/b.yaml" in ln for ln in lines)

    def test_no_op_when_paths_already_match(self, tmp_path: Path, log) -> None:
        log_fn, lines = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "infra/recipes/x.yaml")
        ep = "python train.py --config infra/recipes/x.yaml"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert result == ep
        # No log noise when nothing was actually rewritten.
        assert lines == []

    def test_handles_equals_separator(self, tmp_path: Path, log) -> None:
        # Some shells / hydra invocations write ``--config=path`` with
        # no space. Keep the same separator the user used.
        log_fn, _ = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "x.yaml")
        ep = "python train.py --config=a/b.yaml"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert "--config=x.yaml" in result

    def test_preserves_quoting(self, tmp_path: Path, log) -> None:
        log_fn, _ = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "x.yaml")

        for ep, expected in (
            ('python t.py --config "a/b.yaml"', '--config "x.yaml"'),
            ("python t.py --config 'a/b.yaml'", "--config 'x.yaml'"),
        ):
            result = orchestrate._rewrite_entrypoint_recipe(
                ep, loaded, repo, upload=False, log=log_fn
            )
            assert expected in result

    def test_does_not_touch_config_name_or_config_dir(
        self, tmp_path: Path, log
    ) -> None:
        # Hydra's ``--config-name`` / ``--config-dir`` / ``--config-path``
        # look superficially similar but aren't recipe paths.
        log_fn, lines = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "x.yaml")
        ep = "python train.py --config-name=foo --config-dir=bar --config-path conf"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert result == ep
        assert lines == []

    def test_no_op_when_no_config_flag_present(self, tmp_path: Path, log) -> None:
        log_fn, lines = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "x.yaml")
        ep = "python train.py --some-other-flag"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert result == ep
        assert lines == []

    def test_no_op_when_recipe_outside_repo(self, tmp_path: Path, log) -> None:
        log_fn, lines = log
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "elsewhere.yaml"
        outside.write_text("policy: {}")
        loaded = _loaded()
        loaded.source_path = outside
        ep = "python train.py --config a/b.yaml"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        # Cannot translate path; leave entrypoint alone rather than
        # invent something.
        assert result == ep

    def test_upload_mode_uses_staged_filename(self, tmp_path: Path, log) -> None:
        log_fn, _ = log
        loaded = _loaded()
        loaded.source_path = tmp_path / "anywhere.yaml"
        ep = "python train.py --config a/b.yaml"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, tmp_path, upload=True, log=log_fn
        )

        assert "--config nrl_k8s_run.yaml" in result

    def test_rewrites_multiple_config_flags(self, tmp_path: Path, log) -> None:
        # Defensive: if an entrypoint somehow has multiple ``--config``
        # flags (e.g. an experimental harness chaining), all get the
        # same recipe substitution. Single log line per distinct
        # original path.
        log_fn, lines = log
        repo = tmp_path / "repo"
        loaded = self._loaded_for(repo, "x.yaml")
        ep = "python train.py --config a.yaml ; python verify.py --config a.yaml"

        result = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, repo, upload=False, log=log_fn
        )

        assert result.count("--config x.yaml") == 2
        assert "a.yaml" not in result
        # De-duplicated: same original path → one log line.
        assert sum("rewrote" in ln for ln in lines) == 1
