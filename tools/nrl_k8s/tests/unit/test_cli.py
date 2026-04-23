"""Tests for :mod:`nrl_k8s.cli` — click entrypoints.

Use ``click.testing.CliRunner`` to invoke commands; every downstream
orchestrate / k8s call is mocked so tests never touch a cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from nrl_k8s import cli
from nrl_k8s import config as cfg_mod

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _no_user_defaults(monkeypatch, tmp_path):
    """Don't let a real ``~/.config/nrl-k8s/defaults.yaml`` bleed in."""
    monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", tmp_path / "none.yaml")


@pytest.fixture(autouse=True)
def _force_fallback_loader(monkeypatch):
    """Force the OmegaConf-only recipe loader (no nemo_rl dependency)."""
    import builtins

    real_import = builtins.__import__

    def _fail_nemo_rl(name, *args, **kwargs):
        if name.startswith("nemo_rl"):
            raise ImportError("forced-fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_nemo_rl)


def _write_recipe(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


# =============================================================================
# check — merged validate + plan
# =============================================================================


class TestCheck:
    def test_summary_shows_namespace_and_image(self, tmp_path) -> None:
        recipe = _write_recipe(
            tmp_path, {"infra": {"namespace": "ns-a", "image": "img:1"}}
        )
        runner = CliRunner()
        result = runner.invoke(cli.main, ["check", str(recipe)])
        assert result.exit_code == 0, result.output
        assert "namespace:" in result.output
        assert "ns-a" in result.output
        assert "img:1" in result.output

    def test_summary_lists_each_declared_cluster(self, tmp_path) -> None:
        spec = {
            "headGroupSpec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "h",
                                "image": "old",
                                "resources": {"limits": {"cpu": "8", "memory": "32Gi"}},
                            }
                        ]
                    }
                }
            }
        }
        recipe = _write_recipe(
            tmp_path,
            {
                "infra": {
                    "namespace": "ns-a",
                    "image": "img:new",
                    "clusters": {"training": {"name": "rc-t", "spec": spec}},
                }
            },
        )
        runner = CliRunner()
        result = runner.invoke(cli.main, ["check", str(recipe)])
        assert result.exit_code == 0, result.output
        assert "training: rc-t" in result.output
        assert "cpu=8" in result.output

    def test_output_writes_full_config_and_manifests(self, tmp_path) -> None:
        spec = {
            "headGroupSpec": {
                "template": {"spec": {"containers": [{"name": "h", "image": "old"}]}}
            }
        }
        recipe = _write_recipe(
            tmp_path,
            {
                "infra": {
                    "namespace": "ns-a",
                    "image": "img:new",
                    "clusters": {"training": {"name": "rc-t", "spec": spec}},
                }
            },
        )
        out = tmp_path / "bundle.json"
        runner = CliRunner()
        result = runner.invoke(cli.main, ["check", str(recipe), "-o", str(out)])
        assert result.exit_code == 0, result.output
        parsed = json.loads(out.read_text())
        assert parsed["infra"]["image"] == "img:new"
        assert parsed["manifests"]["training"]["metadata"]["name"] == "rc-t"
        # Image is patched through into the rendered manifest.
        containers = parsed["manifests"]["training"]["spec"]["headGroupSpec"][
            "template"
        ]["spec"]["containers"]
        assert containers[0]["image"] == "img:new"

    def test_reports_validation_error_cleanly(self, tmp_path) -> None:
        """Missing a required field surfaces as a user-facing ``error:`` line,
        not a Python traceback, and exits non-zero. ``image`` is the only
        truly-required string — ``namespace`` auto-fills from the kube
        context if omitted, so we trigger validation by omitting ``image``.
        """
        recipe = _write_recipe(tmp_path, {"infra": {"namespace": "ns-a"}})
        runner = CliRunner()
        result = runner.invoke(cli.main, ["check", str(recipe)])
        assert result.exit_code == 1
        assert "error:" in result.output


# =============================================================================
# --infra combined with recipe infra: block
# =============================================================================


class TestInfraCliOption:
    def test_both_sources_rejected(self, tmp_path) -> None:
        """Passing ``--infra infra.yaml`` while the recipe also has ``infra:``
        errors out instead of silently preferring one.
        """
        infra = tmp_path / "infra.yaml"
        infra.write_text(yaml.safe_dump({"namespace": "ns-file", "image": "img:file"}))

        recipe = _write_recipe(
            tmp_path, {"infra": {"namespace": "ns-inline", "image": "img:inline"}}
        )
        runner = CliRunner()
        result = runner.invoke(cli.main, ["check", str(recipe), "--infra", str(infra)])
        assert result.exit_code == 1
        assert "infra" in result.output


# =============================================================================
# cluster down
# =============================================================================

class TestClusterDashboard:
    """`nrl-k8s cluster dashboard <name>` wraps port-forward + browser
    open, with an optional symlink-fix pre-step. No recipe/infra needed
    — the cluster name is a positional argument, namespace comes from
    --namespace or the active kube context."""

    @staticmethod
    def _stub_env(monkeypatch, browser_opens, pf_started, fix_called,
                  *, pf_cls_args=None, ns="ns-ctx"):
        class _FakePF:
            def __init__(self, cluster_name, namespace, port):
                if pf_cls_args is not None:
                    pf_cls_args.append((cluster_name, namespace, port))
                self._alive = False

            def start(self):
                pf_started.append(True)
                self._alive = False  # exit loop immediately

            def alive(self):
                return self._alive

            def stop(self):
                pass

        monkeypatch.setattr("nrl_k8s.submit._PortForward", _FakePF)
        monkeypatch.setattr("nrl_k8s.submit.is_in_cluster", lambda: True)
        monkeypatch.setattr("nrl_k8s.config._infer_kube_namespace", lambda: ns)
        monkeypatch.setattr(
            "webbrowser.open", lambda url: browser_opens.append(url)
        )
        monkeypatch.setattr(
            "nrl_k8s.cli._reinstall_ray_if_symlinked",
            lambda cluster, ns: fix_called.append([cluster, ns]),
        )

    def test_positional_name_uses_kube_context_namespace(self, monkeypatch):
        browser_opens: list[str] = []
        pf_started: list[bool] = []
        fix_called: list[list[str]] = []
        pf_args: list[tuple[str, str, int]] = []
        self._stub_env(
            monkeypatch, browser_opens, pf_started, fix_called,
            pf_cls_args=pf_args, ns="nemo-rl-testing",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli.main, ["cluster", "dashboard", "raycluster-foo"]
        )
        assert result.exit_code == 0, result.output
        assert pf_started == [True]
        assert fix_called == [["raycluster-foo", "nemo-rl-testing"]]
        assert pf_args == [("raycluster-foo", "nemo-rl-testing", 8265)]
        assert browser_opens == ["http://localhost:8265"]

    def test_namespace_flag_overrides_context(self, monkeypatch):
        browser_opens: list[str] = []
        pf_started: list[bool] = []
        fix_called: list[list[str]] = []
        pf_args: list[tuple[str, str, int]] = []
        self._stub_env(
            monkeypatch, browser_opens, pf_started, fix_called,
            pf_cls_args=pf_args, ns="wrong-ns",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            ["cluster", "dashboard", "rc-x", "-n", "explicit-ns", "--no-open"],
        )
        assert result.exit_code == 0, result.output
        assert fix_called == [["rc-x", "explicit-ns"]]
        assert pf_args == [("rc-x", "explicit-ns", 8265)]
        assert browser_opens == []

    def test_no_fix_skips_reinstall(self, monkeypatch):
        browser_opens: list[str] = []
        pf_started: list[bool] = []
        fix_called: list[list[str]] = []
        self._stub_env(monkeypatch, browser_opens, pf_started, fix_called)

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            ["cluster", "dashboard", "rc-y", "--no-fix", "--no-open"],
        )
        assert result.exit_code == 0, result.output
        assert fix_called == []


# =============================================================================
# rayjob — ephemeral RayJob submission
# =============================================================================


class TestRayJob:
    @staticmethod
    def _recipe_with_training(tmp_path: Path, entrypoint: str | None) -> Path:
        spec = {
            "headGroupSpec": {
                "template": {"spec": {"containers": [{"name": "h", "image": "old"}]}}
            }
        }
        infra = {
            "namespace": "ns",
            "image": "img:new",
            "clusters": {"training": {"name": "rc-train", "spec": spec}},
        }
        if entrypoint is not None:
            infra["launch"] = {"entrypoint": entrypoint}
        return _write_recipe(tmp_path, {"infra": infra})

    def test_dry_run_prints_manifest_without_applying(self, tmp_path, monkeypatch):
        recipe = self._recipe_with_training(tmp_path, "python run.py")

        applied: list[dict] = []
        monkeypatch.setattr(
            "nrl_k8s.k8s.apply_rayjob",
            lambda manifest, ns: applied.append((manifest, ns)),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            ["run", str(recipe), "--rayjob", "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert applied == []
        assert "kind: RayJob" in result.output
        assert "entrypoint: python run.py" in result.output

    def test_apply_then_wait_success(self, tmp_path, monkeypatch):
        recipe = self._recipe_with_training(tmp_path, "echo")

        applied: list[tuple[dict, str]] = []

        def _fake_apply(manifest, ns):
            applied.append((manifest, ns))
            return manifest

        def _fake_wait(name, namespace, *, timeout_s, on_update=None):
            return {
                "metadata": {"name": name},
                "status": {
                    "jobDeploymentStatus": "Complete",
                    "jobStatus": "SUCCEEDED",
                },
            }

        monkeypatch.setattr("nrl_k8s.k8s.apply_rayjob", _fake_apply)
        monkeypatch.setattr("nrl_k8s.k8s.wait_for_rayjob_terminal", _fake_wait)
        monkeypatch.setattr("nrl_k8s.submit.is_in_cluster", lambda: True)

        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(recipe), "--rayjob"])
        assert result.exit_code == 0, result.output
        assert len(applied) == 1
        manifest, ns = applied[0]
        assert ns == "ns"
        assert manifest["kind"] == "RayJob"
        assert manifest["metadata"]["name"] == "rc-train"
        assert manifest["spec"]["entrypoint"] == "echo"
        assert manifest["spec"]["shutdownAfterJobFinishes"] is True

    def test_failed_job_exits_non_zero(self, tmp_path, monkeypatch):
        recipe = self._recipe_with_training(tmp_path, "echo")
        monkeypatch.setattr(
            "nrl_k8s.k8s.apply_rayjob", lambda m, ns: m
        )
        monkeypatch.setattr(
            "nrl_k8s.k8s.wait_for_rayjob_terminal",
            lambda *a, **kw: {
                "status": {"jobDeploymentStatus": "Failed", "jobStatus": "FAILED"}
            },
        )
        monkeypatch.setattr("nrl_k8s.submit.is_in_cluster", lambda: True)

        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(recipe), "--rayjob"])
        assert result.exit_code == 1

    def test_no_wait_skips_poll(self, tmp_path, monkeypatch):
        recipe = self._recipe_with_training(tmp_path, "echo")
        waited: list[int] = []

        monkeypatch.setattr("nrl_k8s.k8s.apply_rayjob", lambda m, ns: m)
        monkeypatch.setattr(
            "nrl_k8s.k8s.wait_for_rayjob_terminal",
            lambda *a, **kw: waited.append(1) or {},
        )
        monkeypatch.setattr("nrl_k8s.submit.is_in_cluster", lambda: True)

        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(recipe), "--rayjob", "--no-wait"])
        assert result.exit_code == 0, result.output
        assert waited == []

    def test_errors_when_entrypoint_missing(self, tmp_path):
        recipe = self._recipe_with_training(tmp_path, entrypoint=None)
        runner = CliRunner()
        result = runner.invoke(cli.main, ["run", str(recipe), "--rayjob", "--dry-run"])
        assert result.exit_code == 1
        assert "entrypoint" in result.output


class TestRunCommand:
    """`nrl-k8s run` delegates to orchestrate.run with the CLI's resolved flags."""

    def test_run_invokes_orchestrate_with_flags(self, tmp_path, monkeypatch) -> None:
        spec = {
            "headGroupSpec": {
                "template": {"spec": {"containers": [{"name": "h", "image": "old"}]}}
            }
        }
        recipe = _write_recipe(
            tmp_path,
            {
                "infra": {
                    "namespace": "ns",
                    "image": "img:new",
                    "launch": {"entrypoint": "python run.py"},
                    "clusters": {"training": {"name": "rc-train", "spec": spec}},
                }
            },
        )

        captured: dict = {}

        class _FakeHandle:
            run_id = "training-1"
            kind = "port-forward"
            cluster_name = "rc-train"
            namespace = "ns"
            pod = None
            tmp_dir = None

        class _FakeResult:
            handle = _FakeHandle()

        def _fake_run(loaded, *, log, repo_root, replace, run_id, skip_daemons, recreate):
            captured["skip_daemons"] = skip_daemons
            captured["recreate"] = recreate
            captured["replace"] = replace
            captured["run_id"] = run_id
            return _FakeResult()

        monkeypatch.setattr("nrl_k8s.orchestrate.run", _fake_run)
        monkeypatch.setattr("nrl_k8s.submit.is_in_cluster", lambda: True)

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [
                "run",
                str(recipe),
                "--raycluster",
                "--mode", "batch",
                "--code-source", "image",
                "--code-path", "/opt/nemo-rl",
                "--run-id", "run-x",
                "--skip-daemons",
                "--recreate",
                "--no-wait",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured == {
            "skip_daemons": True,
            "recreate": True,
            "replace": False,
            "run_id": "run-x",
        }
        assert "run id:  training-1" in result.output


class TestClusterDown:
    def test_errors_without_role_or_name(self, tmp_path, monkeypatch) -> None:
        recipe = _write_recipe(
            tmp_path, {"infra": {"namespace": "ns-a", "image": "img:1"}}
        )
        runner = CliRunner()
        result = runner.invoke(cli.main, ["cluster", "down", str(recipe)])
        assert result.exit_code == 2
        assert "--role" in result.output or "--name" in result.output


# =============================================================================
# --mode resolution (interactive vs batch)
# =============================================================================

class TestModeResolution:
    def test_interactive_defaults(self) -> None:
        from nrl_k8s.schema import CodeSource, RunMode, SubmitterMode

        mode, sub, code, no_wait = cli._resolve_mode_defaults(
            cli_mode=None,
            infra_mode=RunMode.INTERACTIVE,
            cli_submitter=None,
            cli_code_source=None,
            cli_wait=None,
        )
        assert mode is RunMode.INTERACTIVE
        assert sub is SubmitterMode.PORT_FORWARD
        assert code is CodeSource.UPLOAD
        assert no_wait is False

    def test_batch_defaults(self) -> None:
        from nrl_k8s.schema import CodeSource, RunMode, SubmitterMode

        mode, sub, code, no_wait = cli._resolve_mode_defaults(
            cli_mode="batch",
            infra_mode=RunMode.INTERACTIVE,
            cli_submitter=None,
            cli_code_source=None,
            cli_wait=None,
        )
        assert mode is RunMode.BATCH
        assert sub is SubmitterMode.EXEC
        assert code is CodeSource.IMAGE
        assert no_wait is True

    def test_explicit_submitter_overrides_mode(self) -> None:
        """`--mode batch --submitter portForward` keeps the Ray transport."""
        from nrl_k8s.schema import CodeSource, RunMode, SubmitterMode

        _, sub, code, no_wait = cli._resolve_mode_defaults(
            cli_mode="batch",
            infra_mode=RunMode.INTERACTIVE,
            cli_submitter="portForward",
            cli_code_source=None,
            cli_wait=None,
        )
        assert sub is SubmitterMode.PORT_FORWARD
        # codeSource still follows the batch macro.
        assert code is CodeSource.IMAGE
        assert no_wait is True

    def test_explicit_code_source_overrides_mode(self) -> None:
        from nrl_k8s.schema import CodeSource, RunMode

        _, _, code, _ = cli._resolve_mode_defaults(
            cli_mode="batch",
            infra_mode=RunMode.INTERACTIVE,
            cli_submitter=None,
            cli_code_source="lustre",
            cli_wait=None,
        )
        assert code is CodeSource.LUSTRE

    def test_wait_flag_overrides_mode_default(self) -> None:
        """`--mode batch --wait` should keep exec + image but follow logs."""
        _, _, _, no_wait = cli._resolve_mode_defaults(
            cli_mode="batch",
            infra_mode=__import__("nrl_k8s.schema", fromlist=["RunMode"]).RunMode.INTERACTIVE,
            cli_submitter=None,
            cli_code_source=None,
            cli_wait=True,
        )
        assert no_wait is False

    def test_infra_run_mode_used_without_cli_flag(self) -> None:
        """`runMode: batch` in the infra YAML flips defaults even when
        --mode isn't on the command line."""
        from nrl_k8s.schema import CodeSource, RunMode, SubmitterMode

        mode, sub, code, no_wait = cli._resolve_mode_defaults(
            cli_mode=None,
            infra_mode=RunMode.BATCH,
            cli_submitter=None,
            cli_code_source=None,
            cli_wait=None,
        )
        assert mode is RunMode.BATCH
        assert sub is SubmitterMode.EXEC
        assert code is CodeSource.IMAGE
        assert no_wait is True
