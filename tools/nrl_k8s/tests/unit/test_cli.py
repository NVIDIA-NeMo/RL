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


class TestClusterDown:
    def test_errors_without_role_or_name(self, tmp_path, monkeypatch) -> None:
        recipe = _write_recipe(
            tmp_path, {"infra": {"namespace": "ns-a", "image": "img:1"}}
        )
        runner = CliRunner()
        result = runner.invoke(cli.main, ["cluster", "down", str(recipe)])
        assert result.exit_code == 2
        assert "--role" in result.output or "--name" in result.output
