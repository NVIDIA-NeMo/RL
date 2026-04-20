"""Tests for :mod:`nrl_k8s.config` — layered loading of recipe + infra:.

Priority (low → high):

  1. Shipped defaults (``nrl_k8s/defaults/defaults.example.yaml``)
  2. User defaults      (``$NRL_K8S_DEFAULTS`` or ``~/.config/nrl-k8s/defaults.yaml``)
  3. Recipe ``infra:``  (top-level key on the recipe YAML)
  4. CLI overrides      (Hydra-style ``key=value`` list)

Each test pins one boundary of that precedence rule so regressions are caught
with a single-line diff.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nrl_k8s import config as cfg_mod
from nrl_k8s.config import load_recipe_with_infra
from nrl_k8s.schema import InfraConfig, SchedulerKind, WorkspaceKind


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _no_user_defaults(monkeypatch, tmp_path):
    """Point ``_USER_DEFAULTS`` at a non-existent file so user-level config
    never bleeds into tests from whatever laptop this runs on.
    """
    monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", tmp_path / "no-such.yaml")


@pytest.fixture(autouse=True)
def _force_fallback_loader(monkeypatch):
    """Force the OmegaConf-only recipe loader so tests don't depend on nemo_rl.

    The production code prefers nemo_rl's loader (to resolve ``defaults:`` and
    ``${mul:...}``) when available; here we want a hermetic fallback.
    """
    import builtins

    real_import = builtins.__import__

    def _fail_nemo_rl(name, *args, **kwargs):  # noqa: ANN001
        if name.startswith("nemo_rl"):
            raise ImportError("forced-fallback: nemo_rl disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_nemo_rl)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


# =============================================================================
# Layered precedence
# =============================================================================

class TestPrecedence:
    def test_shipped_defaults_alone_plus_required_fields(self, tmp_path) -> None:
        """Minimum recipe (just the required namespace + image) validates."""
        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {"infra": {"namespace": "ns-a", "image": "img:1"}},
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.namespace == "ns-a"
        assert loaded.infra.image == "img:1"
        # Shipped default:
        assert loaded.infra.scheduler.kind is SchedulerKind.DEFAULT
        assert loaded.infra.workspace.kind is WorkspaceKind.RAY_UPLOAD

    def test_recipe_overrides_shipped_defaults(self, tmp_path) -> None:
        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {
                "infra": {
                    "namespace": "ns",
                    "image": "img:1",
                    "scheduler": {"kind": "kai", "queue": "team-a"},
                }
            },
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.scheduler.kind is SchedulerKind.KAI
        assert loaded.infra.scheduler.queue == "team-a"

    def test_user_defaults_beat_shipped(self, tmp_path, monkeypatch) -> None:
        user = _write_yaml(
            tmp_path / "user.yaml",
            {"infra": {"scheduler": {"kind": "kai", "queue": "user-q"}}},
        )
        monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", user)

        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {"infra": {"namespace": "ns", "image": "img:1"}},
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.scheduler.kind is SchedulerKind.KAI
        assert loaded.infra.scheduler.queue == "user-q"

    def test_recipe_beats_user_defaults(self, tmp_path, monkeypatch) -> None:
        user = _write_yaml(
            tmp_path / "user.yaml",
            {"infra": {"scheduler": {"kind": "kai", "queue": "user-q"}}},
        )
        monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", user)

        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {
                "infra": {
                    "namespace": "ns",
                    "image": "img:1",
                    "scheduler": {"kind": "kai", "queue": "recipe-q"},
                }
            },
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.scheduler.queue == "recipe-q"

    def test_cli_overrides_beat_recipe(self, tmp_path) -> None:
        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {
                "infra": {
                    "namespace": "ns",
                    "image": "img:1",
                    "scheduler": {"kind": "kai", "queue": "recipe-q"},
                }
            },
        )
        loaded = load_recipe_with_infra(
            recipe, overrides=["infra.scheduler.queue=cli-q"]
        )
        assert loaded.infra.scheduler.queue == "cli-q"

    def test_user_defaults_without_infra_wrapper(self, tmp_path, monkeypatch) -> None:
        """Users may write either ``{infra: {...}}`` or just the body."""
        user = _write_yaml(
            tmp_path / "user.yaml",
            {"scheduler": {"kind": "kai", "queue": "bare-q"}},
        )
        monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", user)

        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {"infra": {"namespace": "ns", "image": "img:1"}},
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.scheduler.queue == "bare-q"


# =============================================================================
# Recipe handling
# =============================================================================

class TestRecipeBody:
    def test_infra_is_peeled_off(self, tmp_path) -> None:
        """The returned recipe must not contain `infra:` — the NeMo-RL
        entrypoint is never supposed to see it.
        """
        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {
                "infra": {"namespace": "ns", "image": "img:1"},
                "policy": {"train_micro_batch_size": 2},
                "grpo": {"num_generations_per_prompt": 4},
            },
        )
        loaded = load_recipe_with_infra(recipe)
        assert "infra" not in loaded.recipe
        assert loaded.recipe["policy"]["train_micro_batch_size"] == 2

    def test_recipe_without_infra_still_loads(self, tmp_path, monkeypatch) -> None:
        """User defaults should supply namespace+image when the recipe omits infra:."""
        user = _write_yaml(
            tmp_path / "user.yaml",
            {"infra": {"namespace": "ns", "image": "img:1"}},
        )
        monkeypatch.setattr(cfg_mod, "_USER_DEFAULTS", user)

        recipe = _write_yaml(
            tmp_path / "recipe.yaml", {"policy": {"train_micro_batch_size": 2}}
        )
        loaded = load_recipe_with_infra(recipe)
        assert loaded.infra.namespace == "ns"

    def test_recipe_with_non_mapping_infra_rejected(self, tmp_path) -> None:
        recipe = _write_yaml(tmp_path / "recipe.yaml", {"infra": ["not", "a", "dict"]})
        with pytest.raises(ValueError):
            load_recipe_with_infra(recipe)


# =============================================================================
# Return value shape
# =============================================================================

class TestLoadedConfig:
    def test_returns_validated_infra(self, tmp_path) -> None:
        recipe = _write_yaml(
            tmp_path / "recipe.yaml",
            {"infra": {"namespace": "ns", "image": "img:1"}},
        )
        loaded = load_recipe_with_infra(recipe)
        assert isinstance(loaded.infra, InfraConfig)
        assert loaded.source_path == recipe.resolve()
