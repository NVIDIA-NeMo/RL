"""Recipe + infra config loader for ``nrl-k8s``.

The CLI's single source of truth for a run. Given a recipe path, loads and
merges four layers in priority order (low to high):

  1. Shipped defaults         — ``nrl_k8s/defaults/defaults.example.yaml``
  2. User defaults            — ``~/.config/nrl-k8s/defaults.yaml`` (optional)
  3. Recipe-level ``infra:``  — the ``infra:`` key on the recipe YAML, if present
  4. CLI Hydra overrides      — ``infra.scheduler.queue=my-queue``, etc.

Layers 1-3 are YAML; layer 4 is a list of Hydra-style strings from the CLI. The
merged ``infra`` mapping is validated through :class:`nrl_k8s.schema.InfraConfig`.

The rest of the recipe (``policy``, ``grpo``, ``data``, ``logger``, ...) is
loaded via NeMo-RL's own loader when available, so defaults/inheritance
(``defaults: ../../grpo_math_1B.yaml``) and ``${mul:...}`` resolvers are
handled exactly the same way as the existing training entrypoints.

If ``nemo_rl`` is not importable (e.g. the CLI is installed standalone on a
dev machine without the full repo), a minimal OmegaConf fallback handles a
single-file recipe. Inheritance is not supported in fallback mode.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .schema import InfraConfig


def get_username() -> str:
    """Return the local OS username, sanitised for K8s resource names."""
    raw = os.environ.get("NRL_K8S_USER") or getpass.getuser()
    return raw.lower().replace("_", "-").replace(".", "-")


def _register_nrl_resolvers() -> None:
    if not OmegaConf.has_resolver("user"):
        OmegaConf.register_new_resolver("user", lambda: get_username())

_SHIPPED_DEFAULTS = Path(__file__).parent / "defaults" / "defaults.example.yaml"
_USER_DEFAULTS = Path(
    os.environ.get(
        "NRL_K8S_DEFAULTS", Path.home() / ".config" / "nrl-k8s" / "defaults.yaml"
    )
)


@dataclass
class LoadedConfig:
    """Bundle returned by :func:`load_recipe_with_infra`.

    ``recipe`` holds the resolved recipe with ``infra`` removed (so it can be
    passed to the NeMo-RL entry-point as-is). ``infra`` is the validated
    :class:`InfraConfig` instance. ``source_path`` is the recipe path we loaded.
    """

    recipe: DictConfig
    infra: InfraConfig
    source_path: Path


# =============================================================================
# Public API
# =============================================================================


def load_recipe_with_infra(
    recipe_path: str | Path,
    overrides: list[str] | None = None,
    *,
    infra_path: str | Path | None = None,
) -> LoadedConfig:
    """Load a NeMo-RL recipe plus its infra config and return both.

    Two supported layouts:

    * **Split**: pass ``infra_path`` to point at a dedicated infra YAML
      (recommended for any non-trivial infra block — keeps the recipe
      focused on training config). The recipe must not also declare
      an ``infra:`` key in that case.
    * **Bundled**: recipe has an ``infra:`` top-level key. ``infra_path``
      is ``None``.

    Args:
        recipe_path: Path to a recipe YAML (absolute or relative to cwd).
        overrides: Hydra-style overrides. ``infra.*`` overrides apply to
            the infra layer; other overrides apply to the recipe.
        infra_path: Optional path to a standalone infra YAML (see above).
    """
    overrides = overrides or []
    recipe_path = Path(recipe_path).resolve()
    _register_nrl_resolvers()

    recipe_overrides, infra_overrides = _partition_overrides(overrides)
    recipe = _load_recipe(recipe_path, overrides=recipe_overrides)

    infra_raw = _merge_infra(
        recipe,
        infra_path=Path(infra_path).resolve() if infra_path else None,
        overrides=infra_overrides,
    )
    recipe.pop("infra", None)  # peel any recipe-level infra: off

    # Validate; OmegaConf -> plain containers -> pydantic.
    infra_container = OmegaConf.to_container(infra_raw, resolve=True)
    if not infra_container.get("namespace"):
        infra_container["namespace"] = _infer_kube_namespace()
    infra = InfraConfig.model_validate(infra_container)

    return LoadedConfig(recipe=recipe, infra=infra, source_path=recipe_path)


_SA_NS_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def _infer_kube_namespace() -> str:
    """Default ``infra.namespace`` to the active kube context's namespace.

    Tries (in order) the pod service-account file, the current kubeconfig
    context, and finally ``default``.
    """
    try:
        if _SA_NS_PATH.exists():
            ns = _SA_NS_PATH.read_text().strip()
            if ns:
                return ns
    except OSError:
        pass
    try:
        from kubernetes import config as k8s_config

        _contexts, active = k8s_config.list_kube_config_contexts()
        ns = ((active or {}).get("context") or {}).get("namespace")
        if ns:
            return ns
    except Exception:
        pass
    return "default"


def _partition_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Split Hydra overrides: infra.* go to the infra layer, rest to recipe.

    Keeps the recipe loader from seeing (and rejecting) infra.* keys on
    strict NeMo-RL configs that use ``struct`` mode.
    """
    recipe_side: list[str] = []
    infra_side: list[str] = []
    for o in overrides:
        body = o.lstrip("+~")
        if body.startswith("infra.") or body == "infra":
            # Strip "infra." prefix so the override applies directly on
            # the infra DictConfig (which has no wrapping "infra:" key).
            leading = o[: len(o) - len(body)]
            infra_side.append(leading + body[len("infra.") :])
        else:
            recipe_side.append(o)
    return recipe_side, infra_side


# =============================================================================
# Internals
# =============================================================================


def _load_recipe(recipe_path: Path, overrides: list[str]) -> DictConfig:
    """Load a recipe YAML. Uses nemo_rl's loader if available, else OmegaConf fallback."""
    try:
        from nemo_rl.utils.config import (  # type: ignore[import-not-found]
            load_config,
            parse_hydra_overrides,
            register_omegaconf_resolvers,
        )
    except ImportError:
        return _load_recipe_fallback(recipe_path, overrides)

    register_omegaconf_resolvers()
    cfg = load_config(str(recipe_path))

    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)

    if not isinstance(cfg, DictConfig):
        raise ValueError(f"recipe at {recipe_path} did not load as a mapping")
    return cfg


def _load_recipe_fallback(recipe_path: Path, overrides: list[str]) -> DictConfig:
    """OmegaConf-only loader used when nemo_rl is unavailable.

    Handles ``defaults:`` inheritance recursively (same semantics as
    :func:`nemo_rl.utils.config.load_config_with_inheritance`). Custom
    resolvers like ``${mul:...}`` are NOT registered — those are only
    needed by NeMo-RL's own entrypoints, which always run inside the Ray
    container where nemo_rl is importable.
    """
    cfg = _load_with_inheritance(recipe_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"recipe at {recipe_path} did not load as a mapping")
    return cfg


def _load_with_inheritance(path: Path) -> DictConfig:
    """Walk a recipe's ``defaults:`` chain and return the merged DictConfig."""
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{path} did not load as a mapping")

    if "defaults" in cfg:  # type: ignore[operator]
        raw = cfg.pop("defaults")
        defaults: list[str] = (
            [str(raw)] if isinstance(raw, (str, Path)) else [str(x) for x in raw]
        )
        base: DictConfig = OmegaConf.create({})
        for rel in defaults:
            parent = (path.parent / rel).resolve()
            parent_cfg = _load_with_inheritance(parent)
            merged = OmegaConf.merge(base, parent_cfg)
            if not isinstance(merged, DictConfig):
                raise ValueError(f"defaults merge for {path} produced non-mapping")
            base = merged
        merged = OmegaConf.merge(base, cfg)
        if not isinstance(merged, DictConfig):
            raise ValueError(f"inheritance merge for {path} produced non-mapping")
        cfg = merged
    return cfg


def _merge_infra(
    recipe: DictConfig,
    *,
    infra_path: Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Stack shipped defaults < user defaults < (infra file | recipe.infra) < CLI."""
    shipped = _load_yaml_if_present(_SHIPPED_DEFAULTS, required=True)
    user = _load_yaml_if_present(_USER_DEFAULTS, required=False)

    infra_layer: DictConfig
    if infra_path is not None:
        if "infra" in recipe:  # type: ignore[operator]
            raise ValueError(
                "infra config supplied via --infra but the recipe also contains "
                "an `infra:` key — choose one or the other."
            )
        infra_layer = _load_with_inheritance(infra_path)
        infra_layer = _pick_infra(infra_layer)
    else:
        infra_layer = _extract_recipe_infra(recipe)

    # Strip top-level keys starting with "_" — these are anchor-only scratch
    # sections (e.g. `_shared: &foo ...`) that YAML emits to the parsed dict
    # even though they carry no infra meaning.
    for k in list(infra_layer.keys()):  # type: ignore[union-attr]
        if isinstance(k, str) and k.startswith("_"):
            infra_layer.pop(k)

    merged = OmegaConf.merge(
        _pick_infra(shipped),
        _pick_infra(user) if user is not None else OmegaConf.create({}),
        infra_layer,
    )
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    if not isinstance(merged, DictConfig):
        raise RuntimeError("internal: infra merge did not produce a DictConfig")
    return merged


def _pick_infra(cfg: DictConfig) -> DictConfig:
    """A defaults file may be either ``{infra: {...}}`` or just the infra body."""
    if "infra" in cfg:  # type: ignore[operator]
        inner = cfg["infra"]
        if not isinstance(inner, DictConfig):
            raise ValueError("defaults file has non-mapping `infra:` key")
        return inner
    return cfg


def _extract_recipe_infra(recipe: DictConfig) -> DictConfig:
    if "infra" not in recipe:  # type: ignore[operator]
        return OmegaConf.create({})
    inner = recipe["infra"]
    if not isinstance(inner, DictConfig):
        raise ValueError("recipe `infra:` key must be a mapping")
    return inner


def _load_yaml_if_present(path: Path, *, required: bool) -> DictConfig | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"shipped defaults missing: {path}")
        return None
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"{path} did not load as a mapping")
    return loaded


__all__ = ["LoadedConfig", "load_recipe_with_infra"]
