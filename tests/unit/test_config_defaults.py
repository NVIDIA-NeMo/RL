# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for dataclass config defaults (see docs/design-docs/dataclass-config-defaults.md)."""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    AsyncGRPOConfigDefaults,
    ClippedPGLossConfig,
    ClippedPGLossConfigDefaults,
    GRPOConfig,
    GRPOConfigDefaults,
    GRPOMasterConfigDefaults,
    RewardScalingConfig,
    RewardScalingConfigDefaults,
    RewardShapingConfigDefaults,
)
from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.utils.config import (
    apply_config_defaults,
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)

register_omegaconf_resolvers()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(os.path.abspath(__file__)).parent
_REPO_ROOT = _TESTS_DIR.parent.parent
_EXEMPLAR_YAML = _REPO_ROOT / "examples" / "configs" / "grpo_math_1B.yaml"


# ===================================================================
# 1. Field-subset tests — every defaults field must exist in TypedDict
# ===================================================================

_DEFAULTS_TO_TYPEDDICT = [
    (GRPOConfigDefaults, GRPOConfig),
    (ClippedPGLossConfigDefaults, ClippedPGLossConfig),
    (RewardScalingConfigDefaults, RewardScalingConfig),
    (RewardShapingConfigDefaults, RewardShapingConfig),
    (AsyncGRPOConfigDefaults, AsyncGRPOConfig),
]


@pytest.mark.parametrize(
    "defaults_cls,typeddict_cls",
    _DEFAULTS_TO_TYPEDDICT,
    ids=[d.__name__ for d, _ in _DEFAULTS_TO_TYPEDDICT],
)
def test_defaults_fields_subset_of_typeddict(defaults_cls, typeddict_cls):
    """Defaults dataclass must not introduce fields absent from the TypedDict."""
    dc_keys = {f.name for f in dataclasses.fields(defaults_cls)}
    td_keys = set(typeddict_cls.__annotations__.keys())
    extra = dc_keys - td_keys
    assert not extra, (
        f"{defaults_cls.__name__} has fields not in {typeddict_cls.__name__}: {extra}"
    )


# ===================================================================
# 2. Default values match exemplar YAML
# ===================================================================


def _load_exemplar() -> dict:
    cfg = load_config_with_inheritance(_EXEMPLAR_YAML)
    return OmegaConf.to_container(cfg, resolve=True)


def test_grpo_defaults_match_exemplar_yaml():
    """GRPOConfigDefaults values must match the exemplar YAML."""
    yaml_dict = _load_exemplar()
    grpo_yaml = yaml_dict["grpo"]

    for f in dataclasses.fields(GRPOConfigDefaults):
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            continue
        # Skip nested dataclass fields — tested separately.
        default = (
            f.default if f.default is not dataclasses.MISSING else f.default_factory()
        )
        if dataclasses.is_dataclass(default):
            continue
        yaml_val = grpo_yaml.get(f.name)
        if yaml_val is not None:
            assert yaml_val == default, (
                f"grpo.{f.name}: YAML={yaml_val!r}, dataclass={default!r}"
            )


def test_reward_scaling_defaults_match_exemplar_yaml():
    """RewardScalingConfigDefaults values must match the exemplar YAML."""
    yaml_dict = _load_exemplar()
    rs_yaml = yaml_dict["grpo"]["reward_scaling"]

    for f in dataclasses.fields(RewardScalingConfigDefaults):
        yaml_val = rs_yaml.get(f.name)
        if yaml_val is not None:
            assert yaml_val == f.default, (
                f"grpo.reward_scaling.{f.name}: YAML={yaml_val!r}, dataclass={f.default!r}"
            )


def test_loss_fn_defaults_match_exemplar_yaml():
    """ClippedPGLossConfigDefaults values must match the exemplar YAML."""
    yaml_dict = _load_exemplar()
    loss_yaml = yaml_dict["loss_fn"]

    for f in dataclasses.fields(ClippedPGLossConfigDefaults):
        yaml_val = loss_yaml.get(f.name)
        if yaml_val is not None:
            assert yaml_val == f.default, (
                f"loss_fn.{f.name}: YAML={yaml_val!r}, dataclass={f.default!r}"
            )


# ===================================================================
# 3. apply_config_defaults — unit tests
# ===================================================================


def test_missing_keys_filled():
    """Missing keys should be filled with dataclass defaults."""

    @dataclass
    class Defaults:
        a: int = 1
        b: str = "hello"

    config = {"x": 99}
    apply_config_defaults(config, Defaults)
    assert config == {"x": 99, "a": 1, "b": "hello"}


def test_existing_keys_not_overwritten():
    """Existing keys (including None and False) must never be overwritten."""

    @dataclass
    class Defaults:
        a: int = 1
        b: bool = True
        c: str = "default"

    config = {"a": 999, "b": False, "c": None}
    apply_config_defaults(config, Defaults)
    assert config == {"a": 999, "b": False, "c": None}


def test_nested_recursion():
    """Nested dataclass defaults should recurse into existing dicts."""

    @dataclass
    class Inner:
        x: int = 10
        y: bool = False

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)
        top: str = "ok"

    config = {"inner": {"x": 42}}
    apply_config_defaults(config, Outer)
    assert config == {"inner": {"x": 42, "y": False}, "top": "ok"}


def test_missing_nested_section_created():
    """When a nested section is entirely absent, create it from defaults."""

    @dataclass
    class Inner:
        enabled: bool = False
        value: int = 5

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)

    config = {}
    apply_config_defaults(config, Outer)
    assert config == {"inner": {"enabled": False, "value": 5}}


def test_required_fields_skipped():
    """Fields without defaults (required) must be skipped, not injected."""

    @dataclass
    class Defaults:
        required_field: int  # no default — must come from YAML
        optional_field: bool = False

    config = {"required_field": 42}
    apply_config_defaults(config, Defaults)
    # required_field was already present — kept; optional_field filled.
    assert config == {"required_field": 42, "optional_field": False}

    # When required field is absent, it must NOT be injected.
    config2: dict = {}
    apply_config_defaults(config2, Defaults)
    assert "required_field" not in config2
    assert config2 == {"optional_field": False}


def test_non_dataclass_ignored():
    """Passing a non-dataclass as defaults_cls should be a no-op."""
    config = {"a": 1}
    result = apply_config_defaults(config, dict)
    assert result == {"a": 1}


def test_non_type_hint_skipped():
    """When a type hint is not a concrete type (e.g., Union), recursion is skipped."""

    @dataclass
    class Defaults:
        items: list = field(default_factory=list)

    config = {"items": {"nested": True}}
    apply_config_defaults(config, Defaults)
    # items is a dict in config but the hint (list) is not a dataclass → no recursion
    assert config == {"items": {"nested": True}}


def test_grpo_master_defaults_integration():
    """GRPOMasterConfigDefaults should fill missing keys in a realistic config."""
    config = {
        "grpo": {
            "num_prompts_per_step": 32,
            "num_generations_per_prompt": 16,
            "seed": 100,
        },
        "loss_fn": {
            "reference_policy_kl_penalty": 0.01,
        },
    }
    apply_config_defaults(config, GRPOMasterConfigDefaults)

    # seed was already set — must not be overwritten
    assert config["grpo"]["seed"] == 100
    # overlong_filtering was missing — filled from defaults
    assert config["grpo"]["overlong_filtering"] is False
    # calculate_advantages_on_gpu was missing — filled
    assert config["grpo"]["calculate_advantages_on_gpu"] is False
    # nested reward_scaling should be filled
    assert config["grpo"]["reward_scaling"]["enabled"] is False
    # loss_fn NotRequired fields filled
    assert config["loss_fn"]["disable_ppo_ratio"] is False
    assert config["loss_fn"]["force_on_policy_ratio"] is False
