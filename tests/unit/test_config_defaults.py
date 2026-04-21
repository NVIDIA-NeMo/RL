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
import warnings
from dataclasses import field
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict, ValidationError
from pydantic.dataclasses import dataclass

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
    validate_config,
)

register_omegaconf_resolvers()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(os.path.abspath(__file__)).parent
_REPO_ROOT = _TESTS_DIR.parent.parent
_EXEMPLAR_YAML = _REPO_ROOT / "examples" / "configs" / "grpo_math_1B.yaml"

_EXTRA_IGNORE = ConfigDict(extra="ignore")


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
# 3. validate_config — unit tests
# ===================================================================


def test_missing_keys_filled():
    """Missing keys should be filled with dataclass defaults."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        a: int = 1
        b: str = "hello"

    config = {"x": 99}
    result = validate_config(config, Defaults)
    assert result == {"x": 99, "a": 1, "b": "hello"}


def test_existing_keys_not_overwritten():
    """Existing keys (including None and False) must never be overwritten."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        a: int = 1
        b: bool = True
        c: str | None = "default"  # nullable so None is a valid value

    config = {"a": 999, "b": False, "c": None}
    result = validate_config(config, Defaults)
    assert result["a"] == 999
    assert result["b"] is False
    assert result["c"] is None


def test_nested_recursion():
    """Nested dataclass defaults should recurse into existing dicts."""

    @dataclass(config=_EXTRA_IGNORE)
    class Inner:
        x: int = 10
        y: bool = False

    @dataclass(config=_EXTRA_IGNORE)
    class Outer:
        inner: Inner = field(default_factory=Inner)
        top: str = "ok"

    config = {"inner": {"x": 42}}
    result = validate_config(config, Outer)
    assert result == {"inner": {"x": 42, "y": False}, "top": "ok"}


def test_missing_nested_section_created():
    """When a nested section is entirely absent, create it from defaults."""

    @dataclass(config=_EXTRA_IGNORE)
    class Inner:
        enabled: bool = False
        value: int = 5

    @dataclass(config=_EXTRA_IGNORE)
    class Outer:
        inner: Inner = field(default_factory=Inner)

    config = {}
    result = validate_config(config, Outer)
    assert result == {"inner": {"enabled": False, "value": 5}}


def test_extra_keys_preserved():
    """Keys not in the schema must survive the round-trip."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        a: int = 1

    config = {"a": 2, "unknown_key": "keep_me", "another": [1, 2, 3]}
    result = validate_config(config, Defaults)
    assert result["a"] == 2
    assert result["unknown_key"] == "keep_me"
    assert result["another"] == [1, 2, 3]


def test_nested_extra_keys_preserved():
    """Extra keys inside nested dicts must also survive."""

    @dataclass(config=_EXTRA_IGNORE)
    class Inner:
        x: int = 10

    @dataclass(config=_EXTRA_IGNORE)
    class Outer:
        inner: Inner = field(default_factory=Inner)

    config = {"inner": {"x": 42, "extra_inner": "hi"}, "extra_top": 999}
    result = validate_config(config, Outer)
    assert result["inner"]["x"] == 42
    assert result["inner"]["extra_inner"] == "hi"
    assert result["extra_top"] == 999


# ===================================================================
# 4. Type validation tests (NEW — the core upgrade)
# ===================================================================


def test_type_error_raises_validation_error():
    """Passing a wrong type for a known field must raise ValidationError."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        seed: int = 42

    with pytest.raises(ValidationError):
        validate_config({"seed": "not_an_int"}, Defaults)


def test_type_error_nested():
    """Type errors in nested configs must also be caught."""

    @dataclass(config=_EXTRA_IGNORE)
    class Inner:
        value: int = 5

    @dataclass(config=_EXTRA_IGNORE)
    class Outer:
        inner: Inner = field(default_factory=Inner)

    with pytest.raises(ValidationError):
        validate_config({"inner": {"value": "bad"}}, Outer)


def test_type_coercion():
    """Pydantic should coerce compatible types (e.g. str "42" → int 42)."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        count: int = 0
        rate: float = 1.0

    result = validate_config({"count": "42", "rate": "3.14"}, Defaults)
    assert result["count"] == 42
    assert isinstance(result["count"], int)
    assert abs(result["rate"] - 3.14) < 1e-6
    assert isinstance(result["rate"], float)


def test_nullable_fields():
    """Fields typed as T | None should accept None."""

    @dataclass(config=_EXTRA_IGNORE)
    class Defaults:
        threshold: float | None = None

    result = validate_config({"threshold": None}, Defaults)
    assert result["threshold"] is None

    result2 = validate_config({"threshold": 0.5}, Defaults)
    assert result2["threshold"] == 0.5

    result3 = validate_config({}, Defaults)
    assert result3["threshold"] is None


# ===================================================================
# 5. Backward compatibility — deprecated apply_config_defaults
# ===================================================================


def test_apply_config_defaults_deprecated_warns():
    """apply_config_defaults should emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        apply_config_defaults({"grpo": {}}, GRPOMasterConfigDefaults)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert "validate_config" in str(deprecation_warnings[0].message)


def test_apply_config_defaults_still_works():
    """Deprecated apply_config_defaults should produce the same result as validate_config."""
    config_a = {
        "grpo": {"num_prompts_per_step": 32, "seed": 100},
        "loss_fn": {"reference_policy_kl_penalty": 0.01},
    }
    config_b = {
        "grpo": {"num_prompts_per_step": 32, "seed": 100},
        "loss_fn": {"reference_policy_kl_penalty": 0.01},
    }
    result_new = validate_config(config_a, GRPOMasterConfigDefaults)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result_old = apply_config_defaults(config_b, GRPOMasterConfigDefaults)
    assert result_new == result_old


# ===================================================================
# 6. Integration test — GRPOMasterConfigDefaults
# ===================================================================


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
        "policy": {"some_policy_key": True},  # extra top-level section
    }
    result = validate_config(config, GRPOMasterConfigDefaults)

    # seed was already set — must not be overwritten
    assert result["grpo"]["seed"] == 100
    # overlong_filtering was missing — filled from defaults
    assert result["grpo"]["overlong_filtering"] is False
    # calculate_advantages_on_gpu was missing — filled
    assert result["grpo"]["calculate_advantages_on_gpu"] is False
    # nested reward_scaling should be filled
    assert result["grpo"]["reward_scaling"]["enabled"] is False
    # loss_fn NotRequired fields filled
    assert result["loss_fn"]["disable_ppo_ratio"] is False
    assert result["loss_fn"]["force_on_policy_ratio"] is False
    # extra top-level section preserved
    assert result["policy"]["some_policy_key"] is True
    # extra keys inside grpo preserved
    assert result["grpo"]["num_prompts_per_step"] == 32


def test_grpo_master_defaults_with_exemplar_yaml():
    """validate_config on full exemplar YAML should not raise."""
    yaml_dict = _load_exemplar()
    result = validate_config(yaml_dict, GRPOMasterConfigDefaults)
    # All original keys must still be present
    assert "grpo" in result
    assert "loss_fn" in result
    assert "policy" in result
    assert "data" in result
