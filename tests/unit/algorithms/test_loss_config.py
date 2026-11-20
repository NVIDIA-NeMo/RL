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

import warnings

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.loss import (
    CISPOLossConfig,
    ClippedPGLossConfig,
    ClippedPGLossFn,
    ClippedSurrogateLossConfig,
    PolicyGradientLossConfig,
    ReinforceLossConfig,
    as_clipped_pg_config,
    build_policy_gradient_loss,
)
from nemo_rl.algorithms.ppo import MasterConfig as PPOMasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig as SingleControllerMasterConfig,
)

LOSS_CONFIG = TypeAdapter(PolicyGradientLossConfig)
LEGACY_SETTINGS = {"ratio_clip_c": 3.0, "ratio_clip_max": 0.3, "extension_option": 7}


@pytest.mark.parametrize("tag", [{}, {"kind": "legacy_pg"}])
def test_legacy_tag_is_optional_and_never_serialized(tag):
    parsed = LOSS_CONFIG.validate_python({**tag, **LEGACY_SETTINGS})
    untagged = ClippedPGLossConfig(**LEGACY_SETTINGS)

    assert isinstance(parsed, ClippedPGLossConfig)
    assert parsed.kind == "legacy_pg"
    # Dumps feed checkpoint fingerprints, so the tag must not appear in them.
    assert "kind" not in parsed.model_dump()
    assert parsed.model_dump() == untagged.model_dump()
    assert parsed.model_dump(exclude_unset=True) == LEGACY_SETTINGS
    assert LOSS_CONFIG.validate_json(parsed.model_dump_json()) == parsed


def test_unknown_kind_is_rejected():
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        LOSS_CONFIG.validate_python({"kind": "unknown"})


def test_legacy_instances_pass_through_the_union_unchanged():
    legacy = ClippedPGLossConfig(**LEGACY_SETTINGS)
    assert LOSS_CONFIG.validate_python(legacy) is legacy
    assert as_clipped_pg_config(legacy) is legacy


def test_factory_builds_the_clipped_pg_loss_with_trainer_settings():
    cfg = ClippedPGLossConfig(
        force_on_policy_ratio=True, seq_logprob_error_in_loss=True
    )

    loss_fn = build_policy_gradient_loss(
        cfg, use_fused_linear_logprobs=True, seq_logprob_error_threshold=0.5
    )

    assert isinstance(loss_fn, ClippedPGLossFn)
    assert loss_fn.use_fused_linear_logprobs
    assert loss_fn.seq_logprob_error_threshold == 0.5
    assert loss_fn.opd_full is None


IS_CORRECTION = {
    "use_importance_sampling_correction": True,
    "truncated_importance_sampling_ratio": 5.0,
    "truncated_importance_sampling_ratio_min": 0.0,
}
ADAPTIVE_PPO_CLIP_SETTINGS = {
    "truncated_importance_sampling_adaptive_target_entropy",
    "truncated_importance_sampling_adaptive_k_step",
    "truncated_importance_sampling_adaptive_k_max",
    "truncated_importance_sampling_adaptive_outer_bound",
}
# Per variant: its config type, the legacy switches that select the same
# objective, example settings, and the ClippedPGLossConfig settings it has no
# field for because they belong to other objectives.
VARIANTS = {
    "ppo": (
        ClippedSurrogateLossConfig,
        {},
        {
            **IS_CORRECTION,
            "ratio_clip_max": 0.28,
            "ratio_clip_c": 3.0,
            "truncated_importance_sampling_type": "tis",
        },
        {"use_cispo", "disable_ppo_ratio", "sequence_level_importance_ratios"},
    ),
    "cispo": (
        CISPOLossConfig,
        {"use_cispo": True},
        {
            **IS_CORRECTION,
            "ratio_clip_min": 1.0,
            "ratio_clip_max": 5.0,
            "reference_policy_kl_penalty": 0.0,
            "truncated_importance_sampling_type": "tis",
            "force_on_policy_ratio": True,
        },
        {"disable_ppo_ratio", "sequence_level_importance_ratios", "ratio_clip_c"},
    ),
    "reinforce": (
        ReinforceLossConfig,
        {"disable_ppo_ratio": True},
        {
            **IS_CORRECTION,
            "token_level_loss": False,
            "reference_policy_kl_penalty": 0.0,
            "truncated_importance_sampling_type": "icepop",
        },
        {
            "ratio_clip_min",
            "ratio_clip_max",
            "ratio_clip_c",
            "sequence_level_importance_ratios",
            "use_cispo",
            *ADAPTIVE_PPO_CLIP_SETTINGS,
        },
    ),
}
UNSUPPORTED_SETTINGS = [
    (kind, setting)
    for kind, (_, _, _, unsupported) in VARIANTS.items()
    for setting in sorted(unsupported)
]


@pytest.mark.parametrize("kind", VARIANTS)
def test_variant_matches_the_untagged_config_of_its_objective(kind):
    variant, legacy_switches, settings, _ = VARIANTS[kind]
    tagged = LOSS_CONFIG.validate_python({"kind": kind, **settings})
    untagged = ClippedPGLossConfig(**legacy_switches, **settings)

    assert type(tagged) is variant
    assert as_clipped_pg_config(tagged).model_dump() == untagged.model_dump()
    assert vars(build_policy_gradient_loss(tagged)) == vars(ClippedPGLossFn(untagged))
    assert tagged.model_dump()["kind"] == kind
    assert LOSS_CONFIG.validate_json(tagged.model_dump_json()) == tagged


@pytest.mark.parametrize("kind", VARIANTS)
def test_variant_tracks_the_shared_config_fields_and_defaults(kind):
    variant, legacy_switches, _, unsupported = VARIANTS[kind]
    # A setting added to ClippedPGLossConfig must be given to each variant or
    # listed as unsupported; this keeps them from drifting apart silently.
    assert set(variant.model_fields) == (
        set(ClippedPGLossConfig.model_fields) - unsupported
    )
    assert (
        variant().as_clipped_pg_config().model_dump()
        == ClippedPGLossConfig(**legacy_switches).model_dump()
    )


@pytest.mark.parametrize(("kind", "setting"), UNSUPPORTED_SETTINGS)
def test_variant_rejects_settings_of_other_objectives(kind, setting):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        VARIANTS[kind][0](**{setting: None})


@pytest.mark.parametrize(
    ("kind", "setting", "fixed"),
    [
        ("cispo", "use_cispo", True),
        ("cispo", "token_level_loss", True),
        ("reinforce", "disable_ppo_ratio", True),
        ("reinforce", "force_on_policy_ratio", False),
        ("reinforce", "seq_logprob_error_in_loss", False),
    ],
)
def test_variant_pins_the_switches_its_objective_fixes(kind, setting, fixed):
    variant = VARIANTS[kind][0]
    assert getattr(variant(**{setting: fixed}), setting) is fixed
    with pytest.raises(ValidationError, match="literal_error"):
        LOSS_CONFIG.validate_python({"kind": kind, setting: not fixed})


@pytest.mark.parametrize(
    ("kind", "truncation"),
    [
        *((kind, "unknown") for kind in VARIANTS),
        ("reinforce", "adaptive-ppo-clip"),
    ],
)
def test_variant_rejects_a_truncation_type_it_does_not_support(kind, truncation):
    with pytest.raises(ValidationError, match="literal_error"):
        LOSS_CONFIG.validate_python(
            {"kind": kind, "truncated_importance_sampling_type": truncation}
        )


@pytest.mark.parametrize(
    ("switch", "replacement"),
    [
        ("use_cispo", "kind: cispo"),
        ("disable_ppo_ratio", "kind: reinforce"),
    ],
)
def test_deprecated_switch_still_loads_and_warns(switch, replacement):
    with pytest.warns(FutureWarning, match=f"loss_fn.{switch} is deprecated") as caught:
        parsed = LOSS_CONFIG.validate_python({switch: True})

    assert replacement in str(caught[0].message)
    assert isinstance(parsed, ClippedPGLossConfig)
    assert getattr(parsed, switch) is True


@pytest.mark.parametrize("loss_fn", [LEGACY_SETTINGS, {"use_cispo": False}])
def test_untagged_block_without_a_deprecated_switch_does_not_warn(loss_fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(LOSS_CONFIG.validate_python(loss_fn), ClippedPGLossConfig)


@pytest.mark.parametrize(
    "legacy_settings",
    [
        {"use_cispo": True},
        {"use_cispo": False},
        {"disable_ppo_ratio": True},
        {"sequence_level_importance_ratios": True, "token_level_loss": False},
    ],
)
def test_block_inheriting_kind_ppo_with_a_legacy_setting_loads_as_before(
    legacy_settings,
):
    # What a recipe written before the exemplars selected a kind looks like
    # after it is merged with a tagged exemplar.
    inherited = {"kind": "ppo", "ratio_clip_max": 0.28}
    with pytest.warns(FutureWarning, match="together with kind: ppo") as caught:
        parsed = LOSS_CONFIG.validate_python({**inherited, **legacy_settings})

    assert len(caught) == 1
    assert type(parsed) is ClippedPGLossConfig
    assert (
        parsed.model_dump()
        == ClippedPGLossConfig(ratio_clip_max=0.28, **legacy_settings).model_dump()
    )


@pytest.mark.parametrize(
    "master_config", [GRPOMasterConfig, PPOMasterConfig, SingleControllerMasterConfig]
)
@pytest.mark.parametrize(
    ("loss_fn", "expected"),
    [
        ({"ratio_clip_max": 0.3}, ClippedPGLossConfig),
        ({"kind": "legacy_pg", "ratio_clip_max": 0.3}, ClippedPGLossConfig),
        ({"kind": "ppo"}, ClippedSurrogateLossConfig),
        ({"kind": "cispo"}, CISPOLossConfig),
        ({"kind": "reinforce"}, ReinforceLossConfig),
    ],
)
def test_trainer_configs_select_the_loss_config_by_kind(
    master_config, loss_fn, expected
):
    # Validate only the loss block: the rest of a trainer config is unrelated.
    field = master_config.model_fields["loss_fn"]
    adapter = TypeAdapter(field.rebuild_annotation())
    assert type(adapter.validate_python(loss_fn)) is expected
