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
"""Tests of the config rules of ``policy.sequence_packing.share_prefixes``."""

import re
from typing import Any

import pytest

from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.opd import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationFullConfig,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    TokenCaptureConfig,
)
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig
from nemo_rl.models.policy.shared_prefix import (
    bound_by_full_length,
    validate_shared_prefix_config,
)

OPTION = "policy.sequence_packing.share_prefixes"


def _policy_config() -> dict[str, Any]:
    """Return a policy config that meets every requirement of the option."""
    return {
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": 64,
            "logprob_mb_tokens": 64,
            "algorithm": "modified_first_fit_decreasing",
            "share_prefixes": True,
        },
        "dynamic_batching": {"enabled": False},
        "megatron_cfg": {"enabled": True, "pipeline_model_parallel_size": 1},
        "generation": {"top_k": None, "top_p": 1.0},
    }


def _policy_config_with(dotted_key: str, value: Any) -> dict[str, Any]:
    """Return the accepted policy config with one setting changed."""
    policy_config = _policy_config()
    *parents, key = dotted_key.split(".")
    block = policy_config
    for parent in parents:
        block = block[parent]
    block[key] = value
    return policy_config


def _master_config(
    policy_config: dict[str, Any],
    *,
    loss_fn: ClippedPGLossConfig | None = None,
    token_capture: TokenCaptureConfig | None = None,
    on_policy_distillation: OnPolicyDistillationConfig | None = None,
) -> MasterConfig:
    """Build the parts of a single-controller config that the check reads."""
    return MasterConfig.model_construct(
        policy=policy_config,
        loss_fn=ClippedPGLossConfig() if loss_fn is None else loss_fn,
        token_capture=(
            TokenCaptureConfig(enabled=True) if token_capture is None else token_capture
        ),
        on_policy_distillation=on_policy_distillation,
    )


def _assert_rejected(master_config: MasterConfig, requirement: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"{OPTION} requires {requirement}")):
        validate_shared_prefix_config(master_config)


@pytest.mark.parametrize("top_k", [None, 0, -1])
def test_accepts_a_config_that_meets_every_requirement(top_k: int | None) -> None:
    policy_config = _policy_config_with("generation.top_k", top_k)
    validate_shared_prefix_config(_master_config(policy_config))


def test_accepts_forest_training() -> None:
    master_config = _master_config(
        _policy_config(),
        token_capture=TokenCaptureConfig(enabled=True, forest_training=True),
    )
    validate_shared_prefix_config(master_config)


def test_accepts_on_policy_distillation_without_the_full_vocabulary_loss() -> None:
    master_config = _master_config(
        _policy_config(),
        on_policy_distillation=OnPolicyDistillationConfig(enabled=True),
    )
    validate_shared_prefix_config(master_config)


@pytest.mark.parametrize(
    "policy_config",
    [
        {},
        {"sequence_packing": {"enabled": True}},
        {"sequence_packing": {"enabled": True, "share_prefixes": False}},
    ],
    ids=["no sequence_packing block", "key absent", "key false"],
)
def test_reads_no_other_setting_when_the_option_is_off(
    policy_config: dict[str, Any],
) -> None:
    validate_shared_prefix_config(MasterConfig.model_construct(policy=policy_config))


_UNSUPPORTED_POLICY_SETTINGS: list[tuple[str, Any, str]] = [
    ("sequence_packing.enabled", False, "policy.sequence_packing.enabled=true."),
    (
        "sequence_packing.pair_grouping_key",
        "pair_index",
        "policy.sequence_packing.pair_grouping_key to be unset.",
    ),
    (
        "sequence_packing.max_sequences_per_bin",
        1,
        "policy.sequence_packing.max_sequences_per_bin to be unset.",
    ),
    ("dynamic_batching.enabled", True, "policy.dynamic_batching.enabled=false."),
    ("megatron_cfg.enabled", False, "policy.megatron_cfg.enabled=true."),
    (
        "megatron_cfg.pipeline_model_parallel_size",
        2,
        "policy.megatron_cfg.pipeline_model_parallel_size=1.",
    ),
    (
        "megatron_cfg.peft",
        {"enabled": True},
        "policy.megatron_cfg.peft.enabled=false.",
    ),
    (
        "megatron_cfg.use_fused_linear_logprobs",
        True,
        "policy.megatron_cfg.use_fused_linear_logprobs=false.",
    ),
    (
        "megatron_cfg.moe_hybridep_prepad_packed_inputs",
        True,
        "policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false.",
    ),
    (
        "megatron_cfg.zero_train_gen_mismatch",
        True,
        "policy.megatron_cfg.zero_train_gen_mismatch=false.",
    ),
    (
        "megatron_cfg.batch_invariant_mode",
        True,
        "policy.megatron_cfg.batch_invariant_mode=false.",
    ),
    (
        "generation.top_k",
        32,
        "policy.generation.top_k=null and policy.generation.top_p=1.0.",
    ),
    (
        "generation.top_p",
        0.95,
        "policy.generation.top_k=null and policy.generation.top_p=1.0.",
    ),
    ("router_replay", {"enabled": True}, "policy.router_replay.enabled=false."),
    ("is_vlm", True, "policy.is_vlm=false."),
    ("draft", Eagle3DraftConfig(enabled=True), "policy.draft.enabled=false."),
]


@pytest.mark.parametrize(
    ("dotted_key", "value", "requirement"),
    _UNSUPPORTED_POLICY_SETTINGS,
    ids=[dotted_key for dotted_key, _, _ in _UNSUPPORTED_POLICY_SETTINGS],
)
def test_rejects_an_unsupported_policy_setting(
    dotted_key: str, value: Any, requirement: str
) -> None:
    policy_config = _policy_config_with(dotted_key, value)
    _assert_rejected(_master_config(policy_config), requirement)


def test_rejects_a_policy_without_a_megatron_config() -> None:
    policy_config = _policy_config()
    del policy_config["megatron_cfg"]
    _assert_rejected(_master_config(policy_config), "policy.megatron_cfg.enabled=true.")


def test_rejects_token_capture_off() -> None:
    master_config = _master_config(
        _policy_config(), token_capture=TokenCaptureConfig(enabled=False)
    )
    _assert_rejected(master_config, "token_capture.enabled=true:")


@pytest.mark.parametrize(
    ("loss_fn", "requirement"),
    [
        (ClippedPGLossConfig(score_centering=True), "loss_fn.score_centering=false."),
        (
            ClippedPGLossConfig(seq_logprob_error_in_loss=True),
            "loss_fn.seq_logprob_error_in_loss=false.",
        ),
        (
            ClippedPGLossConfig(positive_example_nll_weight=0.1),
            "loss_fn.positive_example_nll_weight=0.",
        ),
    ],
    ids=["score_centering", "seq_logprob_error_in_loss", "positive_example_nll"],
)
def test_rejects_an_unsupported_loss_setting(
    loss_fn: ClippedPGLossConfig, requirement: str
) -> None:
    _assert_rejected(_master_config(_policy_config(), loss_fn=loss_fn), requirement)


def test_rejects_the_full_vocabulary_distillation_loss() -> None:
    master_config = _master_config(
        _policy_config(),
        on_policy_distillation=OnPolicyDistillationConfig(
            enabled=True, full=OnPolicyDistillationFullConfig(enabled=True)
        ),
    )
    _assert_rejected(master_config, "on_policy_distillation.full.enabled=false.")


@pytest.mark.parametrize(
    ("megatron_cfg", "budget_key", "expected"),
    [
        ({"enabled": True, "mtp_num_layers": 2}, "logprob_mb_tokens", False),
        ({"enabled": True}, "logprob_mb_tokens", False),
        ({"enabled": True, "mtp_num_layers": 2}, "train_mb_tokens", True),
        ({"enabled": True}, "train_mb_tokens", True),
        ({"enabled": True, "mtp_num_layers": 0}, "train_mb_tokens", False),
    ],
    ids=[
        "logprobs with MTP",
        "logprobs, MTP left to the checkpoint",
        "train with MTP",
        "train, MTP left to the checkpoint",
        "train without MTP",
    ],
)
def test_bound_by_full_length(
    megatron_cfg: dict[str, Any], budget_key: str, expected: bool
) -> None:
    policy_config = {"megatron_cfg": megatron_cfg}
    assert bound_by_full_length(policy_config, budget_key=budget_key) is expected
