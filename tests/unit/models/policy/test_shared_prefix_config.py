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
"""Shared-prefix configuration errors must precede worker initialization."""

import pytest
from pydantic import ValidationError

from nemo_rl.models.policy import (
    SharedPrefixTrainingConfig,
    validate_shared_prefix_data_parallel_size,
    validate_shared_prefix_training_config,
)


@pytest.fixture
def policy_config():
    return {
        "shared_prefix_training": {"mode": "train"},
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": False,
            "activation_checkpointing": False,
        },
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": 32,
            "logprob_mb_tokens": 32,
        },
        "generation": {"top_p": 1.0, "top_k": 0, "temperature": 0.7},
    }


@pytest.fixture
def megatron_rl():
    """Checks past the topology resolver need the megatron.rl planning modules."""
    pytest.importorskip("megatron.rl.shared_prefix_tensors")


@pytest.mark.parametrize("mode", ["logprobs", "train"])
@pytest.mark.parametrize("top_p,top_k", [(0.95, 0), (1.0, 32), (0.95, 32)])
def test_shared_filtering_rejected_before_backend_setup(
    policy_config, mode, top_p, top_k
):
    policy_config["shared_prefix_training"]["mode"] = mode
    policy_config["generation"].update(top_p=top_p, top_k=top_k)
    with pytest.raises(ValueError, match=r"top_p=1\.0.*top_k"):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize("mode", ["logprobs", "train"])
@pytest.mark.parametrize("top_k", [None, 0, -1])
def test_shared_unfiltered_sampling_and_temperature_allowed(
    policy_config, megatron_rl, mode, top_k
):
    policy_config["shared_prefix_training"]["mode"] = mode
    policy_config["generation"]["top_k"] = top_k
    assert validate_shared_prefix_training_config(policy_config).mode == mode


@pytest.mark.parametrize("mode", ["logprobs", "train"])
def test_shared_policy_without_generation_config_allowed(
    policy_config, megatron_rl, mode
):
    policy_config["shared_prefix_training"]["mode"] = mode
    del policy_config["generation"]
    assert validate_shared_prefix_training_config(policy_config).mode == mode


@pytest.mark.parametrize("mode", [None, "disabled", "dense"])
def test_nonshared_sampling_unchanged(policy_config, mode):
    if mode is None:
        del policy_config["shared_prefix_training"]
    else:
        policy_config["shared_prefix_training"]["mode"] = mode
    policy_config["generation"].update(top_p=0.95, top_k=32)
    assert validate_shared_prefix_training_config(policy_config).mode == (
        "disabled" if mode is None else mode
    )


@pytest.mark.parametrize("megatron", [False, True])
def test_inactive_mtp_bypass_rejected_for_every_backend(policy_config, megatron):
    policy_config["shared_prefix_training"].update(
        mode="disabled", bypass_evaluation_mtp=True
    )
    policy_config["megatron_cfg"]["enabled"] = megatron
    with pytest.raises(ValueError, match="bypass_evaluation_mtp requires"):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize("mode", ["dense", "logprobs", "train"])
def test_mtp_bypass_allowed_in_execution_or_dense_control(
    policy_config, megatron_rl, mode
):
    policy_config["shared_prefix_training"].update(
        mode=mode, bypass_evaluation_mtp=True
    )
    assert validate_shared_prefix_training_config(policy_config).bypass_evaluation_mtp


def test_disabled_mode_remains_backend_neutral_without_bypass():
    assert (
        validate_shared_prefix_training_config(
            {"shared_prefix_training": {"mode": "disabled"}}
        ).mode
        == "disabled"
    )


@pytest.mark.parametrize(
    "block",
    [
        {"mode": "observe"},
        {"mode": "train", "pack_group": True},
        {"mode": "train", "training_dense_bin": True},
        {"mode": "train", "align_dataparallel": True},
    ],
)
def test_unknown_modes_and_misspelled_flags_rejected(block):
    with pytest.raises(ValidationError):
        SharedPrefixTrainingConfig(**block)


# Every flag-combination raise in SharedPrefixTrainingConfig, keyed by the
# block and the error text. These need no other policy config.
_FLAG_COMBINATION_ERRORS = [
    (
        {"mode": "logprobs", "match_logprob_training_layout": True},
        "match_logprob_training_layout requires shared train mode",
    ),
    (
        {"mode": "train", "training_dense_bins": True, "pack_groups": True},
        "training_dense_bins requires",
    ),
    (
        {"mode": "train", "training_shard_work_weights": (-1, 1)},
        "training_shard_work_weights must be nonnegative",
    ),
    (
        {"mode": "logprobs", "training_shard_work_weights": (0, 1)},
        "training_shard_work_weights requires shared train mode",
    ),
    (
        {"mode": "train", "shard_work_weights": (0, 0)},
        "shard_work_weights must be nonnegative",
    ),
    (
        {"mode": "dense", "shard_work_weights": (1, 1)},
        "shard_work_weights requires shared logprobs or train mode",
    ),
    (
        {"mode": "disabled", "uniform_router_gating": True},
        "uniform_router_gating requires",
    ),
    (
        {"mode": "disabled", "bypass_evaluation_mtp": True},
        "bypass_evaluation_mtp requires",
    ),
    (
        {"mode": "disabled", "pack_groups": True, "align_data_parallel": True},
        "mode=disabled ignores align_data_parallel, pack_groups",
    ),
    (
        {"mode": "dense", "evaluation_packing": True},
        "mode=dense ignores evaluation_packing",
    ),
    ({"mode": "train", "repack_groups": True}, "repack_groups requires pack_groups"),
    (
        {"mode": "train", "pack_dense_fallbacks": True},
        "pack_dense_fallbacks requires pack_groups",
    ),
    (
        {"mode": "logprobs", "evaluation_packing": True},
        "evaluation_packing requires pack_groups",
    ),
    (
        {"mode": "train", "pack_groups": True, "merge_dense_fallbacks": True},
        "merge_dense_fallbacks requires pack_groups and pack_dense_fallbacks",
    ),
    (
        {
            "mode": "train",
            "pack_groups": True,
            "evaluation_packing": True,
            "match_logprob_training_layout": True,
        },
        "evaluation_packing has no effect with match_logprob_training_layout",
    ),
    (
        {
            "mode": "train",
            "pack_groups": True,
            "preserve_training_prefixes_during_alignment": True,
        },
        "Preserving training prefixes requires align_data_parallel",
    ),
    (
        {"mode": "train", "pack_groups": True, "align_data_parallel": True},
        "align_data_parallel requires pack_groups and repack_groups",
    ),
    (
        {
            "mode": "train",
            "pack_groups": True,
            "repack_groups": True,
            "pack_dense_fallbacks": True,
            "merge_dense_fallbacks": True,
            "align_data_parallel": True,
        },
        "Distributed packing does not support merge_dense_fallbacks",
    ),
    (
        {
            "mode": "logprobs",
            "pack_groups": True,
            "repack_groups": True,
            "evaluation_packing": True,
            "align_data_parallel": True,
        },
        "Distributed evaluation packing requires uniform MTP bypass",
    ),
]


@pytest.mark.parametrize("block,message", _FLAG_COMBINATION_ERRORS)
def test_flag_combination_rejected_at_config_load(block, message):
    with pytest.raises(ValidationError, match=message):
        SharedPrefixTrainingConfig(**block)
    with pytest.raises(ValueError, match=message):
        validate_shared_prefix_training_config({"shared_prefix_training": block})


@pytest.mark.parametrize(
    "block",
    [
        {"mode": "dense", "uniform_router_gating": True, "bypass_evaluation_mtp": True},
        {"mode": "logprobs", "pack_groups": True, "evaluation_packing": True},
        {
            "mode": "train",
            "pack_groups": True,
            "pack_dense_fallbacks": True,
            "merge_dense_fallbacks": True,
        },
        {
            "mode": "train",
            "pack_groups": True,
            "repack_groups": True,
            "pack_dense_fallbacks": True,
            "align_data_parallel": True,
            "preserve_training_prefixes_during_alignment": True,
            "training_dense_bins": True,
            "match_logprob_training_layout": True,
            "shard_work_weights": (1, 1),
            "training_shard_work_weights": (0, 1),
        },
    ],
)
def test_effective_flag_combinations_accepted(block):
    assert SharedPrefixTrainingConfig(**block).mode == block["mode"]


def _set(path, value):
    def mutate(config):
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(key):
    def mutate(config):
        del config[key]

    return mutate


def _use_tp2(config):
    config["megatron_cfg"].update(tensor_model_parallel_size=2, sequence_parallel=True)


def _chain(*mutations):
    def mutate(config):
        for mutation in mutations:
            mutation(config)

    return mutate


# Requirements an enabled mode places on the rest of the policy config, in the
# order validate_shared_prefix_training_config checks them.
_POLICY_REQUIREMENT_ERRORS = [
    (
        _set(("megatron_cfg", "enabled"), False),
        ValueError,
        r"requires policy\.megatron_cfg\.enabled=true",
    ),
    (_delete("megatron_cfg"), ValueError, r"requires policy\.megatron_cfg\.enabled"),
    (
        _set(("megatron_cfg", "peft"), {"enabled": True}),
        ValueError,
        r"peft\.enabled=false",
    ),
    (
        _set(("sequence_packing", "enabled"), False),
        ValueError,
        r"sequence_packing\.enabled=true",
    ),
    (_delete("sequence_packing"), ValueError, r"sequence_packing\.enabled=true"),
]

# These run after the megatron.rl topology resolver is imported.
_TOPOLOGY_REQUIREMENT_ERRORS = [
    (
        _set(("megatron_cfg", "sequence_parallel"), True),
        ValueError,
        "sequence_parallel that is true exactly when TP>1",
    ),
    (
        _set(("megatron_cfg", "tensor_model_parallel_size"), 0),
        ValueError,
        "positive integer",
    ),
    (
        _chain(
            _use_tp2,
            _set(("make_sequence_length_divisible_by",), 4),
            _set(("sequence_packing", "train_mb_tokens"), 30),
        ),
        ValueError,
        r"sequence_packing\.train_mb_tokens to be a positive multiple of resolved padding M=4",
    ),
    (
        _chain(
            _use_tp2,
            _set(("make_sequence_length_divisible_by",), 4),
            _set(("sequence_packing", "logprob_mb_tokens"), True),
        ),
        ValueError,
        r"sequence_packing\.logprob_mb_tokens",
    ),
    (
        _set(("megatron_cfg", "pipeline_model_parallel_size"), 2),
        ValueError,
        "pipeline_model_parallel_size=1",
    ),
    (
        _set(("megatron_cfg", "cuda_graph_impl"), "transformer_engine"),
        ValueError,
        "cuda_graph_impl='none'",
    ),
    (
        _set(("megatron_cfg", "fp8_cfg"), {"enabled": True}),
        ValueError,
        r"fp8_cfg\.enabled=false",
    ),
    (_set(("quant_cfg",), "nvfp4"), ValueError, r"policy\.quant_cfg=null"),
]


@pytest.mark.parametrize("mode", ["logprobs", "train"])
@pytest.mark.parametrize("mutate,error,message", _POLICY_REQUIREMENT_ERRORS)
def test_enabled_mode_policy_requirements(policy_config, mode, mutate, error, message):
    policy_config["shared_prefix_training"]["mode"] = mode
    mutate(policy_config)
    with pytest.raises(error, match=message):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize("mode", ["logprobs", "train"])
@pytest.mark.parametrize("mutate,error,message", _TOPOLOGY_REQUIREMENT_ERRORS)
def test_enabled_mode_topology_requirements(
    policy_config, megatron_rl, mode, mutate, error, message
):
    policy_config["shared_prefix_training"]["mode"] = mode
    mutate(policy_config)
    with pytest.raises(error, match=message):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize("enabled", [False, True])
def test_dense_control_requires_megatron(policy_config, enabled):
    policy_config["shared_prefix_training"]["mode"] = "dense"
    policy_config["megatron_cfg"]["enabled"] = enabled
    if enabled:
        assert validate_shared_prefix_training_config(policy_config).mode == "dense"
        return
    with pytest.raises(ValueError, match="dense comparison control requires"):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize(
    "block,data_parallel_size,rejected",
    [
        ({"mode": "train", "pack_groups": True}, 2, True),
        ({"mode": "logprobs", "pack_groups": True}, 4, True),
        ({"mode": "train", "pack_groups": True}, 1, False),
        (
            {
                "mode": "train",
                "pack_groups": True,
                "repack_groups": True,
                "align_data_parallel": True,
            },
            2,
            False,
        ),
        ({"mode": "train"}, 2, False),
        ({"mode": "disabled"}, 2, False),
    ],
)
def test_group_packing_across_data_parallel_requires_alignment(
    block, data_parallel_size, rejected
):
    config = SharedPrefixTrainingConfig(**block)
    if not rejected:
        validate_shared_prefix_data_parallel_size(
            config, data_parallel_size=data_parallel_size
        )
        return
    with pytest.raises(ValueError, match="requires .*align_data_parallel=true"):
        validate_shared_prefix_data_parallel_size(
            config, data_parallel_size=data_parallel_size
        )
