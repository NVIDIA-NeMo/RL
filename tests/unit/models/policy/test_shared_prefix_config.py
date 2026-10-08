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

from nemo_rl.models.policy import validate_shared_prefix_training_config


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
def test_shared_unfiltered_sampling_and_temperature_allowed(policy_config, mode, top_k):
    policy_config["shared_prefix_training"]["mode"] = mode
    policy_config["generation"]["top_k"] = top_k
    assert validate_shared_prefix_training_config(policy_config).mode == mode


@pytest.mark.parametrize("mode", ["logprobs", "train"])
def test_shared_policy_without_generation_config_allowed(policy_config, mode):
    policy_config["shared_prefix_training"]["mode"] = mode
    del policy_config["generation"]
    assert validate_shared_prefix_training_config(policy_config).mode == mode


@pytest.mark.parametrize("mode", [None, "disabled", "observe", "dense"])
def test_nonshared_sampling_unchanged(policy_config, mode):
    if mode is None:
        del policy_config["shared_prefix_training"]
    else:
        policy_config["shared_prefix_training"]["mode"] = mode
    policy_config["generation"].update(top_p=0.95, top_k=32)
    assert validate_shared_prefix_training_config(policy_config).mode == (
        "disabled" if mode is None else mode
    )


@pytest.mark.parametrize("mode", ["disabled", "observe"])
@pytest.mark.parametrize("megatron", [False, True])
def test_inactive_mtp_bypass_rejected_for_every_backend(policy_config, mode, megatron):
    policy_config["shared_prefix_training"].update(
        mode=mode, bypass_evaluation_mtp=True
    )
    policy_config["megatron_cfg"]["enabled"] = megatron
    with pytest.raises(ValueError, match="bypass_evaluation_mtp requires"):
        validate_shared_prefix_training_config(policy_config)


@pytest.mark.parametrize("mode", ["dense", "logprobs", "train"])
def test_mtp_bypass_allowed_in_execution_or_dense_control(policy_config, mode):
    policy_config["shared_prefix_training"].update(
        mode=mode, bypass_evaluation_mtp=True
    )
    assert validate_shared_prefix_training_config(policy_config).bypass_evaluation_mtp


@pytest.mark.parametrize("mode", ["disabled", "observe"])
def test_inactive_modes_remain_backend_neutral_without_bypass(mode):
    assert (
        validate_shared_prefix_training_config(
            {"shared_prefix_training": {"mode": mode}}
        ).mode
        == mode
    )
