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
"""Startup-validation matrix for router replay, against the real exemplar YAMLs.

These tests intentionally load the shipped configs (with inheritance) rather
than hand-built dicts, so key-path assumptions in the validator are exercised
against the true config shape.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.megatron.router_replay import validate_router_replay_startup
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()

REPO_ROOT = Path(__file__).parents[4]
QWEN30B_MEGATRON_YAML = REPO_ROOT / "examples/configs/grpo_math_qwen30ba3b_megatron.yaml"

MOE_HF_CONFIG = SimpleNamespace(num_experts=128)
DENSE_HF_CONFIG = SimpleNamespace()


@pytest.fixture()
def moe_model_config(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: MOE_HF_CONFIG,
    )


@pytest.fixture()
def qwen30b_policy():
    cfg = load_config(QWEN30B_MEGATRON_YAML)
    policy = OmegaConf.to_container(cfg, resolve=True)["policy"]
    policy["router_replay"] = {"enabled": True}
    return policy


def test_exemplar_megatron_config_with_r3_passes(qwen30b_policy, moe_model_config):
    validate_router_replay_startup(qwen30b_policy)


def test_r3_disabled_skips_all_checks(qwen30b_policy):
    qwen30b_policy["router_replay"] = {"enabled": False}
    # DTensor on + megatron off would raise if checks ran.
    qwen30b_policy["dtensor_cfg"]["enabled"] = True
    qwen30b_policy["megatron_cfg"]["enabled"] = False
    validate_router_replay_startup(qwen30b_policy)


def test_r3_with_dtensor_backend_raises(qwen30b_policy, moe_model_config):
    qwen30b_policy["dtensor_cfg"]["enabled"] = True
    qwen30b_policy["megatron_cfg"]["enabled"] = False
    with pytest.raises(ValueError, match="Megatron policy backend"):
        validate_router_replay_startup(qwen30b_policy)


def test_r3_with_mtp_raises(qwen30b_policy, moe_model_config):
    qwen30b_policy["megatron_cfg"]["mtp_num_layers"] = 2
    with pytest.raises(ValueError, match="MTP"):
        validate_router_replay_startup(qwen30b_policy)


def test_r3_with_gym_over_sync_dataplane_raises(qwen30b_policy, moe_model_config):
    with pytest.raises(NotImplementedError, match="sync TQ dataplane"):
        validate_router_replay_startup(
            qwen30b_policy,
            data_plane_enabled=True,
            using_nemo_gym=True,
            async_rollouts=False,
        )


def test_r3_with_gym_without_dataplane_passes(qwen30b_policy, moe_model_config):
    validate_router_replay_startup(
        qwen30b_policy,
        data_plane_enabled=False,
        using_nemo_gym=True,
        async_rollouts=False,
    )


def test_r3_with_sync_dataplane_without_gym_passes(qwen30b_policy, moe_model_config):
    validate_router_replay_startup(
        qwen30b_policy,
        data_plane_enabled=True,
        using_nemo_gym=False,
        async_rollouts=False,
    )


def test_r3_on_dense_model_raises(qwen30b_policy, monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: DENSE_HF_CONFIG,
    )
    with pytest.raises(ValueError, match="MoE model"):
        validate_router_replay_startup(qwen30b_policy)


def test_r3_with_unloadable_model_config_defers_to_model_setup(
    qwen30b_policy, monkeypatch
):
    def _raise(*args, **kwargs):
        raise OSError("no such model")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", _raise)
    validate_router_replay_startup(qwen30b_policy)
