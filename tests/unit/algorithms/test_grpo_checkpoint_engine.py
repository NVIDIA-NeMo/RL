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

"""Tests for GRPO checkpoint-engine refit routing."""

from typing import cast
from unittest.mock import MagicMock


def test_reset_encoder_cache_flag_rejected_on_unsupported_refit_transports():
    """The encoder-cache reset is honored only on collective/IPC and nccl_reshard."""
    import pytest

    from nemo_rl.models.generation.vllm.config import (
        VllmConfig,
        normalize_vllm_refit_config,
    )

    def _config(transport):
        return cast(
            VllmConfig,
            {
                "vllm_cfg": {"reset_encoder_cache_after_weight_update": True},
                "refit_transport": transport,
            },
        )

    # Supported transports pass through unchanged.
    assert normalize_vllm_refit_config(_config(None)) is None
    assert normalize_vllm_refit_config(_config("nccl_reshard")) is None

    # Transports whose refit path never resets the encoder cache fail loudly.
    for transport in ("nixl", "vllm_s3_sparse", "vllm_zmq_sparse"):
        with pytest.raises(ValueError, match="reset_encoder_cache_after_weight_update"):
            normalize_vllm_refit_config(_config(transport))


def test_refit_policy_generation_uses_attached_checkpoint_engine_synchronizer():
    from nemo_rl.algorithms import grpo as grpo_mod
    from nemo_rl.models.generation.vllm import VllmGeneration

    policy = MagicMock()
    kv_scales = {"layer_0": 1.0}

    generation = MagicMock(spec=VllmGeneration)
    generation.weight_synchronizer = MagicMock()
    generation.weight_synchronizer.sync_weights.return_value = {"transfer_s": 1.0}

    result = grpo_mod.refit_policy_generation(
        policy=policy,
        policy_generation=generation,
        colocated_inference=False,
        _refit_buffer_size_gb=2,
        timer=None,
        kv_scales=kv_scales,
    )

    generation.weight_synchronizer.sync_weights.assert_called_once_with(
        timer=None, kv_scales=kv_scales
    )
    policy.sync_params_before_refit.assert_called_once_with()
    assert result == {"transfer_s": 1.0}


def test_refit_policy_generation_sglang_uses_attached_synchronizer(monkeypatch):
    """SGLang always refits through its synchronizer, never the inline branches."""
    from nemo_rl.algorithms import grpo as grpo_mod
    from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

    policy = MagicMock()
    generation = MagicMock(spec=SGLangGeneration)
    generation.weight_synchronizer = MagicMock()
    generation.weight_synchronizer.sync_weights.return_value = None
    ray_get = MagicMock()
    monkeypatch.setattr(grpo_mod.ray, "get", ray_get)

    result = grpo_mod.refit_policy_generation(
        policy=policy,
        policy_generation=generation,
        colocated_inference=True,
        _refit_buffer_size_gb=2,
    )

    generation.weight_synchronizer.sync_weights.assert_called_once_with(
        timer=None, kv_scales=None
    )
    assert result == {}
    # The synchronizer owns every phase transition; grpo must not drive them.
    policy.offload_before_refit.assert_not_called()
    policy.offload_after_refit.assert_not_called()
    generation.prepare_for_generation.assert_not_called()
    ray_get.assert_not_called()
