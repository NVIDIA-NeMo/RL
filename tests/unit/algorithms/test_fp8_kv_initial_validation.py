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

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.algorithms.grpo import (
    MasterConfig as GRPOMasterConfig,
)
from nemo_rl.algorithms.grpo import (
    _default_grpo_save_state,
    _validate_calibrated_fp8_kv_scales,
    grpo_train,
    refit_policy_generation,
)
from nemo_rl.algorithms.grpo_sync import grpo_train_sync
from nemo_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    MseValueLossConfig,
)
from nemo_rl.algorithms.ppo import (
    MasterConfig as PPOMasterConfig,
)
from nemo_rl.algorithms.ppo import (
    _default_ppo_save_state,
    ppo_train,
)
from nemo_rl.models.generation.vllm.utils import (
    can_sync_apply_fp8_kv,
    can_sync_apply_modelopt_real_quant_fp8_kv,
    is_modelopt_qarl_fp8_kv,
    is_modelopt_real_quant_fp8_kv,
)


def _kv_sync_policy_generation():
    policy_generation = MagicMock()
    policy_generation.requires_kv_scale_sync = True
    return policy_generation


def test_modelopt_real_quant_fp8_kv_detection():
    sync_cfg = {
        "backend": "vllm",
        "quant_cfg": "NVFP4_DEFAULT_CFG",
        "real_quant": True,
        "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"},
    }
    async_cfg = {
        **sync_cfg,
        "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3", "async_engine": True},
    }

    assert is_modelopt_real_quant_fp8_kv(sync_cfg)
    assert can_sync_apply_fp8_kv(sync_cfg)
    assert can_sync_apply_modelopt_real_quant_fp8_kv(sync_cfg)
    assert is_modelopt_qarl_fp8_kv({"quant_cfg": "NVFP4_DEFAULT_CFG"}, sync_cfg)
    assert not is_modelopt_qarl_fp8_kv({"quant_cfg": None}, sync_cfg)
    assert is_modelopt_real_quant_fp8_kv(async_cfg)
    assert not can_sync_apply_fp8_kv(async_cfg)
    assert not can_sync_apply_modelopt_real_quant_fp8_kv(async_cfg)
    assert can_sync_apply_fp8_kv(
        {
            "backend": "vllm",
            "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3", "async_engine": False},
        }
    )
    assert not is_modelopt_real_quant_fp8_kv(
        {
            "backend": "vllm",
            "quant_cfg": "NVFP4_DEFAULT_CFG",
            "real_quant": False,
            "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"},
        }
    )


def test_validate_calibrated_fp8_kv_scales_accepts_positive_non_default_scales():
    summary = _validate_calibrated_fp8_kv_scales(
        {
            "model.layers.0.self_attn.attn.q_scale": 0.001,
            "model.layers.0.self_attn.k_scale": 0.002,
            "model.layers.0.self_attn.v_scale": 0.003,
        },
        context="unit test",
    )

    assert summary["total"] == 3
    assert summary["counts"] == {"q": 1, "k": 1, "v": 1, "unknown": 0}
    assert summary["non_default"] == 3
    assert summary["zero_or_negative"] == 0
    assert summary["non_finite"] == 0


@pytest.mark.parametrize(
    "kv_scales",
    [
        {},
        {"model.layers.0.self_attn.k_scale": 0.0},
        {"model.layers.0.self_attn.k_scale": float("nan")},
        {
            "model.layers.0.self_attn.attn.q_scale": 1.0,
            "model.layers.0.self_attn.k_scale": 1.0,
            "model.layers.0.self_attn.v_scale": 1.0,
        },
        {
            "model.layers.0.self_attn.attn.q_scale": 0.001,
            "model.layers.0.self_attn.k_scale": 0.002,
        },
        {
            "model.layers.0.self_attn.attn.q_scale": 0.001,
            "model.layers.0.self_attn.k_scale": 0.002,
            "model.layers.0.self_attn.v_scale": 0.003,
            "model.layers.0.self_attn.extra_scale": 0.004,
        },
    ],
)
def test_validate_calibrated_fp8_kv_scales_rejects_invalid_or_incomplete_scales(
    kv_scales,
):
    with pytest.raises(RuntimeError):
        _validate_calibrated_fp8_kv_scales(kv_scales, context="unit test")


@patch("nemo_rl.algorithms.grpo.ray")
def test_refit_policy_generation_colocated_ipc_passes_and_applies_kv_scales(mock_ray):
    mock_ray.get.side_effect = [None, [True]]
    policy = MagicMock()
    policy.stream_weights_via_ipc_zmq.return_value = ["train-future"]
    policy_generation = MagicMock()
    policy_generation.cfg = {
        "backend": "vllm",
        "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"},
    }
    policy_generation.update_weights_via_ipc_zmq.return_value = ["infer-future"]
    kv_scales = {"model.layers.0.self_attn.k_scale": 2.0}

    refit_policy_generation(
        policy,
        policy_generation,
        colocated_inference=True,
        _refit_buffer_size_gb=1,
        kv_scales=kv_scales,
    )

    policy.stream_weights_via_ipc_zmq.assert_called_once()
    assert policy.stream_weights_via_ipc_zmq.call_args.kwargs["kv_scales"] == kv_scales
    policy_generation.prepare_for_generation.assert_any_call(tags=["weights"])
    policy_generation.prepare_for_generation.assert_any_call(tags=["kv_cache"])
    policy_generation.apply_kv_cache_scales.assert_called_once_with(kv_scales)


@patch("nemo_rl.algorithms.grpo.ray")
def test_refit_policy_generation_colocated_ipc_skips_sync_kv_apply_for_async_vllm(
    mock_ray,
):
    mock_ray.get.side_effect = [None, [True]]
    policy = MagicMock()
    policy.stream_weights_via_ipc_zmq.return_value = ["train-future"]
    policy_generation = MagicMock()
    policy_generation.cfg = {
        "backend": "vllm",
        "quant_cfg": "NVFP4_DEFAULT_CFG",
        "real_quant": True,
        "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3", "async_engine": True},
    }
    policy_generation.update_weights_via_ipc_zmq.return_value = ["infer-future"]
    kv_scales = {"model.layers.0.self_attn.k_scale": 2.0}

    refit_policy_generation(
        policy,
        policy_generation,
        colocated_inference=True,
        _refit_buffer_size_gb=1,
        kv_scales=kv_scales,
    )

    assert policy.stream_weights_via_ipc_zmq.call_args.kwargs["kv_scales"] == kv_scales
    policy_generation.prepare_for_generation.assert_any_call(tags=["kv_cache"])
    policy_generation.apply_kv_cache_scales.assert_not_called()


def test_grpo_train_rejects_initial_validation_with_kv_scale_sync():
    master_config = GRPOMasterConfig.model_construct(
        **{
            "policy": {
                "generation": {"colocated": {"enabled": True}},
                "make_sequence_length_divisible_by": 1,
            },
            "loss_fn": ClippedPGLossConfig(),
            "env": {},
            "data": {"use_multiple_dataloader": False},
            "grpo": {
                "max_num_steps": 0,
                "max_num_epochs": 1,
                "val_at_start": True,
                "val_at_end": False,
                "val_period": 0,
                "adv_estimator": {
                    "name": "grpo",
                    "normalize_rewards": False,
                    "use_leave_one_out_baseline": False,
                    "minus_baseline": True,
                },
            },
            "logger": {"num_val_samples_to_print": 0},
            "cluster": {},
            "checkpointing": {"checkpoint_must_save_by": None},
        }
    )

    with pytest.raises(ValueError, match="val_at_start=True"):
        grpo_train(
            policy=MagicMock(),
            policy_generation=_kv_sync_policy_generation(),
            wrapped_dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            tokenizer=MagicMock(),
            loss_fn=MagicMock(),
            task_to_env={},
            val_task_to_env={},
            logger=MagicMock(),
            checkpointer=MagicMock(),
            grpo_save_state=_default_grpo_save_state(),
            master_config=master_config,
        )


def test_grpo_sync_train_rejects_initial_validation_with_kv_scale_sync():
    master_config = GRPOMasterConfig.model_construct(
        **{
            "policy": {
                "generation": {"colocated": {"enabled": True}},
                "make_sequence_length_divisible_by": 1,
            },
            "loss_fn": ClippedPGLossConfig(),
            "env": {},
            "data": {"use_multiple_dataloader": False},
            "grpo": {
                "max_num_steps": 0,
                "max_num_epochs": 1,
                "val_at_start": True,
                "val_at_end": False,
                "val_period": 0,
            },
            "logger": {"num_val_samples_to_print": 0},
            "cluster": {},
            "checkpointing": {"checkpoint_must_save_by": None},
        }
    )

    with pytest.raises(ValueError, match="val_at_start=True"):
        grpo_train_sync(
            policy=MagicMock(),
            policy_generation=_kv_sync_policy_generation(),
            wrapped_dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            tokenizer=MagicMock(),
            loss_fn=MagicMock(),
            task_to_env={},
            val_task_to_env={},
            logger=MagicMock(),
            checkpointer=MagicMock(),
            grpo_save_state=_default_grpo_save_state(),
            master_config=master_config,
        )


def test_ppo_train_rejects_initial_validation_with_kv_scale_sync():
    master_config = PPOMasterConfig.model_construct(
        **{
            "policy": {
                "generation": {"colocated": {"enabled": True}},
                "make_sequence_length_divisible_by": 1,
            },
            "value": {},
            "loss_fn": ClippedPGLossConfig(),
            "value_loss_fn": MseValueLossConfig(),
            "env": {},
            "data": {"use_multiple_dataloader": False},
            "ppo": {
                "max_num_steps": 0,
                "max_num_epochs": 1,
                "ppo_epochs": 1,
                "policy_training_start_step": 0,
                "val_at_start": True,
                "val_at_end": False,
                "val_period": 0,
                "adv_estimator": {
                    "name": "raw_reward",
                    "normalize_advantages": False,
                },
            },
            "logger": {"num_val_samples_to_print": 0},
            "cluster": {},
            "checkpointing": {"checkpoint_must_save_by": None},
        }
    )

    with pytest.raises(ValueError, match="val_at_start=True"):
        ppo_train(
            policy=MagicMock(),
            policy_generation=_kv_sync_policy_generation(),
            value_model=MagicMock(),
            dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            tokenizer=MagicMock(),
            loss_fn=MagicMock(),
            value_loss_fn=MagicMock(),
            task_to_env={},
            val_task_to_env={},
            logger=MagicMock(),
            checkpointer=MagicMock(),
            ppo_save_state=_default_ppo_save_state(),
            master_config=master_config,
        )
