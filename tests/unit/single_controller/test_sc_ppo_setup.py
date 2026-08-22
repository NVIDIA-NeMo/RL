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

"""Unit tests for the PPO half of the SingleController setup path.

Covers what selects the PPO path (`ppo:` present), what that selection then
requires of the rest of the config, and the products setup has to hand the
controller: a GAE estimator, an MSE value loss, and a critic built on the
training cluster after the policy has stepped off it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import nemo_rl.algorithms.single_controller_utils.setup as sc_setup_mod
from nemo_rl.algorithms.advantage_estimator import GeneralizedAdvantageEstimator
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    SingleControllerPPOConfig,
    is_ppo_run,
    setup_single_controller,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    GAEConfig,
    validate_single_controller_config,
)

_NUM_PROMPTS_PER_STEP = 4
_NUM_GENERATIONS_PER_PROMPT = 2
_GLOBAL_BATCH_SIZE = _NUM_PROMPTS_PER_STEP * _NUM_GENERATIONS_PER_PROMPT


def _value_config(
    *,
    megatron_enabled: bool = False,
    train_global_batch_size: int = _GLOBAL_BATCH_SIZE,
) -> dict:
    return {
        "model_name": "Qwen/Qwen3-0.6B",
        "tokenizer": {"name": "Qwen/Qwen3-0.6B"},
        "train_global_batch_size": train_global_batch_size,
        "train_micro_batch_size": 1,
        "max_total_sequence_length": 32,
        "megatron_cfg": {"enabled": megatron_enabled},
        "dtensor_cfg": {"enabled": not megatron_enabled, "_v2": True},
    }


def _make_master_config(
    *,
    ppo: SingleControllerPPOConfig | None = None,
    value: dict | None = None,
    value_loss_fn: MseValueLossConfig | None = None,
    min_groups_for_streaming_train: int = _NUM_PROMPTS_PER_STEP,
    megatron_enabled: bool = False,
    max_num_steps: int = 100,
) -> MasterConfig:
    """A PPO-shaped SC config; model_construct skips the fields setup never reads."""
    return MasterConfig.model_construct(
        data_plane={"enabled": True, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": False,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo=GRPOConfig.model_construct(
            seed=42,
            max_num_steps=max_num_steps,
            max_num_epochs=1,
            num_prompts_per_step=_NUM_PROMPTS_PER_STEP,
            num_generations_per_prompt=_NUM_GENERATIONS_PER_PROMPT,
            max_rollout_turns=1,
            val_period=0,
            val_at_start=False,
            val_at_end=False,
            skip_reference_policy_logprobs_calculation=True,
        ),
        policy={
            "train_global_batch_size": _GLOBAL_BATCH_SIZE,
            "max_total_sequence_length": 32,
            "tokenizer": {"use_fastokens": False},
            "megatron_cfg": {"enabled": megatron_enabled},
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": True, "resources": {}},
            },
        },
        checkpointing={
            "enabled": False,
            "checkpoint_dir": "results/_sc_ppo_setup_test_ckpt",
            "metric_name": None,
            "higher_is_better": False,
            "keep_top_k": None,
            "save_period": 10,
            "save_optimizer": False,
        },
        loss_fn=ClippedPGLossConfig(reference_policy_kl_penalty=0.0),
        env={},
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=min_groups_for_streaming_train,
            max_buffered_rollouts=_NUM_PROMPTS_PER_STEP * 2,
        ),
        ppo=ppo,
        value=value,
        value_loss_fn=value_loss_fn,
    )


def _ppo_master_config(**kwargs) -> MasterConfig:
    kwargs.setdefault("ppo", SingleControllerPPOConfig())
    kwargs.setdefault(
        "value", _value_config(megatron_enabled=kwargs.get("megatron_enabled", False))
    )
    kwargs.setdefault("value_loss_fn", MseValueLossConfig())
    return _make_master_config(**kwargs)


class TestIsPPORun:
    def test_absent_ppo_block_is_grpo(self):
        assert is_ppo_run(_make_master_config()) is False

    def test_present_ppo_block_is_ppo(self):
        assert is_ppo_run(_ppo_master_config()) is True

    def test_missing_attribute_is_grpo(self):
        """model_construct can omit defaulted fields entirely."""
        assert is_ppo_run(MasterConfig.model_construct()) is False


class TestPPOValidation:
    def test_accepts_a_well_formed_ppo_config(self):
        validate_single_controller_config(_ppo_master_config())

    @pytest.mark.parametrize("missing", ["value", "value_loss_fn"])
    def test_rejects_ppo_without_its_critic_blocks(self, missing):
        mc = _ppo_master_config(**{missing: None})

        with pytest.raises(ValueError, match=f"needs `{missing}:`"):
            validate_single_controller_config(mc)

    def test_rejects_value_block_without_ppo_block(self):
        """A value model nothing builds is a silently-downgraded PPO run."""
        mc = _make_master_config(value=_value_config())

        with pytest.raises(ValueError, match="the `ppo:` block is absent"):
            validate_single_controller_config(mc)

    def test_rejects_multi_chunk_streaming(self):
        """The critic has no split API, so it would step once per chunk while
        the policy steps once per RL step."""
        mc = _ppo_master_config(min_groups_for_streaming_train=1)

        with pytest.raises(
            ValueError,
            match=r"min_groups_for_streaming_train \(1\) == "
            rf"grpo.num_prompts_per_step \({_NUM_PROMPTS_PER_STEP}\)",
        ):
            validate_single_controller_config(mc)

    def test_rejects_value_global_batch_size_mismatch(self):
        mc = _ppo_master_config(value=_value_config(train_global_batch_size=4))

        with pytest.raises(
            ValueError,
            match=r"must equal value.train_global_batch_size \(4\)",
        ):
            validate_single_controller_config(mc)

    def test_rejects_non_gae_estimator(self):
        with pytest.raises(ValueError):
            SingleControllerPPOConfig(adv_estimator={"name": "grpo"})


class TestAdvantageEstimatorSelection:
    def test_ppo_gets_gae_with_the_configured_lambdas(self):
        mc = _ppo_master_config(
            ppo=SingleControllerPPOConfig(
                adv_estimator=GAEConfig(gae_lambda=0.9, gae_gamma=0.99)
            )
        )

        estimator = sc_setup_mod._build_advantage_estimator(mc)

        assert isinstance(estimator, GeneralizedAdvantageEstimator)
        assert estimator.gae_lambda == 0.9
        assert estimator.gae_gamma == 0.99

    def test_grpo_delegates_to_the_group_relative_factory(self):
        mc = _make_master_config()
        sentinel = MagicMock(name="grpo_estimator")

        with patch.object(
            sc_setup_mod, "_create_advantage_estimator", return_value=sentinel
        ) as mock_factory:
            assert sc_setup_mod._build_advantage_estimator(mc) is sentinel

        mock_factory.assert_called_once_with(mc)


class TestMegatronTrainIters:
    def test_injects_into_both_policy_and_value(self):
        mc = _ppo_master_config(megatron_enabled=True, max_num_steps=7)

        sc_setup_mod._maybe_inject_megatron_train_iters(mc)

        assert mc.policy["megatron_cfg"]["train_iters"] == 7
        assert mc.value["megatron_cfg"]["train_iters"] == 7

    def test_skips_a_dtensor_critic(self):
        mc = _ppo_master_config(megatron_enabled=False, max_num_steps=7)

        sc_setup_mod._maybe_inject_megatron_train_iters(mc)

        assert "train_iters" not in mc.value["megatron_cfg"]


@pytest.fixture
def patched_ppo_factories():
    """Patch every external factory the PPO setup path calls."""
    fake_dataloader = MagicMock(name="dataloader")
    fake_dataloader.__len__ = MagicMock(return_value=4)
    fake_policy = MagicMock(name="policy")
    fake_value = MagicMock(name="value")

    with (
        patch.object(
            sc_setup_mod,
            "setup_response_data",
            return_value=(list(range(8)), None, {"math": MagicMock()}, {}),
        ),
        patch.object(sc_setup_mod, "StatefulDataLoader", return_value=fake_dataloader),
        patch.object(
            sc_setup_mod,
            "_build_clusters",
            return_value=(MagicMock(name="train"), MagicMock(name="inference")),
        ),
        patch.object(
            sc_setup_mod, "_build_generation", return_value=(MagicMock(), 0.0)
        ),
        patch.object(
            sc_setup_mod, "_build_trainer", return_value=(fake_policy, 1.0)
        ) as mock_trainer,
        patch.object(
            sc_setup_mod, "_build_value", return_value=(fake_value, 2.0)
        ) as mock_value,
        patch.object(sc_setup_mod, "build_data_plane_client", return_value=MagicMock()),
        patch.object(
            sc_setup_mod, "create_weight_synchronizer", return_value=MagicMock()
        ),
        patch.object(sc_setup_mod, "ClippedPGLossFn", return_value=MagicMock()),
        patch.object(
            sc_setup_mod, "_create_advantage_estimator", return_value=MagicMock()
        ),
        patch.object(sc_setup_mod, "_generation_max_seq_len", return_value=32),
    ):
        yield {
            "_build_trainer": mock_trainer,
            "_build_value": mock_value,
            "policy": fake_policy,
            "value": fake_value,
        }


class TestSetupBuildsTheCritic:
    def test_actor_args_carry_the_critic_and_its_loss(self, patched_ppo_factories):
        mc = _ppo_master_config()

        actor_args, timing = setup_single_controller(
            mc, tokenizer=MagicMock(pad_token_id=0)
        )

        assert actor_args.value_handle is patched_ppo_factories["value"]
        assert isinstance(actor_args.value_loss_fn, MseValueLossFn)
        assert timing.value_init_time_s == 2.0

    def test_policy_steps_off_the_gpu_while_the_critic_loads(
        self, patched_ppo_factories
    ):
        """Both worker groups sit on the training cluster; leaving the policy
        resident through the critic's init is what OOMs a tight fit."""
        policy = patched_ppo_factories["policy"]
        value = patched_ppo_factories["value"]

        setup_single_controller(
            _ppo_master_config(), tokenizer=MagicMock(pad_token_id=0)
        )

        policy.offload_to_cpu.assert_called_once_with()
        value.finish_training.assert_called_once_with()
        policy.prepare_for_training.assert_called_once_with()

    def test_grpo_run_builds_no_critic(self, patched_ppo_factories):
        actor_args, timing = setup_single_controller(
            _make_master_config(), tokenizer=MagicMock(pad_token_id=0)
        )

        patched_ppo_factories["_build_value"].assert_not_called()
        assert actor_args.value_handle is None
        assert actor_args.value_loss_fn is None
        assert timing.value_init_time_s is None
        patched_ppo_factories["policy"].offload_to_cpu.assert_not_called()


class TestTrainClusterSizing:
    """The critic is a second worker group on the training GPUs."""

    def test_non_colocated_train_cluster_admits_policy_and_critic(self):
        mc = _ppo_master_config()
        mc.policy["generation"]["colocated"] = {
            "enabled": False,
            "resources": {"gpus_per_node": 1, "num_nodes": 1},
        }
        mc.cluster = {"num_nodes": 1, "gpus_per_node": 2}

        with patch.object(sc_setup_mod, "RayVirtualCluster") as mock_cluster:
            sc_setup_mod._build_clusters(mc)

        train_call = mock_cluster.call_args_list[0]
        assert train_call.kwargs["name"] == "sc_train_cluster"
        assert train_call.kwargs["max_colocated_worker_groups"] == 2

    def test_grpo_non_colocated_train_cluster_stays_at_one(self):
        mc = _make_master_config()
        mc.policy["generation"]["colocated"] = {
            "enabled": False,
            "resources": {"gpus_per_node": 1, "num_nodes": 1},
        }
        mc.cluster = {"num_nodes": 1, "gpus_per_node": 2}

        with patch.object(sc_setup_mod, "RayVirtualCluster") as mock_cluster:
            sc_setup_mod._build_clusters(mc)

        assert mock_cluster.call_args_list[0].kwargs["max_colocated_worker_groups"] == 1

    def test_colocated_cluster_adds_generation_on_top(self):
        mc = _ppo_master_config()
        mc.cluster = {"num_nodes": 1, "gpus_per_node": 8}

        with patch.object(sc_setup_mod, "RayVirtualCluster") as mock_cluster:
            sc_setup_mod._build_clusters(mc)

        # policy + critic + vLLM
        assert mock_cluster.call_args.kwargs["max_colocated_worker_groups"] == 3
