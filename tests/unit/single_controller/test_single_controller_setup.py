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

"""Unit tests for setup_single_controller (factories monkey-patched)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import nemo_rl.algorithms.single_controller_utils.setup as sc_setup_mod
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    RolloutCheckpointingConfig,
    SingleControllerBundle,
    setup_single_controller,
)


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = True,
    backend: str = "vllm",
    megatron_enabled: bool = False,
    env: dict | None = None,
    max_num_steps: int = 100,
    max_num_epochs: int = 1,
    num_prompts_per_step: int = 4,
    val_period: int = 0,
    val_at_start: bool = False,
    val_at_end: bool = False,
    val_batch_size: int | None = 2,
    max_val_samples: int | None = 3,
    max_inflight_prompts: int = 4,
    checkpoint_dir: str = "/nonexistent/nemo-rl-sc-test-checkpoints",
) -> MasterConfig:
    """Build a partially-populated MasterConfig for unit tests.

    Cross-cutting components (cluster/logger/...) are required by pydantic for
    normal load but unused here — model_construct skips validation, and we hand-fill
    only the dict-shaped fields setup reads (including checkpointing; the default
    checkpoint_dir never exists, so setup takes the fresh-start path).
    """
    return MasterConfig.model_construct(
        data_plane={"enabled": dp_enabled, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": use_multiple_dataloader,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo={
            "max_num_steps": max_num_steps,
            "max_num_epochs": max_num_epochs,
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": 2,
            "max_rollout_turns": 1,
            "val_period": val_period,
            "val_at_start": val_at_start,
            "val_at_end": val_at_end,
            "val_batch_size": val_batch_size,
            "max_val_samples": max_val_samples,
            "seed": 42,
        },
        checkpointing={
            "enabled": False,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": True,
            "keep_top_k": None,
            "save_period": 10,
            "save_optimizer": True,
            "checkpoint_must_save_by": None,
        },
        policy={
            "max_total_sequence_length": 32,
            "megatron_cfg": {"enabled": megatron_enabled},
            "generation": {
                "backend": backend,
                "colocated": {"enabled": colocated, "resources": {}},
                "model_name": "test-model",
                "stop_strings": None,
                "stop_token_ids": None,
                "top_k": None,
                "vllm_cfg": {
                    "async_engine": True,
                    "expose_http_server": True,
                },
            },
        },
        loss_fn=ClippedPGLossConfig(),
        env=env if env is not None else {},
        async_rl=AsyncRLConfig(max_inflight_prompts=max_inflight_prompts),
        rollout_checkpointing=RolloutCheckpointingConfig(),
    )


@pytest.fixture
def patched_factories():
    """Patch every external factory setup calls.

    Returns a dict of mocks keyed by name so individual tests can assert on call args
    without re-importing the patch handles.
    """
    fake_dataset = list(range(8))
    fake_val_dataset = list(range(3))
    fake_dataloader = MagicMock(name="dataloader")
    fake_val_dataloader = MagicMock(name="val_dataloader")
    # len(dataloader) used by the Megatron train_iters injection.
    fake_dataloader.__len__ = MagicMock(return_value=4)
    fake_env_handles = {"math": MagicMock(name="math_env")}
    fake_val_env_handles = {"math": MagicMock(name="val_math_env")}

    def _make_dataloader(dataset, *args, **kwargs):
        del args, kwargs
        if dataset is fake_val_dataset:
            return fake_val_dataloader
        return fake_dataloader

    with (
        patch.object(
            sc_setup_mod,
            "setup_response_data",
            return_value=(
                fake_dataset,
                fake_val_dataset,
                fake_env_handles,
                fake_val_env_handles,
            ),
        ) as mock_setup_response,
        patch.object(
            sc_setup_mod,
            "StatefulDataLoader",
            side_effect=_make_dataloader,
        ) as mock_dataloader,
        patch.object(
            sc_setup_mod,
            "_build_clusters",
            return_value=(
                MagicMock(name="train_cluster"),
                MagicMock(name="inference_cluster"),
            ),
        ) as mock_clusters,
        patch.object(
            sc_setup_mod, "_build_generation", return_value=MagicMock(name="gen")
        ) as mock_gen,
        patch.object(
            sc_setup_mod, "_build_trainer", return_value=MagicMock(name="policy")
        ) as mock_trainer,
        patch.object(
            sc_setup_mod,
            "build_data_plane_client",
            return_value=MagicMock(name="dp_client"),
        ) as mock_dp_client,
        patch.object(
            sc_setup_mod,
            "create_weight_synchronizer",
            return_value=MagicMock(name="weight_sync"),
        ) as mock_weight_sync,
        patch.object(
            sc_setup_mod,
            "_create_advantage_estimator",
            return_value=MagicMock(name="adv"),
        ) as mock_adv,
        patch.object(
            sc_setup_mod, "ClippedPGLossFn", return_value=MagicMock(name="loss_fn")
        ) as mock_loss,
        patch.object(
            sc_setup_mod,
            "spinup_nemo_gym_actor",
            return_value=MagicMock(name="nemo_gym_actor"),
        ) as mock_nemo_gym_actor,
        patch.object(
            sc_setup_mod,
            "_generation_max_seq_len",
            return_value=32,
        ),
    ):
        yield {
            "setup_response_data": mock_setup_response,
            "StatefulDataLoader": mock_dataloader,
            "_build_clusters": mock_clusters,
            "_build_generation": mock_gen,
            "_build_trainer": mock_trainer,
            "build_data_plane_client": mock_dp_client,
            "create_weight_synchronizer": mock_weight_sync,
            "_create_advantage_estimator": mock_adv,
            "ClippedPGLossFn": mock_loss,
            "spinup_nemo_gym_actor": mock_nemo_gym_actor,
            "dataloader": fake_dataloader,
            "val_dataloader": fake_val_dataloader,
            "val_dataset": fake_val_dataset,
            "env_handles": fake_env_handles,
            "val_env_handles": fake_val_env_handles,
        }


class TestSetup:
    """setup arg validation + bundle assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_single_controller(mc, MagicMock())

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        with pytest.raises(NotImplementedError, match="use_multiple_dataloader"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_rollout_checkpointing_requires_root_dir(self):
        with pytest.raises(ValueError, match="root_dir is required"):
            RolloutCheckpointingConfig(enabled=True)

    def test_rollout_checkpointing_rejects_non_gym_path(self, tmp_path):
        config = RolloutCheckpointingConfig(enabled=True, root_dir=tmp_path)

        with pytest.raises(NotImplementedError, match="only NeMo-Gym"):
            sc_setup_mod._validate_rollout_checkpointing_setup(
                config,
                use_nemo_gym=False,
            )

    def test_builds_single_rollout_checkpoint_runtime(self, tmp_path):
        config = RolloutCheckpointingConfig(
            enabled=True,
            root_dir=tmp_path,
            writer_concurrency=3,
        )
        assert config.model_dump()["root_dir"] == str(tmp_path)
        tokenizer = MagicMock(
            name_or_path="tokenizer-name",
            chat_template="{{ messages }}",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            unk_token_id=3,
        )
        tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        generation_config = {
            "backend": "vllm",
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": None,
            "stop_token_ids": [2],
            "stop_strings": None,
            "vllm_cfg": {"max_model_len": 256},
        }
        writer = MagicMock(name="rollout_checkpoint_writer")

        with patch.object(sc_setup_mod, "RolloutCheckpointWriter") as writer_cls:
            writer_cls.options.return_value.remote.return_value = writer
            runtime = sc_setup_mod._build_rollout_checkpoint_runtime(
                config,
                tokenizer=tokenizer,
                generation_config=generation_config,
            )

        assert runtime is not None
        assert runtime.writer is writer
        assert len(runtime.run_id) == 32
        assert len(runtime.sampling_fingerprint) == 64
        assert len(runtime.tokenizer_fingerprint) == 64
        writer_cls.options.assert_called_once_with(max_concurrency=3)
        writer_cls.options.return_value.remote.assert_called_once_with(str(tmp_path))

    def test_returns_bundle(self, patched_factories):
        mc = _make_master_config(colocated=True)
        tokenizer = MagicMock(pad_token_id=0)

        bundle = setup_single_controller(mc, tokenizer)

        assert isinstance(bundle, SingleControllerBundle)
        assert bundle.gen_handle is patched_factories["_build_generation"].return_value
        assert bundle.trainer_handle is patched_factories["_build_trainer"].return_value
        assert bundle.env_handles is patched_factories["env_handles"]
        assert (
            bundle.dp_client
            is patched_factories["build_data_plane_client"].return_value
        )
        assert bundle.dataloader is patched_factories["dataloader"]
        assert bundle.val_dataloader is None
        assert bundle.weight_synchronizer is (
            patched_factories["create_weight_synchronizer"].return_value
        )
        assert bundle.advantage_estimator is (
            patched_factories["_create_advantage_estimator"].return_value
        )
        assert bundle.loss_fn is patched_factories["ClippedPGLossFn"].return_value
        # tq_buffer + rollout_manager are constructed inline (not mocked).
        assert bundle.tq_buffer is not None
        assert bundle.rollout_manager is not None
        # rollout_manager binds the same tq_buffer for the writer + sampler.
        assert bundle.rollout_manager._tq_buffer is bundle.tq_buffer
        assert bundle.rollout_manager._env_handles is bundle.env_handles
        assert bundle.rollout_manager._val_env_handles is bundle.val_env_handles
        # tq_buffer wires the dp_client + default partition.
        assert bundle.tq_buffer._dp_client is bundle.dp_client
        assert bundle.partition_id == "rollout_data"
        assert bundle.tq_buffer._partition_id == "rollout_data"
        assert bundle.rollout_checkpoint is None
        assert bundle.rollout_manager._checkpoint_io_policy is None

    def test_enabled_validation_builds_native_resources(self, patched_factories):
        mc = _make_master_config(val_at_start=True)
        tokenizer = MagicMock(pad_token_id=0)

        bundle = setup_single_controller(mc, tokenizer)

        assert bundle.val_dataloader is patched_factories["val_dataloader"]
        assert bundle.val_env_handles is patched_factories["val_env_handles"]
        val_loader_call = patched_factories["StatefulDataLoader"].call_args_list[1]
        assert val_loader_call.args[0] is patched_factories["val_dataset"]
        assert val_loader_call.kwargs == {
            "batch_size": 2,
            "shuffle": False,
            "collate_fn": sc_setup_mod.rl_collate_fn,
            "drop_last": False,
            "num_workers": 0,
        }

        rollout_manager = bundle.rollout_manager
        assert rollout_manager._env_handles is bundle.env_handles
        assert rollout_manager._val_env_handles is bundle.val_env_handles
        assert rollout_manager._tq_buffer is bundle.tq_buffer
        assert (
            rollout_manager._impl._policy_generation
            is patched_factories["_build_generation"].return_value
        )

    def test_enabled_validation_requires_nonempty_dataset(self, patched_factories):
        mc = _make_master_config(val_period=1)
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            None,
            patched_factories["env_handles"],
            patched_factories["val_env_handles"],
        )

        with pytest.raises(ValueError, match="nonempty validation dataset"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    @pytest.mark.parametrize(
        ("config_key", "invalid_value"),
        [
            ("val_batch_size", None),
            ("val_batch_size", 0),
            ("max_val_samples", None),
            ("max_val_samples", 0),
        ],
    )
    def test_enabled_validation_requires_positive_native_limits(
        self,
        patched_factories,
        config_key,
        invalid_value,
    ):
        kwargs = {"val_at_end": True, config_key: invalid_value}
        mc = _make_master_config(**kwargs)

        with pytest.raises(ValueError, match=f"grpo.{config_key}"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    @pytest.mark.parametrize(
        ("val_batch_size", "max_inflight_prompts", "expected_batch_size"),
        [(2, 4, 2), (None, 2, 2)],
    )
    def test_enabled_nemo_gym_validation_reuses_actor(
        self,
        patched_factories,
        val_batch_size,
        max_inflight_prompts,
        expected_batch_size,
    ):
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            patched_factories["val_dataset"],
        )
        mc = _make_master_config(
            val_at_end=True,
            val_batch_size=val_batch_size,
            max_val_samples=None,
            max_inflight_prompts=max_inflight_prompts,
            env={"should_use_nemo_gym": True, "nemo_gym": {}},
        )

        bundle = setup_single_controller(mc, MagicMock(pad_token_id=0))

        nemo_gym_actor = patched_factories["spinup_nemo_gym_actor"].return_value
        assert bundle.env_handles["nemo_gym"] is nemo_gym_actor
        assert bundle.val_env_handles["nemo_gym"] is nemo_gym_actor
        assert bundle.rollout_manager._env_handles["nemo_gym"] is nemo_gym_actor
        assert bundle.rollout_manager._val_env_handles["nemo_gym"] is nemo_gym_actor
        assert bundle.rollout_manager._tq_buffer is bundle.tq_buffer
        val_loader_call = patched_factories["StatefulDataLoader"].call_args_list[1]
        assert val_loader_call.kwargs["batch_size"] == expected_batch_size
        patched_factories["spinup_nemo_gym_actor"].assert_called_once()

    @pytest.mark.parametrize(
        ("config_key", "invalid_value"),
        [("max_val_samples", 3), ("val_batch_size", 0)],
    )
    def test_enabled_nemo_gym_validation_rejects_invalid_limits(
        self,
        patched_factories,
        config_key,
        invalid_value,
    ):
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            patched_factories["val_dataset"],
        )
        kwargs = {
            "val_at_end": True,
            "val_batch_size": 2,
            "max_val_samples": None,
            config_key: invalid_value,
            "env": {"should_use_nemo_gym": True, "nemo_gym": {}},
        }

        with pytest.raises(ValueError, match=f"grpo.{config_key}"):
            setup_single_controller(
                _make_master_config(**kwargs), MagicMock(pad_token_id=0)
            )

    def test_nemo_gym_validation_rejects_effort_before_external_setup(
        self, patched_factories
    ):
        mc = _make_master_config(
            val_at_end=True,
            val_batch_size=None,
            max_val_samples=None,
            env={
                "should_use_nemo_gym": True,
                "nemo_gym": {"effort_levels": {"low_weight": 0.0}},
            },
        )

        with pytest.raises(NotImplementedError, match="effort_levels"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        for factory_name in (
            "setup_response_data",
            "_build_clusters",
            "_build_generation",
            "_build_trainer",
            "spinup_nemo_gym_actor",
        ):
            patched_factories[factory_name].assert_not_called()

    def test_env_handles_sourced_from_setup_response_data(self, patched_factories):
        """setup_response_data receives master_config.env and supplies env handles."""
        math_env_cfg = {"some": "value"}
        mc = _make_master_config(env={"math": math_env_cfg})

        bundle = setup_single_controller(mc, MagicMock(pad_token_id=0))

        _, call_kwargs = patched_factories["setup_response_data"].call_args
        assert call_kwargs["env_configs"] == {"math": math_env_cfg}
        assert bundle.env_handles is patched_factories["env_handles"]

    def test_weight_sync_factory_args(self, patched_factories):
        """create_weight_synchronizer receives policy / generation / topology."""
        mc = _make_master_config(colocated=False, backend="vllm")
        tokenizer = MagicMock(pad_token_id=0)

        setup_single_controller(mc, tokenizer)

        _, factory_kwargs = patched_factories["create_weight_synchronizer"].call_args
        assert (
            factory_kwargs["policy"] is patched_factories["_build_trainer"].return_value
        )
        assert (
            factory_kwargs["generation"]
            is patched_factories["_build_generation"].return_value
        )
        assert factory_kwargs["generation_backend"] == "vllm"
        assert factory_kwargs["colocated"] is False

    def test_custom_partition_id(self, patched_factories):
        mc = _make_master_config()
        tokenizer = MagicMock(pad_token_id=7)

        bundle = setup_single_controller(mc, tokenizer, partition_id="custom_partition")

        assert bundle.partition_id == "custom_partition"
        assert bundle.tq_buffer._partition_id == "custom_partition"
        assert bundle.tq_buffer._pad_value_dict == {
            "token_ids": 7,
            "input_ids": 7,
        }

    def test_max_num_steps_capped_by_self(self, patched_factories):
        """grpo.max_num_steps stays put when smaller than max_num_epochs * len(dl)."""
        mc = _make_master_config(
            megatron_enabled=False,
            max_num_steps=2,
            max_num_epochs=1,
        )
        # patched dataloader has len() == 4, so the min picks max_num_steps.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.grpo["max_num_steps"] == 2

    def test_max_num_steps_capped_by_dataloader_epochs(self, patched_factories):
        """grpo.max_num_steps drops to max_num_epochs * len(dataloader) when smaller."""
        mc = _make_master_config(
            megatron_enabled=False,
            max_num_steps=1000,
            max_num_epochs=2,
        )
        # patched dataloader has len() == 4 → 2 * 4 = 8 < 1000.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.grpo["max_num_steps"] == 8

    def test_megatron_train_iters_capped_by_max_num_steps(self, patched_factories):
        """train_iters = min(max_num_steps, max_num_epochs * len(dataloader))."""
        mc = _make_master_config(
            megatron_enabled=True,
            max_num_steps=2,
            max_num_epochs=1,
        )
        # patched dataloader has len() == 4, so the min picks max_num_steps.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["megatron_cfg"]["train_iters"] == 2

    def test_megatron_train_iters_capped_by_dataloader_epochs(self, patched_factories):
        """train_iters drops to max_num_epochs * len(dataloader) when smaller."""
        mc = _make_master_config(
            megatron_enabled=True,
            max_num_steps=1000,
            max_num_epochs=2,
        )
        # patched dataloader has len() == 4 → 2 * 4 = 8 < 1000.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["megatron_cfg"]["train_iters"] == 8

    def test_megatron_train_iters_not_set_when_disabled(self, patched_factories):
        mc = _make_master_config(megatron_enabled=False)
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert "train_iters" not in mc.policy.get("megatron_cfg", {})
