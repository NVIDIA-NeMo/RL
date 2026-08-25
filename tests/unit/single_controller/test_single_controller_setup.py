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

import hashlib
import io
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

import nemo_rl.algorithms.single_controller_utils.setup as sc_setup_mod
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CustomSamplerConfig,
    WindowedSampler,
    WindowedSamplerConfig,
)
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    GRPOSaveState,
    _initial_grpo_save_state,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    SingleControllerActorArgs,
    setup_single_controller,
)
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptRef,
    RolloutRecoveryLedger,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration


class _CheckpointingCustomSampler(WindowedSampler):
    """Custom sampler whose static capability must be validated during setup."""

    def __init__(self, buffer: Any) -> None:
        super().__init__(buffer, max_staleness_versions=1)


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = True,
    backend: str = "vllm",
    megatron_enabled: bool = False,
    env: dict | None = None,
    max_num_steps: int = 100,
    max_num_epochs: int | None = 1,
    num_prompts_per_step: int = 4,
) -> MasterConfig:
    """Build a partially-populated MasterConfig for unit tests.

    Cross-cutting components (cluster/checkpointing/...) are required by pydantic for
    normal load but unused here — model_construct skips validation, and we hand-fill
    only the dict-shaped fields setup reads.
    """
    generation_config: dict = {
        "backend": backend,
        "colocated": {"enabled": colocated, "resources": {}},
    }
    policy_config: dict = {
        "train_global_batch_size": num_prompts_per_step * 2,
        "max_total_sequence_length": 32,
        "tokenizer": {"use_fastokens": False},
        "megatron_cfg": {"enabled": megatron_enabled},
        "generation": generation_config,
    }
    if backend == "megatron":
        # The megatron build path reads these before any generation factory runs.
        generation_config["mcore_generation_config"] = {"expose_http_server": False}
        policy_config["model_name"] = "test-model"
    return MasterConfig.model_construct(
        data_plane={"enabled": dp_enabled, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": use_multiple_dataloader,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo=GRPOConfig.model_construct(
            seed=42,
            max_num_steps=max_num_steps,
            max_num_epochs=max_num_epochs,
            num_prompts_per_step=num_prompts_per_step,
            num_generations_per_prompt=2,
            max_rollout_turns=1,
            val_period=0,
            val_at_start=False,
            val_at_end=False,
        ),
        policy=policy_config,
        # Full block: setup builds a CheckpointManager unconditionally (resume
        # lookup), which indexes these keys directly. Nothing is written while
        # enabled=False and the dir doesn't exist.
        checkpointing={
            "enabled": False,
            "checkpoint_dir": "results/_sc_setup_test_ckpt",
            "metric_name": None,
            "higher_is_better": False,
            "keep_top_k": None,
            "save_period": 10,
            "save_optimizer": False,
        },
        loss_fn=ClippedPGLossConfig(),
        env=env if env is not None else {},
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=num_prompts_per_step,
            max_buffered_rollouts=num_prompts_per_step * 2,
        ),
    )


def _native_tq_metadata(
    *, step: int = 3, trainer_version: Optional[int] = None, epoch: int = 1
) -> dict[str, Any]:
    return {
        "data_plane_checkpoint_schema_version": (DATA_PLANE_CHECKPOINT_SCHEMA_VERSION),
        "single_controller_train_steps": step,
        "single_controller_trainer_version": (
            step if trainer_version is None else trainer_version
        ),
        "single_controller_epoch": epoch,
        "partition_id": "rollout_data",
        "sampler_name": "in_order",
        "mode": "authoritative",
        "replay_metadata_schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
        "replay_manifest_digest": "digest-1",
        "replay_group_count": 2,
    }


def _save_state(
    *, step: int = 3, trainer_version: Optional[int] = None, epoch: int = 1
) -> GRPOSaveState:
    state = _initial_grpo_save_state()
    state.current_step = step
    state.current_epoch = epoch
    state.trainer_version = trainer_version
    return state


@pytest.fixture
def patched_factories():
    """Patch every external factory setup calls.

    Returns a dict of mocks keyed by name so individual tests can assert on call args
    without re-importing the patch handles.
    """
    fake_dataset = list(range(8))
    fake_dataloader = MagicMock(name="dataloader")
    # len(dataloader) used by the Megatron train_iters injection.
    fake_dataloader.__len__ = MagicMock(return_value=4)
    fake_env_handles = {"math": MagicMock(name="math_env")}
    # Real return objects; _build_generation and _build_trainer return (obj, elapsed_s) tuples.
    fake_gen = MagicMock(name="gen")
    fake_policy = MagicMock(name="policy")

    with (
        patch.object(
            sc_setup_mod,
            "setup_response_data",
            return_value=(fake_dataset, None, fake_env_handles, {}),
        ) as mock_setup_response,
        patch.object(
            sc_setup_mod,
            "StatefulDataLoader",
            return_value=fake_dataloader,
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
            sc_setup_mod, "_build_generation", return_value=(fake_gen, 0.0)
        ) as mock_gen,
        patch.object(
            sc_setup_mod, "_build_trainer", return_value=(fake_policy, 0.0)
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
            "dataloader": fake_dataloader,
            "env_handles": fake_env_handles,
            "fake_gen": fake_gen,
            "fake_policy": fake_policy,
        }


def test_build_generation_passes_sglang_config():
    """SGLangGeneration receives the complete generation config by keyword."""
    master_config = _make_master_config(backend="sglang")
    master_config.policy["model_name"] = "Qwen/Qwen3-0.6B"
    master_config.policy["generation"]["sglang_cfg"] = {}
    inference_cluster = MagicMock(name="inference_cluster")

    with patch.object(sc_setup_mod, "SGLangGeneration") as mock_sglang:
        generation, _ = sc_setup_mod._build_generation(
            inference_cluster,
            master_config,
        )

    mock_sglang.assert_called_once_with(
        cluster=inference_cluster,
        sglang_cfg=master_config.policy["generation"],
    )
    assert master_config.policy["generation"]["sglang_cfg"]["model_path"] == (
        "Qwen/Qwen3-0.6B"
    )
    generation.finish_generation.assert_called_once_with()


class TestSetup:
    """setup arg validation + actor_args assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_single_controller(mc, MagicMock())

    def test_warns_when_rollout_telemetry_lacks_vllm_metrics(self, patched_factories):
        mc = _make_master_config()
        mc.rollout_checkpointing.telemetry_interval_s = 30.0
        mc.policy["generation"]["vllm_cfg"] = {"async_engine": True}

        with pytest.warns(UserWarning, match="vLLM token, request, and KV-cache"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_warns_when_vllm_telemetry_uses_sync_engine(self, patched_factories):
        mc = _make_master_config()
        mc.rollout_checkpointing.telemetry_interval_s = 30.0
        mc.policy["generation"]["vllm_cfg"] = {
            "async_engine": False,
            "enable_vllm_metrics_logger": True,
        }

        with pytest.warns(UserWarning, match="async_engine=true"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_non_vllm_telemetry_warning_has_no_vllm_config_guidance(
        self, patched_factories
    ):
        mc = _make_master_config(backend="sglang")
        mc.rollout_checkpointing.telemetry_interval_s = 30.0

        with pytest.warns(UserWarning) as warning_records:
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        messages = [str(record.message) for record in warning_records]
        assert any("backend='sglang'" in message for message in messages)
        assert all("enable_vllm_metrics_logger" not in message for message in messages)

    def test_rejects_mooncake_data_plane_checkpointing(self):
        mc = _make_master_config()
        mc.data_plane.update(
            {
                "backend": "mooncake_cpu",
                "checkpointing_enabled": True,
            }
        )
        with pytest.raises(NotImplementedError, match="backend='simple'"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_rejects_windowed_checkpointing_without_native_tq(self):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.async_rl.sampler = WindowedSamplerConfig(max_staleness_versions=1)
        mc.data_plane.update(
            {
                "backend": "simple",
                "checkpointing_enabled": False,
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                "replay-checkpoint-capable sampler requires "
                "data_plane.checkpointing_enabled=true"
            ),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_rejects_checkpointing_custom_sampler_without_native_tq(self):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.async_rl.sampler = CustomSamplerConfig(
            target=f"{__name__}:_CheckpointingCustomSampler"
        )
        mc.data_plane.update(
            {
                "backend": "simple",
                "checkpointing_enabled": False,
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                "replay-checkpoint-capable sampler requires "
                "data_plane.checkpointing_enabled=true"
            ),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        with pytest.raises(NotImplementedError, match="use_multiple_dataloader"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    @pytest.mark.parametrize(
        ("invalid_case", "expected_error", "match"),
        [
            ("min_groups", ValueError, "must be >="),
            (
                "global_batch_size",
                ValueError,
                "must equal policy.train_global_batch_size",
            ),
            ("buffer_capacity", ValueError, "required capacity"),
            (
                "streaming_buffer_capacity",
                ValueError,
                "max_buffered_rollouts.*must be >=.*min_groups_for_streaming_train",
            ),
            (
                "deferred_routes_without_capture",
                ValueError,
                "defer_routed_experts_to_policy requires",
            ),
            (
                "recovery_capacity",
                ValueError,
                "num_prompts_per_step.*min_groups_for_streaming_train - 1",
            ),
            ("megatron_dtensor_trainer", ValueError, "megatron_cfg.enabled"),
            ("megatron_colocated", NotImplementedError, "colocated Megatron"),
            ("megatron_gym_without_http_server", ValueError, "expose_http_server"),
            ("gym_on_sglang", NotImplementedError, "vllm and megatron"),
        ],
    )
    def test_invalid_config_fails_before_setup_factories(
        self,
        invalid_case: str,
        expected_error: type[Exception],
        match: str,
        patched_factories,
    ):
        use_gym = invalid_case in ("megatron_gym_without_http_server", "gym_on_sglang")
        if invalid_case == "min_groups":
            mc = _make_master_config()
            mc.async_rl.min_groups_for_streaming_train = 5
        elif invalid_case == "global_batch_size":
            mc = _make_master_config()
            mc.policy["train_global_batch_size"] = 7
        elif invalid_case == "buffer_capacity":
            mc = _make_master_config()
            mc.async_rl.max_buffered_rollouts = 7
        elif invalid_case == "streaming_buffer_capacity":
            mc = _make_master_config()
            # WindowedSampler has no stronger sampler-specific capacity floor. A
            # buffer smaller than the streaming threshold would let the producer
            # consume every permit while the trainer waits for an unreachable count.
            mc.async_rl.sampler = WindowedSamplerConfig(max_staleness_versions=1)
            mc.async_rl.max_buffered_rollouts = 3
        elif invalid_case == "deferred_routes_without_capture":
            mc = _make_master_config()
            mc.token_capture.defer_routed_experts_to_policy = True
        elif invalid_case == "recovery_capacity":
            mc = _make_master_config()
            # Four restored groups may leave one group below the streaming
            # threshold while the producer atomically reserves the next full
            # four-prompt batch. Capacity 4 cannot make progress; 4 + 2 - 1 can.
            mc.async_rl.sampler = WindowedSamplerConfig(max_staleness_versions=1)
            mc.async_rl.min_groups_for_streaming_train = 2
            mc.async_rl.max_buffered_rollouts = 4
            mc.token_capture.enabled = True
            mc.checkpointing["enabled"] = True
            mc.data_plane["checkpointing_enabled"] = True
        elif invalid_case == "megatron_dtensor_trainer":
            mc = _make_master_config(
                colocated=False, backend="megatron", megatron_enabled=False
            )
        elif invalid_case == "megatron_colocated":
            mc = _make_master_config(
                colocated=True, backend="megatron", megatron_enabled=True
            )
        elif invalid_case == "megatron_gym_without_http_server":
            mc = self._make_gym_megatron_config()
            mc.policy["generation"]["mcore_generation_config"]["expose_http_server"] = (
                False
            )
        elif invalid_case == "gym_on_sglang":
            mc = _make_master_config(colocated=True, backend="sglang")
        else:  # pragma: no cover
            raise AssertionError(f"unknown test case {invalid_case}")

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=use_gym),
            patch.object(sc_setup_mod, "spinup_nemo_gym_actor") as mock_spinup,
            pytest.raises(expected_error, match=match),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        patched_factories["setup_response_data"].assert_not_called()
        patched_factories["_build_clusters"].assert_not_called()
        patched_factories["_build_generation"].assert_not_called()
        patched_factories["_build_trainer"].assert_not_called()
        mock_spinup.assert_not_called()

    def test_returns_actor_args(self, patched_factories):
        mc = _make_master_config(colocated=True)
        tokenizer = MagicMock(pad_token_id=0)

        actor_args, _ = setup_single_controller(mc, tokenizer)

        assert isinstance(actor_args, SingleControllerActorArgs)
        assert actor_args.gen_handle is patched_factories["fake_gen"]
        assert actor_args.trainer_handle is patched_factories["fake_policy"]
        assert actor_args.env_handles is patched_factories["env_handles"]
        assert (
            actor_args.dp_client
            is patched_factories["build_data_plane_client"].return_value
        )
        assert actor_args.dataloader is patched_factories["dataloader"]
        assert actor_args.weight_synchronizer is (
            patched_factories["create_weight_synchronizer"].return_value
        )
        # Refit depends on init_communicator running exactly once at setup time.
        actor_args.weight_synchronizer.init_communicator.assert_called_once()
        assert actor_args.advantage_estimator is (
            patched_factories["_create_advantage_estimator"].return_value
        )
        assert actor_args.loss_fn is patched_factories["ClippedPGLossFn"].return_value
        # tq_buffer + rollout_manager are constructed inline (not mocked).
        assert actor_args.tq_buffer is not None
        assert actor_args.rollout_manager is not None
        # rollout_manager binds the same tq_buffer for the writer + sampler.
        assert actor_args.rollout_manager._tq_buffer is actor_args.tq_buffer
        # tq_buffer wires the dp_client + default partition.
        assert actor_args.tq_buffer._dp_client is actor_args.dp_client
        assert actor_args.partition_id == "rollout_data"
        assert actor_args.tq_buffer._partition_id == "rollout_data"
        assert actor_args.tq_buffer._require_routed_experts is False
        assert actor_args.finalizer_actors == []

    def test_router_replay_requires_routes_in_tq_buffer(self, patched_factories):
        mc = _make_master_config(colocated=True)
        mc.policy["router_replay"] = {"enabled": True}

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert actor_args.tq_buffer._require_routed_experts is True

    def test_env_handles_sourced_from_setup_response_data(self, patched_factories):
        """setup_response_data receives master_config.env and supplies env handles."""
        math_env_cfg = {"some": "value"}
        mc = _make_master_config(env={"math": math_env_cfg})

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        _, call_kwargs = patched_factories["setup_response_data"].call_args
        assert call_kwargs["env_configs"] == {"math": math_env_cfg}
        assert actor_args.env_handles is patched_factories["env_handles"]

    def test_weight_sync_factory_args(self, patched_factories):
        """create_weight_synchronizer receives policy / generation / topology."""
        mc = _make_master_config(colocated=False, backend="vllm")
        tokenizer = MagicMock(pad_token_id=0)

        setup_single_controller(mc, tokenizer)

        _, factory_kwargs = patched_factories["create_weight_synchronizer"].call_args
        assert factory_kwargs["policy"] is patched_factories["fake_policy"]
        assert factory_kwargs["generation"] is patched_factories["fake_gen"]
        assert factory_kwargs["generation_backend"] == "vllm"
        assert factory_kwargs["colocated"] is False

    def test_custom_partition_id(self, patched_factories):
        mc = _make_master_config()
        tokenizer = MagicMock(pad_token_id=7)

        actor_args, _ = setup_single_controller(
            mc, tokenizer, partition_id="custom_partition"
        )

        assert actor_args.partition_id == "custom_partition"
        assert actor_args.tq_buffer._partition_id == "custom_partition"
        assert actor_args.tq_buffer._pad_value_dict == {
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

        assert mc.grpo.max_num_steps == 2

    def test_max_num_steps_capped_by_dataloader_epochs(self, patched_factories):
        """grpo.max_num_steps drops to max_num_epochs * len(dataloader) when smaller."""
        mc = _make_master_config(
            megatron_enabled=False,
            max_num_steps=1000,
            max_num_epochs=2,
        )
        # patched dataloader has len() == 4 → 2 * 4 = 8 < 1000.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.grpo.max_num_steps == 8

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

    def test_megatron_train_iters_with_unbounded_epochs(self, patched_factories):
        """None max_num_epochs leaves max_num_steps as the Megatron limit."""
        mc = _make_master_config(
            megatron_enabled=True,
            max_num_steps=100,
            max_num_epochs=None,
        )
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.grpo.max_num_steps == 100
        assert mc.policy["megatron_cfg"]["train_iters"] == 100

    def test_megatron_train_iters_not_set_when_disabled(self, patched_factories):
        mc = _make_master_config(megatron_enabled=False)
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert "train_iters" not in mc.policy.get("megatron_cfg", {})

    def test_nemo_gym_wires_env_handle(self, patched_factories):
        """When _should_use_nemo_gym is True the nemo-gym actor is spun up and stored."""
        mc = _make_master_config(colocated=True, backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            None,
        )
        fake_gym_actor = MagicMock(name="nemo_gym_actor")

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=fake_gym_actor
            ) as mock_spinup,
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        mock_spinup.assert_called_once_with(
            env_configs=mc.env,
            base_urls=patched_factories["fake_gen"].dp_openai_server_base_urls,
            model_name="test-model",
            enable_router_replay=False,
            routed_experts_dtype="int16",
            use_fastokens=False,
            token_capture=None,
        )
        assert actor_args.env_handles["nemo_gym"] is fake_gym_actor

    def test_token_capture_always_creates_finalizer_actor_pool(self, patched_factories):
        mc = _make_master_config(colocated=True, backend="vllm")
        mc.policy["generation"].update(
            {
                "model_name": "test-model",
                "stop_strings": None,
                "stop_token_ids": None,
                "top_k": None,
                "vllm_cfg": {"async_engine": True},
            }
        )
        mc.logger = {"log_dir": "/tmp/test-token-capture"}
        mc.token_capture.enabled = True
        mc.token_capture.num_finalizer_workers = 3
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            None,
        )
        fake_actors = [MagicMock(name=f"finalizer_{index}") for index in range(3)]

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=MagicMock()
            ),
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
            patch(
                "nemo_rl.experience.finalizer_actor.create_finalizer_actors",
                return_value=fake_actors,
            ) as mock_create_finalizer_actors,
        ):
            actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=9))

        (actor_dp_config, actor_config), actor_kwargs = (
            mock_create_finalizer_actors.call_args
        )
        assert actor_dp_config == mc.data_plane
        assert actor_config.partition_id == "rollout_data"
        assert actor_config.staging_partition == mc.token_capture.staging_partition
        assert actor_config.pad_token_id == 9
        assert actor_kwargs == {"num_workers": 3}
        assert actor_args.finalizer_actors == fake_actors
        assert not hasattr(actor_args.rollout_manager, "_finalizer")

    def test_setup_timing_populated_for_colocated_vllm(self, patched_factories):
        """Colocated vLLM records gen+policy+collective+total+worker fields."""
        mc = _make_master_config(colocated=True, backend="vllm")

        _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        for field in (
            "generation_init_time_s",
            "policy_init_time_s",
            "collective_init_time_s",
            "worker_setup_time_s",
            "total_setup_time_s",
            "other_setup_time_s",
        ):
            value = getattr(metrics, field)
            assert value is not None, f"missing {field} on {metrics}"
            assert value >= 0
        # parallel_wall_time_s / parallel_init_enabled are grpo.py-only in the
        # shared SetupTimingMetrics — SC does not emit them.
        assert metrics.parallel_wall_time_s is None
        assert metrics.parallel_init_enabled is None
        # Reserve/load split is populated on the gym-on path only.
        assert metrics.generation_init_reserve_time_s is None
        assert metrics.generation_init_load_time_s is None

    def test_setup_timing_populated_for_noncolocated_vllm(self, patched_factories):
        """Non-colocated vLLM records the same per-phase fields as colocated."""
        mc = _make_master_config(colocated=False, backend="vllm")

        _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert metrics.generation_init_time_s is not None
        assert metrics.policy_init_time_s is not None
        assert metrics.worker_setup_time_s is not None
        # parallel_wall_time_s / parallel_init_enabled are grpo.py-only.
        assert metrics.parallel_wall_time_s is None
        assert metrics.parallel_init_enabled is None
        # Reserve/load split is populated on the gym-on path only.
        assert metrics.generation_init_reserve_time_s is None
        assert metrics.generation_init_load_time_s is None

    def test_setup_timing_backend_agnostic_for_sglang(self, patched_factories):
        """SC uses the backend-agnostic generation_init_time_s regardless of backend."""
        mc = _make_master_config(colocated=True, backend="sglang")

        _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert metrics.generation_init_time_s is not None

    def test_nemo_gym_uses_deferred_vllm_load(self, patched_factories):
        """NeMo-Gym path reserves vLLM ports up-front and finishes the load afterwards."""
        mc = _make_master_config(colocated=True, backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=MagicMock()
            ),
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        # _build_generation must be called with defer_model_load=True so the workers
        # only reserve URLs; load_and_start()+finish_generation() run afterwards.
        _, gen_kwargs = patched_factories["_build_generation"].call_args
        assert gen_kwargs.get("defer_model_load") is True
        deferred_vllm = patched_factories["fake_gen"]
        deferred_vllm.load_and_start.assert_called_once_with()
        deferred_vllm.finish_generation.assert_called_once_with()

    def test_nemo_gym_records_timing_metrics(self, patched_factories):
        """NeMo-Gym path records per-phase timings (vllm/policy/gym/worker)."""
        mc = _make_master_config(colocated=True, backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=MagicMock()
            ),
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert metrics.nemo_gym_init_time_s is not None
        assert metrics.generation_init_time_s is not None
        assert metrics.policy_init_time_s is not None
        assert metrics.worker_setup_time_s is not None
        # parallel_wall_time_s / parallel_init_enabled are grpo.py-only.
        assert metrics.parallel_wall_time_s is None
        assert metrics.parallel_init_enabled is None

    def test_nemo_gym_noncolocated_finishes_deferred_load(self, patched_factories):
        """Non-colocated + gym fans out gym / deferred-load / trainer together."""
        mc = _make_master_config(colocated=False, backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=MagicMock()
            ),
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            actor_args, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        # _build_generation runs once (URL reservation only); the load is finished
        # by _finish_deferred_generation inside the executor.
        patched_factories["_build_generation"].assert_called_once()
        _, gen_kwargs = patched_factories["_build_generation"].call_args
        assert gen_kwargs.get("defer_model_load") is True
        patched_factories["fake_gen"].load_and_start.assert_called_once_with()
        assert actor_args.gen_handle is patched_factories["fake_gen"]
        assert metrics.nemo_gym_init_time_s is not None
        assert metrics.generation_init_time_s is not None
        assert metrics.policy_init_time_s is not None

    @pytest.mark.parametrize("colocated", [True, False])
    def test_nemo_gym_generation_init_time_includes_reserve_time(
        self, patched_factories, colocated
    ):
        """generation_init_time_s folds in the deferred-VllmGeneration reserve time.

        With gym on, _build_generation(defer_model_load=True) does worker-group
        spawn + port bind (no weight load). That elapsed time has to end up in
        generation_init_time_s alongside the deferred-load elapsed; otherwise
        gym-on runs undercount generation setup by the worker-group span. The
        reserve/load split is also exposed for overlap analysis.
        """
        mc = _make_master_config(colocated=colocated, backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)
        # Deferred _build_generation returns 3.0s of reserve time; _build_generation
        # is only called once (for reservation), so this is the reserve span.
        patched_factories["_build_generation"].return_value = (
            patched_factories["fake_gen"],
            3.0,
        )

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=MagicMock()
            ),
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        # gen_load_time (from _finish_deferred_generation, unpatched) is ~0 in
        # the test — the reserve time dominates and must be present.
        assert metrics.generation_init_time_s >= 3.0
        assert metrics.generation_init_reserve_time_s == 3.0
        assert metrics.generation_init_load_time_s is not None

    def _make_gym_megatron_config(self) -> MasterConfig:
        mc = _make_master_config(
            colocated=False, backend="megatron", megatron_enabled=True
        )
        mc.policy["generation"]["mcore_generation_config"]["expose_http_server"] = True
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        return mc

    @pytest.mark.parametrize(
        ("scenario", "error_match"),
        [
            ("gym", None),
            ("gym_served_mismatch", "different address"),
            ("native", None),
        ],
        ids=["gym", "gym_served_mismatch", "native"],
    )
    def test_megatron_setup(
        self, patched_factories, scenario: str, error_match: str | None
    ):
        """Non-colocated Megatron generation setup, gym and native legs.

        gym: reserve rank-0's URL, spin Gym up on it, build trainer + engine
        (weight load skipped, reserved port adopted), cross-check the served
        address, reap the port holder.
        gym_served_mismatch: the served-vs-reserved cross-check fires after the
        builds when the engine comes up on a different address.
        native: expose_http_server=false and no Gym, so nothing reserves a URL,
        no port holder is created, and the cross-check is skipped.
        """
        gym = scenario != "native"
        if gym:
            mc = self._make_gym_megatron_config()
            patched_factories["setup_response_data"].return_value = (
                list(range(8)),
                None,
            )
        else:
            mc = _make_master_config(
                colocated=False, backend="megatron", megatron_enabled=True
            )
        mc.async_rl.recompute_kv_cache_after_weight_updates = True
        tokenizer = MagicMock(pad_token_id=0)
        reserved_url = "http://10.0.0.1:5555/v1"
        served_url = (
            "http://10.0.0.9:7/v1"
            if scenario == "gym_served_mismatch"
            else reserved_url
        )
        port_holder = MagicMock(name="port_holder")
        fake_gym_actor = MagicMock(name="nemo_gym_actor")

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=gym),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=fake_gym_actor
            ) as mock_spinup,
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
            patch.object(sc_setup_mod, "MegatronGeneration") as mock_megatron,
            patch.object(sc_setup_mod, "ray") as mock_ray,
        ):
            mock_megatron.reserve_http_server_address.return_value = (
                reserved_url,
                5555,
                port_holder,
            )
            # Wire the real check through the class mock so the
            # served-vs-reserved legs exercise the genuine logic.
            mock_megatron.verify_served_address = (
                MegatronGeneration.verify_served_address
            )
            mock_megatron.return_value.dp_openai_server_base_urls = [served_url]
            if error_match is None:
                actor_args, metrics = setup_single_controller(mc, tokenizer)
            else:
                with pytest.raises(RuntimeError, match=error_match):
                    setup_single_controller(mc, tokenizer)

        inference_cluster = patched_factories["_build_clusters"].return_value[1]
        # The megatron path never uses the generic generation factory and applies
        # its config overrides before any build (_build_generation normally sets
        # model_name; the kv-cache mode comes from the async_rl flag).
        patched_factories["_build_generation"].assert_not_called()
        assert mc.policy["generation"]["model_name"] == "test-model"
        mcore_cfg = mc.policy["generation"]["mcore_generation_config"]
        assert mcore_cfg["kv_cache_management_mode"] == "recompute"
        assert mc.async_rl.recompute_kv_cache_after_weight_updates is False
        # Reservation + holder lifecycle exist on the gym legs only.
        if gym:
            mock_megatron.reserve_http_server_address.assert_called_once_with(
                inference_cluster,
                mc.policy,
            )
            mock_ray.kill.assert_called_once_with(port_holder)
        else:
            mock_megatron.reserve_http_server_address.assert_not_called()
            mock_ray.kill.assert_not_called()

        # Construction: trainer first, generation from the dedicated cluster,
        # with the weight load skipped and the reserved port adopted (gym) or
        # absent (native).
        patched_factories["_build_trainer"].assert_called_once()
        mock_megatron.assert_called_once_with(
            config=mc.policy,
            tokenizer=tokenizer,
            cluster=inference_cluster,
            processor=None,
            weights_path=None,
            skip_weight_load=True,
            reserved_http_server_port=5555 if gym else None,
        )
        if gym:
            # Gym spins up on the reserved URL, before the served-address
            # cross-check — so the mismatch leg sees it too.
            _, spinup_kwargs = mock_spinup.call_args
            assert spinup_kwargs["base_urls"] == [reserved_url]
        else:
            mock_spinup.assert_not_called()
        if scenario == "gym_served_mismatch":
            return  # raised at the cross-check; no actor_args/metrics exist

        assert actor_args.gen_handle is mock_megatron.return_value
        assert actor_args.trainer_handle is patched_factories["fake_policy"]
        assert metrics.generation_init_time_s is not None
        assert metrics.policy_init_time_s is not None
        _, factory_kwargs = patched_factories["create_weight_synchronizer"].call_args
        assert factory_kwargs["generation_backend"] == "megatron"
        assert factory_kwargs["colocated"] is False
        assert factory_kwargs["inference_cluster"] is inference_cluster
        if gym:
            assert actor_args.env_handles["nemo_gym"] is fake_gym_actor
            assert metrics.nemo_gym_init_time_s is not None
            assert metrics.generation_init_reserve_time_s is not None
        else:
            # Reserve/load split is populated on the gym-on path only.
            assert metrics.generation_init_reserve_time_s is None


class TestNativeTQRecoverySetup:
    def test_setup_loads_tq_before_creating_single_controller_client(
        self, tmp_path, patched_factories
    ):
        checkpoint_path = tmp_path / "step_3"
        (checkpoint_path / DATA_PLANE_CHECKPOINT_DIR).mkdir(parents=True)
        (checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME).touch()
        torch.save({}, checkpoint_path / "train_dataloader.pt")
        save_state = _save_state()
        policy = patched_factories["fake_policy"]
        events: list[str] = []
        policy.load_data_plane_checkpoint.side_effect = (
            lambda checkpoint_dir: events.append("load") or _native_tq_metadata()
        )
        patched_factories["build_data_plane_client"].side_effect = (
            lambda *args, **kwargs: events.append("build")
            or MagicMock(name="dp_client")
        )
        checkpointer = MagicMock()
        checkpointer.get_latest_checkpoint_path.return_value = str(checkpoint_path)
        checkpointer.load_training_info.return_value = vars(save_state)
        checkpointer.get_resume_paths.return_value = (None, None)
        mc = _make_master_config()

        with patch.object(sc_setup_mod, "CheckpointManager", return_value=checkpointer):
            actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert events == ["load", "build"]
        assert actor_args.data_plane_checkpoint_metadata == _native_tq_metadata()

    def test_loads_authoritative_tq_checkpoint_when_metadata_sidecar_exists(
        self, tmp_path
    ):
        checkpoint_path = tmp_path / "step_3"
        (checkpoint_path / DATA_PLANE_CHECKPOINT_DIR).mkdir(parents=True)
        (checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME).touch()
        policy = MagicMock()
        metadata = _native_tq_metadata()
        policy.load_data_plane_checkpoint.return_value = metadata
        save_state = _save_state()

        restored = sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
            policy,
            last_checkpoint_path=str(checkpoint_path),
            save_state=save_state,
            partition_id="rollout_data",
            sampler_name="in_order",
        )

        assert restored == metadata
        policy.load_data_plane_checkpoint.assert_called_once_with(
            checkpoint_path / DATA_PLANE_CHECKPOINT_DIR
        )

    def test_validates_trainer_version_independently_from_train_step(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        (checkpoint_path / DATA_PLANE_CHECKPOINT_DIR).mkdir(parents=True)
        (checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME).touch()
        policy = MagicMock()
        metadata = _native_tq_metadata(step=3, trainer_version=7)
        policy.load_data_plane_checkpoint.return_value = metadata

        restored = sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
            policy,
            last_checkpoint_path=str(checkpoint_path),
            save_state=_save_state(trainer_version=7),
            partition_id="rollout_data",
            sampler_name="in_order",
        )

        assert restored == metadata

    def test_legacy_replay_checkpoint_is_rejected(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        (checkpoint_path / LEGACY_REPLAY_BUFFER_FILENAME).touch()
        policy = MagicMock()

        with pytest.raises(RuntimeError, match="legacy replay_buffer.pt"):
            sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
                policy,
                last_checkpoint_path=str(checkpoint_path),
                save_state=_save_state(),
                partition_id="rollout_data",
                sampler_name="in_order",
            )

        policy.load_data_plane_checkpoint.assert_not_called()

    def test_checkpoint_without_replay_artifacts_does_not_load_tq(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        policy = MagicMock()

        restored = sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
            policy,
            last_checkpoint_path=str(checkpoint_path),
            save_state=_save_state(),
            partition_id="rollout_data",
            sampler_name="in_order",
        )

        assert restored is None
        policy.load_data_plane_checkpoint.assert_not_called()

    def test_metadata_sidecar_requires_matching_tq_directory(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        (checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME).touch()

        with pytest.raises(FileNotFoundError, match="matching native TQ checkpoint"):
            sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
                MagicMock(),
                last_checkpoint_path=str(checkpoint_path),
                save_state=_save_state(),
                partition_id="rollout_data",
                sampler_name="in_order",
            )

    def test_rejects_tq_checkpoint_from_different_training_step(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        (checkpoint_path / DATA_PLANE_CHECKPOINT_DIR).mkdir(parents=True)
        (checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME).touch()
        policy = MagicMock()
        policy.load_data_plane_checkpoint.return_value = _native_tq_metadata(step=2)

        with pytest.raises(ValueError, match="does not match the trainer checkpoint"):
            sc_setup_mod._maybe_restore_native_data_plane_checkpoint(
                policy,
                last_checkpoint_path=str(checkpoint_path),
                save_state=_save_state(),
                partition_id="rollout_data",
                sampler_name="in_order",
            )


class TestRolloutRecoverySetup:
    @staticmethod
    def _write_partial_ledger(checkpoint_path) -> tuple[dict[str, Any], str]:
        ledger = RolloutRecoveryLedger()
        ledger.reserve_group(
            group_id="partial-group",
            prompt_id="2",
            prompt_ref=PromptRef(sample_id="2", task_name="nemo_gym"),
            prompt_payload={"idx": 2, "task_name": "nemo_gym"},
            expected_generations=2,
            target_step=3,
            start_weight_version=3,
        )
        buffer = io.BytesIO()
        torch.save(ledger.state_dict(), buffer)
        payload = buffer.getvalue()
        (checkpoint_path / ROLLOUT_RECOVERY_STATE_FILENAME).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        return (
            {
                "rollout_recovery_schema_version": (ROLLOUT_RECOVERY_SCHEMA_VERSION),
                "rollout_recovery_payload_sha256": digest,
                "rollout_recovery_group_count": 1,
            },
            digest,
        )

    def test_restores_digest_bound_lineage_and_rehydrates_prompt(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        metadata, _ = self._write_partial_ledger(checkpoint_path)

        ledger = sc_setup_mod._maybe_restore_rollout_recovery_ledger(
            last_checkpoint_path=str(checkpoint_path),
            data_plane_checkpoint_metadata=metadata,
            token_capture_enabled=True,
        )
        assert ledger is not None
        dataset = [
            {"idx": index, "task_name": "nemo_gym", "payload": f"prompt-{index}"}
            for index in range(4)
        ]

        sc_setup_mod._rehydrate_rollout_recovery_prompts(ledger, dataset)

        restored = ledger.get_group("partial-group")
        assert restored.runtime_prompt_payload is dataset[2]

    def test_rejects_corrupt_rollout_recovery_sidecar(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        metadata, _ = self._write_partial_ledger(checkpoint_path)
        (checkpoint_path / ROLLOUT_RECOVERY_STATE_FILENAME).write_bytes(b"corrupt")

        with pytest.raises(ValueError, match="digest does not match"):
            sc_setup_mod._maybe_restore_rollout_recovery_ledger(
                last_checkpoint_path=str(checkpoint_path),
                data_plane_checkpoint_metadata=metadata,
                token_capture_enabled=True,
            )

    def test_rejects_native_token_capture_checkpoint_without_lineage(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()

        with pytest.raises(FileNotFoundError, match="older checkpoints"):
            sc_setup_mod._maybe_restore_rollout_recovery_ledger(
                last_checkpoint_path=str(checkpoint_path),
                data_plane_checkpoint_metadata={"mode": "authoritative"},
                token_capture_enabled=True,
            )

    def test_static_prompt_reference_must_match_current_dataset(self, tmp_path):
        checkpoint_path = tmp_path / "step_3"
        checkpoint_path.mkdir()
        metadata, _ = self._write_partial_ledger(checkpoint_path)
        ledger = sc_setup_mod._maybe_restore_rollout_recovery_ledger(
            last_checkpoint_path=str(checkpoint_path),
            data_plane_checkpoint_metadata=metadata,
            token_capture_enabled=True,
        )
        assert ledger is not None
        dataset = [{"idx": index, "task_name": "nemo_gym"} for index in range(4)]
        dataset[2] = {"idx": 99, "task_name": "nemo_gym"}

        with pytest.raises(ValueError, match="durable sample reference"):
            sc_setup_mod._rehydrate_rollout_recovery_prompts(ledger, dataset)
