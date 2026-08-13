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
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    ReadyFirstSamplerConfig,
    SamplerConfig,
)
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    SingleControllerActorArgs,
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
    max_num_epochs: int | None = 1,
    num_prompts_per_step: int = 4,
    sampler_cfg: SamplerConfig | None = None,
    loss_cfg: ClippedPGLossConfig | None = None,
) -> MasterConfig:
    """Build a partially-populated MasterConfig for unit tests.

    Cross-cutting components (cluster/checkpointing/...) are required by pydantic for
    normal load but unused here — model_construct skips validation, and we hand-fill
    only the dict-shaped fields setup reads.
    """
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
        policy={
            "train_global_batch_size": num_prompts_per_step * 2,
            "max_total_sequence_length": 32,
            "tokenizer": {"use_fastokens": False},
            "megatron_cfg": {"enabled": megatron_enabled},
            "generation": {
                "backend": backend,
                "colocated": {"enabled": colocated, "resources": {}},
            },
        },
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
        loss_fn=loss_cfg if loss_cfg is not None else ClippedPGLossConfig(),
        env=env if env is not None else {},
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=num_prompts_per_step,
            max_buffered_rollouts=num_prompts_per_step * 2,
            **({} if sampler_cfg is None else {"sampler": sampler_cfg}),
        ),
    )


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


def test_build_clusters_rejects_non_colocated_megatron_generation():
    """The topology guard identifies Megatron as the generation backend."""
    master_config = _make_master_config(colocated=False, backend="megatron")
    master_config.cluster = {"num_nodes": 2, "gpus_per_node": 8}

    with pytest.raises(
        AssertionError,
        match="Megatron generation backend.*non-colocated inference",
    ):
        sc_setup_mod._build_clusters(master_config)


class TestSetup:
    """setup arg validation + actor_args assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_single_controller(mc, MagicMock())

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        with pytest.raises(NotImplementedError, match="use_multiple_dataloader"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    @pytest.mark.parametrize(
        ("invalid_case", "match"),
        [
            ("min_groups", "must be >="),
            ("global_batch_size", "must equal policy.train_global_batch_size"),
            ("buffer_capacity", "required capacity"),
        ],
    )
    def test_invalid_config_fails_before_setup_factories(
        self,
        invalid_case: str,
        match: str,
        patched_factories,
    ):
        mc = _make_master_config()
        if invalid_case == "min_groups":
            mc.async_rl.min_groups_for_streaming_train = 5
        elif invalid_case == "global_batch_size":
            mc.policy["train_global_batch_size"] = 7
        elif invalid_case == "buffer_capacity":
            mc.async_rl.max_buffered_rollouts = 7
        else:  # pragma: no cover
            raise AssertionError(f"unknown test case {invalid_case}")

        with pytest.raises(ValueError, match=match):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        patched_factories["setup_response_data"].assert_not_called()
        patched_factories["_build_clusters"].assert_not_called()
        patched_factories["_build_generation"].assert_not_called()
        patched_factories["_build_trainer"].assert_not_called()

    @pytest.mark.parametrize(
        ("loss_overrides", "match"),
        [
            (
                {"use_importance_sampling_correction": False},
                "use_importance_sampling_correction=true",
            ),
            (
                {
                    "use_importance_sampling_correction": True,
                    "force_on_policy_ratio": True,
                },
                "force_on_policy_ratio=false",
            ),
        ],
        ids=["no_is_correction", "forced_on_policy_ratio"],
    )
    def test_ready_first_sampler_rejects_incompatible_loss_config(
        self,
        loss_overrides: dict,
        match: str,
        patched_factories,
    ):
        # ready_first is only valid with use_importance_sampling_correction=true
        # and force_on_policy_ratio=false; anything else is rejected at setup,
        # before any factory allocates resources.
        mc = _make_master_config(
            sampler_cfg=ReadyFirstSamplerConfig(max_staleness_versions=1),
            loss_cfg=ClippedPGLossConfig(**loss_overrides),
        )

        with pytest.raises(ValueError, match=match):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        patched_factories["setup_response_data"].assert_not_called()
        patched_factories["_build_clusters"].assert_not_called()

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
        )
        assert actor_args.env_handles["nemo_gym"] is fake_gym_actor

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
        # Backend-specific fields are grpo.py-only; SC does not populate them.
        assert metrics.vllm_init_time_s is None
        assert metrics.sglang_init_time_s is None

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

    @pytest.mark.parametrize("backend", ["sglang", "megatron"])
    def test_nemo_gym_rejects_non_vllm_backend(self, patched_factories, backend):
        """SC nemo-gym wiring only supports vLLM; every other backend must raise."""
        mc = _make_master_config(colocated=True, backend=backend)
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            None,
        )

        with (
            patch.object(sc_setup_mod, "_should_use_nemo_gym", return_value=True),
            patch.object(sc_setup_mod, "spinup_nemo_gym_actor") as mock_spinup,
            pytest.raises(NotImplementedError, match="vllm"),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))
        mock_spinup.assert_not_called()
