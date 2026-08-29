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

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import nemo_rl.algorithms.single_controller_utils.setup as sc_setup_mod
from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    DataPlaneCheckpointMetadata,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CustomSamplerConfig,
    ReadyFirstSamplerConfig,
    SamplerConfig,
    WindowedSampler,
    WindowedSamplerConfig,
)
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    GRPOSaveState,
    _initial_grpo_save_state,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    SingleControllerActorArgs,
    setup_single_controller,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
from nemo_rl.data_plane.schema import SC_ROLLOUT_SCHEMA_FIELDS
from nemo_rl.experience.rollouts import EffortLevelsConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


class _CheckpointingCustomSampler(WindowedSampler):
    """Custom sampler whose static capability must be validated during setup."""

    supports_buffer_checkpoint = True

    def __init__(self, buffer: Any) -> None:
        super().__init__(buffer, max_staleness_versions=1)


class _NonCheckpointingCustomSampler(WindowedSampler):
    """Custom sampler that explicitly opts out of replay recovery."""

    supports_buffer_checkpoint = False

    def __init__(self, buffer: Any) -> None:
        super().__init__(buffer, max_staleness_versions=1)


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = False,
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
        data_plane={
            "enabled": dp_enabled,
            "impl": "transfer_queue",
            "backend": "simple",
        },
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
        cluster={"num_nodes": 2, "gpus_per_node": 8, "segment_size": None},
        loss_fn=loss_cfg if loss_cfg is not None else ClippedPGLossConfig(),
        env=env if env is not None else {},
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=num_prompts_per_step,
            max_buffered_rollouts=num_prompts_per_step * 2,
            **({} if sampler_cfg is None else {"sampler": sampler_cfg}),
        ),
    )


def _native_tq_metadata(
    *, step: int = 3, trainer_version: Optional[int] = None, epoch: int = 1
) -> DataPlaneCheckpointMetadata:
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
                None,
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
        patch(
            "nemo_rl.algorithms.grpo._create_advantage_estimator",
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


def test_build_clusters_rejects_unsupported_topology_backend(monkeypatch):
    """Topology planning reports the supported SC backends instead of KeyError."""
    master_config = _make_master_config(colocated=False, backend="trtllm")
    master_config.cluster = {"num_nodes": 2, "gpus_per_node": 8, "segment_size": 1}
    master_config.policy["generation"]["colocated"]["resources"] = {
        "gpus_per_node": 8,
        "num_nodes": 1,
    }
    monkeypatch.setattr(
        sc_setup_mod,
        "prepare_segment_topology",
        lambda *args, **kwargs: (
            [{"nvlink_domain": 0.001}],
            ["inference"],
            {
                "training": ("nvlink_domain", 0),
                "inference": ("nvlink_domain", 1),
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="only supports vllm or sglang generation; got 'trtllm'",
    ):
        sc_setup_mod._build_clusters(master_config)


def test_build_clusters_leaves_dedicated_teacher_nodes(monkeypatch):
    """Teacher nodes are removed before the student train/inference split."""
    master_config = _make_master_config(colocated=False)
    master_config.cluster = {"num_nodes": 3, "gpus_per_node": 8}
    master_config.policy["generation"]["colocated"]["resources"] = {
        "gpus_per_node": 8,
        "num_nodes": 1,
    }
    master_config.on_policy_distillation = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"default_teacher": "Qwen/Qwen3-1.7B"},
        default_teacher_alias="default_teacher",
        non_colocated_teachers={"enabled": True},
    )
    constructed = []

    class FakeCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            constructed.append(kwargs)

    monkeypatch.setattr(sc_setup_mod, "RayVirtualCluster", FakeCluster)

    _, _, teacher_topology = sc_setup_mod._build_clusters(master_config)

    assert constructed[0]["bundle_ct_per_node_list"] == [8]
    assert constructed[1]["bundle_ct_per_node_list"] == [8]
    assert teacher_topology is None


def test_build_clusters_supports_two_node_shared_student_layout(monkeypatch):
    """One student node can split train/inference while node two hosts teacher."""
    master_config = _make_master_config(colocated=False)
    master_config.cluster = {"num_nodes": 2, "gpus_per_node": 8}
    master_config.policy["generation"]["colocated"]["resources"] = {
        "gpus_per_node": 4,
        "num_nodes": 1,
    }
    master_config.on_policy_distillation = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"default": "/ckpt/teacher"},
        default_teacher_alias="default",
        non_colocated_teachers={
            "enabled": True,
            "default_teacher_cfg": {"num_nodes": 1, "gpus_per_node": 8},
        },
    )
    constructed = []

    class FakeCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            constructed.append(self)

    monkeypatch.setattr(sc_setup_mod, "RayVirtualCluster", FakeCluster)

    train_cluster, inference_cluster, teacher_topology = sc_setup_mod._build_clusters(
        master_config
    )

    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [4]
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [4]
    assert teacher_topology is None


def test_single_controller_mopd_recipe_resolves_to_runtime_contract():
    """The inherited recipe resolves exactly as the SC entrypoint consumes it."""
    register_omegaconf_resolvers()
    repo_root = Path(__file__).resolve().parents[3]
    recipe = repo_root / (
        "examples/configs/recipes/llm/"
        "mopd-qwen3-1.7b-3n8g-megatron-pack-single-controller.yaml"
    )
    resolved = OmegaConf.to_container(load_config(recipe), resolve=True)

    assert isinstance(resolved, dict)
    config = MasterConfig.model_validate(resolved)
    assert config.grpo.async_grpo is None
    assert config.grpo.adv_estimator.name == "opd"
    assert config.grpo.skip_reference_policy_logprobs_calculation is True
    assert config.async_rl.min_groups_for_streaming_train == (
        config.grpo.num_prompts_per_step
    )
    assert config.policy["train_global_batch_size"] == (
        config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    )
    assert config.data_plane["enabled"] is True
    assert config.env["should_use_nemo_gym"] is True
    assert config.on_policy_distillation.enabled is True
    assert config.on_policy_distillation.non_colocated_teachers is not None
    assert config.on_policy_distillation.non_colocated_teachers.enabled is True
    assert (
        config.on_policy_distillation.teacher_model_by_agent_name["default_teacher"]
        == config.policy["model_name"]
    )


def test_sibling_recovery_functional_config_resolves_to_runtime_contract():
    """The two-phase Gym recovery fixture must pass SC config validation."""
    register_omegaconf_resolvers()
    repo_root = Path(__file__).resolve().parents[3]
    config = load_config(
        repo_root / "examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml"
    )
    overrides = [
        "policy.model_name=Qwen/Qwen3-0.6B",
        "policy.dtensor_cfg.enabled=false",
        "policy.megatron_cfg.enabled=true",
        "policy.megatron_cfg.tensor_model_parallel_size=1",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        "policy.megatron_cfg.expert_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=1",
        "policy.megatron_cfg.sequence_parallel=false",
        "policy.generation.vllm_cfg.tensor_parallel_size=1",
        "policy.generation.vllm_cfg.async_engine=true",
        "policy.max_total_sequence_length=512",
        "policy.generation.colocated.enabled=false",
        "policy.generation.colocated.resources.num_nodes=1",
        "policy.generation.colocated.resources.gpus_per_node=1",
        "grpo.num_prompts_per_step=4",
        "grpo.num_generations_per_prompt=2",
        "grpo.max_num_steps=2",
        "grpo.val_period=-1",
        "grpo.val_at_start=false",
        "grpo.async_grpo=null",
        "policy.train_global_batch_size=8",
        "policy.train_micro_batch_size=1",
        "cluster.gpus_per_node=2",
        "loss_fn.reference_policy_kl_penalty=0.01",
        "grpo.skip_reference_policy_logprobs_calculation=false",
        "loss_fn.use_importance_sampling_correction=true",
        "checkpointing.enabled=true",
        "checkpointing.checkpoint_dir=/tmp/sibling-recovery-checkpoints",
        "checkpointing.metric_name=null",
        "checkpointing.save_period=1",
        "+checkpointing.save_data_plane=true",
        "++data_plane.enabled=true",
        "++data_plane.impl=transfer_queue",
        "++data_plane.backend=simple",
        "++data_plane.simple.storage_capacity=1000000",
        "++data_plane.simple.num_storage_units=2",
        "++data_plane.claim_meta_poll_interval_s=0.5",
        "++token_capture.enabled=true",
        "++rollout_recovery.default_granularity=sibling",
        "++async_rl.sampler.name=in_order",
        "++async_rl.sampler.max_lookahead_versions=1",
        "++async_rl.min_groups_for_streaming_train=4",
        "++async_rl.max_inflight_prompts=8",
        "++async_rl.max_buffered_rollouts=8",
        "++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=120",
        "++async_rl.stall_watchdog.interval_s=10",
        "++async_rl.stall_watchdog.stall_timeout_s=300",
        "++async_rl.stall_watchdog.stall_action=abort",
    ]

    resolved = OmegaConf.to_container(
        parse_hydra_overrides(config, overrides),
        resolve=True,
    )

    assert isinstance(resolved, dict)
    master_config = MasterConfig.model_validate(resolved)
    validate_single_controller_config(master_config)
    assert master_config.checkpointing["metric_name"] is None
    assert master_config.checkpointing["save_data_plane"] is True
    assert master_config.async_rl.rollout_failure.native.generation_timeout_s is None
    assert master_config.async_rl.rollout_failure.nemo_gym.rollout_timeout_s == 120


class TestSetup:
    """setup arg validation + actor_args assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_single_controller(mc, MagicMock())

    def test_rejects_mooncake_data_plane_checkpointing(self):
        mc = _make_master_config()
        mc.data_plane["backend"] = "mooncake_cpu"
        mc.checkpointing["save_data_plane"] = True
        with pytest.raises(NotImplementedError, match="backend='mooncake_cpu'"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_rejects_windowed_checkpointing_without_native_tq(self):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.checkpointing["save_data_plane"] = False
        mc.async_rl.sampler = WindowedSamplerConfig(max_staleness_versions=1)
        mc.data_plane["backend"] = "simple"

        with pytest.raises(
            ValueError,
            match=(
                "replay-checkpoint-capable sampler requires "
                "checkpointing.save_data_plane=true"
            ),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_checkpointing_error_explains_mooncake_incompatibility(self):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.checkpointing["save_data_plane"] = False
        mc.async_rl.sampler = WindowedSamplerConfig(max_staleness_versions=1)
        mc.data_plane["backend"] = "mooncake_cpu"

        with pytest.raises(
            ValueError,
            match=(
                "backend='mooncake_cpu'.*backend='simple'.*checkpointing.enabled=false"
            ),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_rejects_checkpointing_custom_sampler_without_native_tq(self):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.checkpointing["save_data_plane"] = False
        mc.async_rl.sampler = CustomSamplerConfig(
            target=f"{__name__}:_CheckpointingCustomSampler"
        )
        mc.data_plane["backend"] = "simple"

        with pytest.raises(
            ValueError,
            match=(
                "replay-checkpoint-capable sampler requires "
                "checkpointing.save_data_plane=true"
            ),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_warns_when_custom_sampler_cannot_recover_buffered_rollouts(
        self, patched_factories
    ):
        mc = _make_master_config()
        mc.checkpointing["enabled"] = True
        mc.checkpointing["save_data_plane"] = False
        mc.async_rl.sampler = CustomSamplerConfig(
            target=f"{__name__}:_NonCheckpointingCustomSampler"
        )
        mc.data_plane["backend"] = "simple"

        with pytest.warns(
            UserWarning, match="cannot recover completed buffered rollouts"
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        with pytest.raises(NotImplementedError, match="use_multiple_dataloader"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    @pytest.mark.parametrize(
        (
            "opd_enabled",
            "teacher_enabled",
            "adv_name",
            "use_nemo_gym",
            "match",
        ),
        [
            (
                False,
                False,
                "opd",
                True,
                "requires on_policy_distillation.enabled=true",
            ),
            (
                True,
                True,
                "grpo",
                True,
                "requires grpo.adv_estimator.name='opd'",
            ),
            (True, False, "opd", True, "non_colocated_teachers.enabled=true"),
            (True, True, "opd", False, "requires env.should_use_nemo_gym=true"),
        ],
    )
    def test_invalid_mopd_config_fails_before_allocating_resources(
        self,
        opd_enabled: bool,
        teacher_enabled: bool,
        adv_name: str,
        use_nemo_gym: bool,
        match: str,
        patched_factories,
    ):
        mc = _make_master_config()
        mc.env["should_use_nemo_gym"] = use_nemo_gym
        mc.grpo.adv_estimator = AdvEstimatorConfig(name=adv_name)
        mc.on_policy_distillation = OnPolicyDistillationConfig(
            enabled=opd_enabled,
            teacher_model_by_agent_name={"teacher": "/ckpt/teacher"},
            non_colocated_teachers={"enabled": teacher_enabled},
        )

        with pytest.raises(ValueError, match=match):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        patched_factories["_build_clusters"].assert_not_called()

    def test_mopd_reserves_before_models_and_initializes_teacher_last(
        self, patched_factories, monkeypatch
    ):
        mc = _make_master_config(env={"should_use_nemo_gym": True})
        mc.cluster = {"num_nodes": 3, "gpus_per_node": 8}
        mc.policy["generation"]["vllm_cfg"] = {
            "async_engine": True,
            "expose_http_server": True,
        }
        mc.policy["generation"].update(
            {"stop_strings": None, "stop_token_ids": None, "top_k": None}
        )
        mc.grpo.adv_estimator = AdvEstimatorConfig(name="opd")
        mc.on_policy_distillation = OnPolicyDistillationConfig(
            enabled=True,
            teacher_model_by_agent_name={"teacher": "/ckpt/teacher"},
            default_teacher_alias="teacher",
            non_colocated_teachers={"enabled": True},
        )
        events = []
        teacher_cluster = MagicMock(name="teacher_cluster")
        teacher_group = MagicMock(name="teacher_group")

        def reserve_teachers(*args, **kwargs):
            del args, kwargs
            events.append("reserve_teacher")
            return {"teacher": teacher_cluster}

        original_build_generation = patched_factories["_build_generation"].return_value

        def build_generation(*args, **kwargs):
            del args, kwargs
            events.append("build_generation")
            return original_build_generation

        def create_teachers(*args, **kwargs):
            del args, kwargs
            events.append("create_teacher")
            return {"teacher": teacher_group}, {"teacher": "teacher"}

        patched_factories["_build_generation"].side_effect = build_generation
        monkeypatch.setattr(
            sc_setup_mod.opd_module,
            "reserve_teacher_clusters",
            reserve_teachers,
        )
        monkeypatch.setattr(
            sc_setup_mod.opd_module,
            "create_teacher_worker_groups",
            create_teachers,
        )
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)
        monkeypatch.setattr(
            sc_setup_mod,
            "_spinup_gym",
            lambda **_kwargs: (MagicMock(name="nemo_gym_actor"), 0.0),
        )

        actor_args, timings = setup_single_controller(mc, MagicMock(pad_token_id=17))

        assert events == ["reserve_teacher", "build_generation", "create_teacher"]
        assert actor_args.teacher_worker_groups == {"teacher": teacher_group}
        assert actor_args.alias_to_group_alias == {"teacher": "teacher"}
        teacher_group.setup_data_plane.assert_called_once_with(mc.data_plane)
        assert timings.teacher_reservation_time_s is not None
        assert timings.teacher_model_init_time_s is not None

    @pytest.mark.parametrize(
        ("invalid_case", "match"),
        [
            ("min_groups", "must be >="),
            ("global_batch_size", "must equal policy.train_global_batch_size"),
            ("buffer_capacity", "required capacity"),
            (
                "deferred_routes_without_capture",
                "defer_routed_experts_to_policy requires",
            ),
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
        elif invalid_case == "deferred_routes_without_capture":
            mc.token_capture.defer_routed_experts_to_policy = True
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
        mc = _make_master_config()
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
        actor_args.dp_client.register_partition.assert_called_once()
        warmup = actor_args.dp_client.register_partition.call_args.kwargs
        assert warmup["partition_id"] == "rollout_data"
        assert set(SC_ROLLOUT_SCHEMA_FIELDS) <= set(warmup["fields"])
        assert "teacher_reference_logprobs" in warmup["fields"]
        assert warmup["num_samples"] == 16
        assert warmup["grpo_group_size"] == 2

    def test_reserves_topology_constrained_training_before_builds(
        self, patched_factories
    ):
        mc = _make_master_config(colocated=False)
        mc.cluster = {"num_nodes": 2, "gpus_per_node": 8, "segment_size": 1}
        mc.policy["generation"]["colocated"]["resources"] = {
            "gpus_per_node": 4,
            "num_nodes": 1,
        }
        train_cluster = patched_factories["_build_clusters"].return_value[0]
        events = []
        train_cluster.get_placement_groups.side_effect = lambda: events.append(
            "reserve_train"
        )
        original_build = patched_factories["_build_generation"].return_value

        def build_generation(*args, **kwargs):
            del args, kwargs
            events.append("build_generation")
            return original_build

        patched_factories["_build_generation"].side_effect = build_generation

        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert events[0] == "reserve_train"
        train_cluster.get_placement_groups.assert_called_once_with()

    def test_effort_levels_reach_the_rollout_manager(self, patched_factories):
        """env.nemo_gym.effort_levels is resolved into RolloutManager's kwarg.

        Asserted on the constructor rather than on ``_impl``: only the NeMo-Gym impl
        keeps the config, while the native impl absorbs it via ``**kwargs``.
        """
        mc = _make_master_config(
            env={
                "nemo_gym": {
                    "effort_levels": {
                        "low_weight": 1.0,
                        "low_penalty": 2.0,
                        "low_ub": 500,
                        "low_string": "<budget>",
                    }
                }
            }
        )

        with patch.object(sc_setup_mod, "RolloutManager") as mock_rollout_manager:
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        _, call_kwargs = mock_rollout_manager.call_args
        assert call_kwargs["effort_config"] == EffortLevelsConfig(
            low_weight=1.0, low_penalty=2.0, low_ub=500, low_string="<budget>"
        )

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({}, id="no_nemo_gym_section"),
            pytest.param({"nemo_gym": {}}, id="no_effort_levels_key"),
        ],
    )
    def test_rollout_manager_gets_no_effort_config_when_unset(
        self, env: dict, patched_factories
    ):
        """Shaping stays off unless env.nemo_gym.effort_levels is configured."""
        mc = _make_master_config(env=env)

        with patch.object(sc_setup_mod, "RolloutManager") as mock_rollout_manager:
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        _, call_kwargs = mock_rollout_manager.call_args
        assert call_kwargs["effort_config"] is None

    def test_router_replay_requires_routes_in_tq_buffer(self, patched_factories):
        mc = _make_master_config()
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
        """When should_use_nemo_gym is True the nemo-gym actor is spun up and stored."""
        mc = _make_master_config(backend="vllm")
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
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
            patch.object(
                sc_setup_mod, "spinup_nemo_gym_actor", return_value=fake_gym_actor
            ) as mock_spinup,
            patch.object(sc_setup_mod, "router_replay_enabled", return_value=False),
        ):
            tokenizer = MagicMock(pad_token_id=0)
            actor_args, _ = setup_single_controller(mc, tokenizer)

        mock_spinup.assert_called_once_with(
            env_configs=mc.env,
            base_urls=patched_factories["fake_gen"].dp_openai_server_base_urls,
            model_name="test-model",
            # Reaches the actor once, at spinup, rather than riding along with every
            # run_rollouts call.
            tokenizer=tokenizer,
            enable_router_replay=False,
            routed_experts_dtype="int16",
            use_fastokens=False,
            token_capture=None,
        )
        assert actor_args.env_handles["nemo_gym"] is fake_gym_actor

    def test_token_capture_always_creates_finalizer_actor_pool(self, patched_factories):
        mc = _make_master_config(colocated=False, backend="vllm")
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
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
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

    def test_setup_timing_populated_for_noncolocated_vllm(self, patched_factories):
        """Non-colocated vLLM records every per-phase field."""
        mc = _make_master_config(colocated=False, backend="vllm")

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
        # parallel_wall_time_s / parallel_init_enabled are grpo.py-only.
        assert metrics.parallel_wall_time_s is None
        assert metrics.parallel_init_enabled is None
        # Reserve/load split is populated on the gym-on path only.
        assert metrics.generation_init_reserve_time_s is None
        assert metrics.generation_init_load_time_s is None

    def test_setup_timing_backend_agnostic_for_sglang(self, patched_factories):
        """SC uses the backend-agnostic generation_init_time_s regardless of backend."""
        mc = _make_master_config(backend="sglang")

        _, metrics = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert metrics.generation_init_time_s is not None

    def test_nemo_gym_uses_deferred_vllm_load(self, patched_factories):
        """NeMo-Gym path reserves vLLM ports up-front and finishes the load afterwards."""
        mc = _make_master_config(backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)

        with (
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
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
        mc = _make_master_config(backend="vllm")
        mc.policy["generation"]["model_name"] = "test-model"
        mc.policy["generation"]["stop_strings"] = None
        mc.policy["generation"]["stop_token_ids"] = None
        mc.policy["generation"]["top_k"] = None
        patched_factories["setup_response_data"].return_value = (list(range(8)), None)

        with (
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
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
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
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

    def test_nemo_gym_generation_init_time_includes_reserve_time(
        self, patched_factories
    ):
        """generation_init_time_s folds in the deferred-VllmGeneration reserve time.

        With gym on, _build_generation(defer_model_load=True) does worker-group
        spawn + port bind (no weight load). That elapsed time has to end up in
        generation_init_time_s alongside the deferred-load elapsed; otherwise
        gym-on runs undercount generation setup by the worker-group span. The
        reserve/load split is also exposed for overlap analysis.
        """
        mc = _make_master_config(colocated=False, backend="vllm")
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
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
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
        mc = _make_master_config(backend=backend)
        patched_factories["setup_response_data"].return_value = (
            list(range(8)),
            None,
        )

        with (
            patch.object(sc_setup_mod, "should_use_nemo_gym", return_value=True),
            patch.object(sc_setup_mod, "spinup_nemo_gym_actor") as mock_spinup,
            pytest.raises(NotImplementedError, match="vllm"),
        ):
            setup_single_controller(mc, MagicMock(pad_token_id=0))
        mock_spinup.assert_not_called()


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
        policy.load_data_plane_checkpoint.side_effect = lambda checkpoint_dir: (
            events.append("load") or _native_tq_metadata()
        )
        patched_factories["build_data_plane_client"].side_effect = (
            lambda *args, **kwargs: (
                events.append("build") or MagicMock(name="dp_client")
            )
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

    def test_loads_authoritative_tq_checkpoint_when_metadata_file_exists(
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

    def test_checkpoint_without_replay_artifacts_does_not_load_tq(
        self, tmp_path, capsys
    ):
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
        output = capsys.readouterr().out
        assert REPLAY_BUFFER_METADATA_FILENAME in output
        assert "matching TQ checkpoint will not be loaded" in output
        assert "dataloader cursor is still restored" in output
        assert "buffered at checkpoint time will be discarded" in output

    def test_metadata_file_requires_matching_tq_directory(self, tmp_path):
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
