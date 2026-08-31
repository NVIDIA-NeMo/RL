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
"""Driver-side factory for the SingleController (async-RL) training path.

setup builds the full SingleControllerActorArgs on the driver and the caller passes it to
SingleControllerActor.remote. Everything lives on the driver because driver-side
TQPolicy owns the worker group directly — running this inside another Ray actor nests
runtime_envs and breaks Ray's resource resolution (see the PR #2692 follow-up).
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    DataPlaneCheckpointMetadata,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    sampler_supports_buffer_checkpoint,
)
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _get_effort_config,
    _get_grpo_save_state,
)
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.loss.loss_functions import MseValueLossFn
from nemo_rl.algorithms.metric_utils import (
    SetupTimingMetrics,
    print_setup_timing_summary,
)
from nemo_rl.algorithms.ppo import MasterConfig as PPOMasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    algo_config,
    is_ppo_run,
    validate_single_controller_config,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import load_dataloader_state, setup_response_data
from nemo_rl.data_plane import (
    DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
    DataPlaneClient,
    build_data_plane_client,
    data_plane_supports_checkpointing,
)
from nemo_rl.data_plane.schema import (
    SC_ROLLOUT_SCHEMA_FIELDS,
    fields_with_optional_routed_experts,
)
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
    prepare_segment_topology,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import should_use_nemo_gym, spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import (
    RolloutManager,
    RolloutRetryPolicy,
    RolloutTimeouts,
)
from nemo_rl.experience.rollouts import (
    get_nemo_gym_thinking_tags,
    resolve_reward_penalty_config,
    should_mask_flagged_samples,
)
from nemo_rl.models.generation import resolve_generation_class
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    HealthyShardSelector,
)
from nemo_rl.models.generation.generation_router import (
    GenerationRouterActor,
    GenerationRouterImpl,
)
from nemo_rl.models.generation.interfaces import (
    resolve_routed_experts_dtype_name_for_model,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
)
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.models.value.tq_value import TQValue
from nemo_rl.utils.checkpoint import (
    CheckpointManager,
    validate_warm_start_checkpoint,
)
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


@dataclass
class SingleControllerActorArgs:
    """All inputs SingleControllerActor needs, built driver-side by setup_single_controller().

    Passed as a single arg to SingleControllerActor.remote so the actor's __init__ does
    no construction work — every heavy object is cloudpickled in.
    """

    gen_handle: Any
    trainer_handle: Any  # driver-side TQPolicy
    env_handles: dict[str, EnvironmentInterface]
    train_cluster: RayVirtualCluster
    inference_cluster: RayVirtualCluster
    dp_client: DataPlaneClient
    dataloader: StatefulDataLoader
    weight_synchronizer: WeightSynchronizer
    advantage_estimator: Any
    loss_fn: LossFunction
    rollout_manager: RolloutManager
    tq_buffer: TQReplayBuffer
    partition_id: str
    save_state: GRPOSaveState
    last_checkpoint_path: Optional[str]
    data_plane_checkpoint_metadata: Optional[DataPlaneCheckpointMetadata] = None
    # None when async_rl.generation_fleet_health is disabled.
    fleet_monitor: Optional[GenerationFleetHealth] = None
    # None unless async_rl.generation_router is enabled.
    generation_router: Optional[ray.actor.ActorHandle[GenerationRouterImpl]] = None
    # Populated only for text MOPD. Aliases may outnumber worker groups when
    # multiple agents share one deduplicated teacher checkpoint.
    teacher_worker_groups: Optional[dict[str, Any]] = None
    alias_to_group_alias: Optional[dict[str, str]] = None
    # None on a GRPO run. Both are set together on the PPO path: the critic and
    # the MSE loss it trains under.
    value_handle: Optional[TQValue] = None
    value_loss_fn: Optional[LossFunction] = None


def _maybe_restore_native_data_plane_checkpoint(
    policy: TQPolicy,
    *,
    last_checkpoint_path: Optional[str],
    save_state: GRPOSaveState,
    partition_id: str,
    sampler_name: str,
) -> Optional[DataPlaneCheckpointMetadata]:
    """Load and validate an authoritative native TQ checkpoint when present.

    The replay metadata file is the format marker. Checkpoints without
    any replay artifact resume trainer state with an empty replay buffer;
    legacy tensor-bearing replay files are rejected rather than silently
    ignored. Rollout tensors are never serialized into a controller-side
    replay checkpoint.
    """
    if last_checkpoint_path is None:
        return None
    checkpoint_path = Path(last_checkpoint_path)
    replay_metadata_path = checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME
    if not replay_metadata_path.is_file():
        legacy_replay_path = checkpoint_path / LEGACY_REPLAY_BUFFER_FILENAME
        if legacy_replay_path.is_file():
            raise RuntimeError(
                "Checkpoint contains legacy replay_buffer.pt state, which "
                "predates authoritative native TQ replay recovery. Resume it "
                "with the older implementation or explicitly start without "
                "restoring buffered rollouts."
            )
        print(
            f"⚠️ No {REPLAY_BUFFER_METADATA_FILENAME} found in checkpoint "
            f"{checkpoint_path}. The matching TQ checkpoint will not be loaded, "
            "and recovery will use an empty replay buffer. The dataloader cursor "
            "is still restored, so any prompt groups buffered at checkpoint time "
            "will be discarded.",
            flush=True,
        )
        return None

    data_plane_path = checkpoint_path / DATA_PLANE_CHECKPOINT_DIR
    if not data_plane_path.is_dir():
        raise FileNotFoundError(
            "Metadata-only replay checkpoint requires a matching native TQ "
            f"checkpoint at {data_plane_path}"
        )

    print(f"📦 Restoring native TQ checkpoint: {data_plane_path}", flush=True)
    raw_metadata = policy.load_data_plane_checkpoint(data_plane_path)
    if not isinstance(raw_metadata, dict):
        raise TypeError(
            "Native TQ checkpoint load must return a metadata dictionary, "
            f"got {type(raw_metadata).__name__}"
        )
    metadata = cast(DataPlaneCheckpointMetadata, raw_metadata)
    expected_values: DataPlaneCheckpointMetadata = {
        "data_plane_checkpoint_schema_version": (DATA_PLANE_CHECKPOINT_SCHEMA_VERSION),
        "single_controller_train_steps": save_state.current_step,
        "single_controller_trainer_version": (
            save_state.trainer_version
            if save_state.trainer_version is not None
            else save_state.current_step
        ),
        "single_controller_epoch": save_state.current_epoch,
        "partition_id": partition_id,
        "sampler_name": sampler_name,
        "mode": "authoritative",
        "replay_metadata_schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    }
    mismatches = {
        key: {"checkpoint": metadata.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Native TQ checkpoint metadata does not match the trainer "
            f"checkpoint: {mismatches}"
        )
    manifest_digest = metadata.get("replay_manifest_digest")
    if not isinstance(manifest_digest, str) or not manifest_digest:
        raise ValueError(
            "Native TQ checkpoint metadata is missing replay_manifest_digest"
        )
    group_count = metadata.get("replay_group_count")
    if not isinstance(group_count, int) or group_count < 0:
        raise ValueError(
            "Native TQ checkpoint metadata has invalid replay_group_count: "
            f"{group_count!r}"
        )
    print(
        f"📦 Native TQ checkpoint restored and validated: groups={group_count}",
        flush=True,
    )
    return metadata


def _non_colocated_teacher_node_count(master_config: MasterConfig) -> int:
    """Validate teacher GPU geometry and return its deduplicated node count."""
    if not opd_module.is_non_colocated_teachers_enabled(master_config):
        return 0

    # Lazy to preserve teacher_worker_group's existing import cycle boundary:
    # that module imports the OPD config schemas.
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    teacher_configs = create_teacher_configs_from_opd_config(
        opd_module._opd_cfg(master_config)
    )
    cluster_gpus_per_node = master_config.cluster["gpus_per_node"]
    for teacher_config in teacher_configs:
        if teacher_config.gpus_per_node > cluster_gpus_per_node:
            raise ValueError(
                f"OPD teacher {teacher_config.alias!r} requests "
                f"gpus_per_node={teacher_config.gpus_per_node}, which exceeds "
                f"cluster.gpus_per_node={cluster_gpus_per_node}."
            )
    return sum(config.num_nodes for config in teacher_configs)


def _build_clusters(
    master_config: MasterConfig,
) -> tuple[
    RayVirtualCluster,
    RayVirtualCluster,
    Optional[dict[str, tuple[str, int]]],
]:
    """Allocate student clusters while leaving validated nodes for teachers.

    The colocated branch is unreachable on a real run -- validation rejects
    colocated.enabled=true -- and is kept for when SC can support that mode.
    """
    cluster_config = master_config.cluster
    generation_config = master_config.policy["generation"]
    colocated = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    num_nodes = cluster_config["num_nodes"]
    gpus_per_node = cluster_config["gpus_per_node"]
    segment_size = cluster_config.get("segment_size")
    port_range_low = cluster_config.get("master_port_range_low")
    port_range_high = cluster_config.get("master_port_range_high")
    teacher_nodes = _non_colocated_teacher_node_count(master_config)
    policy_nodes = num_nodes - teacher_nodes
    if policy_nodes <= 0:
        raise ValueError(
            "cluster.num_nodes must leave at least one node for the student after "
            f"reserving {teacher_nodes} non-colocated teacher node(s); got "
            f"cluster.num_nodes={num_nodes}."
        )

    # Worker groups sharing the training GPUs: the policy, plus the critic on
    # the PPO path.
    train_worker_groups = 2 if is_ppo_run(master_config) else 1

    if colocated:
        # Policy (+ critic) + generation share GPUs — one cluster.
        node_constraints, remaining_ids, topology = prepare_segment_topology(
            segment_size,
            policy_nodes,
            role="policy",
        )
        teacher_topology = (
            {node_id: topology[node_id] for node_id in remaining_ids}
            if segment_size is not None
            else None
        )
        cluster = RayVirtualCluster(
            name="sc_policy_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * policy_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=(
                train_worker_groups
                if backend == "megatron"
                else train_worker_groups + 1
            ),
            port_range_low=port_range_low,
            port_range_high=port_range_high,
            segment_size=segment_size,
            node_resource_constraints=node_constraints,
        )
        return cluster, cluster, teacher_topology

    # Non-colocated: split node into train + inference clusters.
    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
    if inference_gpus_per_node is None:
        raise ValueError(
            "Non-colocated generation requires "
            "policy.generation.colocated.resources.gpus_per_node."
        )
    inference_nodes = inference_resources["num_nodes"] or 1
    if policy_nodes == 1:
        train_gpus_per_node = gpus_per_node - inference_gpus_per_node
        train_nodes = 1
        assert train_gpus_per_node > 0, (
            f"Not enough GPUs for training: {gpus_per_node} - {inference_gpus_per_node} = {train_gpus_per_node}"
        )
    else:
        train_gpus_per_node = gpus_per_node
        train_nodes = policy_nodes - inference_nodes
        assert train_nodes > 0, (
            f"train_nodes must be > 0: {policy_nodes} - {inference_nodes} = {train_nodes}"
        )

    train_constraints = None
    inference_constraints = None
    train_segment_size = None
    inference_segment_size = None
    teacher_topology = None
    if segment_size is not None:
        if policy_nodes == 1:
            # Train and inference intentionally split one physical node by GPU.
            shared_constraints, remaining_ids, topology = prepare_segment_topology(
                segment_size,
                1,
                role="student",
            )
            train_constraints = shared_constraints
            inference_constraints = shared_constraints
            train_segment_size = segment_size
            inference_segment_size = segment_size
            teacher_topology = {node_id: topology[node_id] for node_id in remaining_ids}
        else:
            train_constraints, remaining_ids, topology = prepare_segment_topology(
                segment_size,
                train_nodes,
                role="training",
            )
            train_segment_size = segment_size
            remaining_topology = {
                node_id: topology[node_id] for node_id in remaining_ids
            }
            generation_config_dict = cast(dict[str, Any], generation_config)
            if backend == "vllm":
                vllm_cfg = generation_config_dict["vllm_cfg"]
                gpus_per_instance = vllm_cfg["tensor_parallel_size"] * vllm_cfg.get(
                    "pipeline_parallel_size", 1
                )
            elif backend == "sglang":
                gpus_per_instance = generation_config_dict["sglang_cfg"].get(
                    "gpus_per_server", 1
                )
            elif backend == "megatron":
                gpus_per_instance = MegatronGeneration.nvlink_domain_span(
                    master_config.policy
                )
            else:
                raise ValueError(
                    "single_controller_utils.setup only supports vllm, sglang, "
                    f"or megatron generation; got {backend!r}"
                )
            nodes_per_instance = (
                gpus_per_instance + inference_gpus_per_node - 1
            ) // inference_gpus_per_node
            if inference_nodes % nodes_per_instance == 0:
                inference_segment_size = nodes_per_instance
                (
                    inference_constraints,
                    inference_remaining_ids,
                    _,
                ) = prepare_segment_topology(
                    inference_segment_size,
                    inference_nodes,
                    topology=remaining_topology,
                    role="inference",
                )
                teacher_topology = {
                    node_id: topology[node_id] for node_id in inference_remaining_ids
                }
            else:
                print(
                    f"  ⚠ inference_nodes={inference_nodes} is not divisible by "
                    f"nodes_per_instance={nodes_per_instance}; skipping inference "
                    "topology constraints",
                    flush=True,
                )
                teacher_topology = remaining_topology

    train_cluster = RayVirtualCluster(
        name="sc_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=train_worker_groups,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
        segment_size=train_segment_size,
        node_resource_constraints=train_constraints,
    )
    inference_cluster = RayVirtualCluster(
        name="sc_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
        segment_size=inference_segment_size,
        node_resource_constraints=inference_constraints,
    )
    return train_cluster, inference_cluster, teacher_topology


def _build_generation(
    inference_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    *,
    defer_model_load: bool = False,
    reserved_http_server_port: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    processor: Optional[AutoProcessor] = None,
) -> tuple[Any, float]:
    """Spin up the generation backend (vLLM, SGLang, or Megatron).

    Args:
        inference_cluster: Ray virtual cluster the generation workers run on.
        master_config: SC MasterConfig.
        defer_model_load: If True (for the NeMo-Gym flow), reserve OpenAI server URLs without loading weights; caller runs gen.load_and_start() later (vLLM only).
        reserved_http_server_port: OpenAI server port pre-published to NeMo-Gym (Megatron only).
        tokenizer: Tokenizer for the dedicated Megatron inference policy (Megatron only).
        processor: Optional AutoProcessor for VLM paths (Megatron only).

    Returns:
        A tuple of (generation object, wall time spent in this call). The
        generation object is a VllmGeneration, SGLangGeneration, or MegatronGeneration.
    """
    t0 = time.perf_counter()
    generation_config = master_config.policy["generation"]
    generation_config["model_name"] = master_config.policy["model_name"]
    backend = generation_config["backend"]

    if backend == "vllm":
        vllm_config = cast(VllmConfig, generation_config)
        vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
            master_config.policy.get("hf_config_overrides", {})
        )
        configure_vllm_for_router_replay(master_config.policy)
        gen = VllmGeneration(
            cluster=inference_cluster,
            config=vllm_config,
            defer_model_load=defer_model_load,
        )

    elif backend == "sglang":
        assert not defer_model_load, (
            "defer_model_load is only supported for the vllm backend"
        )
        sglang_config = cast(SGLangConfig, generation_config)
        sglang_config["sglang_cfg"].setdefault(
            "model_path", master_config.policy["model_name"]
        )
        gen = SGLangGeneration(
            cluster=inference_cluster,
            sglang_cfg=sglang_config,
        )

    elif backend == "megatron":
        assert not defer_model_load, (
            "defer_model_load is only supported for the vllm backend"
        )
        assert tokenizer is not None, "Megatron generation requires a tokenizer"
        # Non-colocated only (colocated is rejected at config validation).
        # The inference and trainer policies build in parallel;
        # the inference engine only becomes live at the initial refit, which delivers weights.
        gen = MegatronGeneration(
            config=master_config.policy,
            tokenizer=tokenizer,
            cluster=inference_cluster,
            reserved_http_server_port=reserved_http_server_port,
            processor=processor,
            skip_weight_load=True,
        )

    else:
        raise ValueError(
            "single_controller_utils.setup only supports vllm, sglang, or megatron "
            f"generation; got {backend!r}"
        )

    if not defer_model_load:
        gen.finish_generation()

    return gen, time.perf_counter() - t0


def _finish_deferred_generation(generation: Any) -> tuple[Any, float]:
    """Finish loading and starting the deferred generation.

    Args:
        generation: The deferred generation object.

    Returns:
        A tuple of (finished generation object, wall time spent in this call).
    """
    t0 = time.perf_counter()
    generation.load_and_start()
    generation.finish_generation()
    return generation, time.perf_counter() - t0


def _build_trainer(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer,
    processor,
    *,
    weights_path: Optional[Path],
    optimizer_path: Optional[Path],
) -> tuple[Any, float]:
    """Build the TQ-mediated trainer (driver-side TQPolicy).

    Args:
        train_cluster: Ray virtual cluster the trainer workers run on.
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        weights_path: Checkpointed policy weights to resume from, or None.
        optimizer_path: Checkpointed optimizer state to resume from, or None.

    Returns:
        A tuple of (TQPolicy trainer, wall time spent in this call).
    """
    t0 = time.perf_counter()
    loss_config = master_config.loss_fn
    init_reference_model = loss_config.reference_policy_kl_penalty > 0
    trainer = TQPolicy(
        cluster=train_cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=init_reference_model,
        dp_cfg=master_config.data_plane,
    )
    return trainer, time.perf_counter() - t0


def _build_value(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    weights_path: Optional[Path],
    optimizer_path: Optional[Path],
) -> tuple[TQValue, float]:
    """Build the TQ-mediated PPO critic (driver-side TQValue).

    Args:
        train_cluster: Ray virtual cluster the critic shares with the trainer.
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the value model.
        weights_path: Checkpointed value weights to resume from, or None.
        optimizer_path: Checkpointed value optimizer state to resume from, or None.

    Returns:
        A tuple of (TQValue critic, wall time spent in this call).
    """
    t0 = time.perf_counter()
    value = TQValue(
        cluster=train_cluster,
        config=master_config.value,
        tokenizer=tokenizer,
        name_prefix="lm_value",
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        dp_cfg=master_config.data_plane,
    )
    return value, time.perf_counter() - t0


def _spinup_gym(
    master_config: MasterConfig,
    base_urls: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Any, float]:
    """Spin up the NeMo-Gym actor against the reserved vLLM URLs.

    Args:
        master_config: SC MasterConfig.
        base_urls: Reserved vLLM OpenAI server URLs.
        tokenizer: Installed on the actor at spinup rather than passed per rollout
            call. See NemoGym.set_tokenizer.

    Returns:
        A tuple of (NeMo-Gym actor, wall time spent in this call).
    """
    t0 = time.perf_counter()
    policy_config = master_config.policy
    generation_config = policy_config["generation"]
    enable_router_replay = router_replay_enabled(policy_config)
    routed_experts_dtype = (
        resolve_routed_experts_dtype_name_for_model(generation_config["model_name"])
        if enable_router_replay
        else "int16"
    )
    actor = spinup_nemo_gym_actor(
        env_configs=master_config.env,
        base_urls=base_urls,
        model_name=generation_config["model_name"],
        tokenizer=tokenizer,
        enable_router_replay=enable_router_replay,
        routed_experts_dtype=routed_experts_dtype,
        use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
    )
    return actor, time.perf_counter() - t0


def _generation_max_seq_len(generation_config) -> int:
    """Return the per-backend max sequence length.

    vllm uses vllm_cfg.max_model_len; sglang uses sglang_cfg.context_length;
    megatron uses mcore_generation_config.max_model_len.
    """
    backend = generation_config["backend"]
    if backend == "vllm":
        return generation_config["vllm_cfg"]["max_model_len"]
    if backend == "sglang":
        return generation_config["sglang_cfg"]["context_length"]
    if backend == "megatron":
        return generation_config["mcore_generation_config"]["max_model_len"]
    raise ValueError(f"Unknown generation backend: {backend!r}")


def _clamp_max_num_steps(
    master_config: MasterConfig, dataloader: StatefulDataLoader
) -> None:
    """Clamp max_num_steps to max_num_epochs * len(dataloader)."""
    algo_cfg = algo_config(master_config)
    max_num_epochs = algo_cfg.max_num_epochs
    if max_num_epochs is None:
        return
    algo_cfg.max_num_steps = min(
        algo_cfg.max_num_steps,
        max_num_epochs * len(dataloader),
    )


def _maybe_inject_megatron_train_iters(master_config: MasterConfig) -> None:
    """Set train_iters from max_num_steps after its dataloader clamp."""
    algo_cfg = algo_config(master_config)
    is_ppo = is_ppo_run(master_config)
    # train_iters is a scheduler-tick budget, and each PPO epoch steps both
    # optimizers once, so the configured warmup/decay horizon has to be scaled.
    ppo_epochs = algo_cfg.ppo_epochs if is_ppo else 1
    train_iters = algo_cfg.max_num_steps * ppo_epochs

    # policy
    policy_config = master_config.policy
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        policy_config["megatron_cfg"]["train_iters"] = train_iters

    # value
    if not is_ppo:
        return
    value_config = master_config.value
    if value_config.get("megatron_cfg", {}).get("enabled", False):
        value_config["megatron_cfg"]["train_iters"] = train_iters  # type: ignore[index]


def _maybe_attach_fleet_health(
    generation: Any, master_config: MasterConfig
) -> Optional[GenerationFleetHealth]:
    """Route generation through fleet health, when it is enabled and supported.

    Returns:
        The monitor the SingleController should drive, or None when fleet health is
        disabled or the backend does not support it.
    """
    fleet_config = master_config.async_rl.generation_fleet_health
    if not fleet_config.enabled:
        return None

    monitor = GenerationFleetHealth(
        shard_count=generation.worker_group.dp_size,
        policy=FleetHealthPolicy(
            unhealthy_threshold=fleet_config.unhealthy_threshold,
            healthy_threshold=fleet_config.healthy_threshold,
            max_restart_attempts_per_shard=fleet_config.max_restart_attempts_per_shard,
            min_healthy_shards=fleet_config.min_healthy_shards,
        ),
        # All-None means the backend reports no OpenAI servers (async_engine=false).
        # Health tracking works fine without URLs -- only the router push needs them --
        # so drop the list rather than letting the shard-count check reject it with a
        # message that reads like an internal bug.
        base_urls=_shard_base_urls(generation),
    )
    # Unconditional: GenerationInterface declares attach_fleet_health, so a backend that
    # does not support it raises its own NotImplementedError naming itself.
    generation.attach_fleet_health(monitor, HealthyShardSelector(monitor=monitor))
    return monitor


def _shard_base_urls(generation: Any) -> Optional[list[Optional[str]]]:
    """Per-shard OpenAI base URLs, or None when the backend exposes no servers."""
    urls = list(generation.dp_openai_server_base_urls or [])
    if not any(urls):
        return None
    return urls


def _maybe_start_generation_router(
    base_urls: list[Optional[str]], master_config: MasterConfig
) -> Any:
    """Start the NeMo-Gym-facing router, if enabled.

    Args:
        base_urls: OpenAI server URLs for the router.
        master_config: SingleController MasterConfig.

    Returns:
        The router actor handle, or None when the router is disabled.
    """
    router_config = master_config.async_rl.generation_router
    if not router_config.enabled:
        return None

    if not master_config.async_rl.generation_fleet_health.enabled:
        # Legitimate, but the operator should know what they are not getting: nothing
        # ever calls set_serving_backends, so the router stays health-blind for the run.
        # It still delivers the stable URL Gym never re-resolves, a backend deadline Gym
        # sets nowhere, and least-outstanding balancing -- just no failover.
        print(
            "⚠️  async_rl.generation_router.enabled=true with generation_fleet_health.enabled=false: "
            "the router will never receive a serving-set update, so it cannot route "
            "around a dead shard. Enable async_rl.generation_fleet_health for failover.",
            flush=True,
        )

    backend_urls = [url for url in (base_urls or []) if url]
    if not backend_urls:
        raise ValueError(
            "async_rl.generation_router.enabled=true requires generation backends that "
            "expose OpenAI-compatible servers; none were reported. This needs the vllm "
            "backend with async_engine and expose_http_server enabled."
        )

    # Reserved once and passed in, so Ray recreating a restarted actor rebinds the same
    # address. NeMo-Gym holds this URL for the life of the run and never re-resolves it.
    port = _get_free_port_local(
        router_config.port_range_low, router_config.port_range_high
    )
    router = GenerationRouterActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        )
    ).remote(
        backend_urls=backend_urls,
        host=_get_node_ip_local(),
        port=port,
        backend_timeout_s=router_config.backend_timeout_s,
        connect_timeout_s=router_config.connect_timeout_s,
        no_healthy_backend_status=router_config.no_healthy_backend_status,
        # Only a monitor-driven run ever pushes membership, and the router's reflex drop
        # of a failing backend is only safe because a later push restores it. Without
        # one, arming the reflex would retire backends permanently.
        health_managed=master_config.async_rl.generation_fleet_health.enabled,
    )
    # Resolve the URL now so the driver fails here rather than inside Gym if the actor
    # could not start. The router binds its socket inside __init__, so a port conflict
    # fails actor construction and surfaces here with the port in the traceback.
    base_url = ray.get(router.base_url.remote())
    print(f"📡 Policy router fronting {len(backend_urls)} backend(s) at {base_url}")
    return router


def _build_advantage_estimator(master_config: MasterConfig) -> Any:
    """Build the advantage estimator from whichever algorithm's factory applies."""
    if is_ppo_run(master_config):
        # TODO(#2625): raw_reward passes this factory but yields no returns, so
        # the critic train would then fetch a column nobody wrote.
        from nemo_rl.algorithms.ppo import _create_advantage_estimator

        return _create_advantage_estimator(cast(PPOMasterConfig, master_config))
    else:
        from nemo_rl.algorithms.grpo import _create_advantage_estimator

        return _create_advantage_estimator(cast(GRPOMasterConfig, master_config))


def _build_retry_policy(master_config: MasterConfig) -> RolloutRetryPolicy:
    """Translate ``async_rl.rollout_failure`` into the rollout layer's policy object."""
    failure_config = master_config.async_rl.rollout_failure
    return RolloutRetryPolicy(
        max_infra_attempts=failure_config.max_infra_attempts_per_prompt,
        max_data_attempts=failure_config.max_data_attempts_per_prompt,
        backoff_base_s=failure_config.backoff_base_s,
        max_backoff_s=failure_config.max_backoff_s,
        max_skipped_prompts=failure_config.max_skipped_prompts,
        max_consecutive_dropped_prompts=failure_config.max_consecutive_dropped_prompts,
        max_gym_row_attempts=failure_config.nemo_gym.max_row_attempts,
    )


def setup_single_controller(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    partition_id: str = "rollout_data",
) -> tuple[SingleControllerActorArgs, SetupTimingMetrics]:
    """Build the full SC actor args driver-side.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        A tuple of (pre-built SC actor args, driver-side per-phase timings
        logged by the SC actor).
    """
    validate_single_controller_config(master_config)
    resolved_reward_penalty_config = resolve_reward_penalty_config(
        master_config.reward_penalties,
        tokenizer,
        thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
    )

    # short names for config sections
    algo_cfg = algo_config(master_config)
    dp_config = master_config.data_plane
    policy_config = master_config.policy
    generation_config = policy_config["generation"]
    data_config = master_config.data

    # Every nccl_reshard precondition, checked once, here, before any GPU work.
    #
    # This guard existed but had exactly one production caller -- grpo.setup -- which the
    # single-controller path does not go through: run_grpo_single_controller goes straight
    # to setup_single_controller. So on SC none of it was enforced, and a config violating
    # e.g. colocated.enabled or enable_eplb got as far as the first refit before anything
    # noticed. This PR makes that worse rather than better: recovery rebuilds the reshard
    # communicators, so a bad config now has a second, later chance to fail.
    #
    # Deliberately after validate_single_controller_config, so the SC-specific errors a
    # reader is more likely to have caused come first.
    if generation_config.get("refit_transport") == "nccl_reshard":
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            check_nccl_reshard_refit_support,
        )

        check_nccl_reshard_refit_support(master_config)

    if algo_cfg.val_period > 0 or algo_cfg.val_at_start or algo_cfg.val_at_end:
        raise NotImplementedError(
            "SingleController doesn't support validation now, will support "
            "later. Set val_period=0, val_at_start=false, val_at_end=false."
        )
    if dp_config is None or not dp_config.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )
    data_plane_checkpointing_supported = data_plane_supports_checkpointing(dp_config)
    if (
        master_config.checkpointing.get("save_data_plane")
        and not data_plane_checkpointing_supported
    ):
        raise NotImplementedError(
            "SingleController data-plane checkpointing is not supported for "
            f"data_plane.backend={dp_config['backend']!r}."
        )
    if master_config.checkpointing["enabled"]:
        sampler_supports_replay_recovery = sampler_supports_buffer_checkpoint(
            master_config.async_rl.sampler
        )
        if sampler_supports_replay_recovery and not master_config.checkpointing.get(
            "save_data_plane"
        ):
            error_message = (
                "SingleController checkpointing with a replay-checkpoint-capable "
                "sampler requires checkpointing.save_data_plane=true so "
                "completed, unconsumed rollouts are recoverable."
            )
            if not data_plane_checkpointing_supported:
                error_message += (
                    f" The configured data_plane.backend={dp_config['backend']!r} "
                    "does not support data-plane checkpointing; use "
                    "data_plane.backend='simple' or set "
                    "checkpointing.enabled=false."
                )
            raise ValueError(error_message)
        if not sampler_supports_replay_recovery:
            warnings.warn(
                f"Sampler {master_config.async_rl.sampler.name!r} cannot recover "
                "completed buffered rollouts. On resume, the dataloader cursor "
                "is restored while buffered prompt groups are discarded.",
                UserWarning,
                stacklevel=2,
            )

    assert generation_config is not None, (
        "single_controller_utils.setup requires policy.generation in master_config"
    )

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )
    if opd_module.is_opd_enabled(master_config) and processor is not None:
        raise NotImplementedError(
            "SingleController MOPD currently supports text-only teacher inputs. "
            "Use the legacy controller for multimodal MOPD."
        )

    checkpointing_pretrained = master_config.checkpointing.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    set_seed(algo_cfg.seed)

    # ==========================
    # Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    loaded_state = cast(
        Optional[dict[str, Any]], checkpointer.load_training_info(last_checkpoint_path)
    )
    save_state = _get_grpo_save_state(loaded_state)
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    if is_ppo_run(master_config):
        # Only a fresh run reads this; a resume ignores it and restores the critic
        # from its own checkpoint, so the key can stay in the config.
        warm_start = master_config.ppo.warm_start_value_checkpoint
        if last_checkpoint_path is None and warm_start is not None:
            validate_warm_start_checkpoint(warm_start)
            print(f"🔥 Warm-starting the value model from {warm_start}")
        value_weights_path, value_optimizer_path = checkpointer.get_resume_paths(
            last_checkpoint_path or warm_start,
            model_component="value",
        )

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    # TODO: add validate dataset wiring.
    use_nemo_gym = should_use_nemo_gym(master_config)
    if use_nemo_gym and generation_config["backend"] not in ("vllm", "megatron"):
        raise NotImplementedError(
            "SC NeMo-Gym integration currently supports the vllm and megatron backends only; got "
            f"{generation_config['backend']!r}"
        )
    # Backend settings checks are pure config: run them before anything builds.
    resolve_generation_class(generation_config).validate_settings(master_config)
    if use_nemo_gym:
        # NeMo-Gym creates the env actor outside setup_response_data; we wire
        # it in after generation is up (it needs the OpenAI server URLs).
        response_data = setup_response_data(tokenizer, data_config, env_configs=None)
        assert len(response_data) == 2
        dataset, _val_dataset = response_data
        env_handles: dict[str, EnvironmentInterface] = {}
    else:
        response_data = setup_response_data(
            tokenizer, data_config, env_configs=master_config.env
        )
        assert len(response_data) == 4
        dataset, _val_dataset, env_handles, _val_env_handles = response_data
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=algo_cfg.num_prompts_per_step,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None:
        print(f"📦 Restoring dataloader state from checkpoint: {last_checkpoint_path}")
        load_dataloader_state(dataloader, last_checkpoint_path, data_config)

    _clamp_max_num_steps(master_config, dataloader)
    _maybe_inject_megatron_train_iters(master_config)

    # ==========================
    # Setup Clusters & Workers
    # ==========================
    setup_start_time = time.perf_counter()
    setup_timing_metrics = SetupTimingMetrics()

    # Create clusters
    train_cluster, inference_cluster, teacher_segment_topology = _build_clusters(
        master_config
    )
    colocated = generation_config["colocated"]["enabled"]
    segment_size = getattr(master_config, "cluster", {}).get("segment_size")

    # Claim constrained training nodes before unconstrained inference or Gym
    # tasks can consume them. This matters when inference topology alignment
    # falls back while the training cluster remains topology-constrained.
    if not colocated and segment_size is not None:
        train_cluster.get_placement_groups()

    # Claim teacher placement groups before deferred generation starts NeMo-Gym,
    # whose resource servers may otherwise opportunistically consume those GPUs.
    teacher_clusters: dict[str, RayVirtualCluster] = {}
    if opd_module.is_non_colocated_teachers_enabled(master_config):
        t0 = time.perf_counter()
        teacher_clusters = opd_module.reserve_teacher_clusters(
            master_config,
            segment_size=segment_size,
            teacher_segment_topology=teacher_segment_topology,
        )
        setup_timing_metrics.teacher_reservation_time_s = time.perf_counter() - t0

    # Create build tasks for generation / trainer / (nemo-gym) workers
    build_tasks: dict[str, Callable[[], Any]] = {}
    generation = None
    defer_generation_model_load = False
    gen_reserve_time = 0.0
    # Started inside the use_nemo_gym branch below, not here: main's parallel-build
    # restructure leaves `generation` as None at this point, and the router needs a
    # live generation to front. None is also the correct value whenever the router
    # is disabled or NeMo-Gym is not in play -- it is Gym that needs one stable URL.
    generation_router = None
    megatron_backend = generation_config["backend"] == "megatron"
    megatron_reserved_url = None
    megatron_port_holder = None
    reserved_http_server_port = None
    if megatron_backend:
        generation_config["model_name"] = master_config.policy["model_name"]

    def _build_trainer_and_value() -> tuple[Any, Optional[TQValue], dict[str, float]]:
        """Build the trainer, then the critic when this is a PPO run.

        Serial, and with the trainer offloaded in between, because both worker
        groups live on the same training GPUs: leaving the policy resident
        while the critic loads is what OOMs a tight fit. The trainer comes back
        to GPU before returning so callers see the same state GRPO leaves them.

        Returns:
            A tuple of (TQPolicy trainer, TQValue critic or None, per-phase wall
            times keyed as "trainer_time" and "value_time").
        """
        time_metrics: dict[str, float] = {}
        trainer, time_metrics["trainer_time"] = _build_trainer(
            train_cluster,
            master_config,
            tokenizer,
            processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )
        if not is_ppo_run(master_config):
            return trainer, None, time_metrics

        trainer.offload_to_cpu()
        value, time_metrics["value_time"] = _build_value(
            train_cluster,
            master_config,
            tokenizer,
            weights_path=value_weights_path,
            optimizer_path=value_optimizer_path,
        )
        # Blocks on the critic's async Ray __init__, then parks it on CPU.
        value.finish_training()
        trainer.prepare_for_training()
        return trainer, value, time_metrics

    def _build_generation_then_trainer(
        defer_generation_model_load: bool, generation=None
    ) -> tuple[Any, Any, Optional[TQValue], dict[str, float]]:
        """Build generation then trainer (and critic) serially.

        Args:
            defer_generation_model_load: If True, generation is a pre-reserved handle and this call
                finishes its model load; if False, builds generation from scratch.
            generation: Pre-reserved generation handle when defer_generation_model_load=True; None otherwise.

        Returns:
            A tuple of (finalized generation object, TQPolicy trainer, TQValue
            critic or None, per-phase wall times keyed as "gen_time",
            "trainer_time" and "value_time").
        """
        time_metrics = {}

        # generation
        if defer_generation_model_load:
            generation, time_metrics["gen_time"] = _finish_deferred_generation(
                generation
            )
        else:
            generation, time_metrics["gen_time"] = _build_generation(
                inference_cluster, master_config
            )

        # trainer (+ critic when PPO)
        trainer, value, train_side_metrics = _build_trainer_and_value()
        time_metrics.update(train_side_metrics)

        return generation, trainer, value, time_metrics

    if not use_nemo_gym and master_config.async_rl.generation_router.enabled:
        # The router exists to hand NeMo-Gym one URL; the native path calls generation
        # over Ray and never sees it. Silently ignoring the flag would be the opposite of
        # how generation_fleet_health treats an unsupported backend.
        raise ValueError(
            "async_rl.generation_router.enabled=true has no effect on the native rollout "
            "path: the router fronts NeMo-Gym's HTTP traffic, and this run does not use "
            "NeMo-Gym. Set env.should_use_nemo_gym=true, or disable the router."
        )

    if use_nemo_gym:
        if megatron_backend:
            # Megatron serves from rank 0 of the generation workers; pre-publish that address.
            t0 = time.perf_counter()
            (
                megatron_reserved_url,
                reserved_http_server_port,
                megatron_port_holder,
            ) = MegatronGeneration.reserve_http_server_address(
                inference_cluster,
                master_config.policy,
            )
            gen_reserve_time = time.perf_counter() - t0
            print(
                f"  ✓ Reserved Megatron server URL: {megatron_reserved_url}",
                flush=True,
            )
            gym_base_urls: list[Optional[str]] = [megatron_reserved_url]
        else:
            # defer generation, only get base_urls for nemo_gym spinup
            generation, gen_reserve_time = _build_generation(
                inference_cluster,
                master_config=master_config,
                defer_model_load=True,
            )
            defer_generation_model_load = True
            gym_base_urls = generation.dp_openai_server_base_urls
        # Before the Gym task is built, so Gym can be handed the router's single URL.
        # These two statements are the only failable ones related to the port holder's creation.
        try:
            generation_router = _maybe_start_generation_router(
                gym_base_urls, master_config
            )
            gym_spinup_base_urls = (
                [ray.get(generation_router.base_url.remote())]
                if generation_router is not None
                else gym_base_urls
            )
        except BaseException:
            if megatron_port_holder is not None:
                ray.kill(megatron_port_holder)
            raise
        # add nemo_gym spinup task
        build_tasks["nemo_gym"] = partial(
            _spinup_gym,
            master_config=master_config,
            base_urls=gym_spinup_base_urls,
            tokenizer=tokenizer,
        )

    if colocated:
        # Colocated: vLLM prefers a clean GPU at load time, so generation comes up before the trainer.
        build_tasks["generation_trainer"] = partial(
            _build_generation_then_trainer,
            defer_generation_model_load=defer_generation_model_load,
            generation=generation,
        )
    else:
        # Non-colocated: generation + trainer run on disjoint GPUs, so bring them up in parallel.
        if defer_generation_model_load:
            build_tasks["generation"] = partial(
                _finish_deferred_generation,
                generation=generation,
            )
        else:
            build_tasks["generation"] = partial(
                _build_generation,
                inference_cluster=inference_cluster,
                master_config=master_config,
                reserved_http_server_port=reserved_http_server_port,
                tokenizer=tokenizer,
                processor=processor,
            )
        build_tasks["trainer"] = _build_trainer_and_value

    # Submit build tasks and get results
    weight_synchronizer: Optional[WeightSynchronizer] = None
    try:
        with ThreadPoolExecutor(max_workers=len(build_tasks)) as executor:
            submitted = {k: executor.submit(fn) for k, fn in build_tasks.items()}
            if "generation_trainer" in submitted:
                generation, trainer, value, time_metrics = submitted[
                    "generation_trainer"
                ].result()
                gen_load_time = time_metrics["gen_time"]
            else:
                generation, gen_load_time = submitted["generation"].result()
                trainer, value, time_metrics = submitted["trainer"].result()
            if megatron_reserved_url is not None:
                # Gym initialization needs a live URL that will respond to health checks.
                # Megatron generation can only respond to health checks once initialized.
                # The Megatron engine cannot be initialized with dummy weights.
                # Thus, we must do an initial refit during initialization,
                # before Gym can spin up.
                t0 = time.perf_counter()
                weight_synchronizer = create_weight_synchronizer(
                    policy=trainer,
                    generation=generation,
                    generation_backend=generation_config["backend"],
                    colocated=colocated,
                    train_cluster=train_cluster,
                    inference_cluster=inference_cluster,
                    refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
                )
                weight_synchronizer.init_communicator()
                setup_timing_metrics.collective_init_time_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                weight_synchronizer.sync_weights()
                setup_timing_metrics.weight_sync_time_s = time.perf_counter() - t0
            if use_nemo_gym:
                env_handles["nemo_gym"], gym_time = submitted["nemo_gym"].result()
                setup_timing_metrics.nemo_gym_init_time_s = gym_time
    finally:
        if megatron_port_holder is not None:
            # Rank 0 adopted (or will never adopt) the held socket; drop the holder.
            ray.kill(megatron_port_holder)

    setup_timing_metrics.generation_init_time_s = gen_reserve_time + gen_load_time

    setup_timing_metrics.policy_init_time_s = time_metrics["trainer_time"]
    if "value_time" in time_metrics:
        setup_timing_metrics.value_init_time_s = time_metrics["value_time"]

    # Native TQ restore must run through the trainer's bootstrap client before
    # the normal SC data-plane client is created or any rollout/train data-plane
    # operation starts.
    data_plane_checkpoint_metadata = _maybe_restore_native_data_plane_checkpoint(
        trainer,
        last_checkpoint_path=last_checkpoint_path,
        save_state=save_state,
        partition_id=partition_id,
        sampler_name=master_config.async_rl.sampler.name,
    )

    if use_nemo_gym:
        # the two fields are only meaningful when use_nemo_gym enabled
        setup_timing_metrics.generation_init_reserve_time_s = gen_reserve_time
        setup_timing_metrics.generation_init_load_time_s = gen_load_time

    if megatron_reserved_url is not None:
        MegatronGeneration.verify_served_address(
            generation.dp_openai_server_base_urls, megatron_reserved_url
        )

    # Loading a teacher with the same checkpoint as the student must happen only
    # after student initialization finishes: both use the same HF-to-Megatron
    # cache path, and concurrent conversion can expose a partial checkpoint.
    teacher_worker_groups: dict[str, Any] = {}
    alias_to_group_alias: dict[str, str] = {}
    if teacher_clusters:
        t0 = time.perf_counter()
        teacher_worker_groups, alias_to_group_alias = (
            opd_module.create_teacher_worker_groups(
                master_config,
                cast(dict[str, Any], policy_config),
                tokenizer,
                teacher_clusters=teacher_clusters,
            )
        )
        for teacher in teacher_worker_groups.values():
            teacher.setup_data_plane(dp_config)
        setup_timing_metrics.teacher_model_init_time_s = time.perf_counter() - t0
        setup_timing_metrics.teacher_init_time_s = (
            setup_timing_metrics.teacher_reservation_time_s or 0.0
        ) + setup_timing_metrics.teacher_model_init_time_s

    worker_setup_time = time.perf_counter() - setup_start_time
    setup_timing_metrics.worker_setup_time_s = worker_setup_time

    # Attach fleet health before any rollout runs, so the very first request is
    # already health-aware.
    fleet_monitor = _maybe_attach_fleet_health(generation, master_config)

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_config, bootstrap=False)
    # SingleController reuses one partition for the run. Warm every known
    # tensor field before rollout, policy, and teacher writers become
    # concurrent; TransferQueue otherwise registers field names lazily.
    dp_client.register_partition(
        partition_id=partition_id,
        fields=fields_with_optional_routed_experts(
            SC_ROLLOUT_SCHEMA_FIELDS,
            enabled=router_replay_enabled(policy_config),
        ),
        num_samples=(
            master_config.async_rl.max_buffered_rollouts
            * algo_cfg.num_generations_per_prompt
        ),
        consumer_tasks=["prev_lp", "ref_lp", "train"],
        grpo_group_size=algo_cfg.num_generations_per_prompt,
    )

    if weight_synchronizer is None:
        t0 = time.perf_counter()
        weight_synchronizer = create_weight_synchronizer(
            policy=trainer,
            generation=generation,
            generation_backend=generation_config["backend"],
            colocated=colocated,
            train_cluster=train_cluster,
            inference_cluster=inference_cluster,
            refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
            refit_timeout_s=master_config.async_rl.generation_fleet_health.refit_timeout_s,
        )
        weight_synchronizer.init_communicator()
        setup_timing_metrics.collective_init_time_s = time.perf_counter() - t0

    # ==========================
    # Setup Algorithm + Rollout Wiring
    # ==========================
    advantage_estimator = _build_advantage_estimator(master_config)
    loss_fn: LossFunction = ClippedPGLossFn(master_config.loss_fn)
    value_loss_fn: Optional[LossFunction] = (
        MseValueLossFn(master_config.value_loss_fn)  # type: ignore
        if is_ppo_run(master_config)
        else None
    )

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tq_buffer = TQReplayBuffer(
        dp_client,
        partition_id=partition_id,
        pad_value_dict={"token_ids": pad_id, "input_ids": pad_id},
        require_routed_experts=router_replay_enabled(policy_config),
    )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        task_to_env=env_handles,
        num_generations_per_prompt=algo_cfg.num_generations_per_prompt,
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=algo_cfg.max_rollout_turns,
        policy_generation=generation,
        generation_config=generation_config,
        use_nemo_gym=use_nemo_gym,
        mask_env_flagged_samples=should_mask_flagged_samples(master_config.env),
        reward_penalty_config=resolved_reward_penalty_config,
        tq_buffer=tq_buffer,
        timeouts=RolloutTimeouts(
            rollout_s=master_config.async_rl.rollout_failure.nemo_gym.rollout_timeout_s,
            generation_s=master_config.async_rl.rollout_failure.native.generation_timeout_s,
            env_s=master_config.async_rl.rollout_failure.native.env_timeout_s,
        ),
        retry_policy=_build_retry_policy(master_config),
        effort_config=_get_effort_config(cast(GRPOMasterConfig, master_config)),
    )

    # Print setup timing metrics
    total_setup_time = time.perf_counter() - setup_start_time
    setup_timing_metrics.total_setup_time_s = total_setup_time
    setup_timing_metrics.other_setup_time_s = total_setup_time - worker_setup_time
    print_setup_timing_summary(setup_timing_metrics)

    # Build actor args and return
    actor_args = SingleControllerActorArgs(
        gen_handle=generation,
        trainer_handle=trainer,
        env_handles=env_handles,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        dp_client=dp_client,
        dataloader=dataloader,
        weight_synchronizer=weight_synchronizer,
        advantage_estimator=advantage_estimator,
        loss_fn=loss_fn,
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=partition_id,
        save_state=save_state,
        last_checkpoint_path=last_checkpoint_path,
        data_plane_checkpoint_metadata=data_plane_checkpoint_metadata,
        fleet_monitor=fleet_monitor,
        generation_router=generation_router,
        teacher_worker_groups=teacher_worker_groups,
        alias_to_group_alias=alias_to_group_alias,
        # PPO extras
        value_handle=value,
        value_loss_fn=value_loss_fn,
    )
    return actor_args, setup_timing_metrics
