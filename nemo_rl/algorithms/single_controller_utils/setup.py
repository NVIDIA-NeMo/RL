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

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _create_advantage_estimator,
    _get_effort_config,
    _get_grpo_save_state,
)
from nemo_rl.algorithms.grpo import MasterConfig as GrpoMasterConfig
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.metric_utils import (
    SetupTimingMetrics,
    print_setup_timing_summary,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import load_dataloader_state, setup_response_data
from nemo_rl.data_plane import DataPlaneClient, build_data_plane_client
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import should_use_nemo_gym, spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import (
    RolloutManager,
    RolloutRetryPolicy,
    RolloutTimeouts,
)
from nemo_rl.experience.rollouts import should_mask_flagged_samples
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
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
)
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
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
    # Defaulted fields must follow the required ones above, so these two stay last.
    # None when async_rl.generation_fleet_health is disabled; the SingleController drives the
    # probe loop when it is present.
    fleet_monitor: Optional[GenerationFleetHealth] = None
    # None unless async_rl.generation_router is enabled; the SingleController pushes the
    # serving backend set to it. Parameterized with the Impl class because the decorated
    # GenerationRouterActor name is an ActorClass instance, not a type.
    generation_router: Optional[ray.actor.ActorHandle[GenerationRouterImpl]] = None


def _build_clusters(
    master_config: MasterConfig,
) -> tuple[RayVirtualCluster, RayVirtualCluster]:
    """Allocate train + inference clusters; one shared cluster when colocated."""
    cluster_config = master_config.cluster
    generation_config = master_config.policy["generation"]
    colocated = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    num_nodes = cluster_config["num_nodes"]
    gpus_per_node = cluster_config["gpus_per_node"]
    port_range_low = cluster_config.get("master_port_range_low")
    port_range_high = cluster_config.get("master_port_range_high")

    if colocated:
        # Policy + generation share GPUs — one cluster.
        cluster = RayVirtualCluster(
            name="sc_policy_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1 if backend == "megatron" else 2,
            port_range_low=port_range_low,
            port_range_high=port_range_high,
        )
        return cluster, cluster

    # Non-colocated: split node into train + inference clusters.
    assert backend != "megatron", (
        "The Megatron generation backend does not support non-colocated inference "
        "in SingleController."
    )
    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
    if inference_gpus_per_node is None:
        raise ValueError(
            "Non-colocated generation requires "
            "policy.generation.colocated.resources.gpus_per_node."
        )
    inference_nodes = inference_resources["num_nodes"] or 1
    if num_nodes == 1:
        train_gpus_per_node = gpus_per_node - inference_gpus_per_node
        train_nodes = 1
        assert train_gpus_per_node > 0, (
            f"Not enough GPUs for training: {gpus_per_node} - {inference_gpus_per_node} = {train_gpus_per_node}"
        )
    else:
        train_gpus_per_node = gpus_per_node
        train_nodes = num_nodes - inference_nodes
        assert train_nodes > 0, (
            f"train_nodes must be > 0: {num_nodes} - {inference_nodes} = {train_nodes}"
        )

    train_cluster = RayVirtualCluster(
        name="sc_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    inference_cluster = RayVirtualCluster(
        name="sc_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    return train_cluster, inference_cluster


def _build_generation(
    inference_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    *,
    defer_model_load: bool = False,
) -> tuple[Any, float]:
    """Spin up the generation backend (vLLM or SGLang).

    Args:
        inference_cluster: Ray virtual cluster the generation workers run on.
        master_config: SC MasterConfig.
        defer_model_load: If True (for the NeMo-Gym flow), reserve OpenAI server URLs without loading weights; caller runs gen.load_and_start() later.

    Returns:
        A tuple of (generation object, wall time spent in this call). The
        generation object is a VllmGeneration or SGLangGeneration.
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

    else:
        raise ValueError(
            f"single_controller_utils.setup only supports vllm or sglang generation; got {backend!r}"
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
    megatron generation has no dedicated field and routes max_new_tokens
    through as max_sequence_length on the inference worker.
    """
    backend = generation_config["backend"]
    if backend == "vllm":
        return generation_config["vllm_cfg"]["max_model_len"]
    if backend == "sglang":
        return generation_config["sglang_cfg"]["context_length"]
    if backend == "megatron":
        return generation_config["max_new_tokens"]
    raise ValueError(f"Unknown generation backend: {backend!r}")


def _clamp_max_num_steps(
    master_config: MasterConfig, dataloader: StatefulDataLoader
) -> None:
    """Clamp grpo.max_num_steps to max_num_epochs * len(dataloader)."""
    grpo_config = master_config.grpo
    max_num_epochs = grpo_config.max_num_epochs
    if max_num_epochs is None:
        return
    grpo_config.max_num_steps = min(
        grpo_config.max_num_steps,
        max_num_epochs * len(dataloader),
    )


def _maybe_inject_megatron_train_iters(master_config: MasterConfig) -> None:
    """Set train_iters from max_num_steps after its dataloader clamp."""
    policy_config = master_config.policy
    if not policy_config.get("megatron_cfg", {}).get("enabled", False):
        return
    grpo_config = master_config.grpo
    policy_config["megatron_cfg"]["train_iters"] = grpo_config.max_num_steps


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


def _maybe_start_generation_router(generation: Any, master_config: MasterConfig) -> Any:
    """Start the NeMo-Gym-facing router, if enabled.

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

    backend_urls = [url for url in (generation.dp_openai_server_base_urls or []) if url]
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

    # short names for config sections
    grpo_config = master_config.grpo
    dp_config = master_config.data_plane
    policy_config = master_config.policy
    generation_config = policy_config["generation"]
    data_config = master_config.data

    if grpo_config.val_period > 0 or grpo_config.val_at_start or grpo_config.val_at_end:
        raise NotImplementedError(
            "SingleController doesn't support validation now, will support "
            "later. Set grpo.val_period=0, val_at_start=false, val_at_end=false."
        )
    if dp_config is None or not dp_config.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )

    assert generation_config is not None, (
        "single_controller_utils.setup requires policy.generation in master_config"
    )

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )

    checkpointing_pretrained = master_config.checkpointing.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    set_seed(grpo_config.seed)

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

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    # TODO: add validate dataset wiring.
    use_nemo_gym = should_use_nemo_gym(master_config)
    if use_nemo_gym and generation_config["backend"] != "vllm":
        raise NotImplementedError(
            "SC NeMo-Gym integration currently supports the vllm backend "
            f"only; got {generation_config['backend']!r}"
        )
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
        batch_size=grpo_config.num_prompts_per_step,
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
    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = generation_config["colocated"]["enabled"]

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

    def _build_generation_then_trainer(
        defer_generation_model_load: bool, generation=None
    ) -> tuple[Any, Any, dict[str, float]]:
        """Build generation then trainer serially.

        Args:
            defer_generation_model_load: If True, generation is a pre-reserved handle and this call
                finishes its model load; if False, builds generation from scratch.
            generation: Pre-reserved generation handle when defer_generation_model_load=True; None otherwise.

        Returns:
            A tuple of (finalized generation object, TQPolicy trainer,
            per-phase wall times keyed as "gen_time" and "trainer_time").
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

        # trainer
        trainer, time_metrics["trainer_time"] = _build_trainer(
            train_cluster,
            master_config,
            tokenizer,
            processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )

        return generation, trainer, time_metrics

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
        # defer generation, only get base_urls for nemo_gym spinup
        generation, gen_reserve_time = _build_generation(
            inference_cluster,
            master_config=master_config,
            defer_model_load=True,
        )
        defer_generation_model_load = True
        # Before the Gym task is built, so Gym can be handed the router's single URL.
        generation_router = _maybe_start_generation_router(generation, master_config)
        # add nemo_gym spinup task
        build_tasks["nemo_gym"] = partial(
            _spinup_gym,
            master_config=master_config,
            # The whole point of the router: Gym holds one NeMo-RL-owned URL and
            # never has to fail over, which is the thing it cannot do.
            base_urls=(
                [ray.get(generation_router.base_url.remote())]
                if generation_router is not None
                else generation.dp_openai_server_base_urls
            ),
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
            )
        build_tasks["trainer"] = partial(
            _build_trainer,
            train_cluster=train_cluster,
            master_config=master_config,
            tokenizer=tokenizer,
            processor=processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )

    # Submit build tasks and get results
    with ThreadPoolExecutor(max_workers=len(build_tasks)) as executor:
        submitted = {k: executor.submit(fn) for k, fn in build_tasks.items()}
        results = {k: f.result() for k, f in submitted.items()}

    if colocated:
        generation, trainer, time_metrics = results["generation_trainer"]
        gen_load_time = time_metrics["gen_time"]
        setup_timing_metrics.policy_init_time_s = time_metrics["trainer_time"]
    else:
        generation, gen_load_time = results["generation"]
        trainer, trainer_time = results["trainer"]
        setup_timing_metrics.policy_init_time_s = trainer_time
    setup_timing_metrics.generation_init_time_s = gen_reserve_time + gen_load_time

    if use_nemo_gym:
        env_handles["nemo_gym"], gym_time = results["nemo_gym"]
        setup_timing_metrics.nemo_gym_init_time_s = gym_time
        # the two fields are only meaningful when use_nemo_gym enabled
        setup_timing_metrics.generation_init_reserve_time_s = gen_reserve_time
        setup_timing_metrics.generation_init_load_time_s = gen_load_time

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

    # ==========================
    # Setup Algorithm + Rollout Wiring
    # ==========================
    advantage_estimator = _create_advantage_estimator(
        cast(GrpoMasterConfig, master_config)
    )
    loss_fn: LossFunction = ClippedPGLossFn(master_config.loss_fn)

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
        num_generations_per_prompt=grpo_config.num_generations_per_prompt,
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=grpo_config.max_rollout_turns,
        policy_generation=generation,
        generation_config=generation_config,
        use_nemo_gym=use_nemo_gym,
        mask_env_flagged_samples=should_mask_flagged_samples(master_config.env),
        tq_buffer=tq_buffer,
        timeouts=RolloutTimeouts(
            rollout_s=master_config.async_rl.rollout_failure.nemo_gym.rollout_timeout_s,
            generation_s=master_config.async_rl.rollout_failure.native.generation_timeout_s,
            env_s=master_config.async_rl.rollout_failure.native.env_timeout_s,
        ),
        retry_policy=_build_retry_policy(master_config),
        effort_config=_get_effort_config(cast(GrpoMasterConfig, master_config)),
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
        fleet_monitor=fleet_monitor,
        generation_router=generation_router,
    )
    return actor_args, setup_timing_metrics
