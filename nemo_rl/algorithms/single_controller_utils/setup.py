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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import MasterConfig as GrpoMasterConfig
from nemo_rl.algorithms.grpo import (
    _create_advantage_estimator,
    _should_use_nemo_gym,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data_plane import DataPlaneClient, build_data_plane_client
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import (
    RolloutManager,
    RolloutRetryPolicy,
    RolloutTimeouts,
)
from nemo_rl.experience.rollouts import should_mask_flagged_samples
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetMonitor,
    HealthyShardSelector,
)
from nemo_rl.models.generation.interfaces import (
    resolve_routed_experts_dtype_name_for_model,
)
from nemo_rl.models.generation.policy_router import PolicyRouterActor
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
)
from nemo_rl.models.policy.tq_policy import TQPolicy
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
    # None when async_rl.fleet_health is disabled; the SingleController drives the
    # probe loop when it is present.
    fleet_monitor: Optional[GenerationFleetMonitor] = None
    # None unless async_rl.policy_router is enabled; the SingleController pushes the
    # serving backend set to it.
    policy_router: Any = None


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
):
    """Spin up the generation backend (vLLM or SGLang)."""
    generation_config = master_config.policy["generation"]
    generation_config["model_name"] = master_config.policy["model_name"]
    backend = generation_config["backend"]
    if backend == "vllm":
        vllm_config = cast(VllmConfig, generation_config)
        vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
            master_config.policy.get("hf_config_overrides", {})
        )
        configure_vllm_for_router_replay(master_config.policy)
        gen = VllmGeneration(cluster=inference_cluster, config=vllm_config)
    elif backend == "sglang":
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
    gen.finish_generation()
    return gen


def _build_trainer(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer,
    processor,
):
    """Build the TQ-mediated trainer (driver-side TQPolicy).

    Driver-side on purpose: instantiating TQPolicy inside another Ray
    actor nests runtime_envs and triggers Ray's
    get_accelerator_ids_for_accelerator_resource IndexError. Keep this
    here until PolicyTrainerActor (PR #2692) lands.
    """
    loss_config = master_config.loss_fn
    init_reference_model = loss_config.reference_policy_kl_penalty > 0
    return TQPolicy(
        cluster=train_cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=True,
        init_reference_model=init_reference_model,
        dp_cfg=master_config.data_plane,
    )


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
) -> Optional[GenerationFleetMonitor]:
    """Route generation through fleet health, when it is enabled and supported.

    Returns:
        The monitor the SingleController should drive, or None when fleet health is
        disabled or the backend does not support it.
    """
    fleet_config = master_config.async_rl.fleet_health
    if not fleet_config.enabled:
        return None
    if not hasattr(generation, "attach_fleet_health"):
        # Loud rather than silent: asking for fleet health and not getting it would
        # otherwise look like it was working.
        raise NotImplementedError(
            "async_rl.fleet_health.enabled=true is only supported for the vllm "
            f"generation backend; got {type(generation).__name__}"
        )

    monitor = GenerationFleetMonitor(
        shard_count=generation.worker_group.dp_size,
        policy=FleetHealthPolicy(
            unhealthy_threshold=fleet_config.unhealthy_threshold,
            healthy_threshold=fleet_config.healthy_threshold,
            max_restart_attempts_per_shard=fleet_config.max_restart_attempts_per_shard,
            min_healthy_shards=fleet_config.min_healthy_shards,
        ),
        base_urls=list(generation.dp_openai_server_base_urls or []) or None,
    )
    generation.attach_fleet_health(monitor, HealthyShardSelector(monitor=monitor))
    return monitor


def _maybe_start_policy_router(generation: Any, master_config: MasterConfig) -> Any:
    """Start the NeMo-Gym-facing router, if enabled.

    Returns:
        The router actor handle, or None when the router is disabled.
    """
    router_config = master_config.async_rl.policy_router
    if not router_config.enabled:
        return None

    backend_urls = [url for url in (generation.dp_openai_server_base_urls or []) if url]
    if not backend_urls:
        raise ValueError(
            "async_rl.policy_router.enabled=true requires generation backends that "
            "expose OpenAI-compatible servers; none were reported. This needs the vllm "
            "backend with async_engine and expose_http_server enabled."
        )

    # Reserved once and passed in, so Ray recreating a restarted actor rebinds the same
    # address. NeMo-Gym holds this URL for the life of the run and never re-resolves it.
    port = _get_free_port_local(
        router_config.port_range_low, router_config.port_range_high
    )
    router = PolicyRouterActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        )
    ).remote(
        backend_urls=backend_urls,
        host=_get_node_ip_local(),
        port=port,
        backend_timeout_s=router_config.backend_timeout_s,
        no_healthy_backend_status=router_config.no_healthy_backend_status,
        served_model_name=master_config.policy["generation"]["model_name"],
    )
    # Resolve the URL now so the driver fails here rather than inside Gym if the actor
    # could not start.
    base_url = ray.get(router.base_url.remote())
    print(f"📡 Policy router fronting {len(backend_urls)} backend(s) at {base_url}")
    return router


def _build_retry_policy(master_config: MasterConfig) -> RolloutRetryPolicy:
    """Translate ``async_rl.rollout_failure`` into the rollout layer's policy object."""
    failure_config = master_config.async_rl.rollout_failure
    return RolloutRetryPolicy(
        max_infra_attempts=failure_config.max_attempts_per_prompt,
        max_data_attempts=failure_config.max_data_attempts_per_prompt,
        backoff_base_s=failure_config.backoff_base_s,
        max_backoff_s=failure_config.max_backoff_s,
        skip_on_data_exhausted=failure_config.on_data_exhausted == "skip",
        max_skipped_prompts=failure_config.max_skipped_prompts,
        max_gym_row_attempts=failure_config.max_gym_row_attempts,
    )


def setup_single_controller(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    partition_id: str = "rollout_data",
) -> SingleControllerActorArgs:
    """Build the full SC actor args driver-side.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        SingleControllerActorArgs ready to be passed to SingleControllerActor.
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
    if master_config.checkpointing["enabled"]:
        raise NotImplementedError(
            "SingleController doesn't support checkpointing now, will support "
            "later. Set checkpointing.enabled=false."
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

    set_seed(grpo_config.seed)

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    # TODO: add validate dataset wiring.
    use_nemo_gym = _should_use_nemo_gym(cast(GrpoMasterConfig, master_config))
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

    _clamp_max_num_steps(master_config, dataloader)
    _maybe_inject_megatron_train_iters(master_config)

    # ==========================
    # Setup Clusters & Workers
    # ==========================
    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = generation_config["colocated"]["enabled"]
    if colocated:
        # Colocated: vLLM prefers a clean GPU at load time, so generation
        # comes up before the policy.
        generation = _build_generation(inference_cluster, master_config)
        policy = _build_trainer(train_cluster, master_config, tokenizer, processor)
    else:
        # Non-colocated: generation + policy run on disjoint GPUs, so
        # bring them up in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            gen_future = executor.submit(
                _build_generation, inference_cluster, master_config
            )
            policy_future = executor.submit(
                _build_trainer, train_cluster, master_config, tokenizer, processor
            )
            generation = gen_future.result()
            policy = policy_future.result()

    # ==========================
    # NeMo-Gym actor (after generation is up so OpenAI URLs are available)
    # ==========================
    policy_router = _maybe_start_policy_router(generation, master_config)

    if use_nemo_gym:
        # TODO(#2625): Mirror GRPO's deferred vLLM load so NeMo-Gym spinup
        # overlaps model loading instead of running serially afterward.
        enable_router_replay = router_replay_enabled(policy_config)
        routed_experts_dtype = (
            resolve_routed_experts_dtype_name_for_model(generation_config["model_name"])
            if enable_router_replay
            else "int16"
        )
        env_handles["nemo_gym"] = spinup_nemo_gym_actor(
            env_configs=master_config.env,
            # The whole point of the router: Gym holds one NeMo-RL-owned URL and
            # never has to fail over, which is the thing it cannot do.
            base_urls=(
                [ray.get(policy_router.base_url.remote())]
                if policy_router is not None
                else generation.dp_openai_server_base_urls
            ),
            model_name=generation_config["model_name"],
            enable_router_replay=enable_router_replay,
            routed_experts_dtype=routed_experts_dtype,
            use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
        )

    # Attach fleet health before any rollout runs, so the very first request is
    # already health-aware.
    fleet_monitor = _maybe_attach_fleet_health(generation, master_config)

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_config, bootstrap=False)

    backend = generation_config["backend"]
    weight_synchronizer = create_weight_synchronizer(
        policy=policy,
        generation=generation,
        generation_backend=backend,
        colocated=colocated,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
        # Only armed when configured; None leaves the refit path unchanged.
        refit_timeout_s=master_config.async_rl.fleet_health.refit_timeout_s,
    )
    weight_synchronizer.init_communicator()

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
            rollout_s=master_config.async_rl.rollout_timeout_s,
            generation_s=master_config.async_rl.generation_timeout_s,
            env_s=master_config.async_rl.env_timeout_s,
        ),
        retry_policy=_build_retry_policy(master_config),
    )

    return SingleControllerActorArgs(
        gen_handle=generation,
        trainer_handle=policy,
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
        fleet_monitor=fleet_monitor,
        policy_router=policy_router,
    )
