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

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, cast

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _create_advantage_estimator,
    _get_grpo_save_state,
    _should_use_nemo_gym,
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
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.experience.rollouts import should_mask_flagged_samples
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
    finalizer_actors: list[Any]
    save_state: GRPOSaveState
    last_checkpoint_path: Optional[str]


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


def _spinup_gym(master_config: MasterConfig, base_urls: list[str]) -> tuple[Any, float]:
    """Spin up the NeMo-Gym actor against the reserved vLLM URLs.

    Args:
        master_config: SC MasterConfig.
        base_urls: Reserved vLLM OpenAI server URLs.

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
        enable_router_replay=enable_router_replay,
        routed_experts_dtype=routed_experts_dtype,
        use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
        # Ledger config rides into Gym's policy model server (§ 9.1).
        token_capture=(
            master_config.token_capture.model_dump()
            if master_config.token_capture.enabled
            else None
        ),
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

    # Token capture: validate the MVP matrix loudly at setup (§ 6, § 10) and
    # give capture-enabled vLLM workers a venv that carries nemo_gym (the
    # worker hosts Gym's capture core + adapter in-process).
    token_capture_cfg = master_config.token_capture
    if token_capture_cfg.enabled:
        if not _should_use_nemo_gym(master_config):
            raise ValueError(
                "token_capture.enabled requires the NeMo-Gym rollout path "
                "(env.should_use_nemo_gym=true) — the ledger lives in Gym's "
                "policy model server"
            )
        if generation_config["backend"] != "vllm":
            raise NotImplementedError(
                "token_capture.enabled supports the vllm backend only; got "
                f"{generation_config['backend']!r}"
            )
        if not generation_config["vllm_cfg"]["async_engine"]:
            raise ValueError(
                "token_capture.enabled requires "
                "policy.generation.vllm_cfg.async_engine=true (the capture "
                "host is the worker's in-process HTTP server)"
            )
        from nemo_rl.distributed.ray_actor_environment_registry import (
            ACTOR_ENVIRONMENT_REGISTRY,
        )
        from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

        ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        ] = PY_EXECUTABLES.VLLM_GYM

        # Fill the derived ledger-hosting fields (see TokenCaptureConfig): a
        # per-run control-plane bearer token and the process-shared capture
        # directory used by every Gym worker.
        if token_capture_cfg.control_auth_token is None:
            # Deferred import: only needed on the capture path.
            import secrets

            token_capture_cfg.control_auth_token = secrets.token_hex(32)
        if token_capture_cfg.capture_dir is None:
            token_capture_cfg.capture_dir = os.path.abspath(
                os.path.join(
                    master_config.logger.get("log_dir") or "logs",
                    "gym_token_capture",
                )
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

    if use_nemo_gym:
        # defer generation, only get base_urls for nemo_gym spinup
        generation, gen_reserve_time = _build_generation(
            inference_cluster,
            master_config=master_config,
            defer_model_load=True,
        )
        defer_generation_model_load = True
        # add nemo_gym spinup task
        build_tasks["nemo_gym"] = partial(
            _spinup_gym,
            master_config=master_config,
            base_urls=generation.dp_openai_server_base_urls,
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

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_config, bootstrap=False)

    # Token-capture mode: pre-register both rollout partitions from this
    # single driver thread before any producer is live. TQ's controller
    # registers unseen field names lazily inside update_production_status
    # without a lock, so the first concurrent puts into an unregistered
    # partition can race kv_retrieve_meta and kill the controller thread
    # (see TQDataPlaneClient.register_partition).
    token_capture_cfg = master_config.token_capture
    if token_capture_cfg.enabled:
        from nemo_rl.data_plane.schema import (
            DP_TRAIN_FIELDS,
            fields_with_optional_routed_experts,
        )
        from nemo_rl.data_plane.schema import (
            ROUTED_EXPERTS_FIELD as STAGING_ROUTED_EXPERTS_FIELD,
        )
        from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS

        r3_enabled = router_replay_enabled(master_config.policy)
        if token_capture_cfg.defer_routed_experts_to_policy and not r3_enabled:
            raise ValueError(
                "token_capture.defer_routed_experts_to_policy requires "
                "policy.router_replay.enabled=true"
            )
        group_size = grpo_config.num_generations_per_prompt
        num_rollout_samples = master_config.async_rl.max_buffered_rollouts * group_size
        dp_client.register_partition(
            partition_id=partition_id,
            fields=fields_with_optional_routed_experts(
                DP_TRAIN_FIELDS,
                enabled=r3_enabled
                and not token_capture_cfg.defer_routed_experts_to_policy,
            ),
            num_samples=num_rollout_samples,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
            grpo_group_size=group_size,
        )
        dp_client.register_partition(
            partition_id=token_capture_cfg.staging_partition,
            fields=list(STAGING_FIELDS)
            + ([STAGING_ROUTED_EXPERTS_FIELD] if r3_enabled else []),
            num_samples=num_rollout_samples,
            consumer_tasks=["finalize", "prev_lp", "train"],
        )
        # Host Gym's capture core in every vLLM DP leader (in-worker DP
        # client + TQTokenSink + the single install_capture call), and give
        # workers the initial weight version to stamp on captured calls.
        try:
            generation.setup_token_capture(
                dp_config, token_capture_cfg.staging_partition
            )
        except Exception as error:
            if "No module named 'nemo_gym'" in str(error):
                # Worker venvs are cached by actor class name
                # (nemo_rl/utils/venvs.py), so a venv prebuilt before token
                # capture predates the nemo_gym extra and is reused as-is.
                raise RuntimeError(
                    "token_capture.enabled requires nemo_gym inside the vLLM "
                    "worker venv, but the cached worker venv predates it. "
                    "Rebuild worker venvs (NRL_FORCE_REBUILD_VENVS=true) or "
                    "delete $NEMO_RL_VENV_DIR/nemo_rl.models.generation.vllm."
                    "vllm_worker_async.VllmAsyncGenerationWorker and rerun."
                ) from error
            raise
        generation.set_rollout_weight_version(0)

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
        staging_partition_id=(
            token_capture_cfg.staging_partition if token_capture_cfg.enabled else None
        ),
    )
    finalizer_actors: list[Any] = []
    if token_capture_cfg.enabled:
        from nemo_rl.experience.finalizer_actor import (
            FinalizerActorConfig,
            create_finalizer_actors,
        )

        finalizer_actors = create_finalizer_actors(
            dp_config,
            FinalizerActorConfig(
                partition_id=partition_id,
                staging_partition=token_capture_cfg.staging_partition,
                pad_token_id=pad_id,
                mixed_weight_version_policy=token_capture_cfg.mixed_weight_version_policy,
                min_valid_fraction_per_group=token_capture_cfg.min_valid_fraction_per_group,
                router_replay_enabled=router_replay_enabled(policy_config),
                defer_routed_experts_to_policy=token_capture_cfg.defer_routed_experts_to_policy,
            ),
            num_workers=token_capture_cfg.num_finalizer_workers,
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
        finalizer_actors=finalizer_actors,
        save_state=save_state,
        last_checkpoint_path=last_checkpoint_path,
    )
    return actor_args, setup_timing_metrics
