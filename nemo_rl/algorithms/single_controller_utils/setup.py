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

setup builds the full SingleControllerBundle on the driver and the caller passes it to
SingleControllerActor.remote. Everything lives on the driver because driver-side
TQPolicy owns the worker group directly — running this inside another Ray actor nests
runtime_envs and breaks Ray's resource resolution (see the PR #2692 follow-up).
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _create_advantage_estimator,
    _default_grpo_save_state,
    _should_use_nemo_gym,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    RolloutCheckpointingConfig,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import load_dataloader_state, setup_response_data
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollout_checkpoint import compute_rollout_fingerprint
from nemo_rl.experience.rollout_checkpoint_writer import RolloutCheckpointWriter
from nemo_rl.experience.rollout_manager import (
    RolloutCheckpointIOPolicy,
    RolloutManager,
)
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


@dataclass(frozen=True)
class RolloutCheckpointRuntime:
    """Driver-built handles and run-wide identity used by SingleController."""

    writer: Any
    run_id: str
    sampling_fingerprint: str
    tokenizer_fingerprint: str


@dataclass
class SingleControllerBundle:
    """All inputs SingleControllerActor needs, built driver-side by setup_single_controller().

    Passed as a single arg to SingleControllerActor.remote so the actor's __init__ does
    no construction work — every heavy object is cloudpickled in.
    """

    gen_handle: Any
    trainer_handle: Any  # driver-side TQPolicy
    env_handles: dict[str, EnvironmentInterface]
    val_env_handles: dict[str, EnvironmentInterface]
    train_cluster: RayVirtualCluster
    inference_cluster: RayVirtualCluster
    dp_client: Any
    dataloader: StatefulDataLoader
    val_dataloader: Optional[StatefulDataLoader]
    weight_synchronizer: WeightSynchronizer
    advantage_estimator: Any
    loss_fn: LossFunction
    rollout_manager: RolloutManager
    tq_buffer: TQReplayBuffer
    partition_id: str
    save_state: GRPOSaveState
    last_checkpoint_path: Optional[str]
    rollout_checkpoint: Optional[RolloutCheckpointRuntime]


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
        "Non-colocated inference is not supported for Megatron generation backends."
    )
    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
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
        generation_config["vllm_kwargs"]["hf_overrides"] = master_config.policy.get(
            "hf_config_overrides", {}
        )
        gen = VllmGeneration(cluster=inference_cluster, config=generation_config)
    elif backend == "sglang":
        generation_config["sglang_cfg"].setdefault(
            "model_path", master_config.policy["model_name"]
        )
        gen = SGLangGeneration(cluster=inference_cluster, config=generation_config)
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
    *,
    weights_path: Optional[Path],
    optimizer_path: Optional[Path],
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
        weights_path=weights_path,
        optimizer_path=optimizer_path,
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
    max_num_epochs = grpo_config.get("max_num_epochs")
    if max_num_epochs is None:
        return
    grpo_config["max_num_steps"] = min(
        grpo_config["max_num_steps"],
        max_num_epochs * len(dataloader),
    )


def _maybe_inject_megatron_train_iters(
    master_config: MasterConfig, dataloader: StatefulDataLoader
) -> None:
    """Set megatron_cfg.train_iters; must run before _build_trainer."""
    policy_config = master_config.policy
    if not policy_config.get("megatron_cfg", {}).get("enabled", False):
        return
    grpo_config = master_config.grpo
    policy_config["megatron_cfg"]["train_iters"] = min(
        grpo_config["max_num_steps"],
        grpo_config["max_num_epochs"] * len(dataloader),
    )


def _validate_rollout_checkpointing_setup(
    config: RolloutCheckpointingConfig,
    *,
    use_nemo_gym: bool,
) -> None:
    """Reject unsupported checkpointing configurations before worker setup."""
    if not config.enabled:
        return
    if config.mode != "completed_generations":
        raise ValueError(
            "SingleController rollout checkpointing only supports "
            "mode='completed_generations'"
        )
    if config.root_dir is None:
        raise ValueError(
            "rollout_checkpointing.root_dir is required when checkpointing is enabled"
        )
    if config.writer_concurrency < 1:
        raise ValueError("rollout_checkpointing.writer_concurrency must be >= 1")
    if config.max_pending_writes < 1:
        raise ValueError("rollout_checkpointing.max_pending_writes must be >= 1")
    if not use_nemo_gym:
        raise NotImplementedError(
            "SingleController rollout checkpointing currently supports only NeMo-Gym"
        )


def _build_rollout_checkpoint_runtime(
    config: RolloutCheckpointingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: Any,
) -> Optional[RolloutCheckpointRuntime]:
    """Create the single writer actor and immutable run-wide fingerprints."""
    if not config.enabled:
        return None
    assert config.root_dir is not None

    writer = RolloutCheckpointWriter.options(
        max_concurrency=config.writer_concurrency
    ).remote(str(config.root_dir))
    sampling_fingerprint = compute_rollout_fingerprint(
        {
            "backend": generation_config["backend"],
            "model_name": generation_config.get("model_name"),
            "max_seq_len": _generation_max_seq_len(generation_config),
            "max_new_tokens": generation_config["max_new_tokens"],
            "temperature": generation_config["temperature"],
            "top_p": generation_config["top_p"],
            "top_k": generation_config.get("top_k"),
            "stop_token_ids": generation_config.get("stop_token_ids"),
            "stop_strings": generation_config.get("stop_strings"),
        }
    )
    tokenizer_fingerprint = compute_rollout_fingerprint(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "chat_template": getattr(tokenizer, "chat_template", None),
            "vocab": tokenizer.get_vocab(),
            "special_token_ids": {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "unk_token_id": tokenizer.unk_token_id,
            },
        }
    )
    return RolloutCheckpointRuntime(
        writer=writer,
        run_id=uuid.uuid4().hex,
        sampling_fingerprint=sampling_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def setup_single_controller(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    partition_id: str = "rollout_data",
) -> SingleControllerBundle:
    """Build the full SC bundle driver-side.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        SingleControllerBundle ready to be passed to SingleControllerActor.
    """
    dp_cfg = master_config.data_plane
    if dp_cfg is None or not dp_cfg.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )

    data_config = master_config.data
    grpo_config = master_config.grpo
    generation_config = master_config.policy["generation"]
    assert generation_config is not None, (
        "single_controller_utils.setup requires policy.generation in master_config"
    )

    checkpointing_pretrained = master_config.checkpointing.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        master_config.policy["pretrained_checkpoint"] = checkpointing_pretrained

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )

    validation_enabled = (
        grpo_config["val_period"] > 0
        or grpo_config["val_at_start"]
        or grpo_config["val_at_end"]
    )
    if (
        validation_enabled
        and bool(master_config.env.get("should_use_nemo_gym"))
        and master_config.env.get("nemo_gym", {}).get("effort_levels") is not None
    ):
        raise NotImplementedError(
            "SingleController NeMo-Gym validation does not support "
            "env.nemo_gym.effort_levels yet."
        )

    set_seed(grpo_config["seed"])

    # ==========================
    # Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state = cast(
        Optional[GRPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )
    if save_state is None:
        save_state = _default_grpo_save_state()
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    use_nemo_gym = _should_use_nemo_gym(master_config)
    rollout_checkpoint_config = master_config.rollout_checkpointing
    _validate_rollout_checkpointing_setup(
        rollout_checkpoint_config,
        use_nemo_gym=use_nemo_gym,
    )
    if use_nemo_gym:
        # NeMo-Gym creates the env actor outside setup_response_data; we wire
        # it in after generation is up (it needs the OpenAI server URLs).
        dataset, val_dataset = setup_response_data(
            tokenizer, data_config, env_configs=None
        )
        env_handles: dict[str, EnvironmentInterface] = {}
        val_env_handles: dict[str, EnvironmentInterface] = {}
    else:
        dataset, val_dataset, env_handles, val_env_handles = setup_response_data(
            tokenizer, data_config, env_configs=master_config.env
        )
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    val_dataloader: Optional[StatefulDataLoader] = None
    if validation_enabled:
        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError(
                "A nonempty validation dataset is required when validation is enabled."
            )
        val_batch_size = grpo_config["val_batch_size"]
        max_val_samples = grpo_config["max_val_samples"]
        if use_nemo_gym:
            if max_val_samples is not None:
                raise ValueError(
                    "grpo.max_val_samples must be None for NeMo-Gym "
                    "SingleController validation."
                )
            if val_batch_size is None:
                val_batch_size = min(
                    len(val_dataset), master_config.async_rl.max_inflight_prompts
                )
            elif (
                not isinstance(val_batch_size, int)
                or isinstance(val_batch_size, bool)
                or val_batch_size <= 0
            ):
                raise ValueError(
                    "grpo.val_batch_size must be a positive integer or None for "
                    "NeMo-Gym SingleController validation."
                )
        else:
            if (
                not isinstance(val_batch_size, int)
                or isinstance(val_batch_size, bool)
                or val_batch_size <= 0
            ):
                raise ValueError(
                    "grpo.val_batch_size must be a positive integer for native "
                    "SingleController validation."
                )
            if (
                not isinstance(max_val_samples, int)
                or isinstance(max_val_samples, bool)
                or max_val_samples <= 0
            ):
                raise ValueError(
                    "grpo.max_val_samples must be a positive integer for native "
                    "SingleController validation."
                )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
    if last_checkpoint_path is not None:
        dataloader_state_path = os.path.join(
            last_checkpoint_path, "train_dataloader.pt"
        )
        if os.path.exists(dataloader_state_path):
            print(
                f"📦 Restoring dataloader state from checkpoint: {dataloader_state_path}"
            )
            load_dataloader_state(dataloader, last_checkpoint_path, data_config)
        else:
            # Pre-Phase-2 checkpoints have no dataloader state.
            print(
                f"⚠️ No dataloader state found at {dataloader_state_path}. "
                "Starting with a fresh dataloader position."
            )

    _clamp_max_num_steps(master_config, dataloader)
    _maybe_inject_megatron_train_iters(master_config, dataloader)

    # ==========================
    # Setup Clusters & Workers
    # ==========================
    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = generation_config["colocated"]["enabled"]
    if colocated:
        # Colocated: vLLM prefers a clean GPU at load time, so generation
        # comes up before the policy.
        generation = _build_generation(inference_cluster, master_config)
        policy = _build_trainer(
            train_cluster,
            master_config,
            tokenizer,
            processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )
    else:
        # Non-colocated: generation + policy run on disjoint GPUs, so
        # bring them up in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            gen_future = executor.submit(
                _build_generation, inference_cluster, master_config
            )
            policy_future = executor.submit(
                _build_trainer,
                train_cluster,
                master_config,
                tokenizer,
                processor,
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )
            generation = gen_future.result()
            policy = policy_future.result()

    # ==========================
    # NeMo-Gym actor (after generation is up so OpenAI URLs are available)
    # ==========================
    if use_nemo_gym:
        if generation_config["backend"] != "vllm":
            raise NotImplementedError(
                "SC NeMo-Gym integration currently supports the vllm backend "
                f"only; got {generation_config['backend']!r}"
            )
        enable_router_replay = router_replay_enabled(master_config.policy)
        nemo_gym_actor = spinup_nemo_gym_actor(
            env_configs=master_config.env,
            base_urls=generation.dp_openai_server_base_urls,
            model_name=generation_config["model_name"],
            enable_router_replay=enable_router_replay,
        )
        env_handles["nemo_gym"] = nemo_gym_actor
        val_env_handles["nemo_gym"] = nemo_gym_actor

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    backend = generation_config["backend"]
    weight_synchronizer = create_weight_synchronizer(
        policy=policy,
        generation=generation,
        generation_backend=backend,
        colocated=colocated,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
    )
    weight_synchronizer.init_communicator()

    # ==========================
    # Setup Algorithm + Rollout Wiring
    # ==========================
    advantage_estimator = _create_advantage_estimator(master_config)
    loss_fn: LossFunction = ClippedPGLossFn(master_config.loss_fn)

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tq_buffer = TQReplayBuffer(
        dp_client,
        partition_id=partition_id,
        pad_value_dict={"token_ids": pad_id, "input_ids": pad_id},
    )
    rollout_checkpoint = _build_rollout_checkpoint_runtime(
        rollout_checkpoint_config,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    checkpoint_io_policy = None
    if rollout_checkpoint_config.enabled:
        checkpoint_io_policy = RolloutCheckpointIOPolicy(
            max_pending_writes=rollout_checkpoint_config.max_pending_writes,
            write_timeout_s=rollout_checkpoint_config.write_timeout_s,
            max_retries=rollout_checkpoint_config.max_write_retries,
            retry_backoff_s=rollout_checkpoint_config.write_retry_backoff_s,
            load_timeout_s=rollout_checkpoint_config.load_timeout_s,
            max_load_retries=rollout_checkpoint_config.max_load_retries,
            load_retry_backoff_s=rollout_checkpoint_config.load_retry_backoff_s,
        )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        env_handles=env_handles,
        val_env_handles=val_env_handles,
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=grpo_config.get("max_rollout_turns"),
        policy_generation=generation,
        generation_config=generation_config,
        use_nemo_gym=use_nemo_gym,
        tq_buffer=tq_buffer,
        checkpoint_io_policy=checkpoint_io_policy,
    )

    return SingleControllerBundle(
        gen_handle=generation,
        trainer_handle=policy,
        env_handles=env_handles,
        val_env_handles=val_env_handles,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        dp_client=dp_client,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        weight_synchronizer=weight_synchronizer,
        advantage_estimator=advantage_estimator,
        loss_fn=loss_fn,
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=partition_id,
        save_state=save_state,
        last_checkpoint_path=last_checkpoint_path,
        rollout_checkpoint=rollout_checkpoint,
    )
