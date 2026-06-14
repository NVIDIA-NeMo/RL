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
"""Setup helpers for the SingleController (async-RL) training path.

Two factories split along the SC actor boundary:

  - setup_handle: driver-side. Builds gen_handle, trainer_handle,
    env_handles plus the two Ray clusters.
  - setup_single_controller_component: in-actor. Builds dp_client,
    dataloader, weight_synchronizer, advantage_estimator, rollout_manager,
    tq_buffer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import _create_advantage_estimator
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.models.generation.sglang import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


def _build_env_handles(
    master_config: MasterConfig,
) -> dict[str, EnvironmentInterface]:
    """Build env_name -> EnvironmentInterface from master_config.env."""
    return {
        env_name: create_env(env_name=env_name, env_config=env_config)
        for env_name, env_config in master_config.env.items()
    }


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
            f"single_controller_utils.setup_handle only supports vllm or sglang generation; got {backend!r}"
        )
    gen.finish_generation()
    return gen


def _build_trainer(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer,
    processor,
):
    """Build the TQ-mediated trainer (driver-side TQPolicy)."""
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


def setup_handle(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    env_handles: Optional[dict[str, EnvironmentInterface]] = None,
) -> tuple[
    Any,                                # gen_handle
    Any,                                # trainer_handle
    dict[str, EnvironmentInterface],    # env_handles
    RayVirtualCluster,                  # train_cluster
    RayVirtualCluster,                  # inference_cluster
]:
    """Build the three remote handles + two clusters SC drives.

    The TQ data-plane client is not returned here. Following the
    grpo_sync convention, every consumer (SC, the rollout worker, the
    policy workers) builds its own connect-only client from
    ``master_config.data_plane`` once TQPolicy has bootstrapped the
    controller.

    Args:
        master_config: Resolved SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        env_handles: Pre-built env_name -> EnvironmentInterface mapping.
            If None, iterates master_config.env and calls create_env.

    Returns:
        (gen_handle, trainer_handle, env_handles, train_cluster,
        inference_cluster).
    """
    dp_cfg = master_config.data_plane
    if dp_cfg is None or not dp_cfg.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup_handle requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )

    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = master_config.policy["generation"]["colocated"]["enabled"]
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

    if env_handles is None:
        env_handles = _build_env_handles(master_config)

    # NOTE: trainer_handle should be a Ray actor wrapping TQPolicy (the
    # PolicyTrainerActor from PR #2692). Until that lands, callers that
    # invoke SC.run() will hit AttributeError on `.remote(...)`; surface
    # the policy here so the rest of the wiring can be inspected.
    trainer_handle: Any = policy
    return (
        generation,
        trainer_handle,
        env_handles,
        train_cluster,
        inference_cluster,
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


def setup_single_controller_component(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    gen_handle: Any,
    trainer_handle: Any,
    env_handles: dict[str, EnvironmentInterface],
    train_cluster: RayVirtualCluster,
    inference_cluster: RayVirtualCluster,
    partition_id: str = "rollout_data",
) -> tuple[
    Any,                 # dp_client
    StatefulDataLoader,
    WeightSynchronizer,
    Any,                 # advantage_estimator
    RolloutManager,
    TQReplayBuffer,
]:
    """Build the six local components SC owns inside its actor process.

    One connect-only DataPlane client is built here and shared by every
    in-process consumer (the TQReplayBuffer, and SC's _call_dp helper) —
    matching the per-process pattern used by SyncRolloutActor and the
    policy workers.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer shared with the policy; drives the input_ids
            pad value on the TQReplayBuffer.
        gen_handle: Generation backend.
        trainer_handle: Trainer Ray actor handle (or driver-side TQPolicy
            until PolicyTrainerActor lands).
        env_handles: env_name -> EnvironmentInterface mapping.
        train_cluster: Used by the weight synchronizer for non-colocated
            NCCL bring-up.
        inference_cluster: Same.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        (dp_client, dataloader, weight_synchronizer, advantage_estimator,
        rollout_manager, tq_buffer).
    """
    data_config = master_config.data
    grpo_config = master_config.grpo
    generation_config = master_config.policy["generation"]
    assert generation_config is not None, (
        "single_controller_utils.setup_single_controller_component requires "
        "policy.generation in master_config"
    )

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )

    # ── dp client ──────────────────────────────────────────────────────
    # connect-only; TQPolicy already bootstrapped the controller driver-side.
    dp_client = build_data_plane_client(master_config.data_plane, bootstrap=False)

    # ── dataloader ─────────────────────────────────────────────────────
    # Load the train dataset in-actor via the no-env form of
    # setup_response_data — env_handles are built driver-side.
    # TODO: add validate
    dataset, _val_dataset = setup_response_data(tokenizer, data_config)
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )

    # ── weight synchronizer ────────────────────────────────────────────
    # TODO: weight synchronizer not validated yet, this is a placeholder for now
    colocated = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    refit_buffer_size_gb = (
        generation_config.get("colocated", {})
        .get("resources", {})
        .get("refit_buffer_size_gb")
    )
    weight_synchronizer = create_weight_synchronizer(
        policy=trainer_handle,
        generation=gen_handle,
        generation_backend=backend,
        colocated=colocated,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        refit_buffer_size_gb=refit_buffer_size_gb,
    )

    # ── advantage estimator ────────────────────────────────────────────
    advantage_estimator = _create_advantage_estimator(master_config)

    # ── TQ buffer + rollout manager ────────────────────────────────────
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tq_buffer = TQReplayBuffer(
        dp_client,
        partition_id=partition_id,
        pad_value_dict={"token_ids": pad_id, "input_ids": pad_id},
    )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        env_handles=env_handles,
        num_generations_per_prompt=grpo_config["num_generations_per_prompt"],
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=grpo_config.get("max_rollout_turns"),
        policy_generation=gen_handle,
        generation_config=generation_config,
        use_nemo_gym=False,
        tq_buffer=tq_buffer,
    )

    return (
        dp_client,
        dataloader,
        weight_synchronizer,
        advantage_estimator,
        rollout_manager,
        tq_buffer,
    )
