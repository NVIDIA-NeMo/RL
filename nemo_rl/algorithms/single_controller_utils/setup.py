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

Two-phase setup mirrors :func:`nemo_rl.algorithms.grpo.setup` but is split
to match SC's actor model:

  - :func:`setup_handle`               — builds the four *remote* handles SC
    needs (``dp_client``, ``gen_handle``, ``trainer_handle``,
    ``env_handles``) plus the two Ray clusters. Driver-side; cheap to
    cross the actor boundary (handles are already Ray references).
  - :func:`setup_single_controller_component` — builds the five *local*
    components SC owns inside its actor process (``dataloader``,
    ``weight_synchronizer``, ``advantage_estimator``,
    ``rollout_manager``, ``tq_buffer``). Called from inside SC so the
    heavy Python objects never ride through Ray's cloudpickle.
"""

from __future__ import annotations

from typing import Any, Optional

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.grpo import (
    MasterConfig as GRPOMasterConfig,
    _create_advantage_estimator,
    setup as grpo_setup,
)
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


def setup_handle(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    dataset: AllTaskProcessedDataset | dict[str, AllTaskProcessedDataset],
    val_dataset: Optional[AllTaskProcessedDataset],
    *,
    processor: Optional[AutoProcessor] = None,
    env_handles: Optional[dict[str, EnvironmentInterface]] = None,
) -> tuple[
    Any,                                # dp_client
    Any,                                # gen_handle
    Any,                                # trainer_handle
    dict[str, EnvironmentInterface],    # env_handles
    RayVirtualCluster,                  # train_cluster
    RayVirtualCluster,                  # inference_cluster
]:
    """Build the four remote handles + two clusters SC drives.

    Delegates the cluster / policy / generation bring-up to
    :func:`nemo_rl.algorithms.grpo.setup` (with a TQ-mediated policy
    factory) so the async path stays in sync with the sync trainer. The
    returned :class:`TQPolicy` is exposed as ``trainer_handle``; in
    production it must be wrapped in a Ray actor that exposes the
    split-API ``.remote(...)`` calls SC issues — see
    :class:`PolicyTrainerActor` (PR #2692).

    Args:
        master_config: Resolved SC MasterConfig.
        tokenizer: Tokenizer used by the policy + by setup_response_data
            for env_handles.
        dataset: Train dataset.
        val_dataset: Optional validation dataset.
        processor: Optional ``AutoProcessor`` for VLM paths.
        env_handles: Pre-built ``task_to_env`` mapping. If ``None``,
            this function calls :func:`setup_response_data` to build it
            from the data + env configs.

    Returns:
        Tuple ``(dp_client, gen_handle, trainer_handle, env_handles,
        train_cluster, inference_cluster)``. Unpacked by the launcher
        and passed flat to :class:`SingleControllerActor`.
    """
    dp_cfg = master_config.data_plane
    if dp_cfg is None or not dp_cfg.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup_handle requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )

    # TQ-mediated policy factory — same wiring as run_grpo.py uses for the
    # sync trainer. The TQPolicy ctor bootstraps the data plane controller
    # and attaches it to worker actors.
    from nemo_rl.models.policy.tq_policy import TQPolicy

    def _make_tq_policy(**kwargs):
        return TQPolicy(**kwargs, dp_cfg=dp_cfg)

    # ``grpo.setup`` is typed against its own MasterConfig. We're not
    # subclassing it, so re-emit our config under that schema (the SC
    # specific fields ride along via ``extra="allow"`` and are ignored).
    grpo_mc = GRPOMasterConfig(**master_config.model_dump())

    (
        policy,
        policy_generation,
        _nemo_gym,
        clusters,
        _dataloader,
        _val_dataloader,
        _loss_fn,
        _logger,
        _checkpointer,
        _grpo_save_state,
        _grpo_mc_out,
    ) = grpo_setup(
        grpo_mc,
        tokenizer,
        dataset,
        val_dataset,
        processor=processor,
        policy_factory=_make_tq_policy,
    )
    train_cluster, inference_cluster = clusters

    if env_handles is None:
        _ds, _vds, env_handles, _val_env_handles = setup_response_data(
            tokenizer,
            master_config.data,
            master_config.env,
        )  # type: ignore[misc]

    # NOTE: trainer_handle should be a Ray actor wrapping TQPolicy (the
    # PolicyTrainerActor from PR #2692). Until that lands, callers that
    # invoke SC.run() will hit AttributeError on `.remote(...)`; surface
    # the policy here so the rest of the wiring can be inspected.
    trainer_handle: Any = policy
    return (
        policy.dp_client,
        policy_generation,
        trainer_handle,
        env_handles,
        train_cluster,
        inference_cluster,
    )


def setup_single_controller_component(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dp_client: Any,
    gen_handle: Any,
    trainer_handle: Any,
    env_handles: dict[str, EnvironmentInterface],
    train_cluster: RayVirtualCluster,
    inference_cluster: RayVirtualCluster,
    dataset: Any,
    partition_id: str = "rollout_data",
) -> tuple[
    StatefulDataLoader,
    WeightSynchronizer,
    Any,                 # advantage_estimator
    RolloutManager,
    TQReplayBuffer,
]:
    """Build the five local components SC owns inside its actor process.

    All five are plain Python objects — they live in SC's process and are
    not Ray actors. SC drives them directly via method calls.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer shared with the policy. Passed through to
            ``RolloutManager`` and used to derive the ``input_ids`` pad
            value for ``TQReplayBuffer``.
        dp_client: DataPlane client handle.
        gen_handle: Generation backend (VllmGeneration / SGLangGeneration).
        trainer_handle: Trainer Ray actor handle (or the bare TQPolicy
            until PolicyTrainerActor lands).
        env_handles: ``task_name -> EnvironmentInterface`` mapping.
        train_cluster: Training cluster — used by the weight synchronizer
            for non-colocated NCCL bring-up.
        inference_cluster: Inference cluster — same.
        dataset: Train dataset, wrapped here in a ``StatefulDataLoader``.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        Tuple ``(dataloader, weight_synchronizer, advantage_estimator,
        rollout_manager, tq_buffer)``.
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
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )

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

    advantage_estimator = _create_advantage_estimator(master_config)

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
        max_seq_len=master_config.policy["max_total_sequence_length"],
        max_rollout_turns=grpo_config.get("max_rollout_turns"),
        policy_generation=gen_handle,
        generation_config=generation_config,
        use_nemo_gym=False,
        tq_buffer=tq_buffer,
    )

    return (dataloader, weight_synchronizer, advantage_estimator, rollout_manager, tq_buffer)
