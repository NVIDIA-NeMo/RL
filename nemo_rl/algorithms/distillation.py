# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# See the License for the specific language governing permissions and limitations.
# limitations under the License.
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    # Training configuration
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int  # for multi-turn rollouts. Math Environments just have 1 turn (answering the question)
    max_num_steps: int  # maximum number of steps to train for
    max_num_epochs: int  # maximum number of epochs to train for
    val_batch_size: int
    val_period: int
    val_steps: NotRequired[list[int]]  # explicit steps to validate at (in addition to val_period)
    val_at_start: bool
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int
    topk_logits_k: int
    seed: int
    teacher_student_prefix_fraction: NotRequired[float]
    # Optional overrides for validation rollouts (see validate()).
    val_max_total_sequence_length: NotRequired[Optional[int]]
    val_max_new_tokens: NotRequired[Optional[int]]


class DistillationSaveState(TypedDict):
    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    val_reward: NotRequired[
        float
    ]  # Can be any metric. Setted to 'accuracy' by default in validation.
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,  # Aligned with GRPO
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(TypedDict):
    """Main configuration structure."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Chech model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm.

    Returns:
        tuple of student_policy, teacher_policy, student_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        # DTensor sequence parallel is supported; ensure CP and SP are not enabled together
        # This incompatibility is enforced in DTensor workers during initialization.
        # Additionally, SP may not be compatible with sequence packing for some models.
        # Refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details.
        # Therefore, we disable SP + packing for distillation.
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if (
        distillation_config["val_period"] > 0
        or distillation_config.get("val_steps", [])
        or distillation_config["val_at_start"]
        or distillation_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
            flush=True,
        )
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(
            f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #    Student Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            ## make vllm hf overrides match the training policy
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        # inference cluster + head node of the train cluster
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        # init collective
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Training & Validation
# ===============================================================================


def _align_teacher_topk_to_student(
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_token_mask: torch.Tensor,
    student_token_mask: torch.Tensor,
    student_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher topk logits from teacher sequence positions to student sequence positions.

    When the teacher uses a different prompt than the student, their sequences have different
    lengths but share identical response tokens. This function extracts teacher logits at
    response positions and places them at the corresponding student response positions.

    Because logits[i] predicts token[i+1] (autoregressive convention), the teacher
    logits at the last prompt position predict the first response token. We include
    this position so the loss can properly score the first response token.
    """
    B, _, k = teacher_topk_logits.shape

    aligned_logits = torch.zeros(B, student_seq_len, k, dtype=teacher_topk_logits.dtype)
    aligned_indices = torch.zeros(B, student_seq_len, k, dtype=teacher_topk_indices.dtype)

    for i in range(B):
        teacher_resp = teacher_token_mask[i].nonzero(as_tuple=True)[0]
        student_resp = student_token_mask[i].nonzero(as_tuple=True)[0]
        n = min(len(teacher_resp), len(student_resp))
        if n > 0:
            aligned_logits[i, student_resp[:n]] = teacher_topk_logits[i, teacher_resp[:n]]
            aligned_indices[i, student_resp[:n]] = teacher_topk_indices[i, teacher_resp[:n]]

            # Also copy the teacher logits from the position just before the first
            # response token: logits[resp[0]-1] predicts the first response token.
            t_prev = int(teacher_resp[0].item()) - 1
            s_prev = int(student_resp[0].item()) - 1
            if t_prev >= 0 and s_prev >= 0:
                aligned_logits[i, s_prev] = teacher_topk_logits[i, t_prev]
                aligned_indices[i, s_prev] = teacher_topk_indices[i, t_prev]

    return aligned_logits, aligned_indices


def _get_teacher_student_prefix_indices(
    batch_size: int,
    fraction: float,
    seed: int,
    total_steps: int,
) -> torch.Tensor:
    """Select a deterministic per-step subset that reuses the student prefix."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"distillation.teacher_student_prefix_fraction must be in [0, 1], got {fraction}"
        )
    if batch_size <= 0 or fraction <= 0.0:
        return torch.empty(0, dtype=torch.long)

    selected_count = min(batch_size, int(np.floor(batch_size * fraction + 0.5)))
    if selected_count <= 0:
        return torch.empty(0, dtype=torch.long)
    if selected_count >= batch_size:
        return torch.arange(batch_size, dtype=torch.long)

    step_seed = (seed * 1_000_003) + total_steps
    rng = np.random.default_rng(step_seed)
    selected = np.sort(rng.choice(batch_size, size=selected_count, replace=False))
    return torch.tensor(selected, dtype=torch.long)


def _apply_teacher_student_prefix_mix(
    repeated_batch: BatchedDataDict[DatumSpec],
    fraction: float,
    seed: int,
    total_steps: int,
) -> dict[str, float]:
    """Swap a deterministic subset of teacher prompts to the student prefix."""
    metrics = {
        "teacher_student_prefix_fraction_configured": float(fraction),
        "teacher_student_prefix_selected_count": 0.0,
        "teacher_student_prefix_realized_fraction": 0.0,
        "teacher_prompt_count": 0.0,
        "teacher_student_prefix_mean_teacher_input_length": 0.0,
        "teacher_prompt_mean_teacher_input_length": 0.0,
    }

    selected_indices = _get_teacher_student_prefix_indices(
        batch_size=repeated_batch.size,
        fraction=fraction,
        seed=seed,
        total_steps=total_steps,
    )
    if "teacher_message_log" not in repeated_batch or repeated_batch.size == 0:
        return metrics

    selected_index_set = {int(idx) for idx in selected_indices.tolist()}
    selected_prefix_lengths: list[int] = []
    teacher_prompt_lengths: list[int] = []

    for i in range(repeated_batch.size):
        if i in selected_index_set:
            repeated_batch["teacher_message_log"][i] = repeated_batch["message_log"][i]
            selected_prefix_lengths.append(
                len(repeated_batch["teacher_message_log"][i][0]["token_ids"])
            )
        else:
            teacher_prompt_lengths.append(
                len(repeated_batch["teacher_message_log"][i][0]["token_ids"])
            )

    selected_count = float(len(selected_index_set))
    teacher_prompt_count = float(repeated_batch.size - len(selected_index_set))
    metrics.update(
        {
            "teacher_student_prefix_selected_count": selected_count,
            "teacher_student_prefix_realized_fraction": (
                selected_count / repeated_batch.size if repeated_batch.size > 0 else 0.0
            ),
            "teacher_prompt_count": teacher_prompt_count,
            "teacher_student_prefix_mean_teacher_input_length": (
                float(np.mean(selected_prefix_lengths))
                if selected_prefix_lengths
                else 0.0
            ),
            "teacher_prompt_mean_teacher_input_length": (
                float(np.mean(teacher_prompt_lengths)) if teacher_prompt_lengths else 0.0
            ),
        }
    )
    return metrics


def _inject_student_rollout_into_teacher_message_logs(
    repeated_batch: BatchedDataDict[DatumSpec],
) -> None:
    """Populate teacher logs with student rollout turns without double-injecting."""
    for i, student_ml in enumerate(repeated_batch["message_log"]):
        teacher_ml = repeated_batch["teacher_message_log"][i]
        teacher_already_has_rollout = teacher_ml is student_ml
        if teacher_already_has_rollout:
            teacher_ml = deepcopy(teacher_ml)
            repeated_batch["teacher_message_log"][i] = teacher_ml
        else:
            for msg in student_ml:
                if msg["role"] != "user":
                    teacher_ml.append(deepcopy(msg))

        for msg in teacher_ml:
            if msg["role"] == "assistant":
                msg["token_loss_mask"] = torch.ones_like(msg["token_ids"])
            else:
                msg["token_loss_mask"] = torch.zeros_like(msg["token_ids"])


def _debug_print_first_sample(
    train_data: BatchedDataDict[Any],
    teacher_data: BatchedDataDict[Any],
    tokenizer: TokenizerType,
) -> None:
    """Print one-shot debug traces for sample 0 in distillation."""
    if train_data.size == 0:
        print("[DISTILL_DEBUG] Empty batch; skipping debug print.", flush=True)
        return

    student_ids = train_data["input_ids"][0].detach().cpu()
    teacher_ids = teacher_data["input_ids"][0].detach().cpu()
    teacher_mask = train_data["token_mask"][0].detach().cpu()
    teacher_topk_logits = train_data["teacher_topk_logits"][0].detach().cpu()
    teacher_topk_indices = train_data["teacher_topk_indices"][0].detach().cpu()

    student_text = tokenizer.decode(
        student_ids.tolist(),
        skip_special_tokens=False,
    )
    teacher_text = tokenizer.decode(
        teacher_ids.tolist(),
        skip_special_tokens=False,
    )

    scored_positions = teacher_mask.nonzero(as_tuple=True)[0]
    preview_n = min(10, len(scored_positions))

    print("\n[DISTILL_DEBUG] ===== FIRST-STEP DEBUG (sample 0) =====", flush=True)
    print("[DISTILL_DEBUG] Full student sequence (raw decode):", flush=True)
    print(student_text, flush=True)
    print("[DISTILL_DEBUG] Full teacher input sequence (raw decode):", flush=True)
    print(teacher_text, flush=True)
    print(
        f"[DISTILL_DEBUG] Teacher scored positions count: {len(scored_positions)}",
        flush=True,
    )

    # Print the full scored token sequence (ids + text)
    scored_ids = train_data["input_ids"][0][scored_positions].detach().cpu()
    scored_texts = [
        tokenizer.decode([int(t)], skip_special_tokens=False) for t in scored_ids
    ]
    print(
        "[DISTILL_DEBUG] Scored token sequence (raw decode):",
        flush=True,
    )
    print(
        tokenizer.decode(scored_ids.tolist(), skip_special_tokens=False),
        flush=True,
    )
    print(
        f"[DISTILL_DEBUG] Scored token ids: {scored_ids.tolist()}",
        flush=True,
    )
    print(
        f"[DISTILL_DEBUG] Scored token texts: {scored_texts}",
        flush=True,
    )

    def _print_token_with_topk(rank: int, pos: int, label: str) -> None:
        token_id = int(train_data["input_ids"][0, pos].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        # teacher logits[i] predicts token[i+1], so the distribution used to
        # score the student token at *pos* comes from teacher logits at pos-1.
        teacher_pos = pos - 1
        if teacher_pos < 0:
            print(
                f"[DISTILL_DEBUG] {label} pos={pos} "
                f"student_token_id={token_id} student_token_text={token_text!r} "
                f"(skipped: no teacher logit at pos-1)",
                flush=True,
            )
            return
        logits_at_pos = teacher_topk_logits[teacher_pos]
        indices_at_pos = teacher_topk_indices[teacher_pos]
        top_vals, top_order = torch.topk(logits_at_pos, k=min(5, logits_at_pos.shape[0]))
        print(
            f"[DISTILL_DEBUG] {label} pos={pos} "
            f"student_token_id={token_id} student_token_text={token_text!r} "
            f"(teacher logits from pos={teacher_pos})",
            flush=True,
        )
        for j in range(top_vals.shape[0]):
            idx_in_k = int(top_order[j].item())
            cand_id = int(indices_at_pos[idx_in_k].item())
            cand_text = tokenizer.decode([cand_id], skip_special_tokens=False)
            cand_logit = float(top_vals[j].item())
            print(
                f"[DISTILL_DEBUG]   top{j + 1}: token_id={cand_id} "
                f"token_text={cand_text!r} logit={cand_logit:.6f}",
                flush=True,
            )

    print(
        f"[DISTILL_DEBUG] First scored positions: {scored_positions[:preview_n].tolist()}",
        flush=True,
    )
    for rank, pos in enumerate(scored_positions[:preview_n].tolist(), start=1):
        _print_token_with_topk(rank, pos, f"scored_token_first_{rank:02d}")

    print(
        f"[DISTILL_DEBUG] Last scored positions: {scored_positions[-preview_n:].tolist()}",
        flush=True,
    )
    last_positions = scored_positions[-preview_n:].tolist()
    for rank, pos in enumerate(last_positions, start=len(scored_positions) - len(last_positions) + 1):
        _print_token_with_topk(rank, pos, f"scored_token_last_{rank:02d}")


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Distillation training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If student_generation is None, use the student_policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config["distillation"]["val_period"]
    val_steps_set = set(master_config["distillation"].get("val_steps", []))
    val_at_start = master_config["distillation"]["val_at_start"]
    val_at_end = master_config["distillation"]["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(
                student_policy, student_generation, colocated_inference
            )
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
            logger=logger,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Run distillation training (multi-epoch until reaching max_num_steps or max_num_epochs)
    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["distillation"]["num_generations_per_prompt"]
                        )
                    )

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    # Use async rollouts if vLLM async engine is enabled
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                with timer.time("data_processing"):
                    has_teacher_ml = "teacher_message_log" in repeated_batch
                    teacher_context_metrics = _apply_teacher_student_prefix_mix(
                        repeated_batch,
                        fraction=master_config["distillation"].get(
                            "teacher_student_prefix_fraction", 0.0
                        ),
                        seed=master_config["distillation"]["seed"],
                        total_steps=total_steps,
                    )

                    # Add loss mask and advantages to each message in LLMMessageLogType
                    for message_log in repeated_batch["message_log"]:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )

                    # Copy assistant messages to teacher_message_log and add loss masks
                    if has_teacher_ml:
                        _inject_student_rollout_into_teacher_message_logs(repeated_batch)

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[DistillationLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                    # Build separate teacher_data if teacher has a different prompt
                    if has_teacher_ml:
                        teacher_flat, teacher_lengths = batched_message_log_to_flat_message(
                            repeated_batch["teacher_message_log"],
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["teacher"][
                                "make_sequence_length_divisible_by"
                            ],
                        )
                        teacher_data = BatchedDataDict(
                            {
                                "input_ids": teacher_flat["token_ids"],
                                "input_lengths": teacher_lengths,
                                "token_mask": teacher_flat["token_loss_mask"],
                                "sample_mask": repeated_batch["loss_multiplier"],
                            }
                        )
                        teacher_data.to("cpu")
                    else:
                        teacher_data = train_data

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(
                        teacher_data,
                        k=master_config["distillation"]["topk_logits_k"],
                        timer=timer,
                    )

                    if has_teacher_ml:
                        # Align teacher topk logits to student sequence positions
                        aligned_logits, aligned_indices = _align_teacher_topk_to_student(
                            teacher_topk_logits=teacher_topk["topk_logits"],
                            teacher_topk_indices=teacher_topk["topk_indices"],
                            teacher_token_mask=teacher_data["token_mask"],
                            student_token_mask=train_data["token_mask"],
                            student_seq_len=train_data["input_ids"].shape[1],
                        )
                        train_data["teacher_topk_logits"] = aligned_logits
                        train_data["teacher_topk_indices"] = aligned_indices
                    else:
                        train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
                        train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]
                    if total_steps == 0:
                        _debug_print_first_sample(train_data, teacher_data, tokenizer)

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run validation if it's a validation step or last step with val_at_end
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    (total_steps + 1) in val_steps_set
                ) or (val_at_end and is_last_step):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                metrics.update(teacher_context_metrics)
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["distillation"][
                    "num_prompts_per_step"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because total_steps is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    if val_task_to_env is None:
        print(
            "  ⚠️ No validation task to environment mapping provided, skipping validation",
            flush=True,
        )
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []  # Can be any metric. Setted to 'accuracy' by default.
        total_task_names = []  # Track which dataset each reward belongs to
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["distillation"]["max_val_samples"]
            // master_config["distillation"]["val_batch_size"]
        )
        val_max_seq_len = (
            master_config["distillation"].get("val_max_total_sequence_length")
            or master_config["policy"]["max_total_sequence_length"]
        )

        val_max_new_tokens = master_config["distillation"].get("val_max_new_tokens")
        gen_cfg = master_config["policy"]["generation"]
        train_max_new_tokens = (
            gen_cfg["max_new_tokens"] if gen_cfg is not None else None
        )
        swap_max_new_tokens = (
            val_max_new_tokens is not None
            and train_max_new_tokens is not None
            and val_max_new_tokens != train_max_new_tokens
        )
        if swap_max_new_tokens:
            policy_generation.update_generation_params(max_new_tokens=val_max_new_tokens)

        try:
            for batch_idx, val_batch in enumerate(val_dataloader):
                if batch_idx >= max_batches:
                    break

                # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
                # Use async rollouts if vLLM async engine is enabled
                if _should_use_async_rollouts(master_config):
                    val_batch, gen_metrics = run_async_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=val_max_seq_len,
                        max_rollout_turns=master_config["distillation"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )
                else:
                    val_batch, gen_metrics = run_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=val_max_seq_len,
                        max_rollout_turns=master_config["distillation"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )
                rewards = val_batch["total_reward"]

                total_rewards.extend(rewards.tolist())
                if "task_name" in val_batch:
                    total_task_names.extend(val_batch["task_name"])
                # Collect per-sample generation lengths for per-task breakdown
                for ml in val_batch["message_log"]:
                    total_lengths.append(
                        sum(len(t["token_ids"]) for t in ml if t["role"] == "assistant")
                    )

                # Collect message logs for later display
                to_env = [
                    get_keys_from_message_log(
                        val_batch["message_log"][i], ["role", "content"]
                    )
                    for i in range(len(val_batch["message_log"]))
                ]

                all_message_logs.extend(to_env)
        finally:
            if swap_max_new_tokens:
                policy_generation.update_generation_params(
                    max_new_tokens=train_max_new_tokens
                )

        # Calculate validation metrics
        accuracy = (
            sum(total_rewards) / len(total_rewards) if len(total_rewards) > 0 else 0
        )
        avg_length = (
            sum(total_lengths) / len(total_lengths) if len(total_lengths) > 0 else 0
        )

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Per-dataset breakdown (accuracy and avg_length)
        if total_task_names:
            from collections import defaultdict

            per_task_rewards: dict[str, list[float]] = defaultdict(list)
            per_task_lengths: dict[str, list[float]] = defaultdict(list)
            for task_name, reward, length in zip(
                total_task_names, total_rewards, total_lengths
            ):
                if task_name is not None:
                    per_task_rewards[task_name].append(reward)
                    per_task_lengths[task_name].append(length)
            for task_name in sorted(per_task_rewards):
                rewards_list = per_task_rewards[task_name]
                lengths_list = per_task_lengths[task_name]
                val_metrics[f"accuracy/{task_name}"] = (
                    sum(rewards_list) / len(rewards_list)
                )
                val_metrics[f"avg_length/{task_name}"] = (
                    sum(lengths_list) / len(lengths_list)
                )

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

        # Save full validation rollouts for later inspection.
        if logger is not None:
            val_log_data = {
                "content": all_message_logs,
                "rewards": total_rewards,
            }
            logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    for key, value in sorted(val_metrics.items()):
        if key.startswith("accuracy/") or key.startswith("avg_length/"):
            print(f"    • {key}: {value:.4f}")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
