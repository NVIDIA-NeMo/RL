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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampled Token Distillation algorithm.

On-policy distillation using GRPO-style policy gradient where the only training
signal is reverse KL divergence between student and teacher at each sampled token.

Unlike standard distillation (which matches top-k logit distributions via direct
KL minimization), this approach:
  1. Student generates rollouts (on-policy)
  2. Teacher scores each sampled token with a single log-probability
  3. Reverse KL is used as per-token advantage in a policy gradient loss
  4. Environment reward is zero — KL is the only signal

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/
"""
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.distillation import check_vocab_equality
from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import masked_mean, set_seed
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
from nemo_rl.models.generation.interfaces import GenerationInterface
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


class SampledTokenDistillationConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int
    max_num_steps: int
    max_num_epochs: int
    val_batch_size: int
    val_period: int
    val_at_start: bool
    val_at_end: bool
    max_val_samples: int
    seed: int
    val_max_total_sequence_length: NotRequired[Optional[int]]
    val_max_new_tokens: NotRequired[Optional[int]]
    overlong_filtering: NotRequired[bool]
    # KL advantage parameters
    kl_penalty_coef: float  # scales the KL advantages
    kl_discount_factor: float  # temporal discount (<1.0 discounts earlier tokens)


class SampledTokenDistillationSaveState(TypedDict):
    total_steps: int
    current_epoch: int
    current_step: int
    val_reward: NotRequired[float]
    consumed_samples: int
    total_valid_tokens: int


class MasterConfig(TypedDict):
    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: ClippedPGLossConfig  # Policy gradient loss configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    sampled_token_distillation: SampledTokenDistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


def _default_save_state() -> SampledTokenDistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


# ===============================================================================
# KL Advantage Computation
# ===============================================================================
def _discounted_future_sum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted sum of future values for each position.

    For position i, computes: sum_{k=0}^{T-1-i} gamma^k * x[i+k]

    Matches tinker-cookbook's discounted_future_sum_vectorized exactly.
    """
    result = torch.empty_like(x)
    running = torch.zeros(1, dtype=x.dtype, device=x.device)
    for t in range(len(x) - 1, -1, -1):
        running = x[t] + gamma * running
        result[t] = running
    return result


def compute_kl_advantages(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    kl_penalty_coef: float,
    kl_discount_factor: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute per-token advantages from reverse KL divergence.

    Matches tinker-cookbook's incorporate_kl_penalty:
      reverse_kl = student_logprobs - teacher_logprobs  (masked)
      avg_kl = mean(reverse_kl) over all valid tokens in the batch
      advantages = kl_penalty_coef * mask * (avg_kl - reverse_kl)

    When kl_discount_factor > 0, applies discounted future sum per sample:
      advantages[t] = adv[t] + γ*adv[t+1] + γ²*adv[t+2] + ...

    Args:
        student_logprobs: Student log-probs for sampled tokens [B, S].
        teacher_logprobs: Teacher log-probs for the same sampled tokens [B, S].
        token_mask: Mask for valid response tokens [B, S].
        sample_mask: Mask for valid samples [B].
        kl_penalty_coef: Coefficient scaling the KL advantages.
        kl_discount_factor: Discount factor for future sum. 0.0 = no discounting.
            When > 0, each position's advantage includes discounted future advantages.

    Returns:
        (advantages, metrics_dict)
    """
    mask = token_mask * sample_mask.unsqueeze(-1)

    # Masked reverse KL per token (zeroed where mask=0, matching tinker-cookbook)
    reverse_kl_per_token = (student_logprobs - teacher_logprobs) * mask  # [B, S]

    # Global baseline: mean reverse KL across all valid tokens in batch
    total_kl = reverse_kl_per_token.sum()
    total_mask = mask.sum()
    avg_reverse_kl = total_kl / (total_mask + 1e-8)  # scalar

    # Per-token KL advantages
    kl_advantages = kl_penalty_coef * mask * (avg_reverse_kl - reverse_kl_per_token)

    # Optional: discounted future sum (tinker-cookbook style)
    if kl_discount_factor > 0:
        for i in range(kl_advantages.shape[0]):
            kl_advantages[i] = _discounted_future_sum(kl_advantages[i], kl_discount_factor)

    metrics = {
        "avg_reverse_kl": avg_reverse_kl.item(),
        "kl_advantages_mean": masked_mean(kl_advantages, mask).item(),
        "kl_advantages_std": (kl_advantages * mask).std().item(),
    }

    return kl_advantages, metrics


def _align_teacher_logprobs_to_student(
    teacher_logprobs: torch.Tensor,
    teacher_token_mask: torch.Tensor,
    student_token_mask: torch.Tensor,
    student_seq_len: int,
) -> torch.Tensor:
    """Align teacher token logprobs to student sequence positions.

    Teacher and student may use different prompt prefixes, so response tokens can
    appear at different absolute positions. This function maps teacher response
    logprobs to the corresponding student response positions in-order.
    """
    B = teacher_logprobs.shape[0]
    aligned = torch.zeros(
        B,
        student_seq_len,
        dtype=teacher_logprobs.dtype,
        device=teacher_logprobs.device,
    )

    for i in range(B):
        teacher_resp = teacher_token_mask[i].nonzero(as_tuple=True)[0]
        student_resp = student_token_mask[i].nonzero(as_tuple=True)[0]
        n = min(len(teacher_resp), len(student_resp))
        if n > 0:
            aligned[i, student_resp[:n]] = teacher_logprobs[i, teacher_resp[:n]]

    return aligned


def _debug_print_first_sample_sequences(
    train_data: BatchedDataDict[Any],
    teacher_data: BatchedDataDict[Any],
    teacher_logprobs_aligned: torch.Tensor,
    tokenizer: TokenizerType,
) -> None:
    """Print one-shot first-sample sequence debug for sampled token distillation."""
    if train_data.size == 0:
        print("[STD_DEBUG] Empty batch; skipping debug print.", flush=True)
        return

    student_ids = train_data["input_ids"][0].detach().cpu()
    teacher_ids = teacher_data["input_ids"][0].detach().cpu()
    student_mask = train_data["token_mask"][0].detach().cpu()

    student_text = tokenizer.decode(
        student_ids.tolist(),
        skip_special_tokens=False,
    )
    teacher_text = tokenizer.decode(
        teacher_ids.tolist(),
        skip_special_tokens=False,
    )

    scored_positions = student_mask.nonzero(as_tuple=True)[0]
    preview_n = min(10, len(scored_positions))

    print("\n[STD_DEBUG] ===== FIRST-STEP DEBUG (sample 0) =====", flush=True)
    print("[STD_DEBUG] Full student sequence (raw decode):", flush=True)
    print(student_text, flush=True)
    print("[STD_DEBUG] Full teacher scoring sequence (raw decode):", flush=True)
    print(teacher_text, flush=True)
    print(
        f"[STD_DEBUG] Teacher scored positions count: {len(scored_positions)}",
        flush=True,
    )
    print(
        f"[STD_DEBUG] First scored positions: {scored_positions[:preview_n].tolist()}",
        flush=True,
    )

    for rank, pos in enumerate(scored_positions[:preview_n].tolist(), start=1):
        token_id = int(train_data["input_ids"][0, pos].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        teacher_lp = float(teacher_logprobs_aligned[0, pos].item())
        print(
            (
                f"[STD_DEBUG] scored_token_{rank:02d} pos={pos} "
                f"student_token_id={token_id} student_token_text={token_text!r} "
                f"teacher_logprob={teacher_lp:.6f}"
            ),
            flush=True,
        )


def _log_first_sample_debug(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    kl_advantages: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    kl_metrics: dict,
) -> None:
    """Log detailed debug info for the first sample of the first batch."""
    print("\n" + "=" * 70)
    print("DEBUG: First sample, first batch — token-level breakdown")
    print("=" * 70)

    # Use first sample
    s_lp = student_logprobs[0]  # [S]
    t_lp = teacher_logprobs[0]  # [S]
    adv = kl_advantages[0]      # [S]
    mask = token_mask[0]        # [S]
    ids = input_ids[0]          # [S]

    # Find valid response token positions (mask=1, starting from position 1)
    valid_positions = (mask[1:] > 0).nonzero(as_tuple=True)[0] + 1
    num_valid = len(valid_positions)

    print(f"  Sample mask: {sample_mask[0].item()}")
    print(f"  Sequence length: {s_lp.shape[0]}")
    print(f"  Valid response tokens: {num_valid}")
    print(f"  Batch-level KL metrics: {kl_metrics}")

    if num_valid == 0:
        print("  (no valid tokens to display)")
        print("=" * 70 + "\n", flush=True)
        return

    # Show first 20 and last 5 valid tokens
    show_first = min(20, num_valid)
    show_last = min(5, max(0, num_valid - show_first))
    positions_to_show = list(valid_positions[:show_first])
    if show_last > 0:
        positions_to_show.append(None)  # sentinel for "..."
        positions_to_show.extend(valid_positions[-show_last:])

    print(f"\n  {'pos':>5} | {'token':>20} | {'student_lp':>11} | {'teacher_lp':>11} | {'reverse_kl':>11} | {'advantage':>11}")
    print("  " + "-" * 88)

    for pos in positions_to_show:
        if pos is None:
            print(f"  {'...':>5} | {'...':>20} | {'...':>11} | {'...':>11} | {'...':>11} | {'...':>11}")
            continue
        p = pos.item()
        token_str = tokenizer.decode([ids[p].item()])
        token_str = repr(token_str)[:20]
        s = s_lp[p].item()
        t = t_lp[p].item()
        rkl = s - t
        a = adv[p].item()
        print(f"  {p:>5} | {token_str:>20} | {s:>11.4f} | {t:>11.4f} | {rkl:>11.4f} | {a:>11.4f}")

    # Summary stats over valid tokens
    valid_s = s_lp[valid_positions]
    valid_t = t_lp[valid_positions]
    valid_rkl = valid_s - valid_t
    valid_adv = adv[valid_positions]

    print(f"\n  Summary (this sample):")
    print(f"    student logprobs:  mean={valid_s.mean():.4f}  std={valid_s.std():.4f}")
    print(f"    teacher logprobs:  mean={valid_t.mean():.4f}  std={valid_t.std():.4f}")
    print(f"    reverse KL:        mean={valid_rkl.mean():.4f}  std={valid_rkl.std():.4f}  min={valid_rkl.min():.4f}  max={valid_rkl.max():.4f}")
    print(f"    advantages:        mean={valid_adv.mean():.4f}  std={valid_adv.std():.4f}  min={valid_adv.min():.4f}  max={valid_adv.max():.4f}")
    print("=" * 70 + "\n", flush=True)


# ===============================================================================
# Setup & Initialization
# ===============================================================================
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
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    SampledTokenDistillationSaveState,
    MasterConfig,
]:
    """Setup for sampled token distillation.

    Returns:
        tuple of student_policy, teacher_policy, student_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, save_state, master_config
    """
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    algo_config = master_config["sampled_token_distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for sampled token distillation"
    )

    set_seed(algo_config["seed"])

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
    save_state: Optional[SampledTokenDistillationSaveState] = cast(
        Optional[SampledTokenDistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if save_state is None:
        save_state = _default_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=algo_config["num_prompts_per_step"],
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

    val_dataloader: Optional[StatefulDataLoader] = None
    if algo_config["val_period"] > 0 or algo_config["val_at_start"] or algo_config["val_at_end"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=algo_config["val_batch_size"],
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
            name="sampled_token_distillation_cluster",
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
            "Non-colocated inference is not supported for Megatron generation backends."
        )

        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                f"when cluster.num_nodes = 1, but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            )
            train_nodes -= inference_nodes

        train_cluster = RayVirtualCluster(
            name="std_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="std_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(
            f"  ✓ Separate clusters: train={train_nodes}x{train_gpus_per_node}GPUs, "
            f"inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        total_train_iters = min(
            algo_config["max_num_steps"],
            algo_config["max_num_epochs"] * len(dataloader),
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
    #    Student Generation
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
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

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    # Only init reference model if we need KL regularization in the loss
    need_reference_model = loss_config.get("reference_policy_kl_penalty", 0) != 0

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        total_train_iters = min(
            algo_config["max_num_steps"],
            algo_config["max_num_epochs"] * len(dataloader),
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
        init_reference_model=need_reference_model,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # Non-colocated collective communication setup
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 12 + "SAMPLED TOKEN DISTILLATION SETUP COMPLETE")
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
        save_state,
        master_config,
    )


# ===============================================================================
# Training Loop
# ===============================================================================
def sampled_token_distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: ClippedPGLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: SampledTokenDistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Sampled Token Distillation training."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert student_generation is not None

    algo_config = master_config["sampled_token_distillation"]
    current_epoch = save_state["current_epoch"]
    current_step = save_state["current_step"]
    total_steps = save_state["total_steps"]
    consumed_samples = save_state["consumed_samples"]
    total_valid_tokens = save_state["total_valid_tokens"]
    val_period = algo_config["val_period"]
    val_at_start = algo_config["val_at_start"]
    val_at_end = algo_config["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = algo_config["max_num_epochs"]
    max_steps = algo_config["max_num_steps"]
    kl_penalty_coef = algo_config["kl_penalty_coef"]
    kl_discount_factor = algo_config["kl_discount_factor"]
    need_reference_model = master_config["loss_fn"].get("reference_policy_kl_penalty", 0) != 0

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
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Main training loop
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
                # ==========================================
                # 1. Prepare batch
                # ==========================================
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            algo_config["num_generations_per_prompt"]
                        )
                    )

                # ==========================================
                # 2. Generate student rollouts (on-policy)
                # ==========================================
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
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=algo_config["max_rollout_turns"],
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
                            max_rollout_turns=algo_config["max_rollout_turns"],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                # ==========================================
                # 3. Data processing
                # ==========================================
                with timer.time("data_processing"):
                    has_teacher_ml = "teacher_message_log" in repeated_batch
                    use_overlong_filtering = algo_config.get("overlong_filtering", False)
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]
                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)
                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier

                    # Add token_loss_mask and generation_logprobs to messages
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
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )

                    # Build teacher message logs with the same rollout injections
                    # (assistant/environment turns) but teacher-specific prompt.
                    if has_teacher_ml:
                        for i, student_ml in enumerate(repeated_batch["message_log"]):
                            teacher_ml = repeated_batch["teacher_message_log"][i]
                            if teacher_ml is student_ml:
                                teacher_ml = deepcopy(teacher_ml)
                                repeated_batch["teacher_message_log"][i] = teacher_ml

                            for msg in student_ml:
                                if msg["role"] != "user":
                                    teacher_ml.append(deepcopy(msg))

                            for msg in teacher_ml:
                                if msg["role"] == "assistant":
                                    msg["token_loss_mask"] = torch.ones_like(
                                        msg["token_ids"]
                                    )
                                else:
                                    msg["token_loss_mask"] = torch.zeros_like(
                                        msg["token_ids"]
                                    )
                                if "generation_logprobs" not in msg:
                                    msg["generation_logprobs"] = torch.zeros_like(
                                        msg["token_ids"], dtype=torch.float32
                                    )

                    # Flatten to training format
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False
                    )
                    train_data.update(extra_multimodal_data)
                    train_data.to("cpu")

                    if has_teacher_ml:
                        teacher_flat, teacher_lengths = batched_message_log_to_flat_message(
                            repeated_batch["teacher_message_log"],
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["teacher"][
                                "make_sequence_length_divisible_by"
                            ],
                        )
                        teacher_extra_multimodal_data = teacher_flat.get_multimodal_dict(
                            as_tensors=False
                        )
                        teacher_data = BatchedDataDict[Any](
                            {
                                "input_ids": teacher_flat["token_ids"],
                                "input_lengths": teacher_lengths,
                                "token_mask": teacher_flat["token_loss_mask"],
                                "sample_mask": repeated_batch["loss_multiplier"],
                                **teacher_extra_multimodal_data,
                            }
                        )
                        teacher_data.to("cpu")
                    else:
                        teacher_extra_multimodal_data = None
                        teacher_data = train_data

                # ==========================================
                # 4. Compute student logprobs (prev_logprobs)
                # ==========================================
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    student_policy.prepare_for_lp_inference()

                print("▶ Computing student logprobs...", flush=True)
                with timer.time("student_logprob_inference"):
                    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                            **extra_multimodal_data,
                        }
                    )
                    train_data["prev_logprobs"] = student_policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]

                # ==========================================
                # 5. Compute teacher logprobs (single sampled token)
                # ==========================================
                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_logprob_inference"):
                    teacher_logprob_data = (
                        BatchedDataDict[Any](
                            {
                                "input_ids": teacher_data["input_ids"],
                                "input_lengths": teacher_data["input_lengths"],
                                "token_mask": teacher_data["token_mask"],
                                "sample_mask": teacher_data["sample_mask"],
                                **(teacher_extra_multimodal_data or {}),
                            }
                        )
                        if has_teacher_ml
                        else logprob_data
                    )
                    teacher_logprobs = teacher_policy.get_logprobs(
                        teacher_logprob_data, timer=timer
                    )["logprobs"]
                    if has_teacher_ml:
                        teacher_logprobs = _align_teacher_logprobs_to_student(
                            teacher_logprobs=teacher_logprobs,
                            teacher_token_mask=teacher_data["token_mask"],
                            student_token_mask=train_data["token_mask"],
                            student_seq_len=train_data["input_ids"].shape[1],
                        )

                with timer.time("teacher_offload"):
                    teacher_policy.offload_after_refit()

                # ==========================================
                # 6. Compute reference policy logprobs (optional)
                # ==========================================
                if need_reference_model:
                    print("▶ Computing reference policy logprobs...", flush=True)
                    with timer.time("reference_logprob_inference"):
                        train_data["reference_policy_logprobs"] = (
                            student_policy.get_reference_policy_logprobs(
                                logprob_data, timer=timer
                            )["reference_logprobs"]
                        )
                else:
                    train_data["reference_policy_logprobs"] = torch.zeros_like(
                        train_data["prev_logprobs"]
                    )

                del logprob_data
                del extra_multimodal_data

                # ==========================================
                # 7. Compute KL advantages
                # ==========================================
                print("▶ Computing KL advantages...", flush=True)
                with timer.time("advantage_calculation"):
                    token_mask = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]

                    # Advantages are computed on the next-token positions [:, 1:]
                    # to match the loss function's slicing convention
                    kl_advantages, kl_metrics = compute_kl_advantages(
                        student_logprobs=train_data["prev_logprobs"][:, 1:],
                        teacher_logprobs=teacher_logprobs[:, 1:],
                        token_mask=token_mask[:, 1:],
                        sample_mask=sample_mask,
                        kl_penalty_coef=kl_penalty_coef,
                        kl_discount_factor=kl_discount_factor,
                    )

                    # Pad back to full sequence length (prepend zeros for position 0)
                    # so that the loss function's [:, 1:] slicing recovers the correct values
                    advantages_full = torch.zeros_like(train_data["prev_logprobs"])
                    advantages_full[:, 1:] = kl_advantages
                    train_data["advantages"] = advantages_full

                    # Debug: log first sample of first batch
                    if total_steps == 0:
                        _debug_print_first_sample_sequences(
                            train_data=train_data,
                            teacher_data=teacher_data,
                            teacher_logprobs_aligned=teacher_logprobs,
                            tokenizer=tokenizer,
                        )
                        _log_first_sample_debug(
                            student_logprobs=train_data["prev_logprobs"],
                            teacher_logprobs=teacher_logprobs,
                            kl_advantages=advantages_full,
                            token_mask=token_mask,
                            sample_mask=sample_mask,
                            input_ids=train_data["input_ids"],
                            tokenizer=tokenizer,
                            kl_metrics=kl_metrics,
                        )

                    del teacher_logprobs

                # ==========================================
                # 8. Train
                # ==========================================
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                # ==========================================
                # 9. Validation
                # ==========================================
                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
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
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )

                # ==========================================
                # 10. Metrics & Logging
                # ==========================================
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
                metrics.update(kl_metrics)
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                # Checkpointing
                consumed_samples += algo_config["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    save_state["current_epoch"] = current_epoch
                    save_state["current_step"] = current_step + 1
                    save_state["total_steps"] = total_steps + 1
                    save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in save_state:
                        del save_state["val_reward"]
                    save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in save_state:
                                del save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            save_state[full_metric_name] = metrics_source[metric_name]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, save_state, master_config
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
                f"  • Avg Reverse KL: {kl_metrics['avg_reverse_kl']:.4f}"
            )
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

            print("\n⏱️  Timing:", flush=True)
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
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics.get("global_valid_toks", 0) / total_time / total_num_gpus
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
        current_step = 0


# ===============================================================================
# Validation
# ===============================================================================
def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
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

    algo_config = master_config["sampled_token_distillation"]
    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []
        total_lengths = []
        all_message_logs = []

        max_batches = algo_config["max_val_samples"] // algo_config["val_batch_size"]
        val_max_seq_len = (
            algo_config.get("val_max_total_sequence_length")
            or master_config["policy"]["max_total_sequence_length"]
        )

        val_max_new_tokens = algo_config.get("val_max_new_tokens")
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

                if _should_use_async_rollouts(master_config):
                    val_batch, gen_metrics = run_async_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=val_max_seq_len,
                        max_rollout_turns=algo_config["max_rollout_turns"],
                        greedy=False,
                    )
                else:
                    val_batch, gen_metrics = run_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=val_max_seq_len,
                        max_rollout_turns=algo_config["max_rollout_turns"],
                        greedy=False,
                    )
                rewards = val_batch["total_reward"]

                total_rewards.extend(rewards.tolist())
                total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

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

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    print("\n  ⏱️  Validation Timing:")
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    timer.reset()

    return val_metrics, timing_metrics
