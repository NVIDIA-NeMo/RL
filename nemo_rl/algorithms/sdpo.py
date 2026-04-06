"""Self-Distilled Policy Optimization (SDPO).

Reference: Hübotter et al. (2026) "Reinforcement Learning via Self-Distillation"
           arXiv:2601.20802

This module implements SDPO in NeMo-RL.  The key idea is:
 - Roll out the policy (same as GRPO).
 - For each prompt group find successful responses (reward >= threshold).
 - For every sample build a "teacher" input: original prompt prepended with a
   successful demonstration.  If no demonstration is available the sample is
   excluded from the distillation loss.
 - Compute teacher log-probs (current model, enriched context) and align them
   to the student sequence positions.
 - Train with token-level reverse-KL distillation (SDPOLossFn) instead of a
   clipped policy-gradient loss.
"""

import re
import time
from typing import Any, NotRequired, Optional, TypedDict, TypeVar

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOLoggerConfig,
    _extract_prompt_only_messages,
    _should_use_async_rollouts,
    refit_policy_generation,
    validate,
)
from nemo_rl.algorithms.loss import (
    SDPOLossConfig,
    SDPOLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.dataloader import MultipleDataloaderWrapper
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ============================================================================
# Configuration
# ============================================================================

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

_DEFAULT_REPROMPT_TEMPLATE = (
    "{prompt}\n\n"
    "Here is a correct solution for reference:\n\n"
    "{solution}\n\n"
    "Now solve the original problem."
)


class SDPOConfig(TypedDict):
    """Top-level SDPO training configuration."""

    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batch_size: int
    val_at_start: bool
    val_at_end: bool
    max_val_samples: int
    seed: int
    # SDPO-specific
    success_reward_threshold: float          # reward >= this counts as "successful"
    max_reprompt_len: int                    # max tokens in teacher prompt
    reprompt_template: NotRequired[str]      # uses {prompt} and {solution} placeholders
    remove_thinking_from_demo: NotRequired[bool]  # strip <think>…</think> from demos
    dont_reprompt_on_self_success: NotRequired[bool]  # exclude own success as demo


class SDPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int
    val_reward: NotRequired[float]


def _default_sdpo_save_state() -> SDPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: SDPOLossConfig
    env: dict[str, Any]
    data: DataConfig
    sdpo: SDPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ============================================================================
# Teacher-input construction
# ============================================================================


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from a response string."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_sdpo_teacher_data(
    message_logs: list,
    rewards: torch.Tensor,
    num_generations: int,
    tokenizer: TokenizerType,
    success_reward_threshold: float = 1.0,
    reprompt_template: str = _DEFAULT_REPROMPT_TEMPLATE,
    remove_thinking_from_demo: bool = False,
    dont_reprompt_on_self_success: bool = False,
    max_reprompt_len: int = 8192,
    pad_token_id: int = 0,
    make_seq_len_divisible_by: int = 1,
) -> tuple[BatchedDataDict, torch.Tensor]:
    """Build teacher-input sequences for SDPO self-distillation.

    For each prompt group (N generations per prompt), finds the first
    successful response (reward >= success_reward_threshold) and uses it
    as the demonstration.  Builds teacher sequences as:

        [reprompted_prompt_tokens | original_response_tokens]

    Returns:
        teacher_data: BatchedDataDict with keys input_ids, input_lengths,
                      token_mask, sample_mask.
        sdpo_mask: bool tensor of shape [B], True for samples that have a
                   demonstration and will receive a distillation signal.
    """
    B = len(message_logs)
    P = B // num_generations

    teacher_input_ids_list: list[torch.Tensor] = []
    teacher_token_mask_list: list[torch.Tensor] = []
    sdpo_mask = torch.zeros(B, dtype=torch.bool)

    for p in range(P):
        start = p * num_generations
        end = start + num_generations
        group_rewards = rewards[start:end]
        group_logs = message_logs[start:end]

        # Identify successful samples within this group
        success_indices = [
            i
            for i in range(num_generations)
            if group_rewards[i].item() >= success_reward_threshold
        ]

        # Per-sample: default = use original (no self-distillation signal)
        for i in range(num_generations):
            msg_log = group_logs[i]

            # Extract original assistant response tokens (last assistant turn)
            response_tokens: Optional[torch.Tensor] = None
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    response_tokens = m["token_ids"]
                    break

            if response_tokens is None:
                # No assistant message found — append empty; no signal
                flat_tokens = torch.cat(
                    [m["token_ids"] for m in msg_log], dim=0
                )
                flat_mask = torch.cat(
                    [
                        torch.ones_like(m["token_ids"])
                        if m["role"] == "assistant"
                        else torch.zeros_like(m["token_ids"])
                        for m in msg_log
                    ],
                    dim=0,
                )
                teacher_input_ids_list.append(flat_tokens)
                teacher_token_mask_list.append(flat_mask)
                continue

            # Find a suitable demonstration
            demo_content: Optional[str] = None
            if len(success_indices) > 0:
                for demo_idx in success_indices:
                    if dont_reprompt_on_self_success and demo_idx == i:
                        continue
                    # Extract demo response content
                    for m in reversed(group_logs[demo_idx]):
                        if m["role"] == "assistant":
                            demo_content = m.get("content", "")
                            break
                    if demo_content is not None:
                        break

            if demo_content is None:
                # No valid demonstration — fall back to original sequence
                flat_tokens = torch.cat(
                    [m["token_ids"] for m in msg_log], dim=0
                )
                flat_mask = torch.cat(
                    [
                        torch.ones_like(m["token_ids"])
                        if m["role"] == "assistant"
                        else torch.zeros_like(m["token_ids"])
                        for m in msg_log
                    ],
                    dim=0,
                )
                teacher_input_ids_list.append(flat_tokens)
                teacher_token_mask_list.append(flat_mask)
                continue

            # We have a demonstration — build reprompted teacher input
            sdpo_mask[start + i] = True

            if remove_thinking_from_demo:
                demo_content = _strip_thinking(demo_content)

            # Build the reprompted prompt messages (keep all non-assistant turns,
            # modify last user turn to include the demonstration)
            teacher_messages = []
            user_turn_count = 0
            total_user_turns = sum(1 for m in msg_log if m["role"] == "user")
            for m in msg_log:
                if m["role"] == "assistant":
                    continue  # skip — response is appended separately
                if m["role"] == "user":
                    user_turn_count += 1
                    if user_turn_count == total_user_turns:
                        # Last user turn: inject demonstration
                        original_content = m.get("content", "")
                        reprompted_content = reprompt_template.format(
                            prompt=original_content,
                            solution=demo_content,
                        )
                        teacher_messages.append(
                            {"role": "user", "content": reprompted_content}
                        )
                    else:
                        teacher_messages.append(
                            {"role": "user", "content": m.get("content", "")}
                        )
                else:
                    teacher_messages.append(
                        {"role": m["role"], "content": m.get("content", "")}
                    )

            # Tokenize reprompted prompt
            try:
                teacher_prompt = tokenizer.apply_chat_template(
                    teacher_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    truncation=True,
                    max_length=max_reprompt_len,
                    padding=False,
                )
                prompt_ids = teacher_prompt["input_ids"][0]  # [prompt_len]
            except Exception:
                # Tokenisation failed — fall back to original
                sdpo_mask[start + i] = False
                flat_tokens = torch.cat(
                    [m["token_ids"] for m in msg_log], dim=0
                )
                flat_mask = torch.cat(
                    [
                        torch.ones_like(m["token_ids"])
                        if m["role"] == "assistant"
                        else torch.zeros_like(m["token_ids"])
                        for m in msg_log
                    ],
                    dim=0,
                )
                teacher_input_ids_list.append(flat_tokens)
                teacher_token_mask_list.append(flat_mask)
                continue

            # Teacher sequence = reprompted prompt + original response
            teacher_seq = torch.cat([prompt_ids.cpu(), response_tokens.cpu()])
            prompt_mask = torch.zeros(len(prompt_ids), dtype=torch.long)
            resp_mask = torch.ones(len(response_tokens), dtype=torch.long)
            teacher_mask_seq = torch.cat([prompt_mask, resp_mask])

            teacher_input_ids_list.append(teacher_seq)
            teacher_token_mask_list.append(teacher_mask_seq)

    # Pad all sequences to the same length
    max_len = max(t.shape[0] for t in teacher_input_ids_list)
    if make_seq_len_divisible_by > 1:
        max_len = (
            (max_len + make_seq_len_divisible_by - 1)
            // make_seq_len_divisible_by
            * make_seq_len_divisible_by
        )

    teacher_input_ids = torch.stack(
        [
            torch.nn.functional.pad(
                t, (0, max_len - t.shape[0]), value=pad_token_id
            )
            for t in teacher_input_ids_list
        ]
    )
    teacher_token_mask = torch.stack(
        [
            torch.nn.functional.pad(
                m.long(), (0, max_len - m.shape[0]), value=0
            )
            for m in teacher_token_mask_list
        ]
    )
    teacher_input_lengths = torch.tensor(
        [t.shape[0] for t in teacher_input_ids_list], dtype=torch.long
    )

    teacher_data: BatchedDataDict = BatchedDataDict(
        {
            "input_ids": teacher_input_ids,
            "input_lengths": teacher_input_lengths,
            "token_mask": teacher_token_mask,
            "sample_mask": torch.ones(B, dtype=torch.float32),
        }
    )

    return teacher_data, sdpo_mask


def align_teacher_logprobs(
    teacher_logprobs: torch.Tensor,     # [B, max_teacher_len]
    teacher_token_mask: torch.Tensor,   # [B, max_teacher_len]  1 = response token
    student_seq_len: int,               # target width
    student_token_mask: torch.Tensor,   # [B, student_seq_len]  1 = response token
) -> torch.Tensor:
    """Re-index teacher logprobs to student response positions.

    The teacher sequence has a longer prompt (it includes the demonstration),
    so response tokens sit at different positions.  This function extracts the
    teacher's response-token log-probs and places them at the corresponding
    positions in the student sequence layout, returning a tensor of shape
    [B, student_seq_len].
    """
    B = teacher_logprobs.shape[0]
    aligned = torch.zeros(
        B,
        student_seq_len,
        device=teacher_logprobs.device,
        dtype=teacher_logprobs.dtype,
    )

    for i in range(B):
        teacher_resp_pos = teacher_token_mask[i].bool().nonzero(as_tuple=True)[0]
        student_resp_pos = student_token_mask[i].bool().nonzero(as_tuple=True)[0]

        n = min(len(teacher_resp_pos), len(student_resp_pos))
        if n == 0:
            continue

        # Both sequences contain the same response tokens — map positionally
        aligned[i, student_resp_pos[:n]] = teacher_logprobs[
            i, teacher_resp_pos[:n]
        ]

    return aligned


# ============================================================================
# Setup
# ============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple:
    """Set up SDPO training artefacts by delegating to grpo.setup.

    Builds a minimal GRPO-compatible config from the SDPO config so that all
    cluster/policy/generation initialisation is handled by the single source of
    truth in grpo.setup.  The GRPO loss function returned by grpo.setup is then
    replaced with SDPOLossFn before returning.

    Returns:
        (policy, policy_generation, cluster, dataloader, val_dataloader,
         loss_fn, logger, checkpointer, sdpo_save_state, master_config)
    """
    import copy
    from nemo_rl.algorithms.grpo import setup as grpo_setup
    from nemo_rl.algorithms.grpo import _default_grpo_save_state

    sdpo_config = master_config["sdpo"]
    loss_config = master_config["loss_fn"]

    # Build a GRPO-compatible master config by mapping shared fields.
    # grpo.setup only reads grpo_config for: num_prompts_per_step,
    # num_generations_per_prompt, max_num_steps, max_num_epochs, seed,
    # val_period, val_batch_size, val_at_start, val_at_end, max_val_samples,
    # max_rollout_turns, and a handful of optional flags.
    grpo_config_stub = {
        "num_prompts_per_step": sdpo_config["num_prompts_per_step"],
        "num_generations_per_prompt": sdpo_config["num_generations_per_prompt"],
        "max_rollout_turns": sdpo_config["max_rollout_turns"],
        "max_num_epochs": sdpo_config["max_num_epochs"],
        "max_num_steps": sdpo_config["max_num_steps"],
        "val_period": sdpo_config["val_period"],
        "val_batch_size": sdpo_config["val_batch_size"],
        "val_at_start": sdpo_config["val_at_start"],
        "val_at_end": sdpo_config["val_at_end"],
        "max_val_samples": sdpo_config["max_val_samples"],
        "seed": sdpo_config["seed"],
        # GRPO-specific flags that SDPO doesn't use — set safe defaults
        "normalize_rewards": False,
        "use_leave_one_out_baseline": False,
        "use_dynamic_sampling": False,
        "overlong_filtering": False,
        "skip_reference_policy_logprobs_calculation": True,
        "seq_logprob_error_threshold": None,
        "reward_shaping": {"enabled": False},
        "reward_scaling": {"enabled": False},
        "adv_estimator": {"name": "grpo", "normalize_rewards": False,
                          "use_leave_one_out_baseline": False, "minus_baseline": False},
        "async_grpo": {"enabled": False, "max_trajectory_age_steps": 1,
                       "in_flight_weight_updates": False,
                       "recompute_kv_cache_after_weight_updates": False},
        "batch_multiplier": 1,
    }

    # GRPO loss_fn stub (grpo.setup uses it only to build ClippedPGLossFn,
    # which we discard; the values here don't affect anything else in setup).
    grpo_loss_fn_stub = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "truncated_importance_sampling_ratio_min": None,
        "truncated_importance_sampling_type": "tis",
        "sequence_level_importance_ratios": False,
        "token_level_loss": True,
        "force_on_policy_ratio": False,
        "use_kl_in_reward": False,
    }

    grpo_master_config = copy.copy(master_config)
    grpo_master_config["grpo"] = grpo_config_stub
    grpo_master_config["loss_fn"] = grpo_loss_fn_stub

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        _grpo_loss_fn,   # discard
        logger,
        checkpointer,
        grpo_save_state,
        _,
    ) = grpo_setup(grpo_master_config, tokenizer, dataset, val_dataset)

    # Replace GRPO loss with SDPO loss
    loss_fn = SDPOLossFn(loss_config)

    # Convert grpo save state to sdpo save state (same fields)
    sdpo_save_state: SDPOSaveState = {
        "consumed_samples": grpo_save_state["consumed_samples"],
        "current_step": grpo_save_state["current_step"],
        "current_epoch": grpo_save_state["current_epoch"],
        "total_steps": grpo_save_state["total_steps"],
        "total_valid_tokens": grpo_save_state["total_valid_tokens"],
        "val_reward": grpo_save_state.get("val_reward", -99999999.0),
    }

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sdpo_save_state,
        master_config,
    )


# ============================================================================
# Training loop
# ============================================================================


def sdpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    sdpo_save_state: SDPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run SDPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert policy_generation is not None

    sdpo_cfg = master_config["sdpo"]
    policy_cfg = master_config["policy"]

    current_step = sdpo_save_state["current_step"]
    total_steps = sdpo_save_state["total_steps"]
    current_epoch = sdpo_save_state["current_epoch"]
    max_num_steps = sdpo_cfg["max_num_steps"]
    max_num_epochs = sdpo_cfg["max_num_epochs"]
    consumed_samples = sdpo_save_state["consumed_samples"]
    total_valid_tokens = sdpo_save_state.get("total_valid_tokens", 0)
    val_at_start = sdpo_cfg["val_at_start"]
    val_at_end = sdpo_cfg["val_at_end"]
    val_period = sdpo_cfg["val_period"]
    colocated_inference = policy_cfg["generation"]["colocated"]["enabled"]

    num_generations = sdpo_cfg["num_generations_per_prompt"]
    success_threshold = sdpo_cfg["success_reward_threshold"]
    max_reprompt_len = sdpo_cfg["max_reprompt_len"]
    reprompt_template = sdpo_cfg.get("reprompt_template", _DEFAULT_REPROMPT_TEMPLATE)
    remove_thinking = sdpo_cfg.get("remove_thinking_from_demo", False)
    dont_self = sdpo_cfg.get("dont_reprompt_on_self_success", False)
    make_div_by = policy_cfg.get("make_sequence_length_divisible_by", 1)

    # Run initial validation if requested
    if val_at_start and current_step == 0:
        print("\nRunning initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, _ = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            master_config,
        )
        logger.log_metrics(val_metrics, step=total_steps)
        print(f"  Validation metrics: {val_metrics}", flush=True)

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}"
        )

        for batch in dataloader:
            metrics: dict[str, Any] = {}
            metrics_logging_data: dict[str, Any] = {}

            print(
                f"\n{'=' * 25} Step {current_step + 1} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics = None

            with timer.time("total_step_time"):
                # ── Prepare batch ────────────────────────────────────────────
                print("Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(num_generations)
                    )
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                # ── Generate responses ───────────────────────────────────────
                print(
                    f"Generating responses (batch size {repeated_batch.size})...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                with timer.time("generation"):
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = (
                            run_async_multi_turn_rollout(
                                policy_generation=policy_generation,
                                input_batch=repeated_batch,
                                tokenizer=tokenizer,
                                task_to_env=task_to_env,
                                max_seq_len=policy_cfg[
                                    "max_total_sequence_length"
                                ],
                                max_rollout_turns=sdpo_cfg["max_rollout_turns"],
                                greedy=False,
                            )
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=policy_cfg["max_total_sequence_length"],
                            max_rollout_turns=sdpo_cfg["max_rollout_turns"],
                            greedy=False,
                        )

                metrics.update(rollout_metrics)

                # ── Evaluate rewards ─────────────────────────────────────────
                rewards = repeated_batch["total_reward"]  # [B]

                # ── Build training data (student) ────────────────────────────
                with timer.time("data_processing"):
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
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

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=make_div_by,
                    )

                    train_data = BatchedDataDict(
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages[
                                "generation_logprobs"
                            ],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": torch.ones(
                                repeated_batch.size, dtype=torch.float32
                            ),
                        }
                    )
                    train_data.to("cpu")

                # ── Compute student (prev) logprobs ──────────────────────────
                print("Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("Computing student logprobs...", flush=True)
                with timer.time("student_logprobs"):
                    logprob_data = BatchedDataDict(
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": train_data["sample_mask"],
                        }
                    )
                    train_data["prev_logprobs"] = policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]
                    del logprob_data

                # ── Build teacher inputs ─────────────────────────────────────
                print("Building SDPO teacher inputs...", flush=True)
                with timer.time("teacher_input_construction"):
                    teacher_data, sdpo_mask = build_sdpo_teacher_data(
                        message_logs=repeated_batch["message_log"],
                        rewards=rewards,
                        num_generations=num_generations,
                        tokenizer=tokenizer,
                        success_reward_threshold=success_threshold,
                        reprompt_template=reprompt_template,
                        remove_thinking_from_demo=remove_thinking,
                        dont_reprompt_on_self_success=dont_self,
                        max_reprompt_len=max_reprompt_len,
                        pad_token_id=tokenizer.pad_token_id,
                        make_seq_len_divisible_by=make_div_by,
                    )
                    teacher_data.to("cpu")

                frac_with_demo = sdpo_mask.float().mean().item()
                metrics["sdpo/frac_with_demo_pre_train"] = frac_with_demo
                print(
                    f"  SDPO: {sdpo_mask.sum().item()}/{len(sdpo_mask)} samples "
                    f"have demonstrations ({100 * frac_with_demo:.1f}%)",
                    flush=True,
                )

                # ── Compute teacher logprobs ──────────────────────────────────
                print("Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprobs"):
                    teacher_lp_raw = policy.get_logprobs(
                        teacher_data, timer=timer
                    )["logprobs"]  # [B, max_teacher_len]

                # Align teacher logprobs to student sequence positions
                with timer.time("teacher_logprob_alignment"):
                    student_seq_len = train_data["input_ids"].shape[1]
                    teacher_logprobs_aligned = align_teacher_logprobs(
                        teacher_logprobs=teacher_lp_raw.cpu(),
                        teacher_token_mask=teacher_data["token_mask"].cpu(),
                        student_seq_len=student_seq_len,
                        student_token_mask=train_data["token_mask"].cpu(),
                    )

                train_data["teacher_logprobs"] = teacher_logprobs_aligned
                train_data["sdpo_mask"] = sdpo_mask.float()
                del teacher_data, teacher_lp_raw, teacher_logprobs_aligned

                # ── Train ────────────────────────────────────────────────────
                print("Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("Training policy (SDPO)...", flush=True)
                with timer.time("policy_training"):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                # ── Metrics & logging ────────────────────────────────────────
                metrics["train/loss"] = train_results.get("loss", float("nan"))
                metrics["train/grad_norm"] = train_results.get(
                    "grad_norm", float("nan")
                )
                metrics["train/mean_reward"] = rewards.mean().item()
                metrics["train/success_fraction"] = (
                    (rewards >= success_threshold).float().mean().item()
                )

                # Aggregate SDPO-specific metrics from the loss function
                for k, v in train_results.get("all_mb_metrics", {}).items():
                    if "sdpo" in k:
                        metrics[k] = (
                            sum(v) / len(v) if isinstance(v, list) else v
                        )

                num_valid_tokens = int(
                    (
                        train_data["token_mask"]
                        * train_data["sample_mask"].unsqueeze(-1)
                    )
                    .sum()
                    .item()
                )
                total_valid_tokens += num_valid_tokens
                metrics["train/num_valid_tokens"] = num_valid_tokens
                metrics["train/total_valid_tokens"] = total_valid_tokens
                consumed_samples += repeated_batch.size
                metrics["train/consumed_samples"] = consumed_samples

                # ── Validation ───────────────────────────────────────────────
                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    val_metrics, _ = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        master_config,
                    )
                    metrics.update(val_metrics)

                    # Update best val reward in save state
                    val_reward_key = "val:accuracy"
                    if val_reward_key in val_metrics:
                        sdpo_save_state["val_reward"] = max(
                            sdpo_save_state.get("val_reward", -1e9),
                            val_metrics[val_reward_key],
                        )

                # Log
                logger.log_metrics(metrics, step=total_steps + 1)

                # ── Checkpoint ───────────────────────────────────────────────
                current_step += 1
                total_steps += 1
                sdpo_save_state.update(
                    {
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "current_epoch": current_epoch,
                        "consumed_samples": consumed_samples,
                        "total_valid_tokens": total_valid_tokens,
                    }
                )

                checkpointer.maybe_save_checkpoint(
                    policy,
                    training_info=sdpo_save_state,
                    metrics=metrics,
                    step=total_steps,
                )

                timeout.update()

            if total_steps >= max_num_steps:
                break

            if timeout.timed_out:
                print("Training timeout reached, saving checkpoint...")
                checkpointer.save_checkpoint(
                    policy,
                    training_info=sdpo_save_state,
                    metrics=metrics,
                    step=total_steps,
                    force=True,
                )
                return

        current_epoch += 1
        current_step = 0
        sdpo_save_state["current_epoch"] = current_epoch
        sdpo_save_state["current_step"] = 0

    print("SDPO training complete.", flush=True)
