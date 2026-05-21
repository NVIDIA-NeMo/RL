"""Self-Distilled Policy Optimization (SDPO).

Reference: Hübotter et al. (2026) "Reinforcement Learning via Self-Distillation"
           arXiv:2601.20802

This module implements SDPO in NeMo-RL.  The key idea is:
 - Roll out the policy (same as GRPO).
 - For each prompt group find successful responses (reward >= threshold).
 - For every sample build a "teacher" input: original prompt prepended with a
   successful demonstration.  If no demonstration is available the sample is
   excluded from the distillation loss.
 - Compute teacher top-k logits (current model, enriched context) and align
   them to the student sequence positions.
 - Train with logit-level KL distillation (SDPOLossFn, paper Eq. 1) summed
   over the response tokens, with an optional tail-correction term for vocab
   outside the top-k.
"""

import os
import re
from typing import Any, NotRequired, Optional, TypedDict, TypeVar

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOLoggerConfig,
    _should_use_async_rollouts,
    refit_policy_generation,
    validate,
)
from nemo_rl.algorithms.loss import (
    SDPOLossConfig,
    SDPOLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig
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
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ============================================================================
# Configuration
# ============================================================================

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

_DEFAULT_REPROMPT_TEMPLATE = (
    "{prompt}\n\n" "Here is a correct solution for reference:\n\n" "{solution}\n\n" "Now solve the original problem."
)

# Env-feedback teacher template (paper Table 2, code/LCBv6 mechanism). The
# teacher conditions on the model's *failed* attempt plus the environment's
# textual feedback (test failures, runtime errors), and is asked to solve the
# problem again. Placeholders: {prompt}, {environment_output}.
_DEFAULT_ENV_FEEDBACK_TEMPLATE = (
    "{prompt}\n\n"
    "The following is feedback from your unsuccessful earlier attempt:\n\n"
    "{environment_output}\n\n"
    "Correctly solve the original question."
)

# Combined teacher template — full paper Table 2 form for failed code rollouts:
# peer-rollout demo AND env feedback. Placeholders: {prompt}, {solution},
# {environment_output}.
_DEFAULT_COMBINED_TEMPLATE = (
    "{prompt}\n\n"
    "Correct solution:\n\n{solution}\n\n"
    "The following is feedback from your unsuccessful earlier attempt:\n\n"
    "{environment_output}\n\n"
    "Correctly solve the original question."
)

# Combined teacher template for *successful* rollouts: peer demo only, no env
# feedback line (the attempt didn't fail). Placeholders: {prompt}, {solution}.
_DEFAULT_COMBINED_SUCCESS_TEMPLATE = (
    "{prompt}\n\n"
    "Correct solution:\n\n{solution}\n\n"
    "Correctly solve the original question."
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
    success_reward_threshold: float  # reward >= this counts as "successful"
    max_reprompt_len: int  # max tokens in teacher prompt
    topk_logits_k: int  # K in top-k logit distillation (paper: 100)
    reprompt_template: NotRequired[str]  # uses {prompt} and {solution} placeholders
    remove_thinking_from_demo: NotRequired[bool]  # strip <think>…</think> from demos
    dont_reprompt_on_self_success: NotRequired[bool]  # exclude own success as demo
    # "peer_rollout" (default): teacher conditions on a successful peer rollout.
    # "env_feedback": teacher conditions on the failed attempt's env feedback.
    # "combined": full paper Table 2 form — peer rollout + env feedback together
    #   (failed rollouts) and peer rollout alone (successful rollouts).
    feedback_source: NotRequired[str]
    env_feedback_template: NotRequired[str]  # placeholders: {prompt}, {environment_output}
    combined_template: NotRequired[str]  # placeholders: {prompt}, {solution}, {environment_output}
    combined_success_template: NotRequired[str]  # placeholders: {prompt}, {solution}
    # Trust-region anchor to the frozen-init policy (paper Table 4): adds
    # beta * KL(student || ref) to the loss summed over response positions.
    # 0 disables (no ref-policy forward pass; ref model is not initialized).
    reference_policy_kl_penalty: NotRequired[float]  # beta; default 0.0
    reference_policy_kl_type: NotRequired[str]  # "k1" | "k2" | "k3" (Schulman); default "k3"
    # EMA-regularized self-teacher (paper Table 12 for LCBv6: alpha=0.01).
    # When enabled, the teacher's top-k forward routes through a slow-moving
    # EMA copy of the student weights instead of the live policy, breaking the
    # student-teacher lockstep that otherwise destabilizes training.
    use_ema_teacher: NotRequired[bool]
    ema_alpha: NotRequired[float]
    # Per-step env-feedback sample logging (paper F.3 verification).
    # 0 = never log samples; 1 = every step; N = every N steps.
    log_sample_every_n_steps: NotRequired[int]
    log_sample_max_chars: NotRequired[int]


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


def _extract_last_assistant_content(msg_log: list) -> str:
    """Return the most recent assistant message content (or empty string)."""
    for m in reversed(msg_log):
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def _render_teacher_prompt_for_logging(
    msg_log: list,
    sample_failed: bool,
    feedback_source: str,
    reprompt_template: str,
    env_feedback_template: str,
    combined_template: str,
    combined_success_template: str,
    peer_demo: Optional[str],
    env_feedback: Optional[str],
) -> str:
    """Re-render the same teacher template `build_sdpo_teacher_data` would have used.

    Used purely for stdout / W&B inspection — does not affect training.
    """
    original_prompt = ""
    for m in msg_log:
        if m.get("role") == "user":
            original_prompt = str(m.get("content", ""))
            break

    tpl: Optional[str] = None
    tpl_kwargs: dict[str, str] = {}
    if feedback_source == "peer_rollout":
        if peer_demo is not None:
            tpl = reprompt_template
            tpl_kwargs = {"solution": peer_demo}
    elif feedback_source == "env_feedback":
        if env_feedback is not None:
            tpl = env_feedback_template
            tpl_kwargs = {"environment_output": env_feedback}
    elif feedback_source == "combined":
        if sample_failed:
            if peer_demo is not None and env_feedback is not None:
                tpl = combined_template
                tpl_kwargs = {"solution": peer_demo, "environment_output": env_feedback}
            elif env_feedback is not None:
                tpl = env_feedback_template
                tpl_kwargs = {"environment_output": env_feedback}
            elif peer_demo is not None:
                tpl = reprompt_template
                tpl_kwargs = {"solution": peer_demo}
        else:
            if peer_demo is not None:
                tpl = combined_success_template
                tpl_kwargs = {"solution": peer_demo}

    if tpl is None:
        return "<no teacher signal for this sample>"
    return tpl.format(prompt=original_prompt, **tpl_kwargs)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def _log_env_feedback_sample(
    *,
    step: int,
    message_logs: list,
    rewards: torch.Tensor,
    success_threshold: float,
    num_generations: int,
    feedback_source: str,
    reprompt_template: str,
    env_feedback_template: str,
    combined_template: str,
    combined_success_template: str,
    dont_reprompt_on_self_success: bool,
    max_chars: int,
    metrics: dict,
) -> None:
    """Pick one fail + one pass sample and dump response/env-feedback/teacher-prompt.

    Writes a greppable block to stdout and (when wandb is available) attaches a
    wandb.Table to ``metrics["sdpo/samples"]`` so the dump is browsable per step
    in the W&B UI.
    """
    B = len(message_logs)
    fail_idx: Optional[int] = None
    pass_idx: Optional[int] = None
    for i in range(B):
        r = rewards[i].item()
        if r >= success_threshold and pass_idx is None:
            pass_idx = i
        elif r < success_threshold and fail_idx is None:
            fail_idx = i
        if pass_idx is not None and fail_idx is not None:
            break

    rows: list[list] = []
    for kind, idx in [("fail", fail_idx), ("pass", pass_idx)]:
        if idx is None:
            continue
        msg_log = message_logs[idx]
        reward = rewards[idx].item()
        sample_failed = reward < success_threshold
        response = _extract_last_assistant_content(msg_log)
        env_feedback = _extract_environment_output(msg_log)

        # Find a peer demo (successful peer in the same prompt group)
        group_start = (idx // num_generations) * num_generations
        peer_demo: Optional[str] = None
        for j in range(group_start, group_start + num_generations):
            if j == idx and dont_reprompt_on_self_success:
                continue
            if rewards[j].item() >= success_threshold:
                peer_demo = _extract_last_assistant_content(message_logs[j])
                if peer_demo:
                    break

        teacher_prompt = _render_teacher_prompt_for_logging(
            msg_log,
            sample_failed,
            feedback_source,
            reprompt_template,
            env_feedback_template,
            combined_template,
            combined_success_template,
            peer_demo,
            env_feedback,
        )

        resp_t = _truncate(response, max_chars)
        fb_t = _truncate(env_feedback or "<no env feedback>", max_chars)
        tp_t = _truncate(teacher_prompt, max_chars)

        print(
            f"\n========== SDPO SAMPLE step={step} kind={kind} idx={idx} reward={reward:.2f} ==========",
            flush=True,
        )
        print(f"--- MODEL RESPONSE ---\n{resp_t}", flush=True)
        print(f"--- ENV FEEDBACK ---\n{fb_t}", flush=True)
        print(f"--- TEACHER PROMPT ---\n{tp_t}", flush=True)
        print("=" * 70, flush=True)

        rows.append([step, kind, reward, resp_t, fb_t, tp_t])

    if rows:
        try:
            import wandb

            metrics["sdpo/samples"] = wandb.Table(
                columns=["step", "kind", "reward", "response", "env_feedback", "teacher_prompt"],
                data=rows,
            )
        except ImportError:
            pass


def _extract_environment_output(msg_log: list) -> Optional[str]:
    """Pull the most recent env-role observation out of a rollout message log.

    The rollout machinery appends env observations as messages with
    role == "environment" (see nemo_rl.experience.rollouts). We take the last
    such message — it carries the test-failure / runtime-error feedback for
    SDPO's env-feedback teacher.
    """
    for m in reversed(msg_log):
        if m.get("role") == "environment":
            content = m.get("content", "")
            if content:
                return str(content)
    return None


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
    feedback_source: str = "peer_rollout",
    env_feedback_template: str = _DEFAULT_ENV_FEEDBACK_TEMPLATE,
    combined_template: str = _DEFAULT_COMBINED_TEMPLATE,
    combined_success_template: str = _DEFAULT_COMBINED_SUCCESS_TEMPLATE,
) -> tuple[BatchedDataDict, torch.Tensor]:
    """Build teacher-input sequences for SDPO self-distillation.

    Three modes (selected by ``feedback_source``):

    - ``"peer_rollout"`` (default): for each prompt group, find a successful
      peer rollout (reward >= threshold) and use its response as the teacher's
      conditioning demonstration. The reprompted prompt is built from
      ``reprompt_template`` with ``{prompt}`` and ``{solution}`` placeholders.

    - ``"env_feedback"``: for each *failed* rollout (reward < threshold), pull
      the environment's textual feedback from the last "environment" role
      message in the log and condition the teacher on it via
      ``env_feedback_template`` (placeholders ``{prompt}``,
      ``{environment_output}``). Successful rollouts get no signal in this
      mode.

    - ``"combined"``: full paper Table 2 form. Failed rollouts get a teacher
      conditioned on BOTH a successful peer rollout AND this attempt's env
      feedback (``combined_template``). Successful rollouts get a teacher
      conditioned on a successful peer rollout alone
      (``combined_success_template``). Falls back to env-only or peer-only
      when one piece is missing.

    Both modes build the teacher input as
    ``[reprompted_prompt_tokens | original_response_tokens]``, so teacher and
    student response tokens align positionally for the top-k KL.

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
        success_indices = [i for i in range(num_generations) if group_rewards[i].item() >= success_reward_threshold]

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
                flat_tokens = torch.cat([m["token_ids"] for m in msg_log], dim=0)
                flat_mask = torch.cat(
                    [
                        (
                            torch.ones_like(m["token_ids"])
                            if m["role"] == "assistant"
                            else torch.zeros_like(m["token_ids"])
                        )
                        for m in msg_log
                    ],
                    dim=0,
                )
                teacher_input_ids_list.append(flat_tokens)
                teacher_token_mask_list.append(flat_mask)
                continue

            # Resolve the inputs each mode needs: a peer rollout's successful
            # response (for peer_rollout / combined) and this attempt's env
            # feedback (for env_feedback / combined).
            peer_demo: Optional[str] = None
            env_feedback: Optional[str] = None
            sample_failed = group_rewards[i].item() < success_reward_threshold

            if feedback_source in ("peer_rollout", "combined"):
                if len(success_indices) > 0:
                    for demo_idx in success_indices:
                        if dont_reprompt_on_self_success and demo_idx == i:
                            continue
                        for m in reversed(group_logs[demo_idx]):
                            if m["role"] == "assistant":
                                peer_demo = m.get("content", "")
                                break
                        if peer_demo is not None:
                            break
                    if peer_demo is not None and remove_thinking_from_demo:
                        peer_demo = _strip_thinking(peer_demo)

            if feedback_source in ("env_feedback", "combined") and sample_failed:
                env_feedback = _extract_environment_output(msg_log)

            # Pick a template based on mode + which pieces we have.
            tpl: Optional[str] = None
            tpl_kwargs: dict[str, str] = {}
            if feedback_source == "peer_rollout":
                if peer_demo is not None:
                    tpl = reprompt_template
                    tpl_kwargs = {"solution": peer_demo}
            elif feedback_source == "env_feedback":
                if env_feedback is not None:
                    tpl = env_feedback_template
                    tpl_kwargs = {"environment_output": env_feedback}
            elif feedback_source == "combined":
                if sample_failed:
                    if peer_demo is not None and env_feedback is not None:
                        tpl = combined_template
                        tpl_kwargs = {
                            "solution": peer_demo,
                            "environment_output": env_feedback,
                        }
                    elif env_feedback is not None:
                        tpl = env_feedback_template
                        tpl_kwargs = {"environment_output": env_feedback}
                    elif peer_demo is not None:
                        tpl = reprompt_template
                        tpl_kwargs = {"solution": peer_demo}
                else:
                    if peer_demo is not None:
                        tpl = combined_success_template
                        tpl_kwargs = {"solution": peer_demo}

            if tpl is None:
                # No usable teacher signal — flat fallback (no distillation).
                flat_tokens = torch.cat([m["token_ids"] for m in msg_log], dim=0)
                flat_mask = torch.cat(
                    [
                        (
                            torch.ones_like(m["token_ids"])
                            if m["role"] == "assistant"
                            else torch.zeros_like(m["token_ids"])
                        )
                        for m in msg_log
                    ],
                    dim=0,
                )
                teacher_input_ids_list.append(flat_tokens)
                teacher_token_mask_list.append(flat_mask)
                continue

            sdpo_mask[start + i] = True

            # Build the reprompted prompt messages (keep all non-assistant turns,
            # modify last user turn to include the demonstration). Drop env-role
            # turns so they don't leak the feedback into the prompt twice.
            teacher_messages = []
            user_turn_count = 0
            total_user_turns = sum(1 for m in msg_log if m["role"] == "user")
            for m in msg_log:
                if m["role"] == "assistant" or m["role"] == "environment":
                    continue  # skip — response is appended separately; env feedback goes into the template
                if m["role"] == "user":
                    user_turn_count += 1
                    if user_turn_count == total_user_turns:
                        original_content = m.get("content", "")
                        reprompted_content = tpl.format(prompt=original_content, **tpl_kwargs)
                        teacher_messages.append({"role": "user", "content": reprompted_content})
                    else:
                        teacher_messages.append({"role": "user", "content": m.get("content", "")})
                else:
                    teacher_messages.append({"role": m["role"], "content": m.get("content", "")})

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
                flat_tokens = torch.cat([m["token_ids"] for m in msg_log], dim=0)
                flat_mask = torch.cat(
                    [
                        (
                            torch.ones_like(m["token_ids"])
                            if m["role"] == "assistant"
                            else torch.zeros_like(m["token_ids"])
                        )
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
        max_len = (max_len + make_seq_len_divisible_by - 1) // make_seq_len_divisible_by * make_seq_len_divisible_by

    teacher_input_ids = torch.stack(
        [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_token_id) for t in teacher_input_ids_list]
    )
    teacher_token_mask = torch.stack(
        [torch.nn.functional.pad(m.long(), (0, max_len - m.shape[0]), value=0) for m in teacher_token_mask_list]
    )
    teacher_input_lengths = torch.tensor([t.shape[0] for t in teacher_input_ids_list], dtype=torch.long)

    teacher_data: BatchedDataDict = BatchedDataDict(
        {
            "input_ids": teacher_input_ids,
            "input_lengths": teacher_input_lengths,
            "token_mask": teacher_token_mask,
            "sample_mask": torch.ones(B, dtype=torch.float32),
        }
    )

    return teacher_data, sdpo_mask


def align_teacher_topk(
    teacher_topk_logits: torch.Tensor,  # [B, max_teacher_len, K]
    teacher_topk_indices: torch.Tensor,  # [B, max_teacher_len, K]
    teacher_token_mask: torch.Tensor,  # [B, max_teacher_len]  1 = response token
    student_seq_len: int,  # target width
    student_token_mask: torch.Tensor,  # [B, student_seq_len]  1 = response token
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-index teacher top-k logits to align response-token predictions.

    Maps teacher's response-token PREDICTIONS to student's response-token
    PREDICTIONS (not response-token positions themselves).

    Causal LM: logits at position p predict token at position p+1. So the
    prediction OF the j-th response token (at sequence position resp_start+j)
    is produced by logits at position resp_start+j-1. We map teacher's logits
    at T_prompt+j-1 to student's frame position S_prompt+j-1, so the downstream
    loss (which masks via ``token_mask[:, 1:]`` = shifted_mask, where
    shifted_mask[k]=1 iff k+1 is a response position) finds teacher signal at
    every loss-active position — including the prediction of the FIRST
    response token (the most important position for setting the response
    trajectory).

    Padding strategy:
        - logits at non-(response-prediction) positions: zeros (masked
          downstream by token_mask)
        - indices at non-(response-prediction) positions: zero (a valid vocab
          index; gather remains well-defined and the masked positions are
          ignored by the loss)
    """
    B, _, K = teacher_topk_logits.shape
    aligned_logits = torch.zeros(
        B,
        student_seq_len,
        K,
        device=teacher_topk_logits.device,
        dtype=teacher_topk_logits.dtype,
    )
    aligned_indices = torch.zeros(
        B,
        student_seq_len,
        K,
        device=teacher_topk_indices.device,
        dtype=teacher_topk_indices.dtype,
    )

    for i in range(B):
        teacher_resp_pos = teacher_token_mask[i].bool().nonzero(as_tuple=True)[0]
        student_resp_pos = student_token_mask[i].bool().nonzero(as_tuple=True)[0]

        n = min(len(teacher_resp_pos), len(student_resp_pos))
        if n == 0:
            continue

        # Shift -1: write teacher signal at the LOGIT positions whose predictions
        # are the response tokens, not at the response token positions themselves.
        # This aligns with the loss's `token_mask[:, 1:]` shifted-mask semantics.
        teacher_pred_pos = teacher_resp_pos[:n] - 1
        student_pred_pos = student_resp_pos[:n] - 1

        # Guard against responses that start at position 0 (no prompt; very rare).
        # Drop the leading entry in that edge case.
        if (teacher_pred_pos < 0).any() or (student_pred_pos < 0).any():
            keep = (teacher_pred_pos >= 0) & (student_pred_pos >= 0)
            teacher_pred_pos = teacher_pred_pos[keep]
            student_pred_pos = student_pred_pos[keep]
            if len(student_pred_pos) == 0:
                continue

        aligned_logits[i, student_pred_pos] = teacher_topk_logits[i, teacher_pred_pos]
        aligned_indices[i, student_pred_pos] = teacher_topk_indices[i, teacher_pred_pos]

    return aligned_logits, aligned_indices


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
        "adv_estimator": {
            "name": "grpo",
            "normalize_rewards": False,
            "use_leave_one_out_baseline": False,
            "minus_baseline": False,
        },
        "async_grpo": {
            "enabled": False,
            "max_trajectory_age_steps": 1,
            "in_flight_weight_updates": False,
            "recompute_kv_cache_after_weight_updates": False,
        },
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

    # Propagate the EMA-teacher flag into the policy config so the policy
    # workers allocate the EMA state dict at init time (paper Table 12).
    if sdpo_config.get("use_ema_teacher", False):
        grpo_master_config["policy"] = dict(grpo_master_config["policy"])
        grpo_master_config["policy"]["init_ema_teacher"] = True

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        _grpo_loss_fn,  # discard
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
    topk_logits_k = sdpo_cfg["topk_logits_k"]
    reprompt_template = sdpo_cfg.get("reprompt_template", _DEFAULT_REPROMPT_TEMPLATE)
    remove_thinking = sdpo_cfg.get("remove_thinking_from_demo", False)
    dont_self = sdpo_cfg.get("dont_reprompt_on_self_success", False)
    feedback_source = sdpo_cfg.get("feedback_source", "peer_rollout")
    env_feedback_template = sdpo_cfg.get(
        "env_feedback_template", _DEFAULT_ENV_FEEDBACK_TEMPLATE
    )
    combined_template = sdpo_cfg.get("combined_template", _DEFAULT_COMBINED_TEMPLATE)
    combined_success_template = sdpo_cfg.get(
        "combined_success_template", _DEFAULT_COMBINED_SUCCESS_TEMPLATE
    )
    use_ema_teacher = sdpo_cfg.get("use_ema_teacher", False)
    ema_alpha = sdpo_cfg.get("ema_alpha", 0.01)
    make_div_by = policy_cfg.get("make_sequence_length_divisible_by", 1)

    # Run initial validation if requested
    if val_at_start and current_step == 0:
        print("\nRunning initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        # grpo.validate() reads master_config["grpo"] for these keys; provide a shim.
        _validate_cfg = {
            **master_config,
            "grpo": {
                "max_val_samples": sdpo_cfg["max_val_samples"],
                "val_batch_size": sdpo_cfg["val_batch_size"],
                "max_rollout_turns": sdpo_cfg["max_rollout_turns"],
            },
        }
        val_metrics, _ = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            total_steps,
            _validate_cfg,
        )
        logger.log_metrics(val_metrics, step=total_steps)

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

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
                    repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(num_generations)
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
                        refit_policy_generation(policy, policy_generation, colocated_inference)
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                with timer.time("generation"):
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = run_async_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=policy_cfg["max_total_sequence_length"],
                            max_rollout_turns=sdpo_cfg["max_rollout_turns"],
                            greedy=False,
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
                                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                            else:
                                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])
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
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": torch.ones(repeated_batch.size, dtype=torch.float32),
                        }
                    )
                    train_data.to("cpu")

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
                        feedback_source=feedback_source,
                        env_feedback_template=env_feedback_template,
                        combined_template=combined_template,
                        combined_success_template=combined_success_template,
                    )
                    teacher_data.to("cpu")

                frac_with_demo = sdpo_mask.float().mean().item()
                metrics["sdpo/frac_with_demo_pre_train"] = frac_with_demo
                print(
                    f"  SDPO: {sdpo_mask.sum().item()}/{len(sdpo_mask)} samples "
                    f"have demonstrations ({100 * frac_with_demo:.1f}%)",
                    flush=True,
                )

                # ── Sample logging for paper-F.3 format verification ─────────
                log_every = sdpo_cfg.get("log_sample_every_n_steps", 1)
                if log_every > 0 and ((total_steps + 1) % log_every == 0):
                    _log_env_feedback_sample(
                        step=total_steps + 1,
                        message_logs=repeated_batch["message_log"],
                        rewards=rewards,
                        success_threshold=success_threshold,
                        num_generations=num_generations,
                        feedback_source=feedback_source,
                        reprompt_template=reprompt_template,
                        env_feedback_template=env_feedback_template,
                        combined_template=combined_template,
                        combined_success_template=combined_success_template,
                        dont_reprompt_on_self_success=dont_self,
                        max_chars=sdpo_cfg.get("log_sample_max_chars", 2000),
                        metrics=metrics,
                    )

                # ── Compute teacher top-k logits ──────────────────────────────
                print("Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("Computing teacher top-k logits...", flush=True)
                with timer.time("teacher_topk_logits"):
                    teacher_topk = policy.get_topk_logits(
                        teacher_data,
                        k=topk_logits_k,
                        timer=timer,
                        use_ema=use_ema_teacher,
                    )
                    teacher_topk_logits_raw = teacher_topk["topk_logits"]
                    teacher_topk_indices_raw = teacher_topk["topk_indices"]

                # Align teacher top-k tensors to student sequence positions
                with timer.time("teacher_topk_alignment"):
                    student_seq_len = train_data["input_ids"].shape[1]
                    teacher_topk_logits_aligned, teacher_topk_indices_aligned = align_teacher_topk(
                        teacher_topk_logits=teacher_topk_logits_raw.cpu(),
                        teacher_topk_indices=teacher_topk_indices_raw.cpu(),
                        teacher_token_mask=teacher_data["token_mask"].cpu(),
                        student_seq_len=student_seq_len,
                        student_token_mask=train_data["token_mask"].cpu(),
                    )

                train_data["teacher_topk_logits"] = teacher_topk_logits_aligned
                train_data["teacher_topk_indices"] = teacher_topk_indices_aligned
                train_data["sdpo_mask"] = sdpo_mask.float()
                del (
                    teacher_data,
                    teacher_topk,
                    teacher_topk_logits_raw,
                    teacher_topk_indices_raw,
                    teacher_topk_logits_aligned,
                    teacher_topk_indices_aligned,
                )

                # ── Trust-region anchor to init policy (paper Table 4) ────────
                # When loss_fn.reference_policy_kl_penalty > 0, plumb the
                # student's snapshot logprobs (prev) and the frozen-init
                # reference logprobs at sampled tokens. The SDPO loss adds
                # beta * KL(student || ref) summed over response positions.
                ref_kl_penalty = master_config["loss_fn"].get(
                    "reference_policy_kl_penalty", 0.0
                )
                if ref_kl_penalty > 0.0:
                    print("Computing prev + reference logprobs...", flush=True)
                    logprob_data = BatchedDataDict(
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask": train_data["token_mask"],
                            "sample_mask": train_data["sample_mask"],
                        }
                    )
                    with timer.time("prev_logprobs"):
                        train_data["prev_logprobs"] = policy.get_logprobs(
                            logprob_data, timer=timer
                        )["logprobs"]
                    with timer.time("reference_policy_logprobs"):
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                logprob_data, timer=timer
                            )["reference_logprobs"]
                        )
                    del logprob_data

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

                # ── EMA teacher update (paper Table 12, alpha=0.01) ──────────
                if use_ema_teacher:
                    policy.update_ema_teacher_state(ema_alpha, timer=timer)
                    metrics["sdpo/ema_alpha"] = float(ema_alpha)

                # ── Metrics & logging ────────────────────────────────────────
                metrics["train/loss"] = train_results.get("loss", float("nan"))
                metrics["train/grad_norm"] = train_results.get("grad_norm", float("nan"))
                metrics["train/mean_reward"] = rewards.mean().item()
                metrics["train/success_fraction"] = (rewards >= success_threshold).float().mean().item()

                # Aggregate SDPO-specific metrics from the loss function
                for k, v in train_results.get("all_mb_metrics", {}).items():
                    if "sdpo" in k:
                        metrics[k] = sum(v) / len(v) if isinstance(v, list) else v

                num_valid_tokens = int(
                    (train_data["token_mask"] * train_data["sample_mask"].unsqueeze(-1)).sum().item()
                )
                total_valid_tokens += num_valid_tokens
                metrics["train/num_valid_tokens"] = num_valid_tokens
                metrics["train/total_valid_tokens"] = total_valid_tokens
                consumed_samples += repeated_batch.size
                metrics["train/consumed_samples"] = consumed_samples

                # ── Validation ───────────────────────────────────────────────
                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs) and (current_step + 1 == len(dataloader))
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (val_at_end and is_last_step):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(policy, policy_generation, colocated_inference)
                        POLICY_GENERATION_STALE = False
                    # grpo.validate() reads master_config["grpo"] for these keys; provide a shim.
                    _validate_cfg = {
                        **master_config,
                        "grpo": {
                            "max_val_samples": sdpo_cfg["max_val_samples"],
                            "val_batch_size": sdpo_cfg["val_batch_size"],
                            "max_rollout_turns": sdpo_cfg["max_rollout_turns"],
                        },
                    }
                    val_metrics, _ = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        total_steps,
                        _validate_cfg,
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

                timeout.mark_iteration()
                should_save_by_step = is_last_step or total_steps % master_config["checkpointing"]["save_period"] == 0
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (should_save_by_step or should_save_by_timeout):
                    # Track metric for top-k checkpointing
                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if ":" in full_metric_name:
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if metrics_source and metric_name in metrics_source:
                            sdpo_save_state[full_metric_name] = metrics_source[metric_name]

                    print(f"Saving checkpoint for step {total_steps}...", flush=True)
                    checkpoint_path = checkpointer.init_tmp_checkpoint(total_steps, sdpo_save_state, master_config)
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=(
                            os.path.join(checkpoint_path, "policy", "optimizer")
                            if checkpointer.save_optimizer
                            else None
                        ),
                        tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                        checkpointing_cfg=master_config["checkpointing"],
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

                if should_save_by_timeout:
                    print("Timeout reached, stopping training early.", flush=True)
                    return

        current_epoch += 1
        current_step = 0
        sdpo_save_state["current_epoch"] = current_epoch
        sdpo_save_state["current_step"] = 0

    print("SDPO training complete.", flush=True)
