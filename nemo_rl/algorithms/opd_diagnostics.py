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
#
"""Legacy-compatible OPD diagnostic payload builders.

This module intentionally preserves the payload schemas emitted by the original
MOPD research branch while keeping diagnostic mechanics out of the Super async
training loop.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig


def _opd_config(master_config: Any) -> OnPolicyDistillationConfig:
    cfg = (
        master_config.get("on_policy_distillation")
        if isinstance(master_config, dict)
        else getattr(master_config, "on_policy_distillation", None)
    )
    if cfg is None:
        return OnPolicyDistillationConfig()
    if isinstance(cfg, OnPolicyDistillationConfig):
        return cfg
    return OnPolicyDistillationConfig.model_validate(cfg)


def _should_log_opd_sample_stats(master_config: Any, step: int) -> bool:
    """Whether to write compact per-sample OPD reward/gap stats for this step."""
    if not opd_module.is_opd_enabled(master_config):
        return False

    opd_cfg = _opd_config(master_config)
    if not opd_cfg.log_sample_stats:
        return False

    return (step + 1) % opd_cfg.sample_stats_log_period == 0


def _should_log_opd_token_stats(master_config: Any, step: int) -> bool:
    """Whether to write packed per-token OPD logprob tensors for this step."""
    if not opd_module.is_opd_enabled(master_config):
        return False

    opd_cfg = _opd_config(master_config)
    if not opd_cfg.log_token_stats:
        return False

    return (step + 1) % opd_cfg.token_stats_log_period == 0


def _should_log_opd_topk_stats(master_config: Any, step: int) -> bool:
    """Whether to write packed per-token OPD top-k tensors for this step."""
    if not opd_module.is_opd_enabled(master_config):
        return False

    opd_cfg = _opd_config(master_config)
    if not opd_cfg.log_topk_stats:
        return False

    return (step + 1) % opd_cfg.topk_stats_log_period == 0


def _get_opd_topk_stats_k(master_config: Any) -> int:
    opd_cfg = _opd_config(master_config)
    return opd_cfg.topk_stats_k


def _get_opd_topk_stats_max_tokens(master_config: Any) -> Optional[int]:
    opd_cfg = _opd_config(master_config)
    return opd_cfg.topk_stats_max_tokens


OPD_TOPK_STATS_MODE_STUDENT_ONLINE_TEACHER_DEFERRED = "student_online_teacher_deferred"
OPD_TOPK_STATS_MODE_ONLINE = "online"


def _get_opd_topk_stats_mode(master_config: Any) -> str:
    opd_cfg = _opd_config(master_config)
    return opd_cfg.topk_stats_mode


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return Pearson correlation, or NaN when it is undefined."""
    x = x.detach().float().flatten().cpu()
    y = y.detach().float().flatten().cpu()
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if x.numel() < 2:
        return float("nan")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if denom.item() == 0.0:
        return float("nan")
    return (torch.dot(x_centered, y_centered) / denom).item()


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware zero-based average ranks for Spearman correlation."""
    values = values.detach().float().flatten().cpu()
    sorted_values, order = torch.sort(values)
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while (
            end < values.numel()
            and sorted_values[end].item() == sorted_values[start].item()
        ):
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return Spearman correlation with average ranks for ties."""
    x = x.detach().float().flatten().cpu()
    y = y.detach().float().flatten().cpu()
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if x.numel() < 2:
        return float("nan")
    return _pearson_corr(_average_ranks(x), _average_ranks(y))


def _get_opd_sample_response_logging_config(
    master_config: Any,
) -> tuple[bool, Optional[int]]:
    opd_cfg = _opd_config(master_config)
    return opd_cfg.log_sample_responses, opd_cfg.sample_response_max_tokens


def _prepare_seq_error_logging_fields(
    *,
    sample_mask: torch.Tensor,
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor],
    seq_mult_prob_error: Optional[torch.Tensor],
    masked_by_seq_logprob_error: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_mask = sample_mask.detach().to(device=device).float()
    if pre_seq_error_sample_loss_mask is None:
        pre_seq_error_sample_loss_mask = sample_mask.clone()
    else:
        pre_seq_error_sample_loss_mask = (
            pre_seq_error_sample_loss_mask.detach().to(device=device).float()
        )

    if seq_mult_prob_error is None:
        seq_mult_prob_error = torch.full_like(sample_mask, float("nan"))
    else:
        seq_mult_prob_error = seq_mult_prob_error.detach().to(device=device).float()

    if masked_by_seq_logprob_error is None:
        masked_by_seq_logprob_error = torch.zeros_like(sample_mask, dtype=torch.bool)
    else:
        masked_by_seq_logprob_error = (
            masked_by_seq_logprob_error.detach().to(device=device).bool()
        )

    for name, value in (
        ("pre_seq_error_sample_loss_mask", pre_seq_error_sample_loss_mask),
        ("seq_mult_prob_error", seq_mult_prob_error),
        ("masked_by_seq_logprob_error", masked_by_seq_logprob_error),
    ):
        if value.shape != sample_mask.shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} must match sample_mask "
                f"shape {tuple(sample_mask.shape)}."
            )

    return (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    )


def _decode_sample_responses(
    *,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    tokenizer: Any,
    max_tokens: Optional[int],
) -> tuple[list[str], list[bool]]:
    response_token_ids = []
    response_truncated = []
    for sample_input_ids, sample_token_mask in zip(
        input_ids.detach().cpu(), token_mask.detach().cpu().bool()
    ):
        sample_response_token_ids = sample_input_ids[sample_token_mask].tolist()
        if max_tokens is not None:
            response_truncated.append(len(sample_response_token_ids) > max_tokens)
            response_token_ids.append(sample_response_token_ids[:max_tokens])
        else:
            response_truncated.append(False)
            response_token_ids.append(sample_response_token_ids)

    responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)
    return responses, response_truncated


def _build_opd_sample_stats_log_data(
    *,
    step: int,
    tokenizer: Any,
    log_sample_responses: bool,
    sample_response_max_tokens: Optional[int],
    num_generations_per_prompt: Optional[int],
    input_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prev_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    input_lengths: torch.Tensor,
    repeated_batch: Any,
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor] = None,
    seq_mult_prob_error: Optional[torch.Tensor] = None,
    masked_by_seq_logprob_error: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build compact per-sample OPD stats and aggregate reward/gap correlations."""
    if teacher_logprobs.shape != prev_logprobs.shape:
        raise ValueError(
            f"Teacher logprobs shape {tuple(teacher_logprobs.shape)} does not match "
            f"student logprobs shape {tuple(prev_logprobs.shape)} for OPD sample stats."
        )
    if generation_logprobs.shape != prev_logprobs.shape:
        raise ValueError(
            f"Generation logprobs shape {tuple(generation_logprobs.shape)} does not match "
            f"student logprobs shape {tuple(prev_logprobs.shape)} for OPD sample stats."
        )

    prev_logprobs = prev_logprobs.detach().float()
    device = prev_logprobs.device
    teacher_logprobs = teacher_logprobs.detach().to(device=device).float()
    generation_logprobs = generation_logprobs.detach().to(device=device).float()
    token_mask = token_mask.detach().to(device=device).bool()
    sample_mask = sample_mask.detach().to(device=device).float()
    (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    ) = _prepare_seq_error_logging_fields(
        sample_mask=sample_mask,
        pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
        seq_mult_prob_error=seq_mult_prob_error,
        masked_by_seq_logprob_error=masked_by_seq_logprob_error,
        device=device,
    )
    rewards = rewards.detach().to(device=device).float()
    input_lengths = input_lengths.detach().to(device=device)

    token_count = token_mask.sum(dim=-1)
    token_mask_float = token_mask.to(prev_logprobs.dtype)
    gap = teacher_logprobs - prev_logprobs
    finite_gap_mask = torch.isfinite(gap)
    finite_response_gap_count = (finite_gap_mask & token_mask).sum(dim=-1)
    has_all_finite_response_gaps = finite_response_gap_count == token_count
    finite_gap = torch.where(finite_gap_mask, gap, torch.zeros_like(gap))
    safe_token_count = token_count.clamp_min(1).to(prev_logprobs.dtype)
    gap_sum = (finite_gap * token_mask_float).sum(dim=-1)
    gap_mean = gap_sum / safe_token_count
    abs_gap_mean = (finite_gap.abs() * token_mask_float).sum(dim=-1) / safe_token_count

    student_entropy_approx = (
        -torch.exp(prev_logprobs - generation_logprobs) * prev_logprobs
    )
    teacher_entropy_approx = (
        -torch.exp(teacher_logprobs - generation_logprobs) * teacher_logprobs
    )
    finite_student_entropy_mask = torch.isfinite(student_entropy_approx)
    finite_teacher_entropy_mask = torch.isfinite(teacher_entropy_approx)
    finite_response_student_entropy_count = (
        finite_student_entropy_mask & token_mask
    ).sum(dim=-1)
    finite_response_teacher_entropy_count = (
        finite_teacher_entropy_mask & token_mask
    ).sum(dim=-1)
    has_all_finite_response_student_entropy = (
        finite_response_student_entropy_count == token_count
    )
    has_all_finite_response_teacher_entropy = (
        finite_response_teacher_entropy_count == token_count
    )
    finite_student_entropy = torch.where(
        finite_student_entropy_mask,
        student_entropy_approx,
        torch.zeros_like(student_entropy_approx),
    )
    finite_teacher_entropy = torch.where(
        finite_teacher_entropy_mask,
        teacher_entropy_approx,
        torch.zeros_like(teacher_entropy_approx),
    )
    student_entropy_approx_sum = (finite_student_entropy * token_mask_float).sum(dim=-1)
    student_entropy_approx_mean = student_entropy_approx_sum / safe_token_count
    teacher_entropy_approx_sum = (finite_teacher_entropy * token_mask_float).sum(dim=-1)
    teacher_entropy_approx_mean = teacher_entropy_approx_sum / safe_token_count

    valid_for_reward_gap_correlation = (
        (sample_mask > 0)
        & (token_count > 0)
        & has_all_finite_response_gaps
        & torch.isfinite(rewards)
        & torch.isfinite(gap_mean)
        & torch.isfinite(abs_gap_mean)
    )
    valid_for_reward_entropy_correlation = (
        (sample_mask > 0)
        & (token_count > 0)
        & has_all_finite_response_student_entropy
        & torch.isfinite(rewards)
        & torch.isfinite(student_entropy_approx_mean)
    )
    valid_for_reward_teacher_entropy_correlation = (
        (sample_mask > 0)
        & (token_count > 0)
        & has_all_finite_response_teacher_entropy
        & torch.isfinite(rewards)
        & torch.isfinite(teacher_entropy_approx_mean)
    )
    valid_for_gap_entropy_correlation = (
        valid_for_reward_gap_correlation
        & has_all_finite_response_student_entropy
        & torch.isfinite(student_entropy_approx_mean)
    )
    valid_for_gap_teacher_entropy_correlation = (
        valid_for_reward_gap_correlation
        & has_all_finite_response_teacher_entropy
        & torch.isfinite(teacher_entropy_approx_mean)
    )
    valid_for_student_teacher_entropy_correlation = (
        (sample_mask > 0)
        & (token_count > 0)
        & has_all_finite_response_student_entropy
        & has_all_finite_response_teacher_entropy
        & torch.isfinite(student_entropy_approx_mean)
        & torch.isfinite(teacher_entropy_approx_mean)
    )
    valid_rewards_for_gap = rewards[valid_for_reward_gap_correlation]
    valid_gap_mean = gap_mean[valid_for_reward_gap_correlation]
    valid_abs_gap_mean = abs_gap_mean[valid_for_reward_gap_correlation]
    valid_rewards_for_entropy = rewards[valid_for_reward_entropy_correlation]
    valid_student_entropy_for_reward = student_entropy_approx_mean[
        valid_for_reward_entropy_correlation
    ]
    valid_rewards_for_teacher_entropy = rewards[
        valid_for_reward_teacher_entropy_correlation
    ]
    valid_teacher_entropy_for_reward = teacher_entropy_approx_mean[
        valid_for_reward_teacher_entropy_correlation
    ]
    valid_gap_mean_for_entropy = gap_mean[valid_for_gap_entropy_correlation]
    valid_abs_gap_mean_for_entropy = abs_gap_mean[valid_for_gap_entropy_correlation]
    valid_student_entropy_for_gap = student_entropy_approx_mean[
        valid_for_gap_entropy_correlation
    ]
    valid_gap_mean_for_teacher_entropy = gap_mean[
        valid_for_gap_teacher_entropy_correlation
    ]
    valid_abs_gap_mean_for_teacher_entropy = abs_gap_mean[
        valid_for_gap_teacher_entropy_correlation
    ]
    valid_teacher_entropy_for_gap = teacher_entropy_approx_mean[
        valid_for_gap_teacher_entropy_correlation
    ]
    valid_student_entropy_for_teacher_entropy = student_entropy_approx_mean[
        valid_for_student_teacher_entropy_correlation
    ]
    valid_teacher_entropy_for_student_entropy = teacher_entropy_approx_mean[
        valid_for_student_teacher_entropy_correlation
    ]

    metrics = {
        "on_policy_distillation/sample_stats/logged_samples": float(rewards.numel()),
        "on_policy_distillation/sample_stats/valid_samples": float(
            valid_for_reward_gap_correlation.sum().item()
        ),
        "on_policy_distillation/sample_stats/gap_mean": valid_gap_mean.mean().item()
        if valid_gap_mean.numel() > 0
        else float("nan"),
        "on_policy_distillation/sample_stats/abs_gap_mean": valid_abs_gap_mean.mean().item()
        if valid_abs_gap_mean.numel() > 0
        else float("nan"),
        "on_policy_distillation/sample_stats/student_entropy_approx_mean": valid_student_entropy_for_reward.mean().item()
        if valid_student_entropy_for_reward.numel() > 0
        else float("nan"),
        "on_policy_distillation/sample_stats/teacher_entropy_approx_mean": valid_teacher_entropy_for_reward.mean().item()
        if valid_teacher_entropy_for_reward.numel() > 0
        else float("nan"),
        "on_policy_distillation/sample_stats/reward_gap_pearson": _pearson_corr(
            valid_rewards_for_gap, valid_gap_mean
        ),
        "on_policy_distillation/sample_stats/reward_gap_spearman": _spearman_corr(
            valid_rewards_for_gap, valid_gap_mean
        ),
        "on_policy_distillation/sample_stats/reward_abs_gap_pearson": _pearson_corr(
            valid_rewards_for_gap, valid_abs_gap_mean
        ),
        "on_policy_distillation/sample_stats/reward_abs_gap_spearman": _spearman_corr(
            valid_rewards_for_gap, valid_abs_gap_mean
        ),
        "on_policy_distillation/sample_stats/reward_student_entropy_pearson": _pearson_corr(
            valid_rewards_for_entropy, valid_student_entropy_for_reward
        ),
        "on_policy_distillation/sample_stats/reward_student_entropy_spearman": _spearman_corr(
            valid_rewards_for_entropy, valid_student_entropy_for_reward
        ),
        "on_policy_distillation/sample_stats/reward_teacher_entropy_pearson": _pearson_corr(
            valid_rewards_for_teacher_entropy, valid_teacher_entropy_for_reward
        ),
        "on_policy_distillation/sample_stats/reward_teacher_entropy_spearman": _spearman_corr(
            valid_rewards_for_teacher_entropy, valid_teacher_entropy_for_reward
        ),
        "on_policy_distillation/sample_stats/gap_student_entropy_pearson": _pearson_corr(
            valid_gap_mean_for_entropy, valid_student_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/gap_student_entropy_spearman": _spearman_corr(
            valid_gap_mean_for_entropy, valid_student_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/abs_gap_student_entropy_pearson": _pearson_corr(
            valid_abs_gap_mean_for_entropy, valid_student_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/abs_gap_student_entropy_spearman": _spearman_corr(
            valid_abs_gap_mean_for_entropy, valid_student_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/gap_teacher_entropy_pearson": _pearson_corr(
            valid_gap_mean_for_teacher_entropy, valid_teacher_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/gap_teacher_entropy_spearman": _spearman_corr(
            valid_gap_mean_for_teacher_entropy, valid_teacher_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/abs_gap_teacher_entropy_pearson": _pearson_corr(
            valid_abs_gap_mean_for_teacher_entropy, valid_teacher_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/abs_gap_teacher_entropy_spearman": _spearman_corr(
            valid_abs_gap_mean_for_teacher_entropy, valid_teacher_entropy_for_gap
        ),
        "on_policy_distillation/sample_stats/student_teacher_entropy_pearson": _pearson_corr(
            valid_student_entropy_for_teacher_entropy,
            valid_teacher_entropy_for_student_entropy,
        ),
        "on_policy_distillation/sample_stats/student_teacher_entropy_spearman": _spearman_corr(
            valid_student_entropy_for_teacher_entropy,
            valid_teacher_entropy_for_student_entropy,
        ),
    }

    batch_size = rewards.numel()
    sample_indices = list(range(batch_size))
    log_data: dict[str, Any] = {
        "step": [step + 1] * batch_size,
        "sample_index": sample_indices,
        "reward": rewards.cpu().tolist(),
        "sample_loss_mask": sample_mask.cpu().tolist(),
        "pre_seq_error_sample_loss_mask": pre_seq_error_sample_loss_mask.cpu().tolist(),
        "seq_mult_prob_error": seq_mult_prob_error.cpu().tolist(),
        "masked_by_seq_logprob_error": masked_by_seq_logprob_error.cpu().tolist(),
        "input_length": input_lengths.cpu().tolist(),
        "num_response_tokens": token_count.cpu().tolist(),
        "num_finite_response_token_gaps": finite_response_gap_count.cpu().tolist(),
        "has_nonfinite_response_gap": (~has_all_finite_response_gaps).cpu().tolist(),
        "num_finite_response_token_student_entropy": finite_response_student_entropy_count.cpu().tolist(),
        "has_nonfinite_response_student_entropy": (
            ~has_all_finite_response_student_entropy
        )
        .cpu()
        .tolist(),
        "num_finite_response_token_teacher_entropy": finite_response_teacher_entropy_count.cpu().tolist(),
        "has_nonfinite_response_teacher_entropy": (
            ~has_all_finite_response_teacher_entropy
        )
        .cpu()
        .tolist(),
        "teacher_student_logprob_gap_sum": gap_sum.cpu().tolist(),
        "teacher_student_logprob_gap_mean": gap_mean.cpu().tolist(),
        "teacher_student_abs_logprob_gap_mean": abs_gap_mean.cpu().tolist(),
        "student_entropy_approx_sum": student_entropy_approx_sum.cpu().tolist(),
        "student_entropy_approx_mean": student_entropy_approx_mean.cpu().tolist(),
        "teacher_entropy_approx_sum": teacher_entropy_approx_sum.cpu().tolist(),
        "teacher_entropy_approx_mean": teacher_entropy_approx_mean.cpu().tolist(),
        "valid_for_reward_gap_correlation": valid_for_reward_gap_correlation.cpu().tolist(),
        "valid_for_reward_entropy_correlation": valid_for_reward_entropy_correlation.cpu().tolist(),
        "valid_for_reward_teacher_entropy_correlation": valid_for_reward_teacher_entropy_correlation.cpu().tolist(),
        "valid_for_gap_entropy_correlation": valid_for_gap_entropy_correlation.cpu().tolist(),
        "valid_for_gap_teacher_entropy_correlation": valid_for_gap_teacher_entropy_correlation.cpu().tolist(),
        "valid_for_student_teacher_entropy_correlation": valid_for_student_teacher_entropy_correlation.cpu().tolist(),
    }
    if num_generations_per_prompt is not None:
        num_generations_per_prompt = int(num_generations_per_prompt)
    if num_generations_per_prompt is not None and num_generations_per_prompt > 0:
        log_data["prompt_group_index"] = [
            i // num_generations_per_prompt for i in sample_indices
        ]
        log_data["generation_index"] = [
            i % num_generations_per_prompt for i in sample_indices
        ]
    for key in ("source_dataset_idx", "task_name", "ng_task_index"):
        if key in repeated_batch:
            value = repeated_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            else:
                value = list(value)
            if len(value) == batch_size:
                log_data[key] = value
    if "agent_ref" in repeated_batch:
        log_data["agent_ref"] = repeated_batch["agent_ref"]
    if log_sample_responses:
        (
            student_responses,
            student_response_truncated,
        ) = _decode_sample_responses(
            input_ids=input_ids,
            token_mask=token_mask,
            tokenizer=tokenizer,
            max_tokens=sample_response_max_tokens,
        )
        log_data["student_response"] = student_responses
        log_data["student_response_truncated"] = student_response_truncated

    return log_data, metrics


def _build_opd_token_stats_payload(
    *,
    step: int,
    num_generations_per_prompt: Optional[int],
    input_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prev_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    input_lengths: torch.Tensor,
    repeated_batch: Any,
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor] = None,
    seq_mult_prob_error: Optional[torch.Tensor] = None,
    masked_by_seq_logprob_error: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build packed response-token OPD tensors for torch.save."""
    if teacher_logprobs.shape != prev_logprobs.shape:
        raise ValueError(
            f"Teacher logprobs shape {tuple(teacher_logprobs.shape)} does not match "
            f"student logprobs shape {tuple(prev_logprobs.shape)} for OPD token stats."
        )
    if generation_logprobs.shape != prev_logprobs.shape:
        raise ValueError(
            f"Generation logprobs shape {tuple(generation_logprobs.shape)} does not match "
            f"student logprobs shape {tuple(prev_logprobs.shape)} for OPD token stats."
        )
    if (
        input_ids.shape != prev_logprobs.shape
        or token_mask.shape != prev_logprobs.shape
    ):
        raise ValueError(
            "input_ids, token_mask, and logprob tensors must have matching shapes "
            f"for OPD token stats. Got input_ids={tuple(input_ids.shape)}, "
            f"token_mask={tuple(token_mask.shape)}, logprobs={tuple(prev_logprobs.shape)}."
        )

    device = prev_logprobs.device
    token_mask = token_mask.detach().to(device=device).bool()
    batch_size = token_mask.shape[0]
    token_count = token_mask.sum(dim=-1)
    token_count_cpu = token_count.cpu()

    token_coords = token_mask.nonzero(as_tuple=False)
    token_sample_index = token_coords[:, 0].to(torch.int32).cpu()
    token_sequence_position = token_coords[:, 1].to(torch.int32).cpu()
    if token_sample_index.numel() > 0:
        token_response_position = torch.cat(
            [
                torch.arange(int(count), dtype=torch.int32)
                for count in token_count_cpu.tolist()
            ]
        )
    else:
        token_response_position = torch.empty(0, dtype=torch.int32)

    student_prev_logprobs = (
        prev_logprobs.detach().to(device=device).float()[token_mask].cpu()
    )
    teacher_logprobs = (
        teacher_logprobs.detach().to(device=device).float()[token_mask].cpu()
    )
    generation_logprobs = (
        generation_logprobs.detach().to(device=device).float()[token_mask].cpu()
    )
    (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    ) = _prepare_seq_error_logging_fields(
        sample_mask=sample_mask,
        pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
        seq_mult_prob_error=seq_mult_prob_error,
        masked_by_seq_logprob_error=masked_by_seq_logprob_error,
        device=sample_mask.device,
    )

    sample_indices = torch.arange(batch_size, dtype=torch.int32)
    payload: dict[str, Any] = {
        "format_version": 1,
        "description": (
            "Packed response-token OPD logprobs. "
            "teacher_student_logprob_gap = teacher_logprobs - student_prev_logprobs. "
            "student_entropy_approx and teacher_entropy_approx are importance-weighted "
            "sampled-token entropy estimates using generation_logprobs as the behavior policy."
        ),
        "step": int(step + 1),
        "num_generations_per_prompt": (
            int(num_generations_per_prompt)
            if num_generations_per_prompt is not None
            else None
        ),
        "sample_index": sample_indices,
        "reward": rewards.detach().float().cpu(),
        "sample_loss_mask": sample_mask.detach().float().cpu(),
        "pre_seq_error_sample_loss_mask": pre_seq_error_sample_loss_mask.detach()
        .float()
        .cpu(),
        "seq_mult_prob_error": seq_mult_prob_error.detach().float().cpu(),
        "masked_by_seq_logprob_error": masked_by_seq_logprob_error.detach()
        .bool()
        .cpu(),
        "input_length": input_lengths.detach().cpu(),
        "num_response_tokens": token_count_cpu,
        "token_sample_index": token_sample_index,
        "token_sequence_position": token_sequence_position,
        "token_response_position": token_response_position,
        "token_ids": input_ids.detach().to(device=device)[token_mask].cpu(),
        "student_prev_logprobs": student_prev_logprobs,
        "teacher_logprobs": teacher_logprobs,
        "generation_logprobs": generation_logprobs,
        "teacher_student_logprob_gap": teacher_logprobs - student_prev_logprobs,
        "student_entropy_approx": -torch.exp(
            student_prev_logprobs - generation_logprobs
        )
        * student_prev_logprobs,
        "teacher_entropy_approx": -torch.exp(teacher_logprobs - generation_logprobs)
        * teacher_logprobs,
    }

    if (
        payload["num_generations_per_prompt"] is not None
        and payload["num_generations_per_prompt"] > 0
    ):
        gpp = int(payload["num_generations_per_prompt"])
        payload["prompt_group_index"] = (sample_indices // gpp).to(torch.int32)
        payload["generation_index"] = (sample_indices % gpp).to(torch.int32)

    for key in ("source_dataset_idx", "task_name", "ng_task_index"):
        if key in repeated_batch:
            value = repeated_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            else:
                value = list(value)
            if len(value) == batch_size:
                payload[key] = value
    if "agent_ref" in repeated_batch:
        agent_ref = list(repeated_batch["agent_ref"])
        if len(agent_ref) == batch_size:
            payload["agent_ref"] = agent_ref

    metrics = {
        "on_policy_distillation/token_stats/logged_samples": float(batch_size),
        "on_policy_distillation/token_stats/logged_tokens": float(
            token_sample_index.numel()
        ),
    }
    return payload, metrics


def _align_next_token_topk_to_input_positions(
    topk_tensor: torch.Tensor,
    *,
    target_seq_len: int,
    fill_value: float | int,
) -> torch.Tensor:
    """Shift next-token top-k rows onto the input-token positions they predict."""
    if topk_tensor.ndim not in (2, 3):
        raise ValueError(
            "Expected top-k tensor with shape [B, S] or [B, S, K], got "
            f"{tuple(topk_tensor.shape)}"
        )

    batch_size, topk_seq_len = topk_tensor.shape[:2]
    aligned = torch.full(
        (batch_size, target_seq_len, *topk_tensor.shape[2:]),
        fill_value,
        dtype=topk_tensor.dtype,
        device=topk_tensor.device,
    )
    copy_len = min(max(target_seq_len - 1, 0), topk_seq_len)
    if copy_len > 0:
        aligned[:, 1 : copy_len + 1, ...] = topk_tensor[:, :copy_len, ...]
    return aligned


def _compute_topk_overlap_metrics_chunked(
    *,
    token_ids: torch.Tensor,
    student_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    chunk_size: int = 262_144,
) -> dict[str, torch.Tensor]:
    """Compute top-k set/probability overlap metrics without N x K x K materialization.

    Probability metrics are truncated/conditional: each model's probabilities
    are softmax-normalized over its own saved top-k logits, not over the full
    vocabulary. Missing tokens outside a model's top-k receive zero mass.
    """
    num_tokens = int(token_ids.numel())
    k = int(student_ids.shape[-1]) if student_ids.ndim == 2 else 0

    out_intersection = torch.zeros(num_tokens, dtype=torch.int16)
    out_jaccard = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_weighted_jaccard = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_probability_overlap = torch.full(
        (num_tokens,), float("nan"), dtype=torch.float32
    )
    out_teacher_mass_on_student = torch.full(
        (num_tokens,), float("nan"), dtype=torch.float32
    )
    out_student_mass_on_teacher = torch.full(
        (num_tokens,), float("nan"), dtype=torch.float32
    )
    out_in_support_correction = torch.full(
        (num_tokens,), float("nan"), dtype=torch.float32
    )
    out_top1_agreement = torch.zeros(num_tokens, dtype=torch.bool)
    out_realized_in_teacher = torch.zeros(num_tokens, dtype=torch.bool)
    out_realized_in_student = torch.zeros(num_tokens, dtype=torch.bool)
    out_teacher_realized_rank = torch.full((num_tokens,), -1, dtype=torch.int16)
    out_student_realized_rank = torch.full((num_tokens,), -1, dtype=torch.int16)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        s_ids = student_ids[start:end].to(torch.int64)
        t_ids = teacher_ids[start:end].to(torch.int64)
        tok = token_ids[start:end].to(torch.int64)
        valid = (s_ids >= 0).all(dim=-1) & (t_ids >= 0).all(dim=-1)
        if valid.sum().item() == 0:
            continue

        s_ids_v = s_ids[valid]
        t_ids_v = t_ids[valid]
        tok_v = tok[valid]
        s_probs = torch.softmax(student_logits[start:end][valid].float(), dim=-1)
        t_probs = torch.softmax(teacher_logits[start:end][valid].float(), dim=-1)

        n_valid = int(s_ids_v.shape[0])
        student_token_matched = torch.zeros(n_valid, k, dtype=torch.bool)
        teacher_token_matched = torch.zeros(n_valid, k, dtype=torch.bool)
        teacher_prob_on_student_tokens = torch.zeros(n_valid, k, dtype=torch.float32)
        student_prob_on_teacher_tokens = torch.zeros(n_valid, k, dtype=torch.float32)

        for teacher_rank in range(k):
            matches = s_ids_v == t_ids_v[:, teacher_rank : teacher_rank + 1]
            student_token_matched |= matches
            teacher_token_matched[:, teacher_rank] = matches.any(dim=-1)
            teacher_prob_on_student_tokens += (
                matches.float() * t_probs[:, teacher_rank : teacher_rank + 1]
            )
            student_prob_on_teacher_tokens[:, teacher_rank] = (
                matches.float() * s_probs
            ).sum(dim=-1)

        intersection = student_token_matched.sum(dim=-1).to(torch.int16)
        jaccard = intersection.float() / (2 * k - intersection.float()).clamp_min(1.0)

        overlap_mass = torch.minimum(s_probs, teacher_prob_on_student_tokens).sum(
            dim=-1
        )
        teacher_only_mass = torch.where(
            teacher_token_matched, torch.zeros_like(t_probs), t_probs
        ).sum(dim=-1)
        union_mass = (
            torch.maximum(s_probs, teacher_prob_on_student_tokens).sum(dim=-1)
            + teacher_only_mass
        )
        weighted_jaccard = overlap_mass / union_mass.clamp_min(1e-12)

        teacher_mass_on_student = teacher_prob_on_student_tokens.sum(dim=-1)
        student_mass_on_teacher = student_prob_on_teacher_tokens.sum(dim=-1)

        in_support_correction_mass = torch.clamp(
            teacher_prob_on_student_tokens - s_probs, min=0.0
        ).sum(dim=-1)
        total_correction_mass = in_support_correction_mass + teacher_only_mass
        in_support_correction_fraction = torch.where(
            total_correction_mass > 1e-12,
            in_support_correction_mass / total_correction_mass,
            torch.full_like(total_correction_mass, float("nan")),
        )

        realized_in_student = s_ids_v == tok_v.unsqueeze(-1)
        realized_in_teacher = t_ids_v == tok_v.unsqueeze(-1)
        student_has_realized = realized_in_student.any(dim=-1)
        teacher_has_realized = realized_in_teacher.any(dim=-1)
        student_realized_rank = torch.where(
            student_has_realized,
            realized_in_student.to(torch.int16).argmax(dim=-1).to(torch.int16),
            torch.full((n_valid,), -1, dtype=torch.int16),
        )
        teacher_realized_rank = torch.where(
            teacher_has_realized,
            realized_in_teacher.to(torch.int16).argmax(dim=-1).to(torch.int16),
            torch.full((n_valid,), -1, dtype=torch.int16),
        )

        chunk_indices = torch.arange(start, end)[valid]
        out_intersection[chunk_indices] = intersection
        out_jaccard[chunk_indices] = jaccard
        out_weighted_jaccard[chunk_indices] = weighted_jaccard
        out_probability_overlap[chunk_indices] = overlap_mass
        out_teacher_mass_on_student[chunk_indices] = teacher_mass_on_student
        out_student_mass_on_teacher[chunk_indices] = student_mass_on_teacher
        out_in_support_correction[chunk_indices] = in_support_correction_fraction
        out_top1_agreement[chunk_indices] = s_ids_v[:, 0] == t_ids_v[:, 0]
        out_realized_in_teacher[chunk_indices] = teacher_has_realized
        out_realized_in_student[chunk_indices] = student_has_realized
        out_teacher_realized_rank[chunk_indices] = teacher_realized_rank
        out_student_realized_rank[chunk_indices] = student_realized_rank

    return {
        "topk_intersection_size": out_intersection,
        "topk_jaccard": out_jaccard,
        "topk_weighted_jaccard": out_weighted_jaccard,
        "topk_probability_overlap": out_probability_overlap,
        "teacher_conditional_mass_on_student_topk": out_teacher_mass_on_student,
        "student_conditional_mass_on_teacher_topk": out_student_mass_on_teacher,
        "in_student_topk_correction_fraction": out_in_support_correction,
        "top1_agreement": out_top1_agreement,
        "realized_in_teacher_topk": out_realized_in_teacher,
        "realized_in_student_topk": out_realized_in_student,
        "teacher_realized_token_rank": out_teacher_realized_rank,
        "student_realized_token_rank": out_student_realized_rank,
    }


def _masked_mean_or_nan(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values.detach().float()
    finite = torch.isfinite(values)
    mask = mask.bool() & finite
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


def _masked_min_or_nan(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values.detach().float()
    finite = torch.isfinite(values)
    mask = mask.bool() & finite
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].min().item()


def _masked_max_or_nan(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values.detach().float()
    finite = torch.isfinite(values)
    mask = mask.bool() & finite
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].max().item()


def _add_masked_mean_min_max(
    metrics: dict[str, float],
    *,
    prefix: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    metrics[f"{prefix}_mean"] = _masked_mean_or_nan(values, mask)
    metrics[f"{prefix}_min"] = _masked_min_or_nan(values, mask)
    metrics[f"{prefix}_max"] = _masked_max_or_nan(values, mask)


def _bucket_entropy(mass: torch.Tensor) -> torch.Tensor:
    mass = mass.float().clamp(min=0.0, max=1.0)
    return torch.where(
        mass > 0.0,
        -mass * mass.clamp_min(1e-12).log(),
        torch.zeros_like(mass),
    )


def _bucket_cross_entropy(
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
) -> torch.Tensor:
    source_mass = source_mass.float().clamp(min=0.0, max=1.0)
    target_mass = target_mass.float().clamp(min=0.0, max=1.0)
    return torch.where(
        source_mass > 0.0,
        -source_mass * target_mass.clamp_min(1e-12).log(),
        torch.zeros_like(source_mass),
    )


def _compute_topk_full_vocab_terms_chunked(
    *,
    student_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_V_logsumexp: torch.Tensor,
    teacher_V_logsumexp: torch.Tensor,
    chunk_size: int = 262_144,
) -> dict[str, torch.Tensor]:
    """Compute exact saved top-k mass terms and coarsened entropy/CE terms.

    Own top-k probabilities are exact full-vocab probabilities because logits
    are normalized by V_logsumexp. Entropy is exact for the saved top-k head;
    tail entropy and residual cross-entropy terms treat all non-observed mass
    as a single bucket, so they are coarsened diagnostics rather than exact
    full-vocabulary entropy or cross-entropy.
    """
    num_tokens = int(student_ids.shape[0])
    k = int(student_ids.shape[-1]) if student_ids.ndim == 2 else 0

    out: dict[str, torch.Tensor] = {
        "student_topk_head_prob_mass": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_ex_topk_tail_prob_mass": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_topk_head_prob_mass": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_ex_topk_tail_prob_mass": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_topk_head_entropy": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_ex_topk_tail_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_topk_plus_tail_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_topk_head_entropy": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_ex_topk_tail_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_topk_plus_tail_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_topk_mass_in_teacher_topk": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_topk_mass_in_student_topk": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_to_teacher_topk_intersection_cross_entropy": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_to_student_topk_intersection_cross_entropy": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "student_to_teacher_topk_intersection_residual_cross_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
        "teacher_to_student_topk_intersection_residual_cross_entropy_bucket": torch.full(
            (num_tokens,), float("nan"), dtype=torch.float32
        ),
    }

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        s_ids = student_ids[start:end].to(torch.int64)
        t_ids = teacher_ids[start:end].to(torch.int64)
        valid = (
            (s_ids >= 0).all(dim=-1)
            & (t_ids >= 0).all(dim=-1)
            & torch.isfinite(student_V_logsumexp[start:end].float())
            & torch.isfinite(teacher_V_logsumexp[start:end].float())
        )
        if valid.sum().item() == 0:
            continue

        s_ids_v = s_ids[valid]
        t_ids_v = t_ids[valid]
        s_logprobs = student_logits[start:end][valid].float() - student_V_logsumexp[
            start:end
        ][valid].float().unsqueeze(-1)
        t_logprobs = teacher_logits[start:end][valid].float() - teacher_V_logsumexp[
            start:end
        ][valid].float().unsqueeze(-1)
        s_logprobs = torch.minimum(s_logprobs, torch.zeros_like(s_logprobs))
        t_logprobs = torch.minimum(t_logprobs, torch.zeros_like(t_logprobs))
        s_probs = s_logprobs.exp()
        t_probs = t_logprobs.exp()

        s_head_mass = s_probs.sum(dim=-1).clamp(min=0.0, max=1.0)
        t_head_mass = t_probs.sum(dim=-1).clamp(min=0.0, max=1.0)
        s_tail_mass = (1.0 - s_head_mass).clamp(min=0.0, max=1.0)
        t_tail_mass = (1.0 - t_head_mass).clamp(min=0.0, max=1.0)

        s_head_entropy = -(s_probs * s_logprobs).sum(dim=-1)
        t_head_entropy = -(t_probs * t_logprobs).sum(dim=-1)
        s_tail_entropy = _bucket_entropy(s_tail_mass)
        t_tail_entropy = _bucket_entropy(t_tail_mass)

        n_valid = int(s_ids_v.shape[0])
        s_intersection_mass = torch.zeros(n_valid, dtype=torch.float32)
        t_intersection_mass = torch.zeros(n_valid, dtype=torch.float32)
        s_to_t_intersection_ce = torch.zeros(n_valid, dtype=torch.float32)
        t_to_s_intersection_ce = torch.zeros(n_valid, dtype=torch.float32)

        for teacher_rank in range(k):
            matches = s_ids_v == t_ids_v[:, teacher_rank : teacher_rank + 1]
            if not matches.any():
                continue
            match = matches.float()
            teacher_lp = t_logprobs[:, teacher_rank : teacher_rank + 1]
            teacher_prob = t_probs[:, teacher_rank : teacher_rank + 1]

            s_intersection_mass += (match * s_probs).sum(dim=-1)
            t_intersection_mass += (match * teacher_prob).sum(dim=-1)
            s_to_t_intersection_ce += -(match * s_probs * teacher_lp).sum(dim=-1)
            t_to_s_intersection_ce += -(match * teacher_prob * s_logprobs).sum(dim=-1)

        s_residual_mass = (1.0 - s_intersection_mass).clamp(min=0.0, max=1.0)
        t_residual_mass = (1.0 - t_intersection_mass).clamp(min=0.0, max=1.0)
        s_to_t_bucket_ce = s_to_t_intersection_ce + _bucket_cross_entropy(
            s_residual_mass, t_residual_mass
        )
        t_to_s_bucket_ce = t_to_s_intersection_ce + _bucket_cross_entropy(
            t_residual_mass, s_residual_mass
        )

        chunk_indices = torch.arange(start, end)[valid]
        out["student_topk_head_prob_mass"][chunk_indices] = s_head_mass
        out["student_ex_topk_tail_prob_mass"][chunk_indices] = s_tail_mass
        out["teacher_topk_head_prob_mass"][chunk_indices] = t_head_mass
        out["teacher_ex_topk_tail_prob_mass"][chunk_indices] = t_tail_mass
        out["student_topk_head_entropy"][chunk_indices] = s_head_entropy
        out["student_ex_topk_tail_entropy_bucket"][chunk_indices] = s_tail_entropy
        out["student_topk_plus_tail_entropy_bucket"][chunk_indices] = (
            s_head_entropy + s_tail_entropy
        )
        out["teacher_topk_head_entropy"][chunk_indices] = t_head_entropy
        out["teacher_ex_topk_tail_entropy_bucket"][chunk_indices] = t_tail_entropy
        out["teacher_topk_plus_tail_entropy_bucket"][chunk_indices] = (
            t_head_entropy + t_tail_entropy
        )
        out["student_topk_mass_in_teacher_topk"][chunk_indices] = s_intersection_mass
        out["teacher_topk_mass_in_student_topk"][chunk_indices] = t_intersection_mass
        out["student_to_teacher_topk_intersection_cross_entropy"][chunk_indices] = (
            s_to_t_intersection_ce
        )
        out["teacher_to_student_topk_intersection_cross_entropy"][chunk_indices] = (
            t_to_s_intersection_ce
        )
        out["student_to_teacher_topk_intersection_residual_cross_entropy_bucket"][
            chunk_indices
        ] = s_to_t_bucket_ce
        out["teacher_to_student_topk_intersection_residual_cross_entropy_bucket"][
            chunk_indices
        ] = t_to_s_bucket_ce

    return out


TOPK_FULL_VOCAB_TERM_KEYS = (
    "student_topk_head_prob_mass",
    "student_ex_topk_tail_prob_mass",
    "teacher_topk_head_prob_mass",
    "teacher_ex_topk_tail_prob_mass",
    "student_topk_head_entropy",
    "student_ex_topk_tail_entropy_bucket",
    "student_topk_plus_tail_entropy_bucket",
    "teacher_topk_head_entropy",
    "teacher_ex_topk_tail_entropy_bucket",
    "teacher_topk_plus_tail_entropy_bucket",
    "student_topk_mass_in_teacher_topk",
    "teacher_topk_mass_in_student_topk",
    "student_to_teacher_topk_intersection_cross_entropy",
    "teacher_to_student_topk_intersection_cross_entropy",
    "student_to_teacher_topk_intersection_residual_cross_entropy_bucket",
    "teacher_to_student_topk_intersection_residual_cross_entropy_bucket",
)


def _build_opd_topk_offline_inputs_payload(
    *,
    step: int,
    num_generations_per_prompt: Optional[int],
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    input_lengths: torch.Tensor,
    repeated_batch: Any,
    k: int,
    max_logged_tokens: Optional[int],
    topk_stats_mode: str,
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor] = None,
    seq_mult_prob_error: Optional[torch.Tensor] = None,
    masked_by_seq_logprob_error: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build a lightweight top-k offline-input payload without model top-k inference."""
    if input_ids.shape != token_mask.shape:
        raise ValueError(
            "input_ids and token_mask must have matching shapes for OPD top-k "
            f"offline inputs. Got input_ids={tuple(input_ids.shape)}, "
            f"token_mask={tuple(token_mask.shape)}."
        )

    input_ids = input_ids.detach().cpu()
    token_mask = token_mask.detach().cpu().bool()
    sample_mask = sample_mask.detach().float().cpu()
    (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    ) = _prepare_seq_error_logging_fields(
        sample_mask=sample_mask,
        pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
        seq_mult_prob_error=seq_mult_prob_error,
        masked_by_seq_logprob_error=masked_by_seq_logprob_error,
        device=sample_mask.device,
    )
    rewards = rewards.detach().float().cpu()
    input_lengths = input_lengths.detach().cpu()
    token_count = token_mask.sum(dim=-1)
    total_response_tokens = int(token_count.sum().item())
    batch_size = int(input_ids.shape[0])
    sample_indices = torch.arange(batch_size, dtype=torch.int32)

    payload: dict[str, Any] = {
        "format_version": 1,
        "payload_type": "opd_topk_offline_inputs",
        "description": (
            "Lightweight OPD top-k offline-input payload. It stores sampled "
            "sequences and masks needed to compute teacher top-k diagnostics "
            "offline without running teacher top-k inference in the training loop."
        ),
        "step": int(step + 1),
        "topk_stats_mode": topk_stats_mode,
        "k": int(k),
        "num_generations_per_prompt": (
            int(num_generations_per_prompt)
            if num_generations_per_prompt is not None
            else None
        ),
        "max_logged_tokens": (
            int(max_logged_tokens) if max_logged_tokens is not None else None
        ),
        "num_total_response_tokens": total_response_tokens,
        "sample_index": sample_indices,
        "reward": rewards,
        "sample_loss_mask": sample_mask,
        "pre_seq_error_sample_loss_mask": pre_seq_error_sample_loss_mask,
        "seq_mult_prob_error": seq_mult_prob_error,
        "masked_by_seq_logprob_error": masked_by_seq_logprob_error,
        "input_length": input_lengths,
        "num_response_tokens": token_count.cpu(),
        "input_ids": input_ids,
        "token_mask": token_mask,
    }

    if (
        payload["num_generations_per_prompt"] is not None
        and payload["num_generations_per_prompt"] > 0
    ):
        gpp = int(payload["num_generations_per_prompt"])
        payload["prompt_group_index"] = (sample_indices // gpp).to(torch.int32)
        payload["generation_index"] = (sample_indices % gpp).to(torch.int32)

    for key in ("source_dataset_idx", "task_name", "ng_task_index"):
        if key in repeated_batch:
            value = repeated_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            else:
                value = list(value)
            if len(value) == batch_size:
                payload[key] = value
    if "agent_ref" in repeated_batch:
        agent_ref = list(repeated_batch["agent_ref"])
        if len(agent_ref) == batch_size:
            payload["agent_ref"] = agent_ref

    metrics = {
        "on_policy_distillation/topk_stats/student_online_teacher_deferred": float(
            topk_stats_mode == OPD_TOPK_STATS_MODE_STUDENT_ONLINE_TEACHER_DEFERRED
        ),
        "on_policy_distillation/topk_stats/online": 0.0,
        "on_policy_distillation/topk_stats/offline_inputs/logged_samples": float(
            batch_size
        ),
        "on_policy_distillation/topk_stats/offline_inputs/total_response_tokens": float(
            total_response_tokens
        ),
        "on_policy_distillation/topk_stats/k": float(k),
    }
    return payload, metrics


def _compute_student_topk_terms_chunked(
    *,
    token_ids: torch.Tensor,
    student_ids: torch.Tensor,
    student_logprobs: torch.Tensor,
    chunk_size: int = 262_144,
) -> dict[str, torch.Tensor]:
    """Compute single-model student top-k diagnostics for packed token rows."""
    num_tokens = int(token_ids.numel())
    out_realized_in_student = torch.zeros(num_tokens, dtype=torch.bool)
    out_student_realized_rank = torch.full((num_tokens,), -1, dtype=torch.int16)
    out_head_mass = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_tail_mass = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_head_entropy = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_tail_entropy = torch.full((num_tokens,), float("nan"), dtype=torch.float32)
    out_plus_tail_entropy = torch.full((num_tokens,), float("nan"), dtype=torch.float32)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        s_ids = student_ids[start:end].to(torch.int64)
        tok = token_ids[start:end].to(torch.int64)
        valid = (s_ids >= 0).all(dim=-1) & torch.isfinite(
            student_logprobs[start:end].float()
        ).all(dim=-1)
        if valid.sum().item() == 0:
            continue

        s_ids_v = s_ids[valid]
        tok_v = tok[valid]
        s_logprobs = student_logprobs[start:end][valid].float()
        s_logprobs = torch.minimum(s_logprobs, torch.zeros_like(s_logprobs))
        s_probs = s_logprobs.exp()

        head_mass = s_probs.sum(dim=-1).clamp(min=0.0, max=1.0)
        tail_mass = (1.0 - head_mass).clamp(min=0.0, max=1.0)
        head_entropy = -(s_probs * s_logprobs).sum(dim=-1)
        tail_entropy = _bucket_entropy(tail_mass)

        realized_in_student = s_ids_v == tok_v.unsqueeze(-1)
        student_has_realized = realized_in_student.any(dim=-1)
        student_realized_rank = torch.where(
            student_has_realized,
            realized_in_student.to(torch.int16).argmax(dim=-1).to(torch.int16),
            torch.full((int(s_ids_v.shape[0]),), -1, dtype=torch.int16),
        )

        chunk_indices = torch.arange(start, end)[valid]
        out_realized_in_student[chunk_indices] = student_has_realized
        out_student_realized_rank[chunk_indices] = student_realized_rank
        out_head_mass[chunk_indices] = head_mass
        out_tail_mass[chunk_indices] = tail_mass
        out_head_entropy[chunk_indices] = head_entropy
        out_tail_entropy[chunk_indices] = tail_entropy
        out_plus_tail_entropy[chunk_indices] = head_entropy + tail_entropy

    return {
        "realized_in_student_topk": out_realized_in_student,
        "student_realized_token_rank": out_student_realized_rank,
        "student_topk_head_prob_mass": out_head_mass,
        "student_ex_topk_tail_prob_mass": out_tail_mass,
        "student_topk_head_entropy": out_head_entropy,
        "student_ex_topk_tail_entropy_bucket": out_tail_entropy,
        "student_topk_plus_tail_entropy_bucket": out_plus_tail_entropy,
    }


def _build_opd_student_topk_stats_payload(
    *,
    step: int,
    num_generations_per_prompt: Optional[int],
    input_ids: torch.Tensor,
    student_prev_topk_logprobs: torch.Tensor,
    student_prev_topk_indices: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    input_lengths: torch.Tensor,
    repeated_batch: Any,
    max_logged_tokens: Optional[int],
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor] = None,
    seq_mult_prob_error: Optional[torch.Tensor] = None,
    masked_by_seq_logprob_error: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build packed response-token student top-k tensors for teacher replay mode."""
    if input_ids.shape != token_mask.shape:
        raise ValueError(
            "input_ids and token_mask must have matching shapes for OPD student "
            f"top-k stats. Got input_ids={tuple(input_ids.shape)}, "
            f"token_mask={tuple(token_mask.shape)}."
        )

    input_ids = input_ids.detach().cpu()
    token_mask = token_mask.detach().cpu().bool()
    sample_mask = sample_mask.detach().float().cpu()
    (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    ) = _prepare_seq_error_logging_fields(
        sample_mask=sample_mask,
        pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
        seq_mult_prob_error=seq_mult_prob_error,
        masked_by_seq_logprob_error=masked_by_seq_logprob_error,
        device=sample_mask.device,
    )
    rewards = rewards.detach().float().cpu()
    input_lengths = input_lengths.detach().cpu()
    target_seq_len = input_ids.shape[1]
    token_count = token_mask.sum(dim=-1)
    total_response_tokens = int(token_count.sum().item())

    token_coords = token_mask.nonzero(as_tuple=False)
    num_available_tokens = int(token_coords.shape[0])
    if max_logged_tokens is not None and num_available_tokens > max_logged_tokens:
        selection = (
            torch.linspace(
                0,
                num_available_tokens - 1,
                steps=max_logged_tokens,
                device=token_coords.device,
            )
            .round()
            .long()
        )
        token_coords = token_coords.index_select(0, selection)
    else:
        selection = None
    num_logged_tokens = int(token_coords.shape[0])

    student_prev_topk_indices = _align_next_token_topk_to_input_positions(
        student_prev_topk_indices.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=-1,
    )
    student_prev_topk_logprobs = _align_next_token_topk_to_input_positions(
        student_prev_topk_logprobs.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=0.0,
    )

    token_sample_index = token_coords[:, 0].to(torch.int32).cpu()
    token_sequence_position = token_coords[:, 1].to(torch.int32).cpu()
    response_positions = token_mask.cumsum(dim=-1) - 1
    token_response_position = (
        response_positions[token_coords[:, 0], token_coords[:, 1]].to(torch.int32).cpu()
    )

    rows = token_coords[:, 0]
    cols = token_coords[:, 1]
    token_ids = input_ids[rows, cols].to(torch.int32)
    student_ids = student_prev_topk_indices[rows, cols].to(torch.int32)
    student_logprobs = student_prev_topk_logprobs[rows, cols].to(torch.bfloat16)
    student_terms = _compute_student_topk_terms_chunked(
        token_ids=token_ids,
        student_ids=student_ids,
        student_logprobs=student_logprobs,
    )

    batch_size = int(input_ids.shape[0])
    sample_indices = torch.arange(batch_size, dtype=torch.int32)
    payload: dict[str, Any] = {
        "format_version": 2,
        "payload_type": "opd_topk_student_stats",
        "description": (
            "Packed response-token student-prev top-k diagnostics. Top-k rows "
            "are shifted from next-token outputs onto the input token positions "
            "they predict. Teacher top-k is computed by offline replay."
        ),
        "step": int(step + 1),
        "topk_stats_mode": OPD_TOPK_STATS_MODE_STUDENT_ONLINE_TEACHER_DEFERRED,
        "k": int(student_ids.shape[-1]) if student_ids.ndim == 2 else 0,
        "num_generations_per_prompt": (
            int(num_generations_per_prompt)
            if num_generations_per_prompt is not None
            else None
        ),
        "max_logged_tokens": (
            int(max_logged_tokens) if max_logged_tokens is not None else None
        ),
        "num_total_response_tokens": total_response_tokens,
        "num_logged_tokens": num_logged_tokens,
        "topk_tokens_subsampled": selection is not None,
        "sample_index": sample_indices,
        "reward": rewards,
        "sample_loss_mask": sample_mask,
        "pre_seq_error_sample_loss_mask": pre_seq_error_sample_loss_mask,
        "seq_mult_prob_error": seq_mult_prob_error,
        "masked_by_seq_logprob_error": masked_by_seq_logprob_error,
        "input_length": input_lengths,
        "num_response_tokens": token_count.cpu(),
        "token_sample_index": token_sample_index,
        "token_sequence_position": token_sequence_position,
        "token_response_position": token_response_position,
        "token_ids": token_ids,
        "student_prev_topk_token_ids": student_ids,
        "student_prev_topk_logprobs": student_logprobs,
    }
    payload.update(student_terms)

    if (
        payload["num_generations_per_prompt"] is not None
        and payload["num_generations_per_prompt"] > 0
    ):
        gpp = int(payload["num_generations_per_prompt"])
        payload["prompt_group_index"] = (sample_indices // gpp).to(torch.int32)
        payload["generation_index"] = (sample_indices % gpp).to(torch.int32)

    for key in ("source_dataset_idx", "task_name", "ng_task_index"):
        if key in repeated_batch:
            value = repeated_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            else:
                value = list(value)
            if len(value) == batch_size:
                payload[key] = value
    if "agent_ref" in repeated_batch:
        agent_ref = list(repeated_batch["agent_ref"])
        if len(agent_ref) == batch_size:
            payload["agent_ref"] = agent_ref

    if num_logged_tokens > 0:
        valid_token_mask = sample_mask[token_sample_index.long()] > 0
    else:
        valid_token_mask = torch.empty(0, dtype=torch.bool)

    metrics = {
        "on_policy_distillation/topk_stats/student_online_teacher_deferred": 1.0,
        "on_policy_distillation/topk_stats/online": 0.0,
        "on_policy_distillation/topk_stats/student/logged_samples": float(batch_size),
        "on_policy_distillation/topk_stats/student/total_response_tokens": float(
            total_response_tokens
        ),
        "on_policy_distillation/topk_stats/student/logged_tokens": float(
            num_logged_tokens
        ),
        "on_policy_distillation/topk_stats/student/subsampled": float(
            selection is not None
        ),
        "on_policy_distillation/topk_stats/k": float(payload["k"]),
        "on_policy_distillation/topk_stats/student/realized_in_topk_rate": _masked_mean_or_nan(
            payload["realized_in_student_topk"].float(), valid_token_mask
        ),
    }
    for key in (
        "student_topk_head_prob_mass",
        "student_ex_topk_tail_prob_mass",
        "student_topk_head_entropy",
        "student_ex_topk_tail_entropy_bucket",
        "student_topk_plus_tail_entropy_bucket",
    ):
        _add_masked_mean_min_max(
            metrics,
            prefix=f"on_policy_distillation/topk_stats/{key}",
            values=payload[key],
            mask=valid_token_mask,
        )
    return payload, metrics


def _build_opd_topk_stats_payload(
    *,
    step: int,
    num_generations_per_prompt: Optional[int],
    input_ids: torch.Tensor,
    student_topk_logits: torch.Tensor,
    student_topk_indices: torch.Tensor,
    student_V_logsumexp: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_V_logsumexp: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    input_lengths: torch.Tensor,
    repeated_batch: Any,
    max_logged_tokens: Optional[int],
    pre_seq_error_sample_loss_mask: Optional[torch.Tensor] = None,
    seq_mult_prob_error: Optional[torch.Tensor] = None,
    masked_by_seq_logprob_error: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build packed response-token OPD top-k tensors and truncated overlap metrics."""
    if input_ids.shape != token_mask.shape:
        raise ValueError(
            "input_ids and token_mask must have matching shapes for OPD top-k stats. "
            f"Got input_ids={tuple(input_ids.shape)}, token_mask={tuple(token_mask.shape)}."
        )
    if student_topk_indices.shape[-1] != teacher_topk_indices.shape[-1]:
        raise ValueError(
            "Student and teacher top-k tensors must use the same K. Got "
            f"{student_topk_indices.shape[-1]} and {teacher_topk_indices.shape[-1]}."
        )

    input_ids = input_ids.detach().cpu()
    token_mask = token_mask.detach().cpu().bool()
    sample_mask = sample_mask.detach().float().cpu()
    (
        pre_seq_error_sample_loss_mask,
        seq_mult_prob_error,
        masked_by_seq_logprob_error,
    ) = _prepare_seq_error_logging_fields(
        sample_mask=sample_mask,
        pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
        seq_mult_prob_error=seq_mult_prob_error,
        masked_by_seq_logprob_error=masked_by_seq_logprob_error,
        device=sample_mask.device,
    )
    rewards = rewards.detach().float().cpu()
    input_lengths = input_lengths.detach().cpu()
    target_seq_len = input_ids.shape[1]
    token_count = token_mask.sum(dim=-1)
    total_response_tokens = int(token_count.sum().item())

    token_coords = token_mask.nonzero(as_tuple=False)
    num_available_tokens = int(token_coords.shape[0])
    if max_logged_tokens is not None and num_available_tokens > max_logged_tokens:
        selection = (
            torch.linspace(
                0,
                num_available_tokens - 1,
                steps=max_logged_tokens,
                device=token_coords.device,
            )
            .round()
            .long()
        )
        token_coords = token_coords.index_select(0, selection)
    else:
        selection = None
    num_logged_tokens = int(token_coords.shape[0])

    student_topk_indices = _align_next_token_topk_to_input_positions(
        student_topk_indices.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=-1,
    )
    teacher_topk_indices = _align_next_token_topk_to_input_positions(
        teacher_topk_indices.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=-1,
    )
    student_topk_logits = _align_next_token_topk_to_input_positions(
        student_topk_logits.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=0.0,
    )
    teacher_topk_logits = _align_next_token_topk_to_input_positions(
        teacher_topk_logits.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=0.0,
    )
    student_V_logsumexp = _align_next_token_topk_to_input_positions(
        student_V_logsumexp.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=0.0,
    )
    teacher_V_logsumexp = _align_next_token_topk_to_input_positions(
        teacher_V_logsumexp.detach().cpu(),
        target_seq_len=target_seq_len,
        fill_value=0.0,
    )

    token_sample_index = token_coords[:, 0].to(torch.int32).cpu()
    token_sequence_position = token_coords[:, 1].to(torch.int32).cpu()
    response_positions = token_mask.cumsum(dim=-1) - 1
    token_response_position = (
        response_positions[token_coords[:, 0], token_coords[:, 1]].to(torch.int32).cpu()
    )

    rows = token_coords[:, 0]
    cols = token_coords[:, 1]
    token_ids = input_ids[rows, cols].to(torch.int32)
    student_ids = student_topk_indices[rows, cols].to(torch.int32)
    teacher_ids = teacher_topk_indices[rows, cols].to(torch.int32)
    student_logits = student_topk_logits[rows, cols].to(torch.bfloat16)
    teacher_logits = teacher_topk_logits[rows, cols].to(torch.bfloat16)
    student_logsumexp = student_V_logsumexp[rows, cols].to(torch.float32)
    teacher_logsumexp = teacher_V_logsumexp[rows, cols].to(torch.float32)

    overlap_metrics = _compute_topk_overlap_metrics_chunked(
        token_ids=token_ids,
        student_ids=student_ids,
        teacher_ids=teacher_ids,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )
    full_vocab_terms = _compute_topk_full_vocab_terms_chunked(
        student_ids=student_ids,
        teacher_ids=teacher_ids,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_V_logsumexp=student_logsumexp,
        teacher_V_logsumexp=teacher_logsumexp,
    )

    batch_size = int(input_ids.shape[0])
    sample_indices = torch.arange(batch_size, dtype=torch.int32)
    payload: dict[str, Any] = {
        "format_version": 2,
        "description": (
            "Packed response-token OPD top-k diagnostics. Top-k rows are shifted "
            "from next-token outputs onto the input token positions they predict. "
            "Weighted Jaccard/probability overlap metrics are truncated and "
            "conditional over each model's saved top-k logits, not full-vocab "
            "probability overlaps. V_logsumexp fields are full-vocab "
            "normalizers for exact own-top-k logprobs via logits - logsumexp. "
            "Head/tail mass fields use exact own-top-k full-vocab probabilities. "
            "Tail entropy and residual cross-entropy bucket fields collapse all "
            "unobserved vocabulary mass into one residual bucket."
        ),
        "step": int(step + 1),
        "k": int(student_ids.shape[-1]) if student_ids.ndim == 2 else 0,
        "num_generations_per_prompt": (
            int(num_generations_per_prompt)
            if num_generations_per_prompt is not None
            else None
        ),
        "max_logged_tokens": (
            int(max_logged_tokens) if max_logged_tokens is not None else None
        ),
        "num_total_response_tokens": total_response_tokens,
        "num_logged_tokens": num_logged_tokens,
        "topk_tokens_subsampled": selection is not None,
        "sample_index": sample_indices,
        "reward": rewards,
        "sample_loss_mask": sample_mask,
        "pre_seq_error_sample_loss_mask": pre_seq_error_sample_loss_mask,
        "seq_mult_prob_error": seq_mult_prob_error,
        "masked_by_seq_logprob_error": masked_by_seq_logprob_error,
        "input_length": input_lengths,
        "num_response_tokens": token_count.cpu(),
        "token_sample_index": token_sample_index,
        "token_sequence_position": token_sequence_position,
        "token_response_position": token_response_position,
        "token_ids": token_ids,
        "student_topk_token_ids": student_ids,
        "teacher_topk_token_ids": teacher_ids,
        "student_topk_logits": student_logits,
        "teacher_topk_logits": teacher_logits,
        "student_V_logsumexp": student_logsumexp,
        "teacher_V_logsumexp": teacher_logsumexp,
    }
    payload.update(overlap_metrics)
    payload.update(full_vocab_terms)

    if (
        payload["num_generations_per_prompt"] is not None
        and payload["num_generations_per_prompt"] > 0
    ):
        gpp = int(payload["num_generations_per_prompt"])
        payload["prompt_group_index"] = (sample_indices // gpp).to(torch.int32)
        payload["generation_index"] = (sample_indices % gpp).to(torch.int32)

    for key in ("source_dataset_idx", "task_name", "ng_task_index"):
        if key in repeated_batch:
            value = repeated_batch[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            else:
                value = list(value)
            if len(value) == batch_size:
                payload[key] = value
    if "agent_ref" in repeated_batch:
        agent_ref = list(repeated_batch["agent_ref"])
        if len(agent_ref) == batch_size:
            payload["agent_ref"] = agent_ref

    if num_logged_tokens > 0:
        valid_token_mask = sample_mask[token_sample_index.long()] > 0
    else:
        valid_token_mask = torch.empty(0, dtype=torch.bool)

    metrics = {
        "on_policy_distillation/topk_stats/logged_samples": float(batch_size),
        "on_policy_distillation/topk_stats/total_response_tokens": float(
            total_response_tokens
        ),
        "on_policy_distillation/topk_stats/logged_tokens": float(num_logged_tokens),
        "on_policy_distillation/topk_stats/subsampled": float(selection is not None),
        "on_policy_distillation/topk_stats/k": float(payload["k"]),
        "on_policy_distillation/topk_stats/mean_jaccard": _masked_mean_or_nan(
            payload["topk_jaccard"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/mean_weighted_jaccard": _masked_mean_or_nan(
            payload["topk_weighted_jaccard"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/mean_probability_overlap": _masked_mean_or_nan(
            payload["topk_probability_overlap"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/teacher_mass_on_student_topk": _masked_mean_or_nan(
            payload["teacher_conditional_mass_on_student_topk"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/student_mass_on_teacher_topk": _masked_mean_or_nan(
            payload["student_conditional_mass_on_teacher_topk"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/in_student_topk_correction_fraction": _masked_mean_or_nan(
            payload["in_student_topk_correction_fraction"], valid_token_mask
        ),
        "on_policy_distillation/topk_stats/top1_agreement_rate": _masked_mean_or_nan(
            payload["top1_agreement"].float(), valid_token_mask
        ),
        "on_policy_distillation/topk_stats/realized_in_teacher_topk_rate": _masked_mean_or_nan(
            payload["realized_in_teacher_topk"].float(), valid_token_mask
        ),
        "on_policy_distillation/topk_stats/realized_in_student_topk_rate": _masked_mean_or_nan(
            payload["realized_in_student_topk"].float(), valid_token_mask
        ),
    }
    for key in TOPK_FULL_VOCAB_TERM_KEYS:
        _add_masked_mean_min_max(
            metrics,
            prefix=f"on_policy_distillation/topk_stats/{key}",
            values=payload[key],
            mask=valid_token_mask,
        )
    return payload, metrics
