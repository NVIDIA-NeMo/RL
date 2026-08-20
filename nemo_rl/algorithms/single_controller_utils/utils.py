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

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {"probs_ratio_max", "probs_ratio_clamped_max"}
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def reduce_advantage_pump_metrics(
    rewards: list[torch.Tensor],
    masked_advantages: list[torch.Tensor],
    sequence_lengths: list[int],
    num_invalid_tool_calls: list[torch.Tensor],
    num_malformed_thinking: list[torch.Tensor],
    num_assistant_messages: list[torch.Tensor],
) -> dict[str, float]:
    """Reduce per-step accumulators from _advantage_stage into step scalars.

    Args:
        rewards: One tensor per advantage_stage call; each row a sample reward.
        masked_advantages: Token-masked advantages, one tensor per call.
        sequence_lengths: All input_lengths trained on this step.
        num_invalid_tool_calls: Per-row invalid-tool-call message counts.
        num_malformed_thinking: Per-row malformed-thinking message counts.
        num_assistant_messages: Per-row generated assistant-message counts.

    Returns:
        Reward/advantage/token metrics plus message violation counts and rates.
    """
    out: dict[str, float] = {}
    if rewards:
        out["reward"] = float(torch.cat([r.flatten() for r in rewards]).mean())
    if masked_advantages:
        cat = torch.cat([a.flatten() for a in masked_advantages])
        if cat.numel() > 0:
            out["advantages/mean"] = float(cat.mean())
            out["advantages/max"] = float(cat.max())
            out["advantages/min"] = float(cat.min())
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    if num_assistant_messages:
        assistant_count = sum(
            int(counts.sum().item()) for counts in num_assistant_messages
        )
        invalid_count = sum(
            int(counts.sum().item()) for counts in num_invalid_tool_calls
        )
        malformed_count = sum(
            int(counts.sum().item()) for counts in num_malformed_thinking
        )
        denominator = max(assistant_count, 1)
        out["num_invalid_tool_calls"] = float(invalid_count)
        out["num_malformed_thinking"] = float(malformed_count)
        out["num_assistant_messages"] = float(assistant_count)
        out["invalid_tool_call_rate"] = invalid_count / denominator
        out["malformed_thinking_rate"] = malformed_count / denominator
    return out


def apply_message_level_advantage_penalties(
    advantages: torch.Tensor,
    *,
    invalid_tool_call_mask: torch.Tensor,
    malformed_thinking_mask: torch.Tensor,
    invalid_tool_call_advantage: float | None,
    malformed_thinking_advantage: float | None,
) -> torch.Tensor:
    """Overwrite flagged token advantages while leaving valid tokens unchanged.

    Invalid-tool-call penalties take precedence when both masks select the same
    token, matching the legacy GRPO message-level implementation.

    Args:
        advantages: Per-token advantages of shape (batch, seq).
        invalid_tool_call_mask: Bool mask of the same shape as ``advantages``;
            True at tokens produced by an invalid tool call.
        malformed_thinking_mask: Bool mask of the same shape as ``advantages``;
            True at tokens produced by malformed thinking.
        invalid_tool_call_advantage: Value to overwrite flagged tokens with, or
            ``None`` to leave the invalid-tool-call branch disabled.
        malformed_thinking_advantage: Value to overwrite flagged tokens with, or
            ``None`` to leave the malformed-thinking branch disabled.

    Returns:
        New tensor with penalties applied. The original ``advantages`` object is
        returned unchanged when both advantages are ``None``.

    Raises:
        ValueError: If either mask shape does not match ``advantages``.
    """
    if invalid_tool_call_mask.shape != advantages.shape:
        raise ValueError(
            "invalid_tool_call_mask shape "
            f"{tuple(invalid_tool_call_mask.shape)} does not match advantages "
            f"{tuple(advantages.shape)}"
        )
    if malformed_thinking_mask.shape != advantages.shape:
        raise ValueError(
            "malformed_thinking_mask shape "
            f"{tuple(malformed_thinking_mask.shape)} does not match advantages "
            f"{tuple(advantages.shape)}"
        )

    result = advantages
    if malformed_thinking_advantage is not None:
        result = torch.where(
            malformed_thinking_mask.bool(),
            torch.as_tensor(
                malformed_thinking_advantage,
                dtype=advantages.dtype,
                device=advantages.device,
            ),
            result,
        )
    if invalid_tool_call_advantage is not None:
        result = torch.where(
            invalid_tool_call_mask.bool(),
            torch.as_tensor(
                invalid_tool_call_advantage,
                dtype=advantages.dtype,
                device=advantages.device,
            ),
            result,
        )
    return result


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
