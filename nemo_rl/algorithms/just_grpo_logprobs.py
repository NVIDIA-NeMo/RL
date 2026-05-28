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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import JustGRPOLeftmostRevealLogprobEstimationConfig


def get_leftmost_reveal_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> JustGRPOLeftmostRevealLogprobEstimationConfig:
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "just_grpo_leftmost_reveal":
        raise ValueError(
            "policy.logprob_estimation.type must be 'just_grpo_leftmost_reveal'"
        )
    return estimator_cfg


def build_leftmost_reveal_batch(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    mask_token_id: int,
    sample_mask: torch.Tensor | None = None,
    reveal_schedule: str = "sparse",
    max_reveal_positions: int | None = None,
) -> BatchedDataDict[Any]:
    """Expand sequences into JustGRPO leftmost-reveal scoring rows.

    For every token position p selected by token_mask, this creates one row where
    selected positions >= p are replaced with mask_token_id and all previous
    selected tokens remain visible. The target token is scored from logits at the
    same position p.

    With ``reveal_schedule="fixed_response_window"``, every sample emits exactly
    ``max_reveal_positions`` rows starting at its first response token. Rows that
    do not correspond to valid response tokens are retained with a zero row mask.
    This keeps Megatron data-parallel ranks on the same microbatch schedule.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")
    if token_mask.shape != input_ids.shape:
        raise ValueError(
            f"token_mask shape {token_mask.shape} must match input_ids shape {input_ids.shape}"
        )

    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    token_mask_bool = token_mask.to(device=device).bool()
    row_mask_values = []

    if sample_mask is not None:
        sample_mask_bool = sample_mask.to(device=device).bool()
        token_mask_bool = token_mask_bool & sample_mask_bool.unsqueeze(-1)
    else:
        sample_mask_bool = torch.ones(batch_size, device=device, dtype=torch.bool)

    length_mask = torch.arange(seq_len, device=device).unsqueeze(0) < input_lengths.to(
        device=device
    ).unsqueeze(1)
    token_mask_bool = token_mask_bool & length_mask

    row_tensors = []
    batch_indices = []
    target_positions = []
    target_tokens = []
    response_starts = []

    if reveal_schedule == "fixed_response_window" and max_reveal_positions is None:
        raise ValueError(
            "max_reveal_positions must be set when reveal_schedule is "
            "'fixed_response_window'"
        )
    if reveal_schedule not in {"sparse", "fixed_response_window"}:
        raise ValueError(f"Unsupported reveal_schedule: {reveal_schedule}")

    arange_seq = torch.arange(seq_len, device=device)

    for batch_idx in range(batch_size):
        valid_positions = torch.nonzero(
            token_mask_bool[batch_idx], as_tuple=False
        ).flatten()
        if reveal_schedule == "fixed_response_window":
            if valid_positions.numel() > 0:
                response_start = int(valid_positions[0].item())
            else:
                response_start = int(
                    min(max(input_lengths[batch_idx].item() - 1, 0), seq_len - 1)
                )
            positions = torch.arange(
                response_start,
                response_start + int(max_reveal_positions),
                device=device,
            )
        else:
            positions = valid_positions

        for raw_position in positions.tolist():
            position = min(max(int(raw_position), 0), seq_len - 1)
            row_is_valid = (
                raw_position == position
                and bool(sample_mask_bool[batch_idx].item())
                and bool(token_mask_bool[batch_idx, position].item())
            )
            reveal_ids = input_ids[batch_idx].clone()
            future_positions = torch.nonzero(
                token_mask_bool[batch_idx]
                & (arange_seq >= torch.tensor(position, device=device)),
                as_tuple=False,
            ).flatten()
            reveal_ids[future_positions] = mask_token_id
            row_tensors.append(reveal_ids)
            batch_indices.append(batch_idx)
            target_positions.append(position)
            target_tokens.append(input_ids[batch_idx, position])
            response_starts.append(
                response_start if reveal_schedule == "fixed_response_window" else position
            )
            row_mask_values.append(row_is_valid)

    if row_tensors:
        expanded_input_ids = torch.stack(row_tensors, dim=0)
        expanded_input_lengths = input_lengths.to(device=device)[
            torch.tensor(batch_indices, device=device, dtype=torch.long)
        ]
        batch_indices_tensor = torch.tensor(
            batch_indices, device=device, dtype=torch.long
        )
        target_positions_tensor = torch.tensor(
            target_positions, device=device, dtype=torch.long
        )
        target_tokens_tensor = torch.stack(target_tokens).to(device=device)
        response_starts_tensor = torch.tensor(
            response_starts, device=device, dtype=torch.long
        )
        row_mask_tensor = torch.tensor(
            row_mask_values, device=device, dtype=torch.float32
        )
        output_shape_tensor = torch.tensor(
            [batch_size, seq_len], device=device, dtype=torch.long
        ).repeat(expanded_input_ids.shape[0], 1)
    else:
        expanded_input_ids = input_ids.new_empty((0, seq_len))
        expanded_input_lengths = input_lengths.to(device=device).new_empty((0,))
        batch_indices_tensor = torch.empty(0, device=device, dtype=torch.long)
        target_positions_tensor = torch.empty(0, device=device, dtype=torch.long)
        target_tokens_tensor = torch.empty(0, device=device, dtype=input_ids.dtype)
        response_starts_tensor = torch.empty(0, device=device, dtype=torch.long)
        row_mask_tensor = torch.empty(0, device=device, dtype=torch.float32)
        output_shape_tensor = torch.empty(0, 2, device=device, dtype=torch.long)

    return BatchedDataDict[Any](
        {
            "input_ids": expanded_input_ids,
            "input_lengths": expanded_input_lengths,
            "just_grpo_batch_indices": batch_indices_tensor,
            "just_grpo_target_positions": target_positions_tensor,
            "just_grpo_target_tokens": target_tokens_tensor,
            "just_grpo_response_starts": response_starts_tensor,
            "just_grpo_row_mask": row_mask_tensor,
            "just_grpo_output_shape": output_shape_tensor,
        }
    )


def build_leftmost_reveal_loss_batch(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    reveal_schedule: str = "sparse",
    max_reveal_positions: int | None = None,
) -> BatchedDataDict[Any]:
    """Build reveal rows plus flat per-reveal-token loss tensors.

    The Megatron forward still runs over reveal rows shaped ``[R, S]`` so the
    model can score each leftmost target position. The loss tensors are kept
    flat as ``[R]`` because each reveal row contributes exactly one token to the
    policy-gradient loss.
    """
    grpo_batch_token_keys_to_gather = (
        "advantages",
        "prev_logprobs",
        "generation_logprobs",
        "reference_policy_logprobs",
    )

    reveal_data = build_leftmost_reveal_batch(
        input_ids=data["input_ids"],
        input_lengths=data["input_lengths"],
        token_mask=data["token_mask"],
        sample_mask=data.get("sample_mask", None),
        mask_token_id=mask_token_id,
        reveal_schedule=reveal_schedule,
        max_reveal_positions=max_reveal_positions,
    )
    batch_indices = reveal_data["just_grpo_batch_indices"]
    target_positions = reveal_data["just_grpo_target_positions"]
    expanded = BatchedDataDict[Any]()

    expanded["input_ids"] = reveal_data["input_ids"]
    expanded["input_lengths"] = reveal_data["input_lengths"]

    for key in grpo_batch_token_keys_to_gather:
        if key in data:
            expanded[key] = data[key].to(device=batch_indices.device)[
                batch_indices, target_positions
            ]

    if "sample_mask" in data:
        sample_mask = data["sample_mask"].to(device=batch_indices.device)[batch_indices]
    else:
        sample_mask = torch.ones_like(
            batch_indices, dtype=torch.float32, device=batch_indices.device
        )
    expanded["just_grpo_loss_mask"] = sample_mask * reveal_data[
        "just_grpo_row_mask"
    ].to(dtype=sample_mask.dtype)

    for key in (
        "just_grpo_batch_indices",
        "just_grpo_target_positions",
        "just_grpo_target_tokens",
        "just_grpo_response_starts",
        "just_grpo_row_mask",
        "just_grpo_loss_mask",
        "just_grpo_output_shape",
    ):
        if key not in expanded:
            expanded[key] = reveal_data[key]

    return expanded


def pad_reveal_batch_to_multiple(
    data: BatchedDataDict[Any],
    multiple: int,
) -> int:
    """Pad reveal rows in-place so NemoRL fixed microbatching can consume them.

    Returns the original number of rows. Callers that need exact row counts
    should trim post-processed outputs back to this value.
    """
    original_size = data.size
    if multiple <= 1 or original_size == 0:
        return original_size
    remainder = original_size % multiple
    if remainder == 0:
        return original_size

    pad_count = multiple - remainder
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            pad_value = value[-1:].repeat([pad_count] + [1] * (value.ndim - 1))
            data[key] = torch.cat([value, pad_value], dim=0)
        elif isinstance(value, list) and len(value) == original_size:
            data[key] = list(value) + [value[-1]] * pad_count
        elif isinstance(value, tuple) and len(value) == original_size:
            data[key] = tuple(list(value) + [value[-1]] * pad_count)

    if "token_mask" in data:
        data["token_mask"][original_size:] = 0
    if "sample_mask" in data:
        data["sample_mask"][original_size:] = 0

    return original_size


def scatter_leftmost_reveal_logprobs(
    flat_logprobs: torch.Tensor,
    batch_indices: torch.Tensor,
    target_positions: torch.Tensor,
    output_shape: torch.Tensor,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter flat reveal-row logprobs back to NemoRL's [B, S] convention."""
    if output_shape.ndim == 2:
        output_shape = output_shape[0]
    batch_size = int(output_shape[0].item())
    seq_len = int(output_shape[1].item())
    output = flat_logprobs.new_zeros((batch_size, seq_len))
    if row_mask is None:
        valid_rows = torch.ones_like(flat_logprobs, dtype=torch.bool)
    else:
        valid_rows = row_mask.to(device=flat_logprobs.device).bool()
    if flat_logprobs.numel() > 0 and valid_rows.any():
        output[
            batch_indices.to(device=flat_logprobs.device)[valid_rows],
            target_positions.to(device=flat_logprobs.device)[valid_rows],
        ] = flat_logprobs[valid_rows]
    return output
