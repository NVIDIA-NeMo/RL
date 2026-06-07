# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import DiffuGRPOLogprobEstimationConfig


NOISY_RESPONSE_OFFSET = 0


def get_diffu_grpo_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> DiffuGRPOLogprobEstimationConfig:
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "diffu_grpo_fully_masked_completion":
        raise ValueError(
            "policy.logprob_estimation.type must be "
            "diffu_grpo_fully_masked_completion"
        )
    return estimator_cfg


def _completion_score_mask(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")
    if token_mask.shape != input_ids.shape:
        raise ValueError(
            f"token_mask shape {token_mask.shape} must match input_ids shape {input_ids.shape}"
        )

    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    score_mask = token_mask.to(device=device).bool()
    length_mask = torch.arange(seq_len, device=device).unsqueeze(0) < (
        input_lengths.to(device=device).unsqueeze(1)
    )
    score_mask = score_mask & length_mask
    if sample_mask is not None:
        score_mask = score_mask & sample_mask.to(device=device).bool().unsqueeze(-1)
    else:
        score_mask = score_mask & torch.ones(
            batch_size, 1, device=device, dtype=torch.bool
        )
    return score_mask


def _completion_scoring_input_lengths(
    input_lengths: torch.Tensor,
    score_mask: torch.Tensor,
) -> torch.Tensor:
    """Return lengths ending at the last generated completion token."""
    scoring_input_lengths = input_lengths.to(device=score_mask.device).clone()
    for batch_idx in range(score_mask.shape[0]):
        valid_positions = torch.nonzero(
            score_mask[batch_idx], as_tuple=False
        ).flatten()
        if valid_positions.numel() > 0:
            scoring_input_lengths[batch_idx] = int(valid_positions[-1].item()) + 1
    return scoring_input_lengths.to(device=input_lengths.device)


def _completion_start_positions(score_mask: torch.Tensor) -> torch.Tensor:
    starts = torch.zeros(
        score_mask.shape[0],
        device=score_mask.device,
        dtype=torch.long,
    )
    for batch_idx in range(score_mask.shape[0]):
        valid_positions = torch.nonzero(
            score_mask[batch_idx], as_tuple=False
        ).flatten()
        if valid_positions.numel() > 0:
            starts[batch_idx] = int(valid_positions[0].item())
    return starts


def _completion_response_lengths(
    score_mask: torch.Tensor,
    completion_starts: torch.Tensor,
    scoring_input_lengths: torch.Tensor,
) -> torch.Tensor:
    response_lengths = torch.zeros_like(scoring_input_lengths, dtype=torch.long)
    for batch_idx in range(score_mask.shape[0]):
        if torch.any(score_mask[batch_idx]):
            response_lengths[batch_idx] = (
                scoring_input_lengths[batch_idx].to(response_lengths.device)
                - completion_starts[batch_idx].to(response_lengths.device)
            )
    return response_lengths


def _validate_single_completion_span(score_mask: torch.Tensor) -> None:
    for batch_idx in range(score_mask.shape[0]):
        valid_positions = torch.nonzero(
            score_mask[batch_idx], as_tuple=False
        ).flatten()
        if valid_positions.numel() <= 1:
            continue
        span_length = int(valid_positions[-1].item() - valid_positions[0].item()) + 1
        if span_length != valid_positions.numel():
            raise ValueError(
                "DiffuGRPO fully-masked completion expects one contiguous "
                "generated completion span per sample. Multiple response spans "
                "separated by environment/user tokens would leak future context "
                "under bidirectional diffusion attention."
            )


def _build_completion_only_tensors(
    input_ids: torch.Tensor,
    completion_starts: torch.Tensor,
    response_lengths: torch.Tensor,
    clean_lengths: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    block_size: int | None,
    sequence_length_round: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
]:
    batch_size = input_ids.shape[0]
    if block_size is not None and block_size > 0:
        noisy_valid_lengths = torch.where(
            response_lengths > 0,
            torch.div(
                response_lengths + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            * block_size,
            response_lengths,
        )
    else:
        noisy_valid_lengths = response_lengths
    max_response_len = int(noisy_valid_lengths.max().item()) if batch_size else 0
    max_clean_len = int(clean_lengths.max().item()) if batch_size else 0
    noisy_length = max_response_len
    clean_length = max_clean_len
    total_length = noisy_length + clean_length
    if sequence_length_round is not None and sequence_length_round > 0:
        rounded_total_length = (
            (total_length + sequence_length_round - 1)
            // sequence_length_round
            * sequence_length_round
        )
        clean_length += rounded_total_length - total_length
        total_length = rounded_total_length

    layout_input_ids = torch.full(
        (batch_size, total_length),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    target_ids = torch.full_like(layout_input_ids, int(pad_token_id))
    token_mask = torch.zeros(
        (batch_size, total_length), dtype=torch.float32, device=input_ids.device
    )
    score_mask = torch.zeros_like(token_mask)

    for batch_idx in range(batch_size):
        start = int(completion_starts[batch_idx].item())
        response_len = int(response_lengths[batch_idx].item())
        noisy_valid_len = int(noisy_valid_lengths[batch_idx].item())
        clean_len = int(clean_lengths[batch_idx].item())
        clean_offset = noisy_length

        if clean_len > 0:
            clean_tokens = input_ids[batch_idx, :clean_len]
            layout_input_ids[batch_idx, clean_offset : clean_offset + clean_len] = clean_tokens
            target_ids[batch_idx, clean_offset : clean_offset + clean_len] = clean_tokens

        if noisy_valid_len > 0:
            noisy_start = NOISY_RESPONSE_OFFSET
            noisy_end = noisy_start + noisy_valid_len
            layout_input_ids[batch_idx, noisy_start:noisy_end] = int(mask_token_id)

        if response_len > 0:
            response_tokens = input_ids[batch_idx, start : start + response_len]
            noisy_start = NOISY_RESPONSE_OFFSET
            noisy_end = noisy_start + response_len
            target_ids[batch_idx, noisy_start:noisy_end] = response_tokens
            token_mask[batch_idx, noisy_start:noisy_end] = 1.0
            score_mask[batch_idx, noisy_start:noisy_end] = 1.0

    return (
        layout_input_ids,
        target_ids,
        token_mask,
        score_mask,
        noisy_length,
        clean_length,
        noisy_valid_lengths,
    )


def _scatter_original_response_values(
    values: torch.Tensor,
    total_length: int,
    completion_starts: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    scattered = torch.zeros(
        (values.shape[0], total_length), dtype=values.dtype, device=values.device
    )
    for batch_idx in range(values.shape[0]):
        start = int(completion_starts[batch_idx].item())
        response_len = int(response_lengths[batch_idx].item())
        if response_len > 0:
            scattered[batch_idx, :response_len] = values[
                batch_idx, start : start + response_len
            ]
    return scattered


def build_fully_masked_completion_batch(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    pad_to_length: int | None = None,
    block_size: int | None = None,
) -> BatchedDataDict[Any]:
    """Build a completion-only DiffuGRPO batch as ``[masked_response | clean]``.

    The noisy side contains exactly one masked token for each generated
    response token. The clean side contains the prompt plus clean response,
    with post-response environment suffixes removed.
    """
    sequence_length_round = pad_to_length
    input_ids = data["input_ids"]
    input_lengths = data["input_lengths"]
    token_mask = data["token_mask"]
    sample_mask = data.get("sample_mask", None)

    if torch.any(input_lengths > input_ids.shape[1]):
        raise ValueError("input_lengths cannot exceed input_ids sequence length")

    score_mask = _completion_score_mask(
        input_ids=input_ids,
        input_lengths=input_lengths,
        token_mask=token_mask,
        sample_mask=sample_mask,
    )
    _validate_single_completion_span(score_mask)
    clean_lengths = _completion_scoring_input_lengths(
        input_lengths=input_lengths,
        score_mask=score_mask,
    ).to(device=input_ids.device, dtype=torch.long)
    completion_starts = _completion_start_positions(score_mask).to(
        device=input_ids.device, dtype=torch.long
    )
    response_lengths = _completion_response_lengths(
        score_mask=score_mask,
        completion_starts=completion_starts,
        scoring_input_lengths=clean_lengths,
    ).to(device=input_ids.device, dtype=torch.long)

    (
        layout_input_ids,
        target_ids,
        layout_token_mask,
        layout_score_mask,
        noisy_length,
        clean_length,
        noisy_valid_lengths,
    ) = _build_completion_only_tensors(
        input_ids=input_ids,
        completion_starts=completion_starts,
        response_lengths=response_lengths,
        clean_lengths=clean_lengths,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        block_size=block_size,
        sequence_length_round=sequence_length_round,
    )
    total_length = layout_input_ids.shape[1]
    batch_size = layout_input_ids.shape[0]

    batch = BatchedDataDict[Any](
        {
            "input_ids": layout_input_ids,
            "input_lengths": torch.full(
                (batch_size,), total_length, dtype=torch.long, device=input_ids.device
            ),
            "token_mask": layout_token_mask,
            "diffu_grpo_target_ids": target_ids,
            "diffu_grpo_score_mask": layout_score_mask,
            "diffu_grpo_completion_starts": completion_starts,
            "diffu_grpo_response_lengths": response_lengths,
            "diffu_grpo_noisy_valid_lengths": noisy_valid_lengths,
            "diffu_grpo_clean_lengths": clean_lengths,
            "diffu_grpo_noisy_lengths": torch.full(
                (batch_size,), noisy_length, dtype=torch.long, device=input_ids.device
            ),
            "diffu_grpo_clean_padded_lengths": torch.full(
                (batch_size,), clean_length, dtype=torch.long, device=input_ids.device
            ),
            "diffu_grpo_noisy_response_offsets": torch.full(
                (batch_size,),
                NOISY_RESPONSE_OFFSET,
                dtype=torch.long,
                device=input_ids.device,
            ),
        }
    )
    if sample_mask is not None:
        batch["sample_mask"] = sample_mask
    return batch


def build_fully_masked_completion_loss_batch(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    pad_to_length: int | None = None,
    block_size: int | None = None,
) -> BatchedDataDict[Any]:
    batch = build_fully_masked_completion_batch(
        data,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        pad_to_length=pad_to_length,
        block_size=block_size,
    )
    total_length = batch["input_ids"].shape[1]
    completion_starts = batch["diffu_grpo_completion_starts"]
    response_lengths = batch["diffu_grpo_response_lengths"]
    batch["diffu_grpo_loss_mask"] = batch["diffu_grpo_score_mask"]

    for key in (
        "advantages",
        "prev_logprobs",
        "generation_logprobs",
        "reference_policy_logprobs",
    ):
        if key in data:
            values = data[key]
            if values.shape[1] != data["input_ids"].shape[1]:
                raise ValueError(
                    f"{key} must have sequence length {data['input_ids'].shape[1]}, "
                    f"got {values.shape}"
                )
            batch[key] = _scatter_original_response_values(
                values=values,
                total_length=total_length,
                completion_starts=completion_starts,
                response_lengths=response_lengths,
            )

    return batch
