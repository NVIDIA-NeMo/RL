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
"""Plan causal prefix passes for tokenwise version-point gradients.

Generated-token stamps encode version + 1; zero denotes an unknown/prompt token.
Every pass masks loss to one version, and the passes partition the original loss.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

GENERATION_VERSIONS_FIELD = "generation_versions"


def generated_token_versions(
    policy_epoch: Optional[Sequence[Sequence[int]]],
    prompt_length: int,
    num_generated: int,
) -> Optional[list[int]]:
    """Weight version of each generated token from Megatron-Core's ``policy_epoch``.

    ``policy_epoch`` is the engine's append-only list of ``(position, epoch)``
    boundaries: the forward at sequence positions ``>= position`` ran at ``epoch``.
    Generated token ``j`` sits at position ``prompt_length + j`` and was sampled
    from the logits of position ``prompt_length + j - 1``.

    Returns:
        One version per generated token, or None if the engine stamped no epoch.
    """
    if not policy_epoch:
        return None
    boundaries = [(int(start), int(epoch)) for start, epoch in policy_epoch]
    starts = [start for start, _ in boundaries]
    if starts != sorted(starts):
        raise ValueError(f"policy_epoch boundaries are not ordered: {boundaries}")
    versions: list[int] = []
    k = 0
    current: Optional[int] = None
    for j in range(num_generated):
        position = prompt_length + j - 1
        while k < len(boundaries) and boundaries[k][0] <= position:
            current = boundaries[k][1]
            k += 1
        if current is None:
            return None
        versions.append(current)
    return versions


@dataclass(frozen=True)
class TokenVersionPass:
    """Rows and causal prefix lengths for one behavior-weight version."""

    version: int
    rows: list[int]
    lengths: list[int]
    num_real_rows: int


def token_version_candidates(
    data: BatchedDataDict[Any], has_snapshot: Callable[[int], bool]
) -> tuple[list[int], dict[str, float]]:
    """Qualify the whole batch; unknown/evicted stamps retain the ordinary IS path."""
    if GENERATION_VERSIONS_FIELD not in data:
        return [], {"version_point_unstamped_tokens": 1.0}
    stamps = data[GENERATION_VERSIONS_FIELD]
    positions = torch.arange(stamps.shape[1], device=stamps.device)
    active = (
        (data["token_mask"] > 0)
        & (data["sample_mask"][:, None] > 0)
        & (positions[None, :] < data["input_lengths"][:, None])
        & (positions[None, :] > 0)
    )
    unknown = bool((stamps[active] <= 0).any())
    versions = sorted(
        int(v) - 1 for v in torch.unique(stamps[active]).tolist() if v > 0
    )
    missing = any(not has_snapshot(v) for v in versions)
    metrics = {
        "version_point_unstamped_tokens": float(unknown),
        "version_point_no_snapshot": float(missing),
        "version_point_num_versions": float(len(versions)),
        "version_point_mixed_batch": float(len(versions) > 1),
    }
    return ([] if unknown or missing else versions), metrics


def plan_token_version_passes(
    data: BatchedDataDict[Any], versions: Sequence[int], *, row_multiple: int
) -> list[TokenVersionPass]:
    """Select rows per version, truncating after its last loss token and padding DP."""
    if row_multiple < 1:
        raise ValueError("row_multiple must be positive")
    stamps = data[GENERATION_VERSIONS_FIELD]
    positions = torch.arange(stamps.shape[1], device=stamps.device)[None, :]
    active = (
        (data["token_mask"] > 0)
        & (data["sample_mask"][:, None] > 0)
        & (positions < data["input_lengths"][:, None])
        & (positions > 0)
    )
    passes = []
    for version in versions:
        selected = active & (stamps == version + 1)
        rows = selected.any(dim=1).nonzero().flatten().tolist()
        if not rows:
            raise ValueError(f"version {version} has no loss tokens")
        lengths = ((positions + 1) * selected).amax(dim=1)[rows].tolist()
        num_real_rows = len(rows)
        padding = (-num_real_rows) % row_multiple
        # Duplicate a short prefix with zero sample weight so every DP rank participates.
        rows += [rows[0]] * padding
        lengths += [2] * padding
        passes.append(TokenVersionPass(version, rows, lengths, num_real_rows))
    return passes


def token_version_pass_data(
    data: BatchedDataDict[Any], plan: TokenVersionPass
) -> BatchedDataDict[Any]:
    """Materialize an independent, masked pass; leave the rollout batch unchanged."""
    out = data.select_indices(plan.rows)
    lengths = torch.tensor(
        plan.lengths,
        dtype=out["input_lengths"].dtype,
        device=out["input_lengths"].device,
    )
    out["input_lengths"] = torch.minimum(out["input_lengths"], lengths)
    width = int(out["input_lengths"].max())
    original_width = data["input_ids"].shape[1]
    for key, value in out.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == original_width
        ):
            out[key] = value[:, :width]
    positions = torch.arange(width, device=out["token_mask"].device)[None, :]
    out["token_mask"] = out["token_mask"] * (
        (out[GENERATION_VERSIONS_FIELD] == plan.version + 1)
        & (positions < out["input_lengths"][:, None])
    )
    out["sample_mask"][plan.num_real_rows :] = 0
    out["token_mask"][plan.num_real_rows :] = 0
    # Sequence metrics count each original row once, even when its tokens span passes.
    stamps = data[GENERATION_VERSIONS_FIELD][plan.rows]
    original_positions = torch.arange(stamps.shape[1], device=stamps.device)[None, :]
    active = (
        (data["token_mask"][plan.rows] > 0)
        & (original_positions > 0)
        & (original_positions < data["input_lengths"][plan.rows, None])
    )
    first_version = torch.where(active, stamps, torch.iinfo(stamps.dtype).max).amin(
        dim=1
    )
    out["version_point_sequence_count"] = out["sample_mask"] * (
        first_version == plan.version + 1
    )
    return out
