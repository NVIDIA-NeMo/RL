# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent critic row weights; actor filtering and rewards stay unchanged."""

from typing import Optional

import torch

# Small per-row control metadata. Only TQValue dispatches it to its workers;
# policy workers continue reading the original data-plane sample_mask.
CRITIC_SAMPLE_MASK = "critic_sample_mask_override"


def build_critic_sample_mask(
    sample_mask: torch.Tensor,
    actor_mask: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
    *,
    overlong_filtering: bool,
    train_on_env_masked: bool,
    train_on_truncated: bool,
    seq_error_keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Relax only env/truncation gates, preserving placeholders and other gates."""
    if not train_on_env_masked and not train_on_truncated:
        return actor_mask.clone()
    critic_mask = sample_mask.float().clone()
    if not train_on_env_masked:
        critic_mask *= (~mask_sample.bool()).to(critic_mask.dtype)
    if overlong_filtering and not train_on_truncated:
        critic_mask *= (~truncated.bool()).to(critic_mask.dtype)
    if seq_error_keep is not None:
        critic_mask *= seq_error_keep.to(critic_mask.dtype)
    return critic_mask


def critic_mask_counts(
    actor_mask: torch.Tensor,
    critic_mask: torch.Tensor,
    token_mask: torch.Tensor,
    rewards: torch.Tensor,
    canonical: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
) -> dict[str, float]:
    """Count actual additional supervision, including its unchanged labels."""
    has_tokens = token_mask[:, 1:].bool().any(-1)
    actor = (actor_mask > 0) & has_tokens
    critic = (critic_mask > 0) & has_tokens
    extra = critic & ~actor
    roots = canonical.bool()
    tokens = token_mask[:, 1:].float().sum(-1)
    masks = {
        "actor_rows": actor,
        "critic_rows": critic,
        "additional_rows": extra,
        "actor_roots": actor & roots,
        "critic_roots": critic & roots,
        "additional_roots": extra & roots,
        "additional_env_roots": extra & roots & mask_sample.bool(),
        "additional_truncated_roots": extra & roots & truncated.bool(),
        "additional_reward0_roots": extra & roots & (rewards < 0.5),
        "additional_reward1_roots": extra & roots & (rewards >= 0.5),
    }
    out = {f"critic/mask/{key}": float(value.sum()) for key, value in masks.items()}
    for name, weight in (("actor", actor_mask), ("critic", critic_mask)):
        out[f"critic/mask/{name}_valid_tokens"] = float((tokens * weight).sum())
    return out
