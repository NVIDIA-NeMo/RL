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
"""Record MoE routing during Megatron-inference generation as R3 ``routed_experts``.

Upstream router replay (R3, ``nemo_rl.models.megatron.router_replay``) provides the
*replay* side for the Megatron training/logprob forward, but only records routing on
the vLLM generation path. The colocated Megatron-inference generation path therefore
has no recorder. This module bridges that gap: it converts the dynamic-inference
engine's per-sample ``routing_indices`` into the ``[batch, seq, layers, topk]``
``routed_experts`` tensor that ``router_replay.set_router_replay_forward`` /
``build_router_replay_assignments`` consume (see ``_normalize_routed_experts_for_mcore``,
which accepts ``[B, S, L, K]``).
"""

from typing import Optional

import torch


def coerce_routing_to_3d(routing: torch.Tensor) -> torch.Tensor:
    """Normalize per-sample routing to ``[num_tokens, num_layers, topk]``."""
    if not isinstance(routing, torch.Tensor):
        raise TypeError(f"routing_indices must be a torch.Tensor, got {type(routing)}")
    if routing.ndim == 3:
        return routing
    raise ValueError(
        f"routing_indices must be 3D [tokens, layers, topk], got shape {tuple(routing.shape)}"
    )


def align_routing_rows_to_token_count(
    routing: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Pad/trim routing rows to ``num_tokens`` so they replay on a full-sequence forward.

    Dynamic inference accumulates ``[num_tokens, layers, topk]`` (sometimes one fewer
    row than the padded sequence length). Repeat the last recorded row when an extra
    step is required; trim when there are more.
    """
    if routing.ndim != 3:
        raise ValueError(f"Expected 3D routing tensor, got {tuple(routing.shape)}")
    num_rows = routing.shape[0]
    if num_rows == num_tokens:
        return routing
    if num_rows > num_tokens:
        return routing[:num_tokens].contiguous()
    if num_rows == 0:
        raise ValueError("Cannot align empty routing indices to a non-empty sequence")
    pad_rows = num_tokens - num_rows
    last = routing[-1:].expand(pad_rows, -1, -1)
    return torch.cat([routing, last], dim=0)


def build_routed_experts_batch(
    routing_per_sample: list[Optional[torch.Tensor]],
    seq_lengths: torch.Tensor,
    seq_dim: int,
) -> Optional[torch.Tensor]:
    """Build the R3 ``routed_experts`` tensor ``[batch, seq_dim, layers, topk]``.

    Each sample's routing is aligned to its (unpadded) sequence length and right-padded
    with zeros to ``seq_dim`` (the generation ``output_ids`` sequence length), so that
    ``routed_experts[i][:seq_lengths[i]]`` is the per-token routing and the tail is pad.
    Returns ``None`` when no sample carries routing (e.g. dense models / replay disabled).
    """
    if not any(r is not None for r in routing_per_sample):
        return None
    if any(r is None for r in routing_per_sample):
        raise ValueError(
            "routing_indices must be present for every sample when router replay is enabled"
        )
    aligned = [
        align_routing_rows_to_token_count(
            coerce_routing_to_3d(r), int(seq_lengths[i].item())
        )
        for i, r in enumerate(routing_per_sample)
    ]
    num_layers, topk = aligned[0].shape[1], aligned[0].shape[2]
    out = torch.zeros(
        len(aligned), seq_dim, num_layers, topk, dtype=aligned[0].dtype
    )
    for i, r in enumerate(aligned):
        n = min(r.shape[0], seq_dim)
        out[i, :n] = r[:n]
    return out
