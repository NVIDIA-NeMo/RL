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

"""Block-aware context parallelism plumbing for diffusion GRPO.

Block-aware (segmented) context parallelism gives the noisy and clean sections
of the ``[noisy | clean]`` layout one zigzag each instead of one global zigzag,
so the noisy K/V -- only ever consumed block-diagonally -- never leaves its
owning rank and only the clean K/V is all-gathered.

The per-rank row SET differs between the two layouts (at cp=2 the single-zigzag
rank 0 holds the noisy first half plus the clean second half, while the
segmented rank 0 holds noisy chunks {0,3} and clean chunks {0,3}). That is a
cross-rank difference, not a local permutation, so it cannot be absorbed inside
attention: the data sharding, the attention layer, the target-id slicing and the
logprob re-gather must all agree, or they silently disagree about which token
sits at which row.

This module is the single place that decides whether the segmented layout is in
play, so those call sites cannot drift apart. The flag is the same environment
variable the Megatron-Bridge attention layer reads, for the same reason.
"""

import os
from typing import Any, Optional

import torch

# Same parsing as the Megatron-Bridge attention layer, deliberately: if the
# two disagreed about whether the flag is set, the data path and attention
# would use different layouts and the logprobs would come from wrong tokens.
# Tolerant of any non-off word because a bare 1/true routed through an
# OmegaConf override is coerced to int/bool and rejected by Ray's
# Dict[str, str] check on runtime_env["env_vars"].
_BLOCK_AWARE_CP = os.environ.get("DIFFU_CP_BLOCK_AWARE", "").lower() not in (
    "",
    "0",
    "off",
    "false",
    "no",
)


def block_aware_cp_enabled() -> bool:
    """Whether block-aware (segmented) context parallelism is requested."""
    return _BLOCK_AWARE_CP


def block_aware_segments(data_dict: Any) -> Optional[tuple[int, int]]:
    """Return ``(noisy_length, clean_length)`` when block-aware CP applies.

    Returns ``None`` when the flag is off or the batch carries no diffusion
    asymmetric-layout metadata (so non-diffusion batches keep the single global
    zigzag untouched).

    Args:
        data_dict: Microbatch dict, expected to carry
            ``diffu_grpo_noisy_lengths`` / ``diffu_grpo_clean_padded_lengths``
            for diffusion batches.

    Raises:
        ValueError: if the segment lengths vary within the microbatch, which
            would make a single chunk grid impossible.
    """
    if not _BLOCK_AWARE_CP:
        return None
    if "diffu_grpo_noisy_lengths" not in data_dict:
        return None

    noisy = data_dict["diffu_grpo_noisy_lengths"]
    clean = data_dict["diffu_grpo_clean_padded_lengths"]
    if not bool(torch.all(noisy == noisy[0])) or not bool(torch.all(clean == clean[0])):
        raise ValueError(
            "Block-aware CP requires constant noisy/clean lengths within a "
            "microbatch; got varying diffu_grpo_noisy_lengths / "
            "diffu_grpo_clean_padded_lengths."
        )
    return int(noisy[0].item()), int(clean[0].item())


def assert_block_aware_divisible(
    segments: tuple[int, int], cp_size: int, total_length: int
) -> None:
    """Check the per-segment divisibility block-aware CP needs.

    Each segment is zigzagged on its own, so each must divide by ``2*cp_size``
    -- a single global pad does NOT imply this, and in fact leaves the internal
    segment boundary at an arbitrary position that lands on no chunk grid.
    Padding therefore has to be applied per segment in the batch builder.

    The additional requirement that the noisy chunk size be a multiple of the
    block size is asserted inside the attention layer, which is where the block
    size is known.
    """
    noisy_length, clean_length = segments
    for name, seg_len in (("noisy", noisy_length), ("clean", clean_length)):
        assert seg_len % (2 * cp_size) == 0, (
            f"Block-aware CP requires the {name} segment length ({seg_len}) to be "
            f"divisible by 2*cp_size ({2 * cp_size}). Pad each segment to its own "
            f"multiple in the batch builder -- a single global pad over "
            f"[noisy | clean] does not satisfy this."
        )
    assert noisy_length + clean_length == total_length, (
        f"Block-aware CP segment lengths ({noisy_length} + {clean_length} = "
        f"{noisy_length + clean_length}) do not cover the sequence ({total_length})."
    )


def block_aware_cp_padding(block_size: Optional[int]) -> Optional[tuple[int, int]]:
    """Per-segment padding multiples ``(noisy, clean)``, or ``None`` if N/A.

    Each segment must be padded to ITS OWN divisibility, in the batch builder. A single global pad does not work --
    it leaves the internal ``noisy | clean`` boundary at an arbitrary offset
    that lands on no chunk grid, so block alignment breaks even though the
    total length looks fine.

    The noisy segment needs ``2*cp*block_size`` (its chunk size must be a whole
    number of blocks, or a rank's queries would need noisy keys owned by another
    rank). The clean segment needs only ``2*cp``: it carries no block grid of
    its own here, since the response's grid is anchored at the per-sample prompt
    length rather than at the segment start.

    Returns ``None`` when block-aware CP is off, cp_size is 1, or the parallel
    state is not initialized (e.g. batch construction outside a Megatron
    worker), so non-CP callers are unaffected.
    """
    if not _BLOCK_AWARE_CP:
        return None
    try:
        from megatron.core.parallel_state import get_context_parallel_world_size

        cp_size = get_context_parallel_world_size()
    except (ImportError, AssertionError, RuntimeError):
        return None
    if cp_size <= 1:
        return None
    block = int(block_size) if block_size else 1
    return 2 * cp_size * block, 2 * cp_size


def round_up(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``value``."""
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple
