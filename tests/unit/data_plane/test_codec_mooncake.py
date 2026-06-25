# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for the mooncake_cpu-specific wire workarounds.

Covers:
  P1 — `promote_1d` round-trip: writer unsqueezes 1D → (N,1), reader squeezes back.
  P2 — pack_per_token_field: tolerates SP padding wider than max(lengths).

No Ray, no GPU, no transfer_queue required.
"""

from __future__ import annotations

import torch

from nemo_rl.data_plane.codec import pack_per_token_field, to_nested_by_length

from ._rollout_shapes import make_rollout_batch

# ── P1: promote_1d — writer unsqueezes, reader squeezes ──────────────────────


def test_promote_1d_leaves_unsqueezes_1d() -> None:
    """`_promote_1d_leaves` turns 1D ``(N,)`` leaves into ``(N, 1)``.

    Guards the mooncake_cpu path where TQ's extract_field_schema silently
    unsqueezes 1D fields in metadata; the wire layer pre-unsqueezes so the
    per-row data shape matches the metadata-recorded shape.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _promote_1d_leaves

    n = 8
    t = torch.arange(n, dtype=torch.float32)
    td = TensorDict({"reward": t}, batch_size=[n])

    out = _promote_1d_leaves(td)
    assert out["reward"].shape == (n, 1), (
        f"Expected wire shape ({n}, 1) but got {tuple(out['reward'].shape)}."
    )


def test_promote_1d_roundtrip_via_from_wire() -> None:
    """`_promote_1d_leaves` then `_from_wire` restores the original ``(N,)`` shape and values."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import (
        _from_wire,
        _promote_1d_leaves,
    )

    n = 6
    original = torch.arange(n, dtype=torch.float32)
    td = TensorDict({"reward": original}, batch_size=[n])

    wire = _promote_1d_leaves(td)
    assert wire["reward"].shape == (n, 1)

    back = _from_wire(wire)
    assert back["reward"].shape == (n,)
    assert torch.equal(back["reward"], original)


# ── P2: pack_per_token_field — tolerates SP padding ──────────────────────────


def test_pack_per_token_field_truncates_sp_padding() -> None:
    """pack_per_token_field slices each row to its own length, dropping SP padding.

    mcore SP rounds the forward output's seq dim up to a multiple of TP, so
    val.shape[1] > max(lengths). pack_per_token_field handles this by slicing
    each row to its real length.
    """

    n, max_len, sp_extra = 4, 8, 3  # val is wider by sp_extra tokens
    lengths = torch.tensor([3, 5, 7, 4], dtype=torch.long)
    assert lengths.max().item() == max_len - 1  # max_len=8 > max(lengths)=7
    val = torch.randn(n, max_len + sp_extra)  # (4, 11)

    out = pack_per_token_field(val, lengths)

    assert out.is_nested, "pack_per_token_field must produce a nested tensor."
    rows = list(out.unbind())
    assert len(rows) == n
    for i, row in enumerate(rows):
        expected_len = int(lengths[i].item())
        assert row.shape == (expected_len,), (
            f"Row {i}: expected length {expected_len}, got {tuple(row.shape)}. "
            "SP padding tail was not dropped."
        )
        assert torch.equal(row, val[i, :expected_len]), (
            f"Row {i}: values differ after truncation."
        )


def test_pack_per_token_field_exact_fit_matches_to_nested_by_length() -> None:
    """When val.shape[1] == max(lengths), pack_per_token_field matches
    to_nested_by_length.

    This is the 'no SP padding' case — the two helpers must agree when
    the input is already exactly the right width.
    """
    n = 4
    lengths = torch.tensor([3, 5, 2, 4], dtype=torch.long)
    max_len = int(lengths.max().item())
    val = torch.randn(n, max_len)

    out_pack = pack_per_token_field(val, lengths)
    out_nested = to_nested_by_length(val, lengths)

    assert out_pack.is_nested
    assert out_nested.is_nested

    rows_pack = list(out_pack.unbind())
    rows_nested = list(out_nested.unbind())
    for i, (rp, rn) in enumerate(zip(rows_pack, rows_nested)):
        assert torch.equal(rp, rn), (
            f"Row {i} differs between pack_per_token_field and to_nested_by_length "
            "on an exact-fit input."
        )


# ── Realistic bf16 per-token coverage ──


def test_pack_per_token_field_realistic_bf16_logprobs() -> None:
    """pack_per_token_field on bf16 prev_logprobs (realistic dtype + value distribution)."""

    batch = make_rollout_batch(
        n=6, max_seqlen=96, logprob_dtype=torch.bfloat16, seed=29
    )
    out = pack_per_token_field(batch["prev_logprobs"], batch["input_lengths"])
    assert out.is_nested
    assert out.dtype == torch.bfloat16
    # Per-row valid region matches input — bf16 round-trip is loss-y at the bit
    # level but pack_per_token_field shouldn't change values.
    for i, row in enumerate(out.unbind()):
        valid = int(batch["input_lengths"][i])
        assert row.shape[0] == valid
        assert torch.equal(row, batch["prev_logprobs"][i, :valid])
