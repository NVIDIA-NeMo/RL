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
"""Unit tests for the padded ↔ jagged codec bridge.

Phase 1 of the wire-jagged plan: writer emits nested, reader pads on
demand. These tests cover the conversion helpers in isolation; e2e
parity is validated separately.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import (
    materialize,
    pack_jagged_fields,
    response_from_nested,
    to_nested_by_length,
)

from ._rollout_shapes import make_rollout_batch


def _padded(rows: list[list[int]], pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of int sequences to a rectangle; return (padded, lengths)."""
    n = len(rows)
    s = max(len(r) for r in rows)
    out = torch.full((n, s), pad, dtype=torch.long)
    lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out, lens


# ── to_nested_by_length ───────────────────────────────────────────────


def test_to_nested_by_length_strips_padding() -> None:
    """The right-pad columns must NOT be in the nested output."""
    padded, lens = _padded([[1, 2, 3], [4, 5], [6, 7, 8, 9]], pad=0)
    nested = to_nested_by_length(padded, lens)
    assert nested.is_nested
    rows = list(nested.unbind())
    assert torch.equal(rows[0], torch.tensor([1, 2, 3]))
    assert torch.equal(rows[1], torch.tensor([4, 5]))
    assert torch.equal(rows[2], torch.tensor([6, 7, 8, 9]))


def test_to_nested_by_length_preserves_dtype() -> None:
    """bf16 in → bf16 out."""
    padded = torch.randn((3, 5), dtype=torch.bfloat16)
    lens = torch.tensor([2, 4, 5], dtype=torch.long)
    nested = to_nested_by_length(padded, lens)
    assert nested.dtype == torch.bfloat16


def test_to_nested_by_length_rejects_shape_mismatch() -> None:
    padded = torch.zeros((3, 4))
    bad_lens = torch.tensor([1, 2])  # only 2, not 3
    with pytest.raises(ValueError, match=r"lengths shape"):
        to_nested_by_length(padded, bad_lens)


def test_to_nested_by_length_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match=r"\(N, S"):
        to_nested_by_length(torch.zeros(5), torch.tensor([5]))


# ── materialize: jagged → padded ──────────────────────────────────────


def test_materialize_pads_nested_with_field_specific_pad_value() -> None:
    """Token field padded with pad_token_id; mask padded with 0.

    This is the contract worker code expects: the padded view it
    receives looks identical to a rectangular tensor produced by
    batched_message_log_to_flat_message.
    """
    ids_padded, lens = _padded([[10, 20, 30], [40, 50], [60, 70, 80, 90]], pad=0)
    mask_padded, _ = _padded([[1, 1, 1], [1, 1], [1, 1, 1, 1]], pad=0)
    ids_nested = to_nested_by_length(ids_padded, lens)
    mask_nested = to_nested_by_length(mask_padded, lens)

    td = TensorDict(
        {"input_ids": ids_nested, "token_mask": mask_nested},
        batch_size=[3],
    )

    bdd = materialize(
        td,
        layout="padded",
        pad_value_dict={"input_ids": 999, "token_mask": 0},
    )

    # Tokens are padded with the requested ID, not 0.
    assert bdd["input_ids"].shape == (3, 4)
    assert bdd["input_ids"][0, 3].item() == 999  # row 0 needs 1 pad
    assert bdd["input_ids"][1, 2].item() == 999  # row 1 needs 2 pads
    assert bdd["input_ids"][1, 3].item() == 999
    assert bdd["input_ids"][2, 3].item() == 90  # row 2 needs no padding

    # Mask uses the default 0 — match the source.
    assert bdd["token_mask"].shape == (3, 4)
    assert bdd["token_mask"][0, 3].item() == 0
    assert bdd["token_mask"][2, 3].item() == 1


def test_materialize_passes_through_rectangular_tensors() -> None:
    """Already-padded fields are emitted unchanged (no spurious copy)."""
    rect = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    td = TensorDict({"sample_mask": rect}, batch_size=[2])
    bdd = materialize(td, layout="padded")
    assert torch.equal(bdd["sample_mask"], rect)


def test_materialize_jagged_layout_passes_nested_through() -> None:
    """``layout='jagged'`` is the path for callers that consume nested."""
    padded, lens = _padded([[1, 2], [3, 4, 5]], pad=0)
    nested = to_nested_by_length(padded, lens)
    td = TensorDict({"x": nested}, batch_size=[2])
    bdd = materialize(td, layout="jagged")
    assert bdd["x"].is_nested


def test_materialize_default_pad_value_is_zero() -> None:
    """No pad_value_dict → fields pad with 0."""
    padded, lens = _padded([[1, 2, 3], [4]], pad=0)
    nested = to_nested_by_length(padded, lens)
    td = TensorDict({"x": nested}, batch_size=[2])
    bdd = materialize(td, layout="padded")
    assert bdd["x"][1, 1].item() == 0
    assert bdd["x"][1, 2].item() == 0


# ── response_from_nested ──────────────────────────────────────────────


def test_response_from_nested_extracts_response_slice() -> None:
    """Worker write-back path: jagged (prompt+response) → response only.

    With the verl convention, output position i corresponds to predicting
    input token i+1 — so the slice is left-shifted by one.
    """
    # Two samples: prompt_len=2, resp_len=3 / prompt_len=1, resp_len=2
    full_rows = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),  # prompt 0,1; resp 2,3,4
        torch.tensor([1.1, 1.2, 1.3]),  # prompt 0;   resp 1,2
    ]
    full = torch.nested.as_nested_tensor(full_rows, layout=torch.jagged)
    resp_mask_rows = [
        torch.tensor([1.0, 1.0, 1.0]),  # response_len = 3
        torch.tensor([1.0, 1.0]),  # response_len = 2
    ]
    response_mask = torch.nested.as_nested_tensor(resp_mask_rows, layout=torch.jagged)

    out = response_from_nested(full, response_mask)
    assert out.is_nested
    rows = list(out.unbind())
    # Row 0: full has 5 tokens; resp_len=3 → values[5-3-1:5-1] = values[1:4] = [0.2, 0.3, 0.4]
    assert torch.allclose(rows[0], torch.tensor([0.2, 0.3, 0.4]))
    # Row 1: full has 3 tokens; resp_len=2 → values[3-2-1:3-1] = values[0:2] = [1.1, 1.2]
    assert torch.allclose(rows[1], torch.tensor([1.1, 1.2]))


# ── Realistic-shape coverage using ``_rollout_shapes.make_rollout_batch`` ──
# These exercise the same codec helpers with the exact dtypes + value
# distributions a real GRPO rollout produces (bf16 logprobs, int64 ids,
# int32 masks, variable per-row lengths). Catches dtype-narrowing and
# padding-arithmetic bugs that pass on the toy data above.


@pytest.mark.parametrize(
    "logprob_dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "fp32"],
)
def test_to_nested_by_length_realistic_logprobs(logprob_dtype: torch.dtype) -> None:
    """``generation_logprobs`` shape (bf16/fp32) from a real rollout shape round-trips."""

    batch = make_rollout_batch(n=8, max_seqlen=128, logprob_dtype=logprob_dtype, seed=7)
    nested = to_nested_by_length(batch["generation_logprobs"], batch["input_lengths"])
    # dtype must survive the conversion (bf16 in → bf16 out).
    assert nested.dtype == logprob_dtype
    # Per-row valid region matches the input.
    for i, row in enumerate(nested.unbind()):
        valid = int(batch["input_lengths"][i])
        assert row.shape[0] == valid
        assert torch.equal(
            row, batch["generation_logprobs"][i, :valid].to(logprob_dtype)
        )


def test_materialize_realistic_full_field_set_preserves_dtypes() -> None:
    """All rollout fields round-trip through ``materialize`` with correct dtypes.

    Catches the class of bugs where padding silently upcasts bf16 → fp32 or
    coerces int64 → int32 because pad_value_dict's defaults were the wrong type.
    """

    batch = make_rollout_batch(n=4, max_seqlen=64, seed=11)
    # Build a wire TD with jagged leaves keyed by field name.
    td = TensorDict(
        {
            "input_ids": to_nested_by_length(
                batch["input_ids"], batch["input_lengths"]
            ),
            "generation_logprobs": to_nested_by_length(
                batch["generation_logprobs"], batch["input_lengths"]
            ),
            "token_mask": to_nested_by_length(
                batch["token_mask"], batch["input_lengths"]
            ),
        },
        batch_size=[4],
    )
    out = materialize(td, layout="padded", pad_value_dict={"input_ids": 0})

    # Each field comes back at its original dtype.
    assert out["input_ids"].dtype == torch.long
    assert out["generation_logprobs"].dtype == torch.bfloat16
    assert out["token_mask"].dtype == torch.int32


def test_pack_jagged_fields_forced_per_token_field_drops_extra_padding() -> None:
    """Known per-token writebacks can be wider than ``max(input_lengths)``."""

    lengths = torch.tensor([3, 5], dtype=torch.long)
    advantages = torch.arange(16, dtype=torch.float32).view(2, 8)
    extra = torch.arange(16, dtype=torch.float32).view(2, 8)

    td = pack_jagged_fields(
        {
            "advantages": advantages,
            "extra_2d": extra,
        },
        lengths=lengths,
        token_aligned_fields=frozenset({"advantages"}),
    )

    out = materialize(td, layout="padded")

    assert out["advantages"].shape == (2, 5)
    assert torch.equal(out["advantages"][0, :3], advantages[0, :3])
    assert torch.equal(out["advantages"][1], advantages[1, :5])
    assert torch.equal(out["advantages"][0, 3:], torch.zeros(2))
    assert torch.equal(out["extra_2d"], extra)
