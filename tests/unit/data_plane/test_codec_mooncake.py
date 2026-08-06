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

import pytest
import torch

from nemo_rl.data_plane.codec import pack_per_token_field, to_nested_by_length

from ._rollout_shapes import make_rollout_batch

# ── P1: promote_1d — writer unsqueezes, reader squeezes ──────────────────────


def test_promote_1d_leaves_unsqueezes_declared_1d_field() -> None:
    """`_promote_1d_leaves` promotes only fields declared in PROMOTE_1D_FIELDS."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _promote_1d_leaves

    n = 8
    # `input_lengths` is declared in PROMOTE_1D_FIELDS; `input_ids` (2D) is not
    # touched since it isn't 1D.
    td = TensorDict(
        {
            "input_lengths": torch.arange(n, dtype=torch.long),
            "input_ids": torch.arange(n * 3).reshape(n, 3),
        },
        batch_size=[n],
    )

    out = _promote_1d_leaves(td)
    assert out["input_lengths"].shape == (n, 1)
    assert out["input_ids"].shape == (n, 3)


def test_promote_1d_leaves_skips_undeclared_fields() -> None:
    """1D fields not declared in PROMOTE_1D_FIELDS pass through unchanged.

    The writer-side guard (`_assert_promote_1d_contract`) raises on this
    case in production; `_promote_1d_leaves` itself is silent.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _promote_1d_leaves

    td = TensorDict(
        {"custom_undeclared": torch.arange(4, dtype=torch.long)},
        batch_size=[4],
    )
    out = _promote_1d_leaves(td)
    assert out["custom_undeclared"].shape == (4,)  # unchanged


def test_promote_1d_roundtrip_via_from_wire() -> None:
    """`_promote_1d_leaves` then `_from_wire` restores the original (N,) shape."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import (
        _from_wire,
        _promote_1d_leaves,
    )

    n = 6
    original = torch.arange(n, dtype=torch.float32)
    td = TensorDict({"total_reward": original}, batch_size=[n])

    wire = _promote_1d_leaves(td)
    assert wire["total_reward"].shape == (n, 1)

    back = _from_wire(wire)
    assert back["total_reward"].shape == (n,)
    assert torch.equal(back["total_reward"], original)


def test_from_wire_densifies_uniform_nested_and_squeezes_declared_scalar() -> None:
    """TQ v0.1.9's uniform nested reads densify; declared 1D fields squeeze back."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    rows = [torch.tensor([i], dtype=torch.float32) for i in range(4)]
    wire = TensorDict(
        {"total_reward": torch.nested.as_nested_tensor(rows, layout=torch.jagged)},
        batch_size=[len(rows)],
    )

    back = _from_wire(wire)

    assert not back["total_reward"].is_nested
    assert back["total_reward"].shape == (len(rows),)
    assert torch.equal(
        back["total_reward"], torch.arange(len(rows), dtype=torch.float32)
    )


def test_from_wire_preserves_genuine_length_one_token_column() -> None:
    """Fields NOT in PROMOTE_1D_FIELDS retain (N, 1) shape after densification.

    Distinguishes a promoted per-sample scalar (`total_reward`) from a
    per-token field that happens to have length 1 across the batch
    (`input_ids`) — the schema lookup does the disambiguation, not any
    per-row wire metadata.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    n = 4
    wire = TensorDict(
        {
            "total_reward": torch.nested.as_nested_tensor(
                [torch.tensor([float(i)]) for i in range(n)], layout=torch.jagged
            ),
            "input_ids": torch.nested.as_nested_tensor(
                [torch.tensor([i]) for i in range(n)], layout=torch.jagged
            ),
        },
        batch_size=[n],
    )

    back = _from_wire(wire)

    assert back["total_reward"].shape == (n,)         # in schema → squeezed
    assert back["input_ids"].shape == (n, 1)          # not in schema → preserved
    assert torch.equal(back["input_ids"], torch.arange(n).unsqueeze(-1))


def test_assert_contract_raises_on_undeclared_1d_field() -> None:
    """Writer-side guard: dense 1D field not in PROMOTE_1D_FIELDS raises."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import (
        _assert_promote_1d_contract,
    )

    td = TensorDict(
        {"unregistered_scalar": torch.arange(4, dtype=torch.long)}, batch_size=[4]
    )
    with pytest.raises(ValueError, match="not declared in PROMOTE_1D_FIELDS"):
        _assert_promote_1d_contract(td)


def test_assert_contract_raises_on_declared_field_wrong_shape() -> None:
    """Writer-side guard: declared field arriving as non-1D raises."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import (
        _assert_promote_1d_contract,
    )

    td = TensorDict(
        {"input_lengths": torch.arange(4 * 2).reshape(4, 2)}, batch_size=[4]
    )
    with pytest.raises(ValueError, match="declared in PROMOTE_1D_FIELDS"):
        _assert_promote_1d_contract(td)


def test_assert_contract_accepts_valid_batch() -> None:
    """Writer-side guard: valid TD (declared 1D, undeclared 2D, nested) passes."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import (
        _assert_promote_1d_contract,
    )

    n = 4
    td = TensorDict(
        {
            "input_lengths": torch.arange(n, dtype=torch.long),
            "input_ids": torch.arange(n * 3).reshape(n, 3),
            "logprobs": torch.nested.as_nested_tensor(
                [torch.randn(i + 1) for i in range(n)], layout=torch.jagged
            ),
        },
        batch_size=[n],
    )
    # Does not raise.
    _assert_promote_1d_contract(td)


def test_get_samples_single_code_path_across_backends(monkeypatch) -> None:
    """`get_samples` uses one read path for both backends — no custom_meta inspection."""
    from tensordict import TensorDict

    import nemo_rl.data_plane.adapters.transfer_queue as tq_adapter

    n = 3
    # Writer produced `total_reward` (in PROMOTE_1D_FIELDS) and `input_ids`
    # (not in PROMOTE_1D_FIELDS). Both round-trip through TQ v0.1.9 as
    # uniform-nested; densification + name-based squeeze restores shapes.
    wire_data = TensorDict(
        {
            "total_reward": torch.nested.as_nested_tensor(
                [torch.tensor([float(i)]) for i in range(n)], layout=torch.jagged
            ),
            "input_ids": torch.nested.as_nested_tensor(
                [torch.tensor([i]) for i in range(n)], layout=torch.jagged
            ),
        },
        batch_size=[n],
    )

    def fake_kv_batch_get(
        *, keys: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        assert keys == ["a", "b", "c"]
        assert partition_id == "train"
        assert select_fields == ["total_reward", "input_ids"]
        return wire_data

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_get", fake_kv_batch_get, raising=False)

    # Same code path exercises both backends — set self._promote_1d to True.
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._promote_1d = True

    restored = client.get_samples(
        ["a", "b", "c"], "train", ["total_reward", "input_ids"]
    )

    assert restored["total_reward"].shape == (n,)          # in schema → squeezed
    assert restored["input_ids"].shape == (n, 1)           # not in schema → preserved


def test_get_samples_densifies_uniform_rows_without_1d_promotion(monkeypatch) -> None:
    """The simple backend normalizes uniform nested rows without squeezing."""
    from tensordict import TensorDict

    import nemo_rl.data_plane.adapters.transfer_queue as tq_adapter

    rows = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    wire_data = TensorDict(
        {"input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged)},
        batch_size=[len(rows)],
    )

    def fake_kv_batch_get(
        *, keys: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        assert keys == ["a", "b"]
        assert partition_id == "train"
        assert select_fields == ["input_ids"]
        return wire_data

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_get", fake_kv_batch_get, raising=False)
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._promote_1d = False

    restored = client.get_samples(["a", "b"], "train", ["input_ids"])

    assert not restored["input_ids"].is_nested
    assert restored["input_ids"].shape == (2, 2)
    assert torch.equal(restored["input_ids"], torch.stack(rows))


def test_from_wire_preserves_ragged_nested_rows() -> None:
    """Variable-length rollout fields must remain nested."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    rows = [torch.arange(i + 1) for i in range(3)]
    nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    wire = TensorDict({"token_ids": nested}, batch_size=[len(rows)])

    back = _from_wire(wire)

    assert back["token_ids"].is_nested
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(back["token_ids"].unbind(), rows, strict=True)
    )


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
