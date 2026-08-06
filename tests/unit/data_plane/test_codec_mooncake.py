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
  P1 — schema-declared 1D scalar round-trip through the Mooncake workaround.
  P2 — pack_per_token_field: tolerates SP padding wider than max(lengths).

No Ray, no GPU, no transfer_queue required.
"""

from __future__ import annotations

import pytest
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
    td = TensorDict({"input_lengths": t}, batch_size=[n])

    out = _promote_1d_leaves(td)
    assert out["input_lengths"].shape == (n, 1), (
        "Expected input_lengths to use the Mooncake wire shape "
        f"({n}, 1), got {tuple(out['input_lengths'].shape)}."
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
    td = TensorDict({"input_lengths": original}, batch_size=[n])

    wire = _promote_1d_leaves(td)
    assert wire["input_lengths"].shape == (n, 1)

    back = _from_wire(wire)
    assert back["input_lengths"].shape == (n,)
    assert torch.equal(back["input_lengths"], original)


def test_from_wire_densifies_uniform_nested_rows() -> None:
    """TQ v0.1.9's uniform nested reads are restored to dense tensors."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    rows = [torch.tensor([i, i + 1], dtype=torch.float32) for i in range(4)]
    wire = TensorDict(
        {"input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged)},
        batch_size=[len(rows)],
    )

    back = _from_wire(wire)

    assert not back["input_ids"].is_nested
    assert back["input_ids"].shape == (len(rows), 2)
    assert torch.equal(back["input_ids"], torch.stack(rows))


def test_from_wire_preserves_genuine_length_one_token_column() -> None:
    """Only fields promoted from ``(N,)`` are squeezed after a TQ read."""
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

    assert back["total_reward"].shape == (n,)
    assert back["input_ids"].shape == (n, 1)
    assert torch.equal(back["input_ids"], torch.arange(n).unsqueeze(-1))


def test_from_wire_rejects_invalid_declared_field_shape() -> None:
    """A corrupted scalar wire shape fails at the data-plane boundary."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    wire = TensorDict({"input_lengths": torch.ones(3, 2)}, batch_size=[3])

    with pytest.raises(ValueError, match=r"input_lengths.*\(N, 1\)"):
        _from_wire(wire)


def test_promote_1d_leaves_rejects_undeclared_1d_field() -> None:
    """New scalar fields must be added to the authoritative schema."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _promote_1d_leaves

    fields = TensorDict({"new_scalar": torch.arange(3)}, batch_size=[3])

    with pytest.raises(ValueError, match="not declared.*PROMOTE_1D_FIELDS"):
        _promote_1d_leaves(fields)


def test_promote_1d_leaves_rejects_invalid_declared_field_shape() -> None:
    """A schema-declared scalar cannot silently change its user-level rank."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _promote_1d_leaves

    fields = TensorDict({"input_lengths": torch.ones(3, 2)}, batch_size=[3])

    with pytest.raises(ValueError, match=r"input_lengths.*shape \(N,\)"):
        _promote_1d_leaves(fields)


def test_put_samples_uses_schema_without_private_shape_tags(monkeypatch) -> None:
    """Mooncake promotion changes tensors but not user-provided TQ tags."""
    from tensordict import TensorDict

    import nemo_rl.data_plane.adapters.transfer_queue as tq_adapter

    n = 3
    original_fields = TensorDict(
        {
            "input_lengths": torch.arange(n),
            "input_ids": torch.arange(n).unsqueeze(-1),
        },
        batch_size=[n],
    )
    user_tags = [{"weight_version": 7} for _ in range(n)]

    def fake_kv_batch_put(
        *,
        keys: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict[str, object]],
    ) -> None:
        assert keys == ["a", "b", "c"]
        assert partition_id == "train"
        assert fields["input_lengths"].shape == (n, 1)
        assert fields["input_ids"].shape == (n, 1)
        assert tags == user_tags

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", fake_kv_batch_put)
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._promote_1d = True

    meta = client.put_samples(
        ["a", "b", "c"], "train", fields=original_fields, tags=user_tags
    )

    assert meta.tags == user_tags


def test_get_samples_uses_static_shape_schema(monkeypatch) -> None:
    """The Mooncake adapter restores scalar ranks without row metadata."""
    from tensordict import TensorDict

    import nemo_rl.data_plane.adapters.transfer_queue as tq_adapter

    n = 3
    original = TensorDict(
        {
            "total_reward": torch.arange(n, dtype=torch.float32),
            "input_ids": torch.arange(n).unsqueeze(-1),
        },
        batch_size=[n],
    )
    wire_data = TensorDict(
        {
            "total_reward": torch.nested.as_nested_tensor(
                [row for row in original["total_reward"].unsqueeze(-1)],
                layout=torch.jagged,
            ),
            "input_ids": torch.nested.as_nested_tensor(
                [row for row in original["input_ids"]], layout=torch.jagged
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

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_get", fake_kv_batch_get)
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._promote_1d = True

    restored = client.get_samples(
        ["a", "b", "c"], "train", ["total_reward", "input_ids"]
    )

    assert restored["total_reward"].shape == (n,)
    assert restored["input_ids"].shape == (n, 1)
    assert torch.equal(restored["total_reward"], original["total_reward"])
    assert torch.equal(restored["input_ids"], original["input_ids"])


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
