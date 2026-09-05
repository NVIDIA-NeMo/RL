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


@pytest.mark.parametrize("field_name", ["mask_sample", "truncated"])
def test_raw_sample_filter_fields_roundtrip_as_dense_1d(field_name: str) -> None:
    """Raw loss-filter fields survive the Mooncake scalar wire as dense ``(N,)``.

    Ported from the ``_promote_1d_leaves`` pair this file used to carry. That
    writer/reader pair was replaced by ``_patch_scalar_field_schema``, which
    fixes the same TQ bug at the schema layer and covers *every* dense 1-D
    field instead of a hand-kept allowlist — so there is no promote step to
    assert on any more, and the property to pin is just the round trip.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    n = 4
    original = torch.tensor([False, True, False, True])
    td = TensorDict({field_name: original}, batch_size=[n])

    back = _from_wire(td)
    assert back[field_name].shape == (n,)
    assert torch.equal(back[field_name], original)


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


def test_from_wire_squeezes_nothing_even_for_scalar_field_names() -> None:
    """``_from_wire`` densifies; it never reinterprets a row's rank.

    A ``(1,)`` row is genuine data now. Per-sample scalar columns are
    stored as 0-d rows (``_patch_scalar_field_schema``) and TQ stacks them
    into a dense ``(N,)`` before this function runs, so a nested ``(1,)``
    row arriving here means the producer really wrote length-1 rows —
    squeezing it would corrupt them. Field name must not change that.
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

    # ``total_reward`` is a per-sample scalar by name, but these rows are
    # length-1 vectors — both columns densify identically.
    assert back["total_reward"].shape == (n, 1)
    assert back["input_ids"].shape == (n, 1)
    assert torch.equal(back["input_ids"], torch.arange(n).unsqueeze(-1))


def test_from_wire_passes_dense_fields_through_untouched() -> None:
    """Dense inputs are returned as-is — no rank policing by field name.

    Replaces a guard that rejected a declared scalar arriving as ``(3, 2)``.
    That check belonged to the writer-unsqueeze/reader-squeeze pair, which
    no longer exists: the schema now reports the shape the rows actually
    have, so there is no promoted encoding for a malformed value to
    violate.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    wire = TensorDict({"input_lengths": torch.ones(3, 2)}, batch_size=[3])

    back = _from_wire(wire)
    assert back["input_lengths"].shape == (3, 2)


def test_put_samples_passes_fields_and_tags_through_unchanged(monkeypatch) -> None:
    """``put_samples`` reshapes nothing and does not touch user tags.

    The writer-unsqueeze half of the old 1-D workaround is gone — the
    schema now reports the ``()`` rows TQ actually stores — so a ``(N,)``
    column reaches the wire as ``(N,)``. Tag passthrough was the other
    half of this test's intent and is unchanged.
    """
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
        assert fields["input_lengths"].shape == (n,)  # not promoted any more
        assert fields["input_ids"].shape == (n, 1)
        assert tags == user_tags

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", fake_kv_batch_put)
    client = object.__new__(tq_adapter.TQDataPlaneClient)

    meta = client.put_samples(
        ["a", "b", "c"], "train", fields=original_fields, tags=user_tags
    )

    assert meta.tags == user_tags


def test_get_samples_returns_scalar_columns_dense(monkeypatch) -> None:
    """A per-sample scalar column arrives dense and is passed through.

    TQ stores these as 0-d rows and ``_merge_tensors_to_tensordict`` stacks
    them into ``(N,)`` before the adapter sees them, so ``get_samples`` has
    no rank to restore — it just must not disturb the column. Previously
    this arrived nested with ``(1,)`` rows and was squeezed back.
    """
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
            # Dense: what TQ hands back for a 0-d-row scalar column.
            "total_reward": original["total_reward"],
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
    client._data_operations_started = False

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
    client._data_operations_started = False

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
