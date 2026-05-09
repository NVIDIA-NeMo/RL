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
"""Unit tests for the mooncake_cpu-specific codec flags.

Covers:
  P1 — _KV_PROMOTE_1D flag: writer unsqueezes 1D → (N,1), reader squeezes back.
  P2 — pack_per_token_field: tolerates SP padding wider than max(lengths).

No Ray, no GPU, no transfer_queue required.
"""

from __future__ import annotations

import pytest
import torch

# ── Module-level state restoration fixture ───────────────────────────────────


@pytest.fixture
def codec_flags():
    """Save and restore module-level flags after each test.

    Tests that mutate _KV_PROMOTE_1D must use this fixture so they
    cannot pollute other tests in the session.
    """
    from nemo_rl.data_plane import codec

    saved = codec._KV_PROMOTE_1D
    yield codec
    codec._KV_PROMOTE_1D = saved


# ── P1: _KV_PROMOTE_1D — writer unsqueezes, reader squeezes ──────────────────


def test_promote_1d_unsqueezes_on_write(codec_flags) -> None:
    """When _KV_PROMOTE_1D is True, writing a (N,) tensor through _to_wire
    produces an (N, 1) tensor on the wire.

    This guards the mooncake_cpu path where TQ's extract_field_schema silently
    unsqueezes 1D fields in metadata (metadata.py:171-173). The fix is to
    pre-unsqueeze at the wire layer so per-row shape matches the metadata shape.
    """
    # Import the adapter's _to_wire directly so this test stays unit-level.
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _to_wire

    codec_flags.set_kv_promote_1d(True)

    n = 8
    t = torch.arange(n, dtype=torch.float32)
    td = TensorDict({"reward": t}, batch_size=[n])

    out = _to_wire(td)
    assert out["reward"].shape == (n, 1), (
        f"Expected wire shape ({n}, 1) but got {tuple(out['reward'].shape)}. "
        "1D→2D promotion must happen when _KV_PROMOTE_1D is True."
    )


def test_promote_1d_squeezes_on_read_roundtrip(codec_flags) -> None:
    """After a write-unsqueeze, the reader squeezes back so consumers see (N,).

    Simulates the full write → read round-trip through materialize().
    """
    from tensordict import TensorDict

    codec_flags.set_kv_promote_1d(True)

    n = 6
    original = torch.arange(n, dtype=torch.float32)

    # Simulate what _to_wire does on the mooncake_cpu path.
    wire_tensor = original.unsqueeze(-1).contiguous()  # (N, 1)
    td = TensorDict({"reward": wire_tensor}, batch_size=[n])

    # materialize squeezes (N, 1) back to (N,) when _KV_PROMOTE_1D is True.
    from nemo_rl.data_plane.codec import _KV_PROMOTE_1D as flag_before  # noqa: F401

    # The flag is now True (set above). Directly call the squeeze logic.
    from nemo_rl.data_plane.codec import materialize

    bdd = materialize(td, layout="padded")

    assert bdd["reward"].shape == (n,), (
        f"Expected shape ({n},) after read squeeze but got {tuple(bdd['reward'].shape)}."
    )
    assert torch.equal(bdd["reward"], original), (
        "Values changed during 1D round-trip unsqueeze→squeeze."
    )


def test_promote_1d_off_leaves_shape_unchanged(codec_flags) -> None:
    """When _KV_PROMOTE_1D is False (the default), 1D tensors pass through
    the wire layer without modification."""
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _to_wire

    codec_flags.set_kv_promote_1d(False)

    n = 5
    t = torch.arange(n, dtype=torch.long)
    td = TensorDict({"idx": t}, batch_size=[n])

    out = _to_wire(td)
    assert out["idx"].shape == (n,), (
        f"Expected shape ({n},) when _KV_PROMOTE_1D=False but got {tuple(out['idx'].shape)}."
    )


# ── P2: pack_per_token_field — tolerates SP padding ──────────────────────────


def test_pack_per_token_field_truncates_sp_padding() -> None:
    """pack_per_token_field slices each row to its own length, dropping SP padding.

    mcore SP rounds the forward output's seq dim up to a multiple of TP, so
    val.shape[1] > max(lengths). maybe_pack_jagged would skip this field
    (wrong shape); pack_per_token_field handles it correctly.
    """
    from nemo_rl.data_plane.codec import pack_per_token_field

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


def test_pack_per_token_field_exact_fit_equals_maybe_pack_jagged() -> None:
    """When val.shape[1] == max(lengths), pack_per_token_field and
    maybe_pack_jagged produce identical jagged outputs.

    This is the 'no SP padding' case — the two helpers must agree when
    the input is already exactly the right width.
    """
    from nemo_rl.data_plane.codec import maybe_pack_jagged, pack_per_token_field

    n = 4
    lengths = torch.tensor([3, 5, 2, 4], dtype=torch.long)
    max_len = int(lengths.max().item())
    val = torch.randn(n, max_len)

    out_pack = pack_per_token_field(val, lengths)
    out_maybe = maybe_pack_jagged(val, lengths)

    assert out_pack.is_nested
    assert out_maybe.is_nested

    rows_pack = list(out_pack.unbind())
    rows_maybe = list(out_maybe.unbind())
    for i, (rp, rm) in enumerate(zip(rows_pack, rows_maybe)):
        assert torch.equal(rp, rm), (
            f"Row {i} differs between pack_per_token_field and maybe_pack_jagged "
            "on an exact-fit input."
        )
