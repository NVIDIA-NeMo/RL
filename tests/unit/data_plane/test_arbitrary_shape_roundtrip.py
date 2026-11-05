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
"""Can a row of arbitrary shape survive put -> get unchanged?

The producer hands the data plane **one entry per row** and gets the same
rows back. No ``torch.nested`` container, so no layout can reject a shape
and nothing has to be padded to make a container accept it.

Deliberately does NOT go through ``kv_first_write`` / ``pack_jagged_fields``:
those apply the application layer's padded-rectangle conventions, and
whether those are convenient is a separate argument from whether the
transport can carry the shape at all.

Three escalating cases, matching what real VLM fields look like:

  A. ragged dim 0, uniform trailing dims   -- ``pixel_values [n_patches, D]``
  B. non-uniform trailing dims, same rank  -- dynamic-resolution ``[n, 3, H_i, W_i]``
  C. differing rank per row                -- image ``[p, D]`` beside video ``[T, p, D]``

All three work today. ``PackedTensor.to_wire`` flattens each row to 1-D
before handing it to ``torch.jagged``, so rows differ only in dim 0 and B and
C encode as easily as A -- see ``test_to_wire_carries_mixed_rank_rows``. The
true shapes ride beside the payload on ``KVBatchMeta.tags``. What
``pad_to_max_shape`` still materializes is applied in worker memory by
``as_tensor``, never on the wire, and mixed rank is rejected only there.

The ``test_jagged_rejects_*`` cases need no backend and never skip. They
pin *why* the padding exists, so the workaround can be deleted with
evidence rather than by assertion.

Open assumption this exists to settle: TQ's msgpack encoder must
round-trip a ``torch.Tensor`` inside a ``NonTensorStack``.
``codec.unwrap_wire_stripped_payload`` exists because that path has bitten
before, and nothing currently verifies it for tensor payloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# --------------------------------------------------------------------------
# Row fixtures. Small, but structurally faithful to the real fields.
# --------------------------------------------------------------------------

# A: only dim 0 varies. Trailing dim D is fixed by the vision tower.
ROWS_RAGGED_DIM0 = [
    torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8),
    torch.arange(7 * 8, dtype=torch.float32).reshape(7, 8),
    torch.arange(1 * 8, dtype=torch.float32).reshape(1, 8),
]

# B: same rank, H/W differ per row (dynamic-resolution processors).
ROWS_RAGGED_TRAILING = [
    torch.zeros(2, 3, 4, 4),
    torch.ones(1, 3, 6, 6),
    torch.full((3, 3, 2, 8), 7.0),
]

# C: rank differs per row (a still image beside a video clip).
ROWS_MIXED_RANK = [
    torch.zeros(5, 8),  # image: [patches, D]
    torch.ones(2, 5, 8),  # video: [frames, patches, D]
    torch.full((3, 8), 2.0),  # image: [patches, D]
]

ALL_CASES = [
    (ROWS_RAGGED_DIM0, "ragged_dim0"),
    (ROWS_RAGGED_TRAILING, "ragged_trailing"),
    (ROWS_MIXED_RANK, "mixed_rank"),
]
CASE_IDS = ["ragged_dim0", "ragged_trailing", "mixed_rank"]


def _to_wire(rows: list[torch.Tensor]) -> np.ndarray:
    """One object cell per row -- the whole encoder.

    ``pack_jagged_fields`` already forwards ``np.ndarray(dtype=object)``
    untouched and ``materialize`` already returns it without padding, so
    this needs no new field registry, no ``__lengths`` companion, and no
    shape bookkeeping in the application layer. TQ stores one entry per
    (sample, field) either way -- see ``base.py::_generate_values`` -- so
    this is not a fragmentation regression against the nested form.
    """
    arr = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        arr[i] = row
    return arr


def _rows_back(value) -> list:
    """Rows out of whatever container the adapter handed back.

    Deliberately permissive: ``_from_wire`` may densify or re-nest
    depending on what the storage manager reassembled. The container is
    not the contract -- the rows are.
    """
    if isinstance(value, torch.Tensor):
        return list(value.unbind()) if value.is_nested else list(value)
    return list(value)


def _assert_rows_equal(got: list, want: list[torch.Tensor]) -> None:
    assert len(got) == len(want), f"row count {len(got)} != {len(want)}"
    for i, (have, expect) in enumerate(zip(got, want, strict=True)):
        assert isinstance(have, torch.Tensor), (
            f"row {i} came back as {type(have).__name__}, not a Tensor -- "
            "the wire path did not preserve the payload"
        )
        assert have.shape == expect.shape, (
            f"row {i}: got {tuple(have.shape)}, want {tuple(expect.shape)}"
        )
        assert have.dtype == expect.dtype, f"row {i}: dtype changed"
        assert torch.equal(have, expect), f"row {i}: values differ"


# --------------------------------------------------------------------------
# No backend needed: pin the torch.nested constraint that makes ``to_wire``
# flatten each segment to 1-D before building the jagged value.
# --------------------------------------------------------------------------


def test_jagged_rejects_nonuniform_trailing_dims() -> None:
    """Why ``to_wire`` flattens rather than handing rows over as-is.

    ``torch.jagged`` allows exactly one ragged dim; every other dim must agree
    across rows, so these rows cannot be a nested value in their natural shape.
    Flattening each segment to 1-D moves all the variation onto dim 0, which is
    what makes the encoding total -- and it is why nothing has to be padded to
    satisfy the container. ``test_arbitrary_shape_rows_roundtrip[ragged_trailing]``
    is the same rows going through the real encoder.
    """
    with pytest.raises((RuntimeError, TypeError)):
        torch.nested.as_nested_tensor(ROWS_RAGGED_TRAILING, layout=torch.jagged)


def test_jagged_rejects_mixed_rank() -> None:
    """Same constraint, the case padding could never have solved.

    No amount of padding reconciles rank 2 with rank 3, so image-beside-video in
    one field is unrepresentable in a nested value built from natural shapes.
    ``to_wire`` carries it anyway because 1-D rows have no rank to disagree on
    -- see ``test_to_wire_carries_mixed_rank_rows``.
    """
    with pytest.raises((RuntimeError, TypeError)):
        torch.nested.as_nested_tensor(ROWS_MIXED_RANK, layout=torch.jagged)


@pytest.mark.parametrize("rows,case", ALL_CASES, ids=CASE_IDS)
def test_wire_form_preserves_rows_locally(rows, case) -> None:
    """``_to_wire`` round-trips in-process, before any transport.

    Separated from the backend tests so a failure here is unambiguously
    an encoding problem rather than a storage one.
    """
    _assert_rows_equal(_rows_back(_to_wire(rows)), rows)


# --------------------------------------------------------------------------
# Through the real adapter, on both backends.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows,case", ALL_CASES, ids=CASE_IDS)
def test_arbitrary_shape_rows_roundtrip(tq_client_backends, rows, case) -> None:
    """Put arbitrary-shaped rows, get identical rows back.

    ``ragged_dim0`` should pass today. ``ragged_trailing`` and
    ``mixed_rank`` are the open question: TQ's controller can represent
    them (``extract_field_schema`` records ``per_sample_shapes`` as full
    per-row tuples), but nothing verifies the storage managers accept
    rows that disagree on trailing dims or rank.
    """
    client = tq_client_backends
    # conftest requires a partition id unique to each test.
    partition_id = f"arbshape-{case}"
    sample_ids = [f"s{i}" for i in range(len(rows))]

    client.register_partition(
        partition_id=partition_id,
        fields=["pixel_values"],
        num_samples=len(rows),
        consumer_tasks=["train"],
    )
    try:
        client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=TensorDict({"pixel_values": _to_wire(rows)}, batch_size=[len(rows)]),
        )
        out = client.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=["pixel_values"],
        )
        _assert_rows_equal(_rows_back(out["pixel_values"]), rows)
    finally:
        client.clear_samples(sample_ids=sample_ids, partition_id=partition_id)


def test_no_padding_is_introduced(tq_client_backends) -> None:
    """Element count must not grow -- the regression this change is for.

    ``pad_to_max_shape`` inflates every row to the batch max on each
    non-ragged dim; for these rows that is 3*3*6*8 = 432 elements each
    (1296 total) against 336 actual. Asserting on the total catches a
    silent reintroduction of padding anywhere in the path.
    """
    client = tq_client_backends
    partition_id = "arbshape-nopad"
    sample_ids = [f"p{i}" for i in range(len(ROWS_RAGGED_TRAILING))]
    want_elems = sum(r.numel() for r in ROWS_RAGGED_TRAILING)

    client.register_partition(
        partition_id=partition_id,
        fields=["pixel_values"],
        num_samples=len(ROWS_RAGGED_TRAILING),
        consumer_tasks=["train"],
    )
    try:
        client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=TensorDict(
                {"pixel_values": _to_wire(ROWS_RAGGED_TRAILING)},
                batch_size=[len(ROWS_RAGGED_TRAILING)],
            ),
        )
        out = client.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=["pixel_values"],
        )
        got_elems = sum(r.numel() for r in _rows_back(out["pixel_values"]))
        assert got_elems == want_elems, (
            f"element count changed: {got_elems} != {want_elems} -- "
            "padding was reintroduced somewhere in the path"
        )
    finally:
        client.clear_samples(sample_ids=sample_ids, partition_id=partition_id)
