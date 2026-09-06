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
"""The dense 1-D sample shape TQ reports must match the rows it stores.

Upstream ``extract_field_schema`` rebinds a *local* for 1-D inputs
(``value = value.unsqueeze(-1)``) and derives the sample shape from it,
while ``_generate_values`` iterates the *original* ``(N,)`` tensor into
``N`` 0-d rows. Schema says ``(1,)``, storage holds ``()``.

Only the KV path notices, because it is the only one that reconstructs
from the schema: ``BatchMeta.get_shapes`` repeats the uniform ``shape``
per sample and the mooncake client reshapes raw bytes with it. That turns
a scalar column into ``(1,)`` rows, which then re-nest instead of taking
``_merge_tensors_to_tensordict``'s ``all(dim() == 0) -> torch.stack``
branch. ``SimpleStorage`` fetches stored objects by ``(index, field)`` and
never consults the schema, so it is unaffected.

These tests pin the invariant on TQ's own function rather than on a
recorded shape, so they fail if a future TQ revision changes the
derivation in either direction — the fix landing upstream shows up here as
a still-passing test, and a different 1-D convention shows up as a failure
rather than as silent nested scalars in production.
"""

from __future__ import annotations

import pytest
import torch

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

tq_metadata = pytest.importorskip(
    "transfer_queue.metadata",
    reason="transfer_queue not installed",
)
TensorDict = pytest.importorskip("tensordict").TensorDict


@pytest.fixture(autouse=True)
def _patched():
    """Apply the patch for each test; it is idempotent and process-global."""
    tq_adapter._patch_scalar_field_schema()
    yield


def _schema(fields: dict, n: int) -> dict:
    return tq_metadata.extract_field_schema(TensorDict(fields, batch_size=[n]))


def test_dense_1d_field_reports_scalar_sample_shape() -> None:
    """``(N,)`` in means ``()`` per sample — what ``_generate_values`` stores.

    A ``(1,)`` here is the upstream bug: the KV read would reshape each
    row to ``(1,)`` and the column would come back nested.
    """
    schema = _schema({"input_lengths": torch.arange(4, dtype=torch.int64)}, 4)
    assert tuple(schema["input_lengths"]["shape"]) == ()


def test_byte_count_is_unchanged_by_the_fix() -> None:
    """``prod(()) == prod((1,)) == 1`` — this is a reshape fix, not a sizing
    one. Pinned because a sizing change would corrupt reads rather than
    mis-shape them, and that is a much worse failure to discover late."""
    from transfer_queue.utils.tensor_utils import get_nbytes

    n = 4
    shape = _schema({"input_lengths": torch.arange(n, dtype=torch.int64)}, n)[
        "input_lengths"
    ]["shape"]
    assert get_nbytes([torch.int64] * n, [shape] * n) == [8] * n


def test_2d_and_nested_fields_are_left_alone() -> None:
    """The patch must touch dense 1-D only.

    ``(N, S)`` already agrees with its stored rows, and nested fields carry
    exact ``per_sample_shapes`` — rewriting either would break the fields
    that were never broken.
    """
    rows = [torch.ones(3), torch.ones(5)]
    schema = _schema(
        {
            "input_ids": torch.zeros(2, 7, dtype=torch.long),
            "logprobs": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
        },
        2,
    )
    assert tuple(schema["input_ids"]["shape"]) == (7,)
    assert schema["logprobs"]["is_nested"]
    assert [tuple(s) for s in schema["logprobs"]["per_sample_shapes"]] == [(3,), (5,)]


def test_probe_confirms_tq_stores_scalar_rows_as_0d() -> None:
    """The premise the whole patch rests on, checked against TQ itself.

    There is no payload-side fallback any more, so if a TQ revision started
    storing dense 1-D fields as ``(1,)`` rows the rewrite would turn a
    correct schema into a wrong one — and the symptom would be corrupt
    reads, not an import error. ``_patch_scalar_field_schema`` runs this
    probe before installing itself; here it is asserted directly so the
    failure names the cause.
    """
    from transfer_queue.storage.managers.base import KVStorageManager

    rows = KVStorageManager._generate_values(
        TensorDict({"probe": torch.zeros(2)}, batch_size=[2])
    )
    assert len(rows) == 2
    assert all(r.ndim == 0 for r in rows), (
        f"TQ now stores dense 1-D rows as {[tuple(r.shape) for r in rows]}; "
        "the scalar schema patch must be re-checked"
    )


def test_patch_is_idempotent() -> None:
    """Every process that builds a client calls it; double-application must
    not stack wrappers (which would still be correct but unboundedly deep)."""
    first = tq_metadata.extract_field_schema
    tq_adapter._patch_scalar_field_schema()
    assert tq_metadata.extract_field_schema is first


def test_storage_managers_see_the_patched_function() -> None:
    """Both managers bind the name at import time
    (``from transfer_queue.metadata import extract_field_schema``), so
    rebinding only the defining module would leave them on the original and
    the fix would silently not apply where it is actually called."""
    from transfer_queue.storage.managers import base as _base

    assert _base.extract_field_schema is tq_metadata.extract_field_schema


def test_scalar_rows_round_trip_as_a_dense_column() -> None:
    """End state the fix exists for: reconstructing with the reported shape
    yields 0-d rows, which ``_merge_tensors_to_tensordict`` stacks back into
    a dense ``(N,)`` column instead of re-nesting it."""
    n = 4
    src = torch.arange(n, dtype=torch.int64)
    shape = _schema({"input_lengths": src}, n)["input_lengths"]["shape"]

    # What the KV client does: one stored row -> reshape(reported shape).
    rebuilt = [row.reshape(tuple(shape)) for row in src]
    assert all(r.dim() == 0 for r in rebuilt)
    assert torch.equal(torch.stack(rebuilt), src)
