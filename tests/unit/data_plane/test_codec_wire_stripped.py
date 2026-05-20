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
"""Regression tests for the wire-stripped ``NonTensorStack`` path.

TQ's simple-backend ``MsgpackEncoder._encode_tensordict`` serializes any
``TensorDictBase`` via ``dict(obj.items())`` — only the tensor backing
dict. ``NonTensorData`` stores its payload in ``_non_tensordict["data"]``,
so it round-trips through ZMQ as an empty
``TensorDict({}, batch_size=[])`` — the string payload is silently
dropped. The simple-backend storage manager's ``_pack_field_values``
then assembles those stripped TDs into a ``NonTensorStack`` that
``materialize`` has to defend against. The pre-fix path crashed with
``RuntimeError: generator raised StopIteration``.

Construction note: ``tensordict>=0.12.2`` rejects
``NonTensorStack(TensorDict({}, batch_size=[]), ...)`` at construction
time (``All tensordicts must be non-tensors``). To validate
``materialize``'s decode without skirting tensordict's invariants we:

* test :func:`unwrap_wire_stripped_payload` directly — pure per-item
  helper, accepts the wire-stripped ``TensorDict`` shape without
  needing the stack constructor at all;
* drive :func:`materialize` end-to-end by patching ``.tolist()`` on a
  constructed (valid) ``NonTensorStack`` so it returns the wire-stripped
  items list — preserves the data-in / data-out contract while routing
  around the constructor's homogeneity check.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data.llm_message_utils import decompose_message_log
from nemo_rl.data_plane.codec import (
    materialize,
    to_nested_by_length,
    unwrap_wire_stripped_payload,
)

from ._rollout_shapes import make_multi_turn_message_log

# ── unwrap_wire_stripped_payload — direct per-item coverage ───────────


def test_unwrap_wire_stripped_payload_empty_td_to_none() -> None:
    """An empty ``TensorDict`` (batch_dims=0, no keys) → ``None``."""
    assert unwrap_wire_stripped_payload(TensorDict({}, batch_size=[])) is None


def test_unwrap_wire_stripped_payload_real_nontensor_data_passes_through() -> None:
    """A live ``NonTensorData`` payload survives unwrap."""
    assert unwrap_wire_stripped_payload(NonTensorData(data="hello")) == "hello"


# ── materialize — end-to-end with the wire-stripped tolist shape ──────


def _valid_stack(n: int) -> NonTensorStack:
    """A real ``NonTensorStack`` we can patch ``.tolist()`` on.

    Contents are irrelevant — ``materialize`` only iterates the items
    returned by ``tolist()``, which we override below.
    """
    return NonTensorStack(*(NonTensorData(data=None) for _ in range(n)))


def test_materialize_handles_wire_stripped_nontensor_stack() -> None:
    """A stack of empty TDs materializes to an object array of ``None``."""
    items = [TensorDict({}, batch_size=[]) for _ in range(4)]
    stack = _valid_stack(4)
    with patch.object(stack, "tolist", return_value=items):
        td = TensorDict({"content": stack}, batch_size=[4])
        bdd = materialize(td, layout="padded")

    arr = bdd["content"]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.shape == (4,)
    assert list(arr) == [None, None, None, None]


def test_materialize_preserves_real_nontensor_data() -> None:
    """Real ``NonTensorStack`` of strings materializes to the raw strings.

    Guards against the wire-stripped fix accidentally substituting
    ``None`` for legitimate string content (the happy path that
    Mooncake's pickle wire and the patched simple-backend wire produce).
    """
    real = NonTensorStack(
        NonTensorData(data="hello"),
        NonTensorData(data="world"),
        NonTensorData(data="!"),
    )
    td = TensorDict({"content": real}, batch_size=[3])

    bdd = materialize(td, layout="padded")

    arr = bdd["content"]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.shape == (3,)
    assert list(arr) == ["hello", "world", "!"]


def test_materialize_decodes_nontensor_stack_with_tensor_field() -> None:
    """Per-field decode: tensor fields stay padded while object fields ride.

    Guards the invariant that ``materialize``'s object-decode is
    per-field, not all-or-nothing — a TensorDict can mix jagged tensor
    leaves and ``NonTensorStack`` leaves in the same put.
    """
    ids_padded = torch.tensor(
        [[10, 20, 30, 0], [40, 50, 0, 0], [60, 70, 80, 90]], dtype=torch.long
    )
    lens = torch.tensor([3, 2, 4], dtype=torch.long)
    ids_nested = to_nested_by_length(ids_padded, lens)
    msg = NonTensorStack({"id": 0}, {"id": 1}, {"id": 2})

    td = TensorDict(
        {"input_ids": ids_nested, "message_log": msg},
        batch_size=[3],
    )

    bdd = materialize(
        td,
        layout="padded",
        pad_value_dict={"input_ids": 999},
    )

    # Tensor field padded with 999 as usual.
    assert bdd["input_ids"][1, 2].item() == 999
    # Object field comes back as np.ndarray(object).
    assert isinstance(bdd["message_log"], np.ndarray)
    assert bdd["message_log"].dtype == object
    assert [d["id"] for d in bdd["message_log"]] == [0, 1, 2]


# Real production end-to-end coverage of object columns (put → wire →
# get → decode) against both TQ backends lives in
# tests/data_plane/functional/test_tq_lifecycle.py::test_object_round_trip_backends
# and ::test_object_and_tensor_mixed_round_trip_backends. The unit
# tests above cover the decode path in isolation; the functional tests
# cover the full wire round-trip.


def test_materialize_realistic_message_log_object_field() -> None:
    """Realistic multi-turn message_log decomposes into ``turn_roles`` /
    ``turn_contents`` as ``np.ndarray(dtype=object)`` and materializes back."""

    n = 4
    ml_batch = make_multi_turn_message_log(n=n, turns_per_sample=[1, 2, 3, 4], seed=51)
    decomposed = decompose_message_log(ml_batch)

    # The wire-shape: turn_roles + turn_contents are per-sample lists.
    # Build a TD with a NonTensorStack of those lists.
    roles_stack = NonTensorStack(*[list(r) for r in decomposed["turn_roles"]])
    contents_stack = NonTensorStack(*[list(c) for c in decomposed["turn_contents"]])
    td = TensorDict(
        {
            "turn_lengths": decomposed["turn_lengths"],
            "turn_roles": roles_stack,
            "turn_contents": contents_stack,
        },
        batch_size=[n],
    )

    out = materialize(td, layout="padded")
    # Object fields come back as np.ndarray(dtype=object) — the codec's
    # canonical decode of NonTensorStack.
    assert isinstance(out["turn_roles"], np.ndarray)
    assert out["turn_roles"].dtype == object
    assert isinstance(out["turn_contents"], np.ndarray)
    # Per-sample identity survives the decode.
    for i in range(n):
        assert list(out["turn_roles"][i]) == list(decomposed["turn_roles"][i])
