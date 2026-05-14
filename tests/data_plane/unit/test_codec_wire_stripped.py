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
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data_plane.codec import materialize, unwrap_wire_stripped_payload


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


# Real production end-to-end coverage of object columns (put → wire →
# get → decode) against both TQ backends lives in
# tests/data_plane/functional/test_tq_lifecycle.py::test_object_round_trip_backends
# and ::test_object_and_tensor_mixed_round_trip_backends. The unit
# tests above cover the decode path in isolation; the functional tests
# cover the full wire round-trip.
