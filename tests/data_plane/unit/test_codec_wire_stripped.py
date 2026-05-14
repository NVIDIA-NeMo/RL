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
"""Regression test for the wire-stripped ``NonTensorStack`` case.

TQ's simple-backend ``MsgpackEncoder._encode_tensordict`` serializes any
``TensorDictBase`` via ``dict(obj.items())`` — which only iterates the
tensor backing dict. ``NonTensorData`` stores its payload in
``_non_tensordict["data"]`` (a separate dict), so a ``NonTensorData``
round-trips through ZMQ as an empty ``TensorDict({}, batch_size=[])`` —
the string payload is silently dropped. The simple-backend storage
manager's ``_pack_field_values`` then assembles those stripped TDs
into a ``NonTensorStack`` that ``materialize`` has to defend against.

The pre-fix path crashed with::

    RuntimeError: generator raised StopIteration

…because ``np.asarray(val.tolist(), dtype=object)`` iterates each item
to detect nested arrays; an empty TD's ``__iter__`` raises
``StopIteration`` (`tensordict.base:576`, ``batch_dims=0`` guard).

The fix uses ``np.empty + per-index assignment`` and substitutes
``None`` for any wire-stripped TD so downstream JSONL logging gets a
serializable leaf.
"""

from __future__ import annotations

import numpy as np
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data_plane.codec import materialize


def test_materialize_handles_wire_stripped_nontensor_stack() -> None:
    """A ``NonTensorStack`` of empty TDs materializes to an object array of None.

    Simulates TQ's simple-backend wire path where ``NonTensorData``
    payloads have been dropped on the get-response — each per-sample
    leaf is a ``TensorDict({}, batch_size=[])`` instead of a
    ``NonTensorData("…")``.
    """
    stripped = NonTensorStack(
        TensorDict({}, batch_size=[]),
        TensorDict({}, batch_size=[]),
        TensorDict({}, batch_size=[]),
        TensorDict({}, batch_size=[]),
    )
    td = TensorDict({"content": stripped}, batch_size=[4])

    bdd = materialize(td, layout="padded")

    arr = bdd["content"]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.shape == (4,)
    assert list(arr) == [None, None, None, None]


def test_materialize_preserves_real_nontensor_data() -> None:
    """A normal ``NonTensorStack`` of strings materializes to the raw strings.

    Guards against the wire-stripped fix accidentally substituting
    ``None`` for legitimate string content (the happy path that
    Mooncake's pickle wire and the patched simple-backend wire
    produce).
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


def test_materialize_mixed_wire_stripped_and_real() -> None:
    """A mixed stack — some payloads survived, some were stripped.

    Survivors keep their data; stripped TDs become ``None``.
    """
    mixed = NonTensorStack(
        NonTensorData(data="kept"),
        TensorDict({}, batch_size=[]),
        NonTensorData(data="also_kept"),
        TensorDict({}, batch_size=[]),
    )
    td = TensorDict({"content": mixed}, batch_size=[4])

    bdd = materialize(td, layout="padded")

    arr = bdd["content"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)
    assert list(arr) == ["kept", None, "also_kept", None]
