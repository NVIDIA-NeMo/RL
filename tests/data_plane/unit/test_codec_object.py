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
"""Unit tests for non-tensor passthrough on the wire.

Object fields ride the wire as ``NonTensorStack`` leaves (TQ-native);
``materialize`` decodes them back to ``np.ndarray(dtype=object)`` for
the trainer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from nemo_rl.data_plane.codec import (
    materialize,
    pack_object_array,
    to_nested_by_length,
    unpack_object_array,
)


def test_pack_unpack_roundtrip_strings() -> None:
    arr = np.array(["alpha", "beta", "gamma"], dtype=object)
    packed = pack_object_array(arr)
    assert packed.is_nested and packed.dtype == torch.uint8
    out = unpack_object_array(packed)
    assert isinstance(out, np.ndarray) and out.dtype == object
    assert list(out) == ["alpha", "beta", "gamma"]


def test_pack_unpack_roundtrip_message_log_shape() -> None:
    """The actual message_log shape: list[list[dict[str, str|Tensor]]]."""
    sample_a = [
        {"role": "user", "content": "hi", "token_ids": torch.tensor([1, 2, 3])},
        {"role": "assistant", "content": "hello", "token_ids": torch.tensor([4, 5])},
    ]
    sample_b = [
        {"role": "user", "content": "what's up?", "token_ids": torch.tensor([6])},
    ]
    arr = np.array([sample_a, sample_b], dtype=object)
    packed = pack_object_array(arr)
    out = unpack_object_array(packed)
    assert len(out) == 2
    assert out[0][0]["role"] == "user"
    assert out[0][1]["content"] == "hello"
    assert torch.equal(out[1][0]["token_ids"], torch.tensor([6]))


def test_pack_accepts_python_list() -> None:
    """list passes through the same path as np.ndarray(object)."""
    packed = pack_object_array([{"a": 1}, {"a": 2}, {"a": 3}])
    out = unpack_object_array(packed)
    assert [d["a"] for d in out] == [1, 2, 3]


def test_pack_rejects_non_object_ndarray() -> None:
    with pytest.raises(TypeError, match=r"dtype=object"):
        pack_object_array(np.array([1, 2, 3], dtype=np.int64))


def test_unpack_rejects_rectangular_tensor() -> None:
    with pytest.raises(ValueError, match=r"nested"):
        unpack_object_array(torch.zeros(3, dtype=torch.uint8))


def test_materialize_decodes_nontensor_stack() -> None:
    """``NonTensorStack`` leaves are decoded back to ``np.ndarray(object)``.

    Tensor fields in the same TensorDict are still padded as before —
    object support is per-field, not all-or-nothing.
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
