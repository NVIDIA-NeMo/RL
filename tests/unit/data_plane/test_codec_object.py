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
import torch
from tensordict import NonTensorStack, TensorDict

from nemo_rl.data_plane.codec import materialize, to_nested_by_length


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
