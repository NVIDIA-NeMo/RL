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
"""Wire <-> trainer codec.

Phase 1 ships a minimal materialize() that converts a TensorDict (the
wire format) back into a BatchedDataDict (what the existing trainer body
consumes). The wire format today is *padded* — the seed-put in
grpo_train writes already-padded tensors. So this is a thin translation,
not a real jagged → padded transform.

Stage 2 will land:
  * ``FIELD_SCHEMA`` table + per-field encoding.
  * ``to_csr`` / ``from_csr`` for variable-length list[list[primitive]].
  * ``StringEnum`` for fixed-vocab strings.
  * Real jagged ``materialize(layout='padded')`` that pads
    ``torch.nested.nested_tensor`` fields per ``pad_value_dict``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    # Type-only import. At runtime, BatchedDataDict is loaded lazily
    # inside materialize() — see comment there for rationale.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def materialize(
    td: TensorDict,
    layout: Literal["padded", "jagged"] = "padded",
    pad_value_dict: dict[str, int | float] | None = None,
) -> "BatchedDataDict[Any]":
    """Convert a wire TensorDict to a BatchedDataDict.

    Phase 1 contract: the wire is padded already, so this is a thin
    translation (no nested → padded transform). ``layout`` and
    ``pad_value_dict`` are accepted for forward compatibility with
    Stage 2's real jagged path; ``layout='jagged'`` is not yet supported.

    Note on import: ``BatchedDataDict`` lives in ``nemo_rl.distributed``
    which transitively pulls the multimodal stack (``decord``,
    ``torchvision``, ``transformers``) at module load. Lazy-importing
    here keeps ``import nemo_rl.data_plane`` cheap so unit tests that
    don't actually call this function can run in a slim env.
    """
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if layout != "padded":
        raise NotImplementedError(
            f"materialize(layout={layout!r}) is Stage 2 work. "
            "Phase 1 wire format is padded — use layout='padded'."
        )
    del pad_value_dict  # accepted for forward-compat; unused in Phase 1

    out: dict[str, torch.Tensor] = {}
    for key, val in td.items(include_nested=False):
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"materialize() received non-tensor leaf {key!r}: {type(val)}. "
                "Wire format must be tensor-only (P3)."
            )
        out[key] = val
    return BatchedDataDict(out)
