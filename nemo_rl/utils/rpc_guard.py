# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Keep cohort-sized payloads off Ray RPC boundaries.

Actors that exist to take work off the controller only help if the work's
inputs and outputs stay small. Both the token-capture finalizer and the
advantage stage move tensors through DataPlane and return metadata, so this
guard is what makes "metadata-only" checkable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch

# Field names whose values are per-token and therefore large, but whose Python
# type is indistinguishable from metadata -- a list[int] of token ids looks just
# like a short list of ids. assert_metadata_only() below already rejects tensors
# and unrecognised types; this list is only for heavy values that would otherwise
# pass it. Add a name here whenever a new per-token field could reach an RPC
# boundary, and update the dataclass inventory test that guards this file.
FORBIDDEN_RPC_KEYS = frozenset(
    {
        "input_ids",
        "token_ids",
        "token_ids_delta",
        "token_mask",
        "token_mask_delta",
        "generation_logprobs",
        "generation_logprobs_delta",
        "generation_log_probs_delta",
        "logprobs",
        "logprobs_delta",
        "routed_experts",
    }
)


def assert_metadata_only(value: Any, *, path: str = "rpc") -> None:
    """Reject tensors and known heavy row fields reachable from an RPC graph."""
    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"{path} contains a torch.Tensor with shape {tuple(value.shape)}"
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            assert_metadata_only(
                getattr(value, field_info.name),
                path=f"{path}.{field_info.name}",
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_RPC_KEYS:
                raise TypeError(f"{path} contains forbidden heavy field {key!r}")
            assert_metadata_only(key, path=f"{path}.key")
            assert_metadata_only(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_metadata_only(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported RPC type {type(value).__name__}")
