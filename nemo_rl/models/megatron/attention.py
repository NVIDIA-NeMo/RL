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

"""Optional Transformer Engine packing and workspace integration."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

import torch
from transformer_engine.pytorch import attention

# Older TE releases do not expose these optional APIs.
G_REGISTER_CU_SEQLENS = getattr(attention, "register_cu_seqlens", None)
G_ATTENTION_WORKSPACE = getattr(attention, "attention_backend_workspace", None)


def register_cu_seqlens(tensor: torch.Tensor, offsets: Sequence[int]) -> None:
    """Register the host values used to construct a packed prefix tensor."""
    if G_REGISTER_CU_SEQLENS is not None:
        G_REGISTER_CU_SEQLENS(tensor, offsets)


def attention_backend_workspace() -> AbstractContextManager[None]:
    """Release optional attention scratch before policy refit or generation."""
    if G_ATTENTION_WORKSPACE is None:
        return nullcontext()
    return G_ATTENTION_WORKSPACE()
