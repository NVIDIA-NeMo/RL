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

"""Unified model interface for single nn.Module and AutoPipeline (PP).

Normalizes the API so callers never need ``if pp_enabled`` checks for basic
model operations (state_dict, eval, train, parameters, buffers, config, etc.).
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn


class ModelHandle:
    """Thin wrapper that normalizes nn.Module vs AutoPipeline interface.

    With pipeline parallelism, ``from_pretrained`` returns an ``AutoPipeline``
    object instead of an ``nn.Module``.  AutoPipeline lacks standard nn.Module
    methods (``state_dict``, ``eval``, ``train``, ``parameters``, etc.).  This
    wrapper provides a unified API so that downstream code can treat both cases
    identically.

    Usage::

        handle = ModelHandle(model)  # model is nn.Module or AutoPipeline
        handle.eval()                # works for both
        for k, v in handle.state_dict_items():
            ...
        handle.config.pad_token_id   # works for both
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._pp_enabled = hasattr(model, "parts") and hasattr(model, "info")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Any:
        """Access the underlying model/AutoPipeline directly.

        Use for PP-specific APIs like ``raw.info.schedule``,
        ``raw.update_seq_len()``, ``raw.pp_batch_size``, etc.
        """
        return self._model

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def parts(self) -> list[nn.Module]:
        """List of nn.Module model parts (always a list, even for non-PP)."""
        if self._pp_enabled:
            return list(self._model.parts)
        return [self._model]

    @property
    def config(self) -> Any:
        """Model config (HF AutoConfig or similar)."""
        if self._pp_enabled:
            return self._model.parts[0].config
        return self._model.config

    @property
    def has_first_stage(self) -> bool:
        if self._pp_enabled:
            return self._model.info.has_first_stage
        return True

    @property
    def has_last_stage(self) -> bool:
        if self._pp_enabled:
            return self._model.info.has_last_stage
        return True

    # ------------------------------------------------------------------
    # nn.Module-like API
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Merged state dict across all parts."""
        if self._pp_enabled:
            state: dict[str, Any] = {}
            for part in self._model.parts:
                state.update(part.state_dict())
            return state
        return self._model.state_dict()

    def state_dict_items(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate ``(key, tensor)`` pairs of state dict (memory-efficient)."""
        for part in self.parts:
            yield from part.state_dict().items()

    def state_dict_keys(self) -> list[str]:
        """All state dict keys across all parts."""
        keys: list[str] = []
        for part in self.parts:
            keys.extend(part.state_dict().keys())
        return keys

    def eval(self) -> "ModelHandle":
        for part in self.parts:
            part.eval()
        return self

    def train(self, mode: bool = True) -> "ModelHandle":
        for part in self.parts:
            part.train(mode)
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        for part in self.parts:
            yield from part.parameters()

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for part in self.parts:
            yield from part.named_parameters()

    def buffers(self) -> Iterator[torch.Tensor]:
        for part in self.parts:
            yield from part.buffers()

    def to(self, device: str | torch.device) -> "ModelHandle":
        for part in self.parts:
            part.to(device)
        return self

    def move_buffers_to(self, device: str | torch.device) -> None:
        """Move buffers to device (FSDP modules don't move buffers automatically)."""
        for part in self.parts:
            for v in part.buffers():
                torch.utils.swap_tensors(v, v.to(device))
