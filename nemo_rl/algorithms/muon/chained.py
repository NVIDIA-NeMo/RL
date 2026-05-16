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
"""Compose two ``torch.optim.Optimizer`` instances behind a single ``step``.

NeMo-RL's policy worker treats ``self.optimizer`` as a single object: it
calls ``zero_grad``, ``step``, ``state_dict``, and reads ``param_groups``
for LR logging. Muon needs to drive linear weights via ``DTensorMuon`` and
everything else (embeddings, layer norms, biases) via ``torch.optim.AdamW``,
so this thin wrapper presents the two as one.
"""

from __future__ import annotations

from typing import Any

import torch


class ChainedTorchOptimizer(torch.optim.Optimizer):
    """Sequence of ``torch.optim.Optimizer`` instances stepped in order.

    The wrapper deliberately bypasses ``torch.optim.Optimizer.__init__``;
    each underlying optimizer already owns its parameters, state, and
    defaults, and we just expose flat views so callers that iterate
    ``param_groups`` or serialize ``state_dict`` keep working.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("ChainedTorchOptimizer requires at least one optimizer")
        self.optimizers: list[torch.optim.Optimizer] = list(optimizers)
        self.defaults: dict[str, Any] = dict(optimizers[0].defaults)

    @property  # type: ignore[override]
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, _value: Any) -> None:  # noqa: D401 - setter required by base
        # torch.optim.Optimizer.__init__ tries to assign param_groups; we never
        # call it, so this setter is here only to satisfy the descriptor contract.
        raise AttributeError("ChainedTorchOptimizer.param_groups is read-only")

    @property  # type: ignore[override]
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    @state.setter
    def state(self, _value: Any) -> None:  # noqa: D401 - setter required by base
        raise AttributeError("ChainedTorchOptimizer.state is read-only")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Adds to the first underlying optimizer; callers that need explicit
        # routing should mutate the underlying optimizer directly.
        self.optimizers[0].add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        for i, opt in enumerate(self.optimizers):
            key = f"optimizer_{i}"
            if key not in state_dict:
                raise KeyError(
                    f"ChainedTorchOptimizer.load_state_dict: missing '{key}'"
                )
            opt.load_state_dict(state_dict[key])
