# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Temporal alignment helpers for PPO value predictions.

DTensor value training shifts logits inside a loss wrapper, while value
inference shifts the returned tensor directly. Keeping the shared helper here
lets both paths use the same alignment logic without importing the heavier
worker module in lightweight unit tests.
"""

import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction


def right_shift_values(values: torch.Tensor) -> torch.Tensor:
    """Shift values right by 1 along the sequence dim (V(s_{t+1}) -> V(s_t)).

    Aligns value predictions with the Megatron value worker convention so GAE
    (rewards, returns), value targets, and value clipping all see the same
    V(s_t) semantics across backends. Preserves the input tensor shape: the
    first column becomes zeros and column t (t>=1) takes the value from
    column t-1.
    """
    return torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)


class RightShiftLossWrapper:
    """Wrap a LossFunction so value logits are right-shifted before loss."""

    def __init__(self, inner: LossFunction):
        self._inner = inner

    def __call__(self, *args, logits=None, **kwargs):
        if logits is not None:
            logits = right_shift_values(logits)
            return self._inner(*args, logits=logits, **kwargs)
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        # __getattr__ is only called when normal attribute lookup fails, so
        # _inner itself is still found through __dict__. Everything else
        # (input_type, aggregation_type, etc.) delegates to the wrapped
        # loss function.
        return getattr(self._inner, name)
