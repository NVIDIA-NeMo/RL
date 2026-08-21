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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_DRAFT_LOSS_PATH = Path(__file__).parents[3] / "nemo_rl/algorithms/loss/draft.py"
_SPEC = importlib.util.spec_from_file_location(
    "draft_loss_zero_owner_under_test",
    _DRAFT_LOSS_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_zero_owner_emits_fixed_bins_with_autograd_connection() -> None:
    source = torch.nn.Parameter(torch.randn(1, 3, 4))
    draft_hidden = source[:0]
    output_weight = torch.randn(8, 4)

    stats = _MODULE.dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=draft_hidden,
        output_weight=output_weight,
        teacher_logits=torch.randn(1, 4, 8),
        sample_rows=torch.empty(0, dtype=torch.int64),
        label_positions=torch.empty(0, 3, dtype=torch.int64),
        loss_mask=torch.empty(0, 3, dtype=torch.bool),
        position_decay=0.5,
        token_chunk_size=4,
        tp_group=None,
    )

    assert torch.equal(stats.counts, torch.zeros(2))
    assert torch.equal(stats.weights, torch.tensor([1.0, 0.5]))
    assert stats.numerators.requires_grad
    assert torch.equal(stats.numerators, torch.zeros(2))

    stats.numerators.sum().backward()
    assert source.grad is not None
    assert torch.equal(source.grad, torch.zeros_like(source))
