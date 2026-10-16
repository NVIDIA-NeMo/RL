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
"""Named policy/draft optimizer groups: order, per-group LR, scheduler ratio."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# Skip entire module if nemo_automodel is not available
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.models.automodel.draft.checkpoint import optimizer_layout_record
from nemo_rl.models.automodel.draft.setup import build_policy_and_draft_optimizer

pytestmark = pytest.mark.automodel

_POLICY_LR = 1.0e-6
_DRAFT_LR = 1.0e-4


def _make_two_group_optimizer():
    return build_policy_and_draft_optimizer(
        torch.optim.AdamW,
        {"lr": _POLICY_LR},
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 4, bias=False),
        SimpleNamespace(learning_rate=_DRAFT_LR),
    )


def test_optimizer_layout_record_rejects_reordered_groups():
    optimizer = torch.optim.AdamW(
        [
            {"name": "draft", "params": [nn.Parameter(torch.zeros(1))]},
            {"name": "policy", "params": [nn.Parameter(torch.zeros(1))]},
        ]
    )
    with pytest.raises(ValueError, match="expects optimizer param groups named"):
        optimizer_layout_record(optimizer)


def test_named_groups_stable_order_and_per_group_lr():
    optimizer = _make_two_group_optimizer()
    assert [g["name"] for g in optimizer.param_groups] == ["policy", "draft"]
    assert optimizer.param_groups[0]["lr"] == _POLICY_LR
    assert optimizer.param_groups[1]["lr"] == _DRAFT_LR


def test_scheduler_scales_groups_multiplicatively():
    optimizer = _make_two_group_optimizer()
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=4
    )
    ratio0 = optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        ratio = optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
        # The warmup factor applies per group's own base LR, so the draft's
        # LR advantage over the policy stays constant through the schedule.
        assert abs(ratio - ratio0) / ratio0 < 1e-6
    assert abs(optimizer.param_groups[1]["lr"] - _DRAFT_LR) / _DRAFT_LR < 1e-6


def test_v2_workers_do_not_tuple_unpack_model_state():
    """Tripwire: ModelAndOptimizerState has optional appended fields, so the
    v2 workers must access it by field name; a positional unpack would break
    the next time a field is added."""
    repo = Path(__file__).resolve().parents[5]
    for rel in (
        "nemo_rl/models/policy/workers/automodel_policy_worker.py",
        "nemo_rl/models/value/workers/automodel_value_worker.py",
    ):
        source = (repo / rel).read_text()
        assert not re.search(r"\)\s*=\s*model_and_optimizer_state\b", source), (
            f"{rel} positionally unpacks ModelAndOptimizerState; use named "
            "field access instead."
        )
