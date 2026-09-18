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

import torch
from torch import nn

_POLICY_LR = 1.0e-6
_DRAFT_LR = 1.0e-4


def _make_two_group_optimizer():
    policy = nn.Linear(4, 4, bias=False)
    draft = nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(
        [
            {"name": "policy", "params": list(policy.parameters())},
            {"name": "draft", "params": list(draft.parameters()), "lr": _DRAFT_LR},
        ],
        lr=_POLICY_LR,
    )
    return optimizer


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
    repo = Path(__file__).resolve().parents[4]
    for rel in (
        "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py",
        "nemo_rl/models/value/workers/dtensor_value_worker_v2.py",
    ):
        source = (repo / rel).read_text()
        assert not re.search(r"\)\s*=\s*model_and_optimizer_state\b", source), (
            f"{rel} positionally unpacks ModelAndOptimizerState; use named "
            "field access instead."
        )
