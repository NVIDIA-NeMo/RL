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

import pytest
import torch

from nemo_rl.models.megatron.draft.diagnostics import (
    finalize_draft_update_probe,
    format_draft_update_probe,
    require_draft_update,
    start_draft_update_probe,
)


def test_draft_update_probe_reports_gradient_and_parameter_change() -> None:
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.grad = torch.ones_like(module.weight)

    probe = start_draft_update_probe(module)
    with torch.no_grad():
        module.weight.add_(0.25)
    result = finalize_draft_update_probe(module, probe)

    require_draft_update(result)
    marker = format_draft_update_probe(result)
    assert "draft_update_probe=complete" in marker
    assert "checksum_sum_before=" in marker
    assert "checksum_l2_after=" in marker
    assert result.grad_l2 > 0
    assert result.checksum_delta > 0


def test_require_draft_update_tolerates_consistent_noop() -> None:
    """Zero eligible draft windows yields grad 0 + unchanged params: a no-op."""
    module = torch.nn.Linear(3, 2, bias=False)

    probe = start_draft_update_probe(module)
    result = finalize_draft_update_probe(module, probe)

    require_draft_update(result)  # must not raise
    assert result.grad_l2 == 0


def test_require_draft_update_rejects_inconsistent_evidence() -> None:
    module = torch.nn.Linear(3, 2, bias=False)

    # Parameter change without a gradient.
    probe = start_draft_update_probe(module)
    with torch.no_grad():
        module.weight.add_(0.25)
    result = finalize_draft_update_probe(module, probe)
    with pytest.raises(RuntimeError, match="without a gradient"):
        require_draft_update(result)

    # Gradient without a parameter change.
    module.weight.grad = torch.ones_like(module.weight)
    probe = start_draft_update_probe(module)
    result = finalize_draft_update_probe(module, probe)
    with pytest.raises(RuntimeError, match="requires a parameter change"):
        require_draft_update(result)
