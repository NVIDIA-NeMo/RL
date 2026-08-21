from __future__ import annotations

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
