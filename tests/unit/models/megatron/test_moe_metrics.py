# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict

import pytest
import torch


def _make_fake_tracker(values: Dict[str, torch.Tensor]) -> dict[str, Any]:
    tracker: dict[str, Any] = {}
    for name, tensor in values.items():
        tracker[name] = {"values": tensor}
    return tracker


@pytest.mark.mcore
def test_get_moe_metrics_empty_tracker(monkeypatch):
    """If no aux losses are tracked, get_moe_metrics should return an empty dict."""

    from nemo_rl.models import megatron as megatron_module
    from nemo_rl.models.megatron.common import get_moe_metrics

    # Patch the imported functions in nemo_rl.models.megatron.common
    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "reduce_aux_losses_tracker_across_ranks",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "get_moe_layer_wise_logging_tracker",
        lambda: {},
    )

    cleared = {"called": False}

    def _clear():
        cleared["called"] = True

    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "clear_aux_losses_tracker",
        _clear,
    )

    metrics = get_moe_metrics(loss_scale=1.0)
    assert metrics == {}
    assert cleared["called"], "clear_aux_losses_tracker should be called"


@pytest.mark.mcore
def test_get_moe_metrics_aggregation_and_per_layer_logging(monkeypatch):
    """Validate aggregation logic and optional per-layer logging."""

    from nemo_rl.models import megatron as megatron_module
    from nemo_rl.models.megatron.common import get_moe_metrics

    # Fake tracker contents: two aux losses, each with per-layer values.
    load_balancing = torch.tensor([1.0, 3.0])
    z_loss = torch.tensor([2.0, 4.0])

    tracker = _make_fake_tracker(
        {
            "load_balancing_loss": load_balancing,
            "z_loss": z_loss,
        }
    )

    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "reduce_aux_losses_tracker_across_ranks",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "get_moe_layer_wise_logging_tracker",
        lambda: tracker,
    )

    cleared = {"called": False}

    def _clear():
        cleared["called"] = True

    monkeypatch.setattr(
        megatron_module.common,  # type: ignore[attr-defined]
        "clear_aux_losses_tracker",
        _clear,
    )

    # loss_scale = 0.5 means each per-layer value is halved before aggregation.
    metrics = get_moe_metrics(loss_scale=0.5, per_layer_logging=True)

    # Aggregated values: mean over layers after scaling.
    # load_balancing: (1 + 3) / 2 * 0.5 = 1.0
    # z_loss: (2 + 4) / 2 * 0.5 = 1.5
    assert metrics["load_balancing_loss"] == pytest.approx(1.0)
    assert metrics["z_loss"] == pytest.approx(1.5)

    # Per-layer logging should be present.
    assert metrics["moe/load_balancing_loss_layer_0"] == pytest.approx(0.5)
    assert metrics["moe/load_balancing_loss_layer_1"] == pytest.approx(1.5)
    assert metrics["moe/z_loss_layer_0"] == pytest.approx(1.0)
    assert metrics["moe/z_loss_layer_1"] == pytest.approx(2.0)

    assert cleared["called"], "clear_aux_losses_tracker should be called"
