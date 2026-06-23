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
import pytest
import torch


def _seed_tracker(monkeypatch, tracker):
    """Seed MTPLossLoggingHelper with a fixed tracker and disable cross-rank reduce.

    reduce_metrics_in_tracker would otherwise all-reduce over a process group;
    stubbing it lets get_mtp_metrics run single-process on CPU tensors.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    monkeypatch.setattr(MTPLossLoggingHelper, "reduce_metrics_in_tracker", lambda: None)
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", tracker)
    return MTPLossLoggingHelper


@pytest.mark.mcore
def test_get_mtp_metrics_empty_tracker(monkeypatch):
    """No tracked MTP losses -> empty dict, and clean is not required."""
    from nemo_rl.models.megatron.common import get_mtp_metrics

    _seed_tracker(monkeypatch, {})
    assert get_mtp_metrics() == {}


@pytest.mark.mcore
def test_get_mtp_metrics_per_layer_loss_and_acceptance(monkeypatch):
    """Per-layer loss/acceptance are 1-indexed and the tracker is cleaned."""
    from nemo_rl.models.megatron.common import get_mtp_metrics

    tracker = {
        "loss_values": torch.tensor([2.0, 4.0]),
        "correct_values": torch.tensor([1.0, 3.0]),
        "total_values": torch.tensor([2.0, 6.0]),
    }
    helper = _seed_tracker(monkeypatch, tracker)

    cleaned = {"called": False}
    monkeypatch.setattr(
        helper,
        "clean_metrics_in_tracker",
        lambda: cleaned.__setitem__("called", True),
    )

    metrics = get_mtp_metrics()

    # 1-indexed keys matching Megatron-LM; acceptance = correct / total * 100.
    assert metrics["mtp_1_loss"] == pytest.approx(2.0)
    assert metrics["mtp_1_acceptance_rate"] == pytest.approx(50.0)
    assert metrics["mtp_2_loss"] == pytest.approx(4.0)
    assert metrics["mtp_2_acceptance_rate"] == pytest.approx(50.0)
    assert cleaned["called"], "clean_metrics_in_tracker should be called"


@pytest.mark.mcore
def test_get_mtp_metrics_defaults_when_only_loss_tracked(monkeypatch):
    """correct/total absent -> acceptance defaults (correct=0, total=1) => 0%."""
    from nemo_rl.models.megatron.common import get_mtp_metrics

    _seed_tracker(monkeypatch, {"loss_values": torch.tensor([1.5])})

    metrics = get_mtp_metrics()
    assert metrics["mtp_1_loss"] == pytest.approx(1.5)
    assert metrics["mtp_1_acceptance_rate"] == pytest.approx(0.0)
