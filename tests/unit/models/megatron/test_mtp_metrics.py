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
from types import SimpleNamespace

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


@pytest.mark.mcore
def test_get_mtp_metrics_applies_loss_scale(monkeypatch):
    """loss_scale multiplies each layer's loss; acceptance (a count ratio) is unscaled.

    The MTPLossLoggingHelper accumulates the per-microbatch loss across microbatches
    without dividing, so the worker passes loss_scale=1/num_microbatches to recover the
    mean (mirroring get_moe_metrics). Acceptance rate must stay independent of loss_scale.
    """
    from nemo_rl.models.megatron.common import get_mtp_metrics

    tracker = {
        "loss_values": torch.tensor([2.0, 4.0]),
        "correct_values": torch.tensor([1.0, 3.0]),
        "total_values": torch.tensor([2.0, 6.0]),
    }
    _seed_tracker(monkeypatch, tracker)

    # e.g. 4 microbatches accumulated -> loss_scale = 1/4.
    metrics = get_mtp_metrics(loss_scale=0.25)

    assert metrics["mtp_1_loss"] == pytest.approx(0.5)  # 2.0 * 0.25
    assert metrics["mtp_2_loss"] == pytest.approx(1.0)  # 4.0 * 0.25
    assert metrics["mtp_1_acceptance_rate"] == pytest.approx(50.0)
    assert metrics["mtp_2_acceptance_rate"] == pytest.approx(50.0)


@pytest.mark.mcore
def test_get_mtp_metrics_loss_scale_recovers_mean_over_microbatches(monkeypatch):
    """loss_scale=1/num_microbatches turns the accumulated loss sum into the mean.

    loss_values here stands in for the sum of 4 per-microbatch losses of 2.0 each.
    """
    from nemo_rl.models.megatron.common import get_mtp_metrics

    num_microbatches = 4
    _seed_tracker(monkeypatch, {"loss_values": torch.tensor([8.0])})  # 4 * 2.0

    metrics = get_mtp_metrics(loss_scale=1.0 / num_microbatches)
    assert metrics["mtp_1_loss"] == pytest.approx(2.0)


@pytest.mark.mcore
def test_get_mtp_metrics_default_loss_scale_is_identity(monkeypatch):
    """Omitting loss_scale (default 1.0) leaves the loss unchanged (backward compatible)."""
    from nemo_rl.models.megatron.common import get_mtp_metrics

    _seed_tracker(monkeypatch, {"loss_values": torch.tensor([3.0])})
    assert get_mtp_metrics()["mtp_1_loss"] == pytest.approx(3.0)


def _fake_worker(mtp_num_layers):
    """A minimal stand-in for MegatronPolicyWorkerImpl for calling _collect_mtp_metrics."""
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(mtp_num_layers=mtp_num_layers))
    )


@pytest.mark.mcore
def test_collect_mtp_metrics_scales_loss_and_adds_grad_norm(monkeypatch):
    """_collect_mtp_metrics passes loss_scale=1/num_microbatches and surfaces the MTP grad norm.

    The grad norm is placed inside the mtp_metrics dict as "grad_norm" so grpo.py flattens it
    to mtp/grad_norm (logged as train/mtp/grad_norm).
    """
    from nemo_rl.models.policy.workers import megatron_policy_worker as mpw

    captured = {}

    def fake_get_mtp_metrics(loss_scale=1.0):
        captured["loss_scale"] = loss_scale
        return {"mtp_1_loss": 0.5, "mtp_1_acceptance_rate": 40.0}

    # get_mtp_metrics is imported lazily inside the method, so patch it on its source module.
    monkeypatch.setattr(
        "nemo_rl.models.megatron.common.get_mtp_metrics", fake_get_mtp_metrics
    )
    # Single-process test: the last-stage broadcast is an identity passthrough.
    monkeypatch.setattr(mpw, "broadcast_loss_metrics_from_last_stage", lambda d: d)

    metrics: dict = {}
    mpw.MegatronPolicyWorkerImpl._collect_mtp_metrics(
        _fake_worker(mtp_num_layers=1),
        metrics,
        total_num_microbatches=8,
        mtp_grad_norm=1.25,
    )

    assert captured["loss_scale"] == pytest.approx(1.0 / 8)
    assert metrics["mtp_metrics"]["mtp_1_loss"] == pytest.approx(0.5)
    assert metrics["mtp_metrics"]["grad_norm"] == pytest.approx(1.25)


@pytest.mark.mcore
def test_collect_mtp_metrics_omits_grad_norm_when_none(monkeypatch):
    """A None MTP grad norm (e.g. clip_grad==0 or mtp_detach_heads=False) is not logged."""
    from nemo_rl.models.policy.workers import megatron_policy_worker as mpw

    monkeypatch.setattr(
        "nemo_rl.models.megatron.common.get_mtp_metrics",
        lambda loss_scale=1.0: {"mtp_1_loss": 0.5},
    )
    monkeypatch.setattr(mpw, "broadcast_loss_metrics_from_last_stage", lambda d: d)

    metrics: dict = {}
    mpw.MegatronPolicyWorkerImpl._collect_mtp_metrics(
        _fake_worker(mtp_num_layers=1),
        metrics,
        total_num_microbatches=4,
        mtp_grad_norm=None,
    )
    assert "grad_norm" not in metrics["mtp_metrics"]


@pytest.mark.mcore
def test_collect_mtp_metrics_noop_when_mtp_disabled(monkeypatch):
    """MTP disabled (mtp_num_layers=0) -> nothing added and get_mtp_metrics is not called."""
    from nemo_rl.models.policy.workers import megatron_policy_worker as mpw

    called = {"get_mtp_metrics": False}

    def fake_get_mtp_metrics(loss_scale=1.0):
        called["get_mtp_metrics"] = True
        return {}

    monkeypatch.setattr(
        "nemo_rl.models.megatron.common.get_mtp_metrics", fake_get_mtp_metrics
    )

    metrics: dict = {}
    mpw.MegatronPolicyWorkerImpl._collect_mtp_metrics(
        _fake_worker(mtp_num_layers=0),
        metrics,
        total_num_microbatches=4,
        mtp_grad_norm=1.0,
    )
    assert "mtp_metrics" not in metrics
    assert not called["get_mtp_metrics"]
