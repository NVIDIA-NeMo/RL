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


@pytest.mark.mcore
@pytest.mark.parametrize(
    "routing_type,aux_loss_coeff,z_loss_coeff,expected",
    [
        # Load balancing disabled: nothing to pre-initialize, so no permanently-zero
        # metric is reported for the many MoE configs that use "none".
        ("none", 0.0, None, []),
        # Configured type but zero coefficient: the router returns before recording.
        ("aux_loss", 0.0, None, []),
        ("aux_loss", 1e-2, None, ["load_balancing_loss"]),
        # seq_aux_loss must not be reported as "load_balancing_loss": the router
        # records a distinct name, each driving its own all_reduce.
        ("seq_aux_loss", 1e-2, None, ["seq_load_balancing_loss"]),
        ("global_aux_loss", 1e-2, None, ["global_load_balancing_loss"]),
        # A list of balancing types pairs with a list of coefficients; only the
        # entries with a non-zero coefficient are live.
        (
            ["aux_loss", "seq_aux_loss"],
            [1e-2, 1e-2],
            None,
            ["load_balancing_loss", "seq_load_balancing_loss"],
        ),
        (["aux_loss", "seq_aux_loss"], [0.0, 1e-2], None, ["seq_load_balancing_loss"]),
        # z_loss is tracked independently of the balancing type.
        ("none", 0.0, 1e-3, ["z_loss"]),
        ("aux_loss", 1e-2, 1e-3, ["load_balancing_loss", "z_loss"]),
        # Unknown/unsupported balancing types record no aux loss.
        ("sinkhorn", 1e-2, None, []),
    ],
)
def test_get_aux_loss_track_names(routing_type, aux_loss_coeff, z_loss_coeff, expected):
    """track_names must mirror exactly the aux losses the router records."""
    from nemo_rl.models.megatron.common import get_aux_loss_track_names

    model_config = SimpleNamespace(
        moe_router_load_balancing_type=routing_type,
        moe_aux_loss_coeff=aux_loss_coeff,
        moe_z_loss_coeff=z_loss_coeff,
    )

    assert get_aux_loss_track_names(model_config) == expected


@pytest.mark.mcore
def test_get_aux_loss_track_names_tolerates_missing_attrs():
    """A config without MoE attributes must yield no names rather than raise."""
    from nemo_rl.models.megatron.common import get_aux_loss_track_names

    assert get_aux_loss_track_names(SimpleNamespace()) == []
    # z_loss is gated only on its own coefficient, independent of the balancing type.
    assert get_aux_loss_track_names(SimpleNamespace(moe_z_loss_coeff=1e-3)) == [
        "z_loss"
    ]


@pytest.mark.mcore
def test_get_aux_loss_track_names_ignores_non_numeric_coeffs():
    """Non-numeric placeholders must not enable a loss (and must not raise).

    Worker tests build ``model.config`` as a ``MagicMock``, whose attribute access
    returns a truthy mock rather than a number. Comparing that against 0 raises
    ``TypeError``, so the coefficient checks are type-guarded.
    """
    from unittest.mock import MagicMock

    from nemo_rl.models.megatron.common import get_aux_loss_track_names

    mock_config = MagicMock()
    mock_config.num_moe_experts = 4
    assert get_aux_loss_track_names(mock_config) == []

    # A real balancing type paired with a non-numeric coefficient is still "off".
    partial_config = MagicMock()
    partial_config.moe_router_load_balancing_type = "aux_loss"
    assert get_aux_loss_track_names(partial_config) == []


@pytest.mark.mcore
def test_preinit_registers_entry_on_live_tracker(monkeypatch):
    """A PP rank that recorded no aux loss must still enter the collective.

    This asserts against Megatron's real tracker rather than a stubbed dict:
    ``get_moe_layer_wise_logging_tracker()`` is a deprecated shim that returns a fresh
    dict copy, so writing the pre-initialized entry through it would be silently lost.
    """
    from megatron.core.transformer.moe.moe_logging import (
        destroy_moe_metrics_tracker,
        get_moe_metrics_tracker,
    )

    from nemo_rl.models import megatron as megatron_module
    from nemo_rl.models.megatron.common import get_moe_metrics

    destroy_moe_metrics_tracker()
    try:
        monkeypatch.setattr(
            megatron_module.common,  # type: ignore[attr-defined]
            "reduce_aux_losses_tracker_across_ranks",
            lambda *args, **kwargs: None,
        )

        get_moe_metrics(
            loss_scale=1.0,
            num_layers=4,
            mtp_num_layers=1,
            track_names=["load_balancing_loss"],
        )

        live = get_moe_metrics_tracker().metrics
        assert set(live) == {"load_balancing_loss"}
        # Size must be num_layers + mtp_num_layers to match what the router records.
        assert live["load_balancing_loss"].values.numel() == 5
    finally:
        destroy_moe_metrics_tracker()


@pytest.mark.mcore
@pytest.mark.parametrize(
    "kwargs",
    [
        # No pre-initialization requested at all (existing callers / dense models).
        {},
        # num_layers known but no aux loss is live (e.g. balancing type "none"):
        # pre-initializing would report a permanently-zero metric.
        {"num_layers": 4, "track_names": []},
        # track_names known but num_layers unavailable: cannot size the tensor.
        {"track_names": ["load_balancing_loss"]},
    ],
)
def test_no_preinit_leaves_tracker_empty(monkeypatch, kwargs):
    """Without both num_layers and track_names, no tracker entry may be created."""
    from megatron.core.transformer.moe.moe_logging import (
        destroy_moe_metrics_tracker,
        get_moe_metrics_tracker,
    )

    from nemo_rl.models import megatron as megatron_module
    from nemo_rl.models.megatron.common import get_moe_metrics

    destroy_moe_metrics_tracker()
    try:
        monkeypatch.setattr(
            megatron_module.common,  # type: ignore[attr-defined]
            "reduce_aux_losses_tracker_across_ranks",
            lambda *args, **kwargs: None,
        )

        metrics = get_moe_metrics(loss_scale=1.0, **kwargs)

        assert metrics == {}
        assert get_moe_metrics_tracker().metrics == {}
    finally:
        destroy_moe_metrics_tracker()
