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

from nemo_rl.algorithms.loss.loss_functions import (
    CategoricalValueSupport,
    ClassificationValueLossFn,
    MseValueLossConfig,
    MseValueLossFn,
    make_value_loss_fn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

HEAD = {"type": "categorical", "num_bins": 61, "v_min": -0.1, "v_max": 1.1}


def _data(returns: torch.Tensor) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "returns": returns,
            "token_mask": torch.ones_like(returns),
            "sample_mask": torch.ones(returns.shape[0]),
        }
    )


def test_support_geometry():
    support = CategoricalValueSupport.from_value_head_cfg(HEAD)
    centers = support.centers(torch.device("cpu"))
    assert centers.shape == (61,)
    assert support.bin_width == pytest.approx(0.02)
    assert centers[0].item() == pytest.approx(-0.09)
    assert centers[-1].item() == pytest.approx(1.09)


@pytest.mark.parametrize("target_type", ["hl_gauss", "two_hot"])
def test_targets_are_distributions_centred_on_return(target_type):
    support = CategoricalValueSupport.from_value_head_cfg(HEAD)
    returns = torch.tensor([[0.0, 0.37, 1.0]])
    if target_type == "hl_gauss":
        target = support.hl_gauss_target(returns, sigma=0.75 * support.bin_width)
    else:
        target = support.two_hot_target(returns)
    assert target.shape == (1, 3, 61)
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 3))
    assert (target >= 0).all()
    mean = target @ support.centers(torch.device("cpu"))
    torch.testing.assert_close(mean, returns, atol=1e-4, rtol=0)


def test_two_hot_clamps_to_edge_centers():
    support = CategoricalValueSupport.from_value_head_cfg(HEAD)
    target = support.two_hot_target(torch.tensor([-5.0, 5.0]))
    assert target[0, 0].item() == pytest.approx(1.0)
    assert target[1, -1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("target_type", ["hl_gauss", "two_hot"])
def test_loss_optimum_recovers_returns(target_type):
    cfg = MseValueLossConfig(type=target_type)
    loss_fn = make_value_loss_fn(cfg, HEAD)
    assert isinstance(loss_fn, ClassificationValueLossFn)
    returns = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    data = _data(returns)
    logits = torch.zeros(2, 3, 61, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = loss_fn(logits, data, torch.tensor(2), torch.tensor(6))
        loss.backward()
        opt.step()
    _, metrics = loss_fn(logits.detach(), data, torch.tensor(2), torch.tensor(6))
    assert metrics["residual_sq_mean"] < 1e-3
    assert metrics["returns_mean"] == pytest.approx(0.5)
    assert metrics["values_min"] < 0.05 and metrics["values_max"] > 0.95


def test_loss_respects_mask_and_global_normalization():
    loss_fn = make_value_loss_fn(MseValueLossConfig(type="hl_gauss"), HEAD)
    returns = torch.tensor([[1.0, 0.0]])
    data = _data(returns)
    data["token_mask"] = torch.tensor([[1.0, 0.0]])
    logits = torch.randn(1, 2, 61)
    loss_one, _ = loss_fn(logits, data, torch.tensor(1), torch.tensor(1))
    loss_two, _ = loss_fn(logits, data, torch.tensor(1), torch.tensor(2))
    torch.testing.assert_close(loss_two, loss_one / 2)
    logits_changed = logits.clone()
    logits_changed[0, 1] += 100.0
    loss_masked, _ = loss_fn(logits_changed, data, torch.tensor(1), torch.tensor(1))
    torch.testing.assert_close(loss_masked, loss_one)


def test_factory_validation():
    assert isinstance(make_value_loss_fn(MseValueLossConfig(), None), MseValueLossFn)
    assert isinstance(
        make_value_loss_fn(MseValueLossConfig(), {"type": "scalar"}), MseValueLossFn
    )
    with pytest.raises(ValueError, match="categorical"):
        make_value_loss_fn(MseValueLossConfig(type="hl_gauss"), None)
    with pytest.raises(ValueError, match="hl_gauss"):
        make_value_loss_fn(MseValueLossConfig(), HEAD)
    with pytest.raises(ValueError, match="cliprange"):
        MseValueLossConfig(type="hl_gauss", cliprange=0.5)
    with pytest.raises(ValueError, match="num_bins"):
        CategoricalValueSupport(1, 0.0, 1.0)


def test_expected_value_matches_softmax_mean():
    support = CategoricalValueSupport.from_value_head_cfg(HEAD)
    logits = torch.randn(2, 5, 61)
    expected = support.expected_value(logits)
    manual = (logits.softmax(-1) * support.centers(torch.device("cpu"))).sum(-1)
    torch.testing.assert_close(expected, manual)
    assert expected.shape == (2, 5)


def test_label_smoothing_bounds_optimum():
    cfg = MseValueLossConfig(type="hl_gauss", hl_gauss_label_smoothing=0.1)
    loss_fn = make_value_loss_fn(cfg, HEAD)
    returns = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    data = _data(returns)
    logits = torch.zeros(2, 2, 61, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.5)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = loss_fn(logits, data, torch.tensor(2), torch.tensor(4))
        loss.backward()
        opt.step()
    values = CategoricalValueSupport.from_value_head_cfg(HEAD).expected_value(
        logits.detach()
    )
    torch.testing.assert_close(values[0], torch.full((2,), 0.9), atol=0.01, rtol=0)
    torch.testing.assert_close(values[1], torch.full((2,), 0.1), atol=0.01, rtol=0)


def test_label_smoothing_validation():
    with pytest.raises(ValueError, match="label_smoothing"):
        MseValueLossConfig(type="hl_gauss", hl_gauss_label_smoothing=0.6)
    with pytest.raises(ValueError, match="categorical"):
        MseValueLossConfig(hl_gauss_label_smoothing=0.05)
