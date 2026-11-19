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
"""The critic's output head (nemo_rl/algorithms/value_head.py) and HLGaussValueLossFn."""

import math

import pytest
import torch
import yaml
from pydantic import ValidationError

from nemo_rl.algorithms.loss.loss_functions import (
    HLGaussValueLossFn,
    MseValueLossConfig,
    MseValueLossFn,
    build_value_loss_fn,
)
from nemo_rl.algorithms.value_head import (
    ValueHeadConfig,
    bin_centers,
    check_warm_start_value_head,
    decode_values,
    hl_gauss_targets,
    resolve_value_head,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

HL_GAUSS = ValueHeadConfig(type="hl_gauss")


def _reference_targets(y: float, head: ValueHeadConfig) -> list[float]:
    """The paper's formula, written out with math.erf."""
    y = min(max(y, head.v_min), head.v_max)

    def phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    edges = [head.v_min + i * head.bin_width for i in range(head.num_bins + 1)]
    mass = [
        phi((edges[i + 1] - y) / head.smoothing) - phi((edges[i] - y) / head.smoothing)
        for i in range(head.num_bins)
    ]
    total = sum(mass)
    return [m / total for m in mass]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("value_config", [None, {}, {"head": None}])
def test_absent_head_is_scalar(value_config):
    head = resolve_value_head(value_config)
    assert head.type == "scalar" and not head.categorical
    assert head.output_size == 1


def test_hl_gauss_defaults_follow_the_paper():
    head = resolve_value_head({"head": {"type": "hl_gauss"}})
    assert head.output_size == 101
    assert head.bin_width == pytest.approx(1.2 / 101)
    # 0.75 bin widths: the paper's sigma = 0.009 on [-0.1, 1.1].
    assert head.smoothing == pytest.approx(0.75 * 1.2 / 101)
    assert head.smoothing == pytest.approx(0.009, abs=2e-4)
    centers = bin_centers(head)
    assert centers.dtype == torch.float32 and centers.shape == (101,)
    assert centers[0].item() == pytest.approx(-0.1 + head.bin_width / 2)
    assert centers[-1].item() == pytest.approx(1.1 - head.bin_width / 2)


@pytest.mark.parametrize(
    "block",
    [
        {"type": "hl_gauss", "v_min": 1.0, "v_max": 1.0},
        {"type": "hl_gauss", "num_bins": 1},
        {"type": "hl_gauss", "sigma": 0.0},
        {"type": "categorical"},
        {"type": "hl_gauss", "bins": 51},
    ],
)
def test_bad_head_blocks_are_rejected(block):
    with pytest.raises(ValidationError):
        resolve_value_head({"head": block})


# ----------------------------------------------------------------------------
# Targets and decoding
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("y", [-0.5, -0.1, 0.0, 0.37, 0.5, 1.0, 1.1, 2.0])
def test_targets_match_the_paper_formula(y):
    head = ValueHeadConfig(type="hl_gauss", num_bins=21, v_min=-0.1, v_max=1.1)
    got = hl_gauss_targets(torch.tensor([y]), head)[0]
    torch.testing.assert_close(
        got, torch.tensor(_reference_targets(y, head)), atol=1e-6, rtol=1e-5
    )


def test_targets_are_distributions_centered_on_the_return():
    returns = torch.tensor([[0.0, 0.25, 0.5], [0.8, 1.0, -3.0]], dtype=torch.bfloat16)
    targets = hl_gauss_targets(returns, HL_GAUSS)
    assert targets.shape == (2, 3, 101) and targets.dtype == torch.float32
    assert (targets >= 0).all()
    torch.testing.assert_close(targets.sum(-1), torch.ones(2, 3))
    means = (targets * bin_centers(HL_GAUSS)).sum(-1)
    # Interior returns keep their mean; -3 is clamped to the support's floor.
    expected = returns.float().clamp(-0.1, 1.1)
    interior = (expected > -0.05) & (expected < 1.05)
    torch.testing.assert_close(means[interior], expected[interior], atol=1e-3, rtol=0)
    # -3 sits on the floor: half a Gaussian, mean just inside v_min.
    assert -0.1 < means[1, 2].item() < -0.08


def test_decode_scalar_head_squeezes():
    out = torch.randn(2, 3, 1)
    torch.testing.assert_close(decode_values(out, ValueHeadConfig()), out.squeeze(-1))


def test_decode_hl_gauss_is_the_expectation_over_bin_centers():
    # Zero logits: uniform over the bins, the support's midpoint.
    zeros = torch.zeros(2, 4, 101)
    torch.testing.assert_close(
        decode_values(zeros, HL_GAUSS), torch.full((2, 4), 0.5), atol=1e-6, rtol=0
    )
    # A dominant bin decodes to its center.
    peaked = torch.full((1, 1, 101), -50.0)
    peaked[0, 0, 30] = 50.0
    assert decode_values(peaked, HL_GAUSS).item() == pytest.approx(
        bin_centers(HL_GAUSS)[30].item(), abs=1e-5
    )
    # Logits equal to the log-targets decode back to the return; output is fp32.
    returns = torch.tensor([0.1, 0.42, 0.9])
    logits = hl_gauss_targets(returns, HL_GAUSS).clamp_min(1e-30).log().bfloat16()
    decoded = decode_values(logits, HL_GAUSS)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, returns, atol=5e-3, rtol=0)


# ----------------------------------------------------------------------------
# Warm-start compatibility
# ----------------------------------------------------------------------------


def _seed(tmp_path, head_block):
    step = tmp_path / "step_20"
    step.mkdir()
    value = {"model_name": "m"} if head_block is None else {"head": head_block}
    (step / "config.yaml").write_text(yaml.safe_dump({"value": value}))
    return step


def test_warm_start_without_config_is_not_checked(tmp_path):
    check_warm_start_value_head(tmp_path, {"head": {"type": "hl_gauss"}})


@pytest.mark.parametrize(
    ("seed_head", "ours"),
    [
        (None, None),
        ({"type": "hl_gauss"}, {"type": "hl_gauss"}),
        # sigma only shapes the targets, not what the weights mean.
        ({"type": "hl_gauss", "sigma": 0.02}, {"type": "hl_gauss", "sigma": 0.009}),
    ],
)
def test_warm_start_with_the_same_head_passes(tmp_path, seed_head, ours):
    check_warm_start_value_head(_seed(tmp_path, seed_head), {"head": ours})


@pytest.mark.parametrize(
    ("seed_head", "ours"),
    [
        (None, {"type": "hl_gauss"}),
        ({"type": "hl_gauss"}, None),
        ({"type": "hl_gauss"}, {"type": "hl_gauss", "num_bins": 51}),
        ({"type": "hl_gauss"}, {"type": "hl_gauss", "v_max": 2.0}),
    ],
)
def test_warm_start_with_another_head_is_refused(tmp_path, seed_head, ours):
    with pytest.raises(ValueError, match="critic head"):
        check_warm_start_value_head(_seed(tmp_path, seed_head), {"head": ours})


# ----------------------------------------------------------------------------
# HLGaussValueLossFn
# ----------------------------------------------------------------------------


def _loss_inputs(batch: int = 2, seq: int = 5, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, 101, generator=gen, requires_grad=True)
    returns = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.3], [1.0, 0.4, 0.0, -0.2, 0.0]])
    token_mask = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.float32)
    sample_mask = torch.ones(batch)
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "returns": returns,
            "values": torch.zeros(batch, seq),
        }
    )
    mask = token_mask * sample_mask.unsqueeze(-1)
    return logits, data, sample_mask.sum(), mask.sum(), mask


def test_build_value_loss_fn_follows_the_head():
    cfg = MseValueLossConfig(cliprange=0.5)
    assert isinstance(build_value_loss_fn(cfg, None), MseValueLossFn)
    assert isinstance(build_value_loss_fn(cfg, {"model_name": "m"}), MseValueLossFn)
    hl = build_value_loss_fn(MseValueLossConfig(), {"head": {"type": "hl_gauss"}})
    assert isinstance(hl, HLGaussValueLossFn)
    assert hl.cliprange is None
    with pytest.raises(ValueError, match="cliprange"):
        build_value_loss_fn(cfg, {"head": {"type": "hl_gauss"}})


def test_hl_gauss_loss_is_the_masked_token_mean_cross_entropy():
    logits, data, valid_seqs, valid_toks, mask = _loss_inputs()
    loss_fn = HLGaussValueLossFn(MseValueLossConfig(scale=2.0), HL_GAUSS)
    loss, _ = loss_fn(logits, data, valid_seqs, valid_toks)

    targets = hl_gauss_targets(data["returns"], HL_GAUSS)
    token_ce = -(targets * torch.log_softmax(logits.float(), -1)).sum(-1)
    expected = 2.0 * (token_ce * mask).sum() / valid_toks
    torch.testing.assert_close(loss, expected)
    assert loss.requires_grad


def test_hl_gauss_loss_metrics_read_like_the_mse_loss():
    logits, data, valid_seqs, valid_toks, mask = _loss_inputs()
    loss_fn = HLGaussValueLossFn(MseValueLossConfig(), HL_GAUSS)
    _, metrics = loss_fn(logits, data, valid_seqs, valid_toks)
    _, mse_metrics = MseValueLossFn(MseValueLossConfig())(
        torch.zeros(2, 5), data, valid_seqs, valid_toks
    )
    assert set(mse_metrics) <= set(metrics)
    assert {"hl_gauss_entropy", "hl_gauss_clip_frac"} <= set(metrics)

    values = decode_values(logits.detach(), HL_GAUSS)
    returns = data["returns"]
    assert metrics["values_mean"] == pytest.approx(
        ((values * mask).sum() / valid_toks).item()
    )
    assert metrics["residual_sq_mean"] == pytest.approx(
        (((returns - values) ** 2 * mask).sum() / valid_toks).item()
    )
    assert metrics["values_min"] == pytest.approx(
        values[mask.bool()].min().item(), abs=1e-5
    )
    # 1.3 and -0.2 lie outside [-0.1, 1.1]: 2 of 6 valid tokens.
    assert metrics["hl_gauss_clip_frac"] == pytest.approx(2 / 6)
    assert metrics["vf_clipfrac"] == 0.0


def test_hl_gauss_loss_on_uniform_logits():
    _, data, valid_seqs, valid_toks, _ = _loss_inputs()
    logits = torch.zeros(2, 5, 101, requires_grad=True)
    loss, metrics = HLGaussValueLossFn(MseValueLossConfig(), HL_GAUSS)(
        logits, data, valid_seqs, valid_toks
    )
    # CE of any distribution against the uniform one is log(num_bins).
    assert loss.item() == pytest.approx(math.log(101), rel=1e-5)
    assert metrics["hl_gauss_entropy"] == pytest.approx(math.log(101), rel=1e-5)
    assert metrics["values_mean"] == pytest.approx(0.5, abs=1e-6)


def test_hl_gauss_loss_takes_packed_rows():
    logits = torch.randn(1, 7, 101, requires_grad=True)
    data = BatchedDataDict(
        {
            "token_mask": torch.ones(1, 7),
            "sample_mask": torch.ones(1),
            "returns": torch.ones(1, 7),
        }
    )
    loss, metrics = HLGaussValueLossFn(MseValueLossConfig(), HL_GAUSS)(
        logits, data, torch.tensor(1.0), torch.tensor(7.0)
    )
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert metrics["num_valid_samples"] == 1


def test_hl_gauss_loss_rejects_the_wrong_number_of_logits():
    _, data, valid_seqs, valid_toks, _ = _loss_inputs()
    with pytest.raises(ValueError, match="101 logits"):
        HLGaussValueLossFn(MseValueLossConfig(), HL_GAUSS)(
            torch.zeros(2, 5, 1), data, valid_seqs, valid_toks
        )


def test_hl_gauss_training_moves_the_value_to_the_return():
    """Gradient steps on the logits pull the decoded value onto the return."""
    data = BatchedDataDict(
        {
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
            "returns": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        }
    )
    logits = torch.zeros(1, 4, 101, requires_grad=True)
    loss_fn = HLGaussValueLossFn(MseValueLossConfig(), HL_GAUSS)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    losses = []
    for _ in range(300):
        optimizer.zero_grad()
        loss, _ = loss_fn(logits, data, torch.tensor(1.0), torch.tensor(4.0))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    values = decode_values(logits.detach(), HL_GAUSS)
    torch.testing.assert_close(
        values, torch.tensor([[1.0, 1.0, 0.0, 0.0]]), atol=1e-2, rtol=0
    )
    assert losses[-1] < losses[0]
