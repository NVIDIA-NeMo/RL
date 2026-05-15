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
"""Single-GPU bit-exact tests for the DTensor Muon adapter.

DTensorMuon delegates the orthogonalize math to the same primitives that
``emerging_optimizers.Muon`` uses, so these tests assert that the per-step
update is bit-identical (rtol=0, atol=0) for any combination of scale mode,
Newton-Schulz coefficient set, momentum scheme, and weight-decay method.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

emerging = pytest.importorskip("emerging_optimizers.orthogonalized_optimizers")

from emerging_optimizers.orthogonalized_optimizers import Muon  # noqa: E402

from nemo_rl.algorithms.muon import (  # noqa: E402
    ChainedTorchOptimizer,
    DTensorMuon,
    build_dtensor_muon,
)


def _seed(seed: int = 42) -> None:
    torch.manual_seed(seed)


def _equal(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
@pytest.mark.parametrize(
    "coefficient_type", ["simple", "quintic", "polar_express"]
)
def test_dtensor_muon_matches_vanilla_muon_single_step(scale_mode, coefficient_type):
    _seed()
    w_ref = nn.Parameter(torch.randn(64, 32))
    w_dt = nn.Parameter(w_ref.detach().clone())
    grad = torch.randn(64, 32)
    w_ref.grad = grad.clone()
    w_dt.grad = grad.clone()

    common = dict(
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.01,
        nesterov=False,
        weight_decay_method="decoupled",
        fp32_matmul_prec="medium",
        coefficient_type=coefficient_type,
        num_ns_steps=5,
        scale_mode=scale_mode,
        extra_scale_factor=1.0,
    )
    Muon([w_ref], **common).step()
    DTensorMuon([w_dt], **common).step()
    _equal(w_dt.data, w_ref.data)


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "independent", "l2"])
def test_dtensor_muon_matches_vanilla_muon_multi_step(nesterov, weight_decay_method):
    _seed()
    w_ref = nn.Parameter(torch.randn(48, 24))
    w_dt = nn.Parameter(w_ref.detach().clone())
    common = dict(
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.05,
        nesterov=nesterov,
        weight_decay_method=weight_decay_method,
    )
    ref_opt = Muon([w_ref], **common)
    dt_opt = DTensorMuon([w_dt], **common)
    for _ in range(8):
        g = torch.randn_like(w_ref)
        w_ref.grad = g.clone()
        w_dt.grad = g.clone()
        ref_opt.step()
        dt_opt.step()
        _equal(w_dt.data, w_ref.data)


def test_dtensor_muon_extra_scale_factor_scales_update():
    """delta == -lr * orth * scale * extra_scale_factor, so doubling
    extra_scale_factor doubles the per-step delta when nothing else changes."""
    _seed()
    init = torch.randn(32, 16)
    grad = torch.randn(32, 16)
    base = dict(
        lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False,
    )

    def run(factor: float) -> torch.Tensor:
        w = nn.Parameter(init.detach().clone())
        w.grad = grad.clone()
        DTensorMuon([w], **base, extra_scale_factor=factor).step()
        return w.data - init

    delta_1x = run(1.0)
    delta_2x = run(2.0)
    # fp32 multiplication order ((orth * scale) * 2.0) vs (2 * (orth * scale))
    # leaves up to 1 ulp of slop, so allow a tiny absolute tolerance.
    torch.testing.assert_close(delta_2x, 2 * delta_1x, rtol=1e-6, atol=1e-6)


def test_dtensor_muon_state_serialization_roundtrip():
    """A continuous 2-step run must equal a 1-step + checkpoint resume + 1-step run."""
    _seed()
    init = torch.randn(16, 8)
    g1 = torch.randn(16, 8)
    g2 = torch.randn(16, 8)

    # Continuous reference.
    w_ref = nn.Parameter(init.clone())
    opt_ref = DTensorMuon([w_ref], lr=1e-3, momentum=0.95)
    w_ref.grad = g1.clone(); opt_ref.step()
    w_ref.grad = g2.clone(); opt_ref.step()

    # Step 1, then save weight + optimizer state.
    w_a = nn.Parameter(init.clone())
    opt_a = DTensorMuon([w_a], lr=1e-3, momentum=0.95)
    w_a.grad = g1.clone(); opt_a.step()
    saved_state = opt_a.state_dict()
    saved_w = w_a.detach().clone()

    # Resume into a fresh instance and take step 2.
    w_b = nn.Parameter(saved_w.clone())
    opt_b = DTensorMuon([w_b], lr=1e-3, momentum=0.95)
    opt_b.load_state_dict(saved_state)
    w_b.grad = g2.clone(); opt_b.step()

    _equal(w_b.data, w_ref.data)


def test_chained_optimizer_steps_all_underlying():
    _seed()
    w_muon = nn.Parameter(torch.randn(32, 16))
    w_adam = nn.Parameter(torch.randn(8))
    muon = DTensorMuon([w_muon], lr=1e-3)
    adam = torch.optim.AdamW([w_adam], lr=1e-3)
    chained = ChainedTorchOptimizer([muon, adam])

    w_muon.grad = torch.randn_like(w_muon)
    w_adam.grad = torch.randn_like(w_adam)
    before_muon = w_muon.detach().clone()
    before_adam = w_adam.detach().clone()
    chained.step()
    assert not torch.equal(w_muon, before_muon)
    assert not torch.equal(w_adam, before_adam)


def test_chained_optimizer_state_dict_roundtrip():
    w_muon = nn.Parameter(torch.randn(8, 4))
    w_adam = nn.Parameter(torch.randn(4))
    muon = DTensorMuon([w_muon], lr=1e-3)
    adam = torch.optim.AdamW([w_adam], lr=1e-3)
    chained = ChainedTorchOptimizer([muon, adam])
    w_muon.grad = torch.randn_like(w_muon)
    w_adam.grad = torch.randn_like(w_adam)
    chained.step()
    state = chained.state_dict()
    assert set(state.keys()) == {"optimizer_0", "optimizer_1"}

    # Build a clean instance and load.
    w_muon2 = nn.Parameter(torch.zeros_like(w_muon))
    w_adam2 = nn.Parameter(torch.zeros_like(w_adam))
    chained2 = ChainedTorchOptimizer(
        [DTensorMuon([w_muon2], lr=1e-3), torch.optim.AdamW([w_adam2], lr=1e-3)]
    )
    chained2.load_state_dict(state)


def test_build_dtensor_muon_splits_linear_vs_norm_embed_lmhead():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 8)
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(8, 16),
                        nn.LayerNorm(16),
                        nn.Linear(16, 8),
                    )
                ]
            )
            self.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 10, bias=False)

    model = Tiny()
    opt = build_dtensor_muon(model, lr=1e-3)
    assert isinstance(opt, ChainedTorchOptimizer)
    assert isinstance(opt.optimizers[0], DTensorMuon)
    assert isinstance(opt.optimizers[1], torch.optim.AdamW)

    muon_ids = {
        id(p) for g in opt.optimizers[0].param_groups for p in g["params"]
    }
    adam_ids = {
        id(p) for g in opt.optimizers[1].param_groups for p in g["params"]
    }
    assert id(model.layers[0][0].weight) in muon_ids
    assert id(model.layers[0][2].weight) in muon_ids
    assert id(model.layers[0][1].weight) in adam_ids  # LayerNorm.weight (1D)
    assert id(model.norm.weight) in adam_ids
    assert id(model.embed_tokens.weight) in adam_ids
    assert id(model.lm_head.weight) in adam_ids
    # Biases (1D) go through AdamW.
    assert id(model.layers[0][0].bias) in adam_ids
    assert id(model.layers[0][2].bias) in adam_ids


def test_build_dtensor_muon_param_groups_flat_view():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    opt = build_dtensor_muon(Tiny(), lr=1e-3)
    # Flat view should expose at least one group from each underlying optimizer.
    assert len(opt.param_groups) >= 2
    # LR logging path used by dtensor_policy_worker.py:888
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_dtensor_muon_raises_when_no_trainable_params():
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError):
        build_dtensor_muon(model, lr=1e-3)
