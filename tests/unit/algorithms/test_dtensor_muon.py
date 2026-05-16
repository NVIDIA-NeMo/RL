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


def test_optimizer_factory_dispatches_default_adamw_path():
    """The factory must keep the legacy `cls(model.parameters(), **kwargs)`
    semantics for vanilla optimizers so existing AdamW recipes are
    untouched."""
    from nemo_rl.utils.optimizer_factory import build_optimizer_from_cfg

    model = nn.Linear(4, 4)
    cfg = {
        "name": "torch.optim.AdamW",
        "kwargs": {"lr": 1e-3, "weight_decay": 0.01},
    }
    opt = build_optimizer_from_cfg(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_dtensor_muon_separated_qkv_skips_split_path():
    """HF-style separated q/k/v projections are normal 2D weights and must
    end up as ordinary Muon params; the QKV-split path stays inactive."""

    class TinyAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)
            self.norm = nn.LayerNorm(8)

    opt = build_dtensor_muon(TinyAttn(), lr=1e-3)
    muon_opt = opt.optimizers[0]
    assert isinstance(muon_opt, DTensorMuon)
    assert muon_opt._split_qkv is False
    assert muon_opt._is_qkv_fn is None


def test_build_dtensor_muon_fused_qkv_with_explicit_shapes():
    """A fused ``linear_qkv.weight`` parameter must enable the split path
    when ``qkv_split_shapes`` is supplied (or derivable)."""

    class FusedAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Single fused QKV weight of shape (q+k+v rows, hidden).
            self.self_attention = nn.Module()
            self.self_attention.linear_qkv = nn.Linear(8, 24, bias=False)

    model = FusedAttn()
    opt = build_dtensor_muon(
        model,
        lr=1e-3,
        muon_split_qkv=True,
        qkv_split_shapes=(8, 8, 8),
    )
    muon_opt = opt.optimizers[0]
    assert muon_opt._split_qkv is True
    assert muon_opt._is_qkv_fn is not None
    assert muon_opt._is_qkv_fn(model.self_attention.linear_qkv.weight)
    assert muon_opt._qkv_split_shapes == (8, 8, 8)


def test_dtensor_muon_qkv_split_matches_per_slice_orthogonalize():
    """The fused-QKV split-orth-concat must equal stacking the result of
    orthogonalizing each Q/K/V slice independently."""
    _seed()
    # Per-head shapes: q=8 rows, k=4 rows, v=4 rows; 2 query groups; hidden=16.
    shapes = (8, 4, 4)
    num_query_groups = 2
    hidden = 16
    full_rows = num_query_groups * sum(shapes)
    grad = torch.randn(full_rows, hidden)
    fused_param = nn.Parameter(torch.zeros(full_rows, hidden))

    # Reference: split, orthogonalize each slice with vanilla Muon math, concat.
    ref_optim = DTensorMuon(
        [nn.Parameter(torch.empty(0, hidden))],  # placeholder; we just want the fn
        lr=1e-3,
    )
    grad_view = grad.view(num_query_groups, sum(shapes), -1)
    slices = torch.split(grad_view, list(shapes), dim=1)
    flat_slices = [s.reshape(-1, hidden) for s in slices]
    orth_slices = [ref_optim.scaled_orthogonalize_fn(s) for s in flat_slices]
    reshaped = [s.view(num_query_groups, -1, hidden) for s in orth_slices]
    expected = torch.cat(reshaped, dim=1).view(full_rows, hidden)

    # Subject under test.
    optim = DTensorMuon(
        [fused_param],
        lr=1e-3,
        split_qkv=True,
        is_qkv_fn=lambda p: p is fused_param,
        qkv_split_shapes=shapes,
    )
    actual = optim._qkv_split_orthogonalize(grad)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_optimizer_factory_dispatches_muon_builder_path():
    """A builder marked with `_builds_optimizer_from_model = True` must
    receive the model directly so it can split parameters."""
    from nemo_rl.utils.optimizer_factory import build_optimizer_from_cfg

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)

    cfg = {
        "name": "nemo_rl.algorithms.muon.build_dtensor_muon",
        "kwargs": {"lr": 1e-3, "weight_decay": 0.01},
    }
    opt = build_optimizer_from_cfg(Tiny(), cfg)
    assert isinstance(opt, ChainedTorchOptimizer)
    assert isinstance(opt.optimizers[0], DTensorMuon)
    assert isinstance(opt.optimizers[1], torch.optim.AdamW)


# ---------------------------------------------------------------------------
# Operation-level unit tests for DTensorMuon-specific logic
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_rank_pg():
    """Initialize a single-process gloo group so DTensor APIs work without
    torchrun. Cheap enough to keep around for the whole module."""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo", rank=0, world_size=1
        )
        owned = True
    else:
        owned = False
    yield
    if owned:
        torch.distributed.destroy_process_group()


def test_placement_parser_replicate_only_dtensor_matches_plain_tensor(
    single_rank_pg,
):
    """A DTensor with all-Replicate placements must route through the same
    primitives as a plain tensor input — the per-step update is bit-identical."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    _seed()
    full = torch.randn(16, 8)
    grad = torch.randn(16, 8)

    # Plain-tensor reference.
    w_ref = nn.Parameter(full.clone())
    w_ref.grad = grad.clone()
    DTensorMuon([w_ref], lr=1e-3, momentum=0.0).step()

    # Replicate-only DTensor.
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    w_dt = nn.Parameter(
        DTensor.from_local(full.clone(), mesh, [Replicate()], run_check=False)
    )
    w_dt.grad = DTensor.from_local(
        grad.clone(), mesh, [Replicate()], run_check=False
    )
    DTensorMuon([w_dt], lr=1e-3, momentum=0.0).step()

    _equal(w_dt.full_tensor(), w_ref.data)


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_placement_parser_picks_shard_axis_on_2d_mesh(single_rank_pg, shard_dim):
    """A 2D mesh that mirrors FSDP+TP composition must route through the
    shard axis when one of its placements is Shard. Single-rank meshes make
    the all-gather trivial but exercise the parsing branch."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    _seed()
    full = torch.randn(16, 8)
    grad = torch.randn(16, 8)

    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    placements = [Replicate(), Shard(shard_dim)]
    w_dt = nn.Parameter(
        DTensor.from_local(full.clone(), mesh, placements, run_check=False)
    )
    w_dt.grad = DTensor.from_local(
        grad.clone(), mesh, placements, run_check=False
    )
    # Should not crash on a multi-axis mesh; the TP axis is the second one.
    DTensorMuon([w_dt], lr=1e-3, momentum=0.0, tp_mode="duplicated").step()

    # Compare against the plain-tensor reference; single-rank TP collapses
    # to the no-comm path, so the result should be bit-identical.
    w_ref = nn.Parameter(full.clone())
    w_ref.grad = grad.clone()
    DTensorMuon([w_ref], lr=1e-3, momentum=0.0).step()
    _equal(w_dt.full_tensor(), w_ref.data)


# ---------------------------------------------------------------------------
# QKV reshape invariants
# ---------------------------------------------------------------------------


def _identity_orthogonalize_optim(qkv_shapes, hidden):
    """Build a DTensorMuon whose orthogonalize is the identity so we can
    isolate the QKV reshape logic from the Newton-Schulz math."""
    optim = DTensorMuon(
        [nn.Parameter(torch.empty(0, hidden))],
        lr=1e-3,
        split_qkv=True,
        is_qkv_fn=lambda p: True,
        qkv_split_shapes=qkv_shapes,
    )
    optim.scaled_orthogonalize_fn = lambda g: g  # identity for the reshape test
    return optim


def test_qkv_split_preserves_total_rows_and_hidden_dim():
    """The split-orth-concat must reconstruct the same shape it consumed."""
    hidden = 8
    qkv_shapes = (4, 2, 2)
    num_query_groups = 3
    rows = num_query_groups * sum(qkv_shapes)
    grad = torch.randn(rows, hidden)
    out = _identity_orthogonalize_optim(qkv_shapes, hidden)._qkv_split_orthogonalize(grad)
    assert out.shape == grad.shape
    # With identity orthogonalize, the reshape round-trip must be lossless.
    _equal(out, grad)


@pytest.mark.parametrize("num_query_groups", [1, 2, 5])
def test_qkv_split_preserves_slice_order_q_then_k_then_v(num_query_groups):
    """Splits emerge in Q -> K -> V order (matches Megatron's convention at
    muon.py:144-159). Build a grad with distinct constant values per slice
    and assert the reconstructed output still has the same partition."""
    hidden = 4
    qkv_shapes = (3, 2, 2)
    per_head = sum(qkv_shapes)
    rows = num_query_groups * per_head
    # Build a grad where each (query_group, slice) pair has a unique constant.
    grad = torch.zeros(num_query_groups, per_head, hidden)
    grad[:, 0 : qkv_shapes[0], :] = 1.0
    grad[:, qkv_shapes[0] : qkv_shapes[0] + qkv_shapes[1], :] = 2.0
    grad[:, qkv_shapes[0] + qkv_shapes[1] :, :] = 3.0
    grad_flat = grad.view(rows, hidden)

    out = _identity_orthogonalize_optim(qkv_shapes, hidden)._qkv_split_orthogonalize(
        grad_flat
    )
    out_view = out.view(num_query_groups, per_head, hidden)
    assert (out_view[:, 0 : qkv_shapes[0], :] == 1.0).all()
    assert (
        out_view[:, qkv_shapes[0] : qkv_shapes[0] + qkv_shapes[1], :] == 2.0
    ).all()
    assert (out_view[:, qkv_shapes[0] + qkv_shapes[1] :, :] == 3.0).all()


def test_qkv_split_rejects_misaligned_grad_rows():
    """Row count not divisible by sum(qkv_shapes) is an error: the per-head
    reshape would silently bend the slice boundaries otherwise."""
    optim = _identity_orthogonalize_optim((4, 2, 2), hidden=4)
    bad_grad = torch.randn(9, 4)  # 9 is not a multiple of 8
    with pytest.raises(ValueError, match="not divisible"):
        optim._qkv_split_orthogonalize(bad_grad)


def test_qkv_split_mha_equal_q_k_v_shapes():
    """MHA (no GQA) has q == k == v per-head; the split path must still work."""
    optim = _identity_orthogonalize_optim((4, 4, 4), hidden=8)
    grad = torch.randn(24, 8)  # 2 query groups
    out = optim._qkv_split_orthogonalize(grad)
    assert out.shape == grad.shape
    _equal(out, grad)


# ---------------------------------------------------------------------------
# Multi-rank TP equivalence tests (CPU + gloo backend via mp.spawn)
# ---------------------------------------------------------------------------


def _tp_worker(
    rank: int,
    world_size: int,
    init_full: torch.Tensor,
    grad_full: torch.Tensor,
    tp_mode: str,
    shard_dim: int,
    out_path: str,
) -> None:
    import os
    import pickle

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    # Overwrite (not setdefault) so a parent-process group's env doesn't leak
    # through into the spawned worker and pin it to the wrong port / size.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        # Build a DTensor by sharding the full init tensor.
        full = init_full.detach().clone()
        param = nn.Parameter(
            DTensor.from_local(
                full.tensor_split(world_size, dim=shard_dim)[rank],
                mesh,
                [Shard(shard_dim)],
                run_check=False,
            )
        )
        grad_local = grad_full.tensor_split(world_size, dim=shard_dim)[rank]
        param.grad = DTensor.from_local(
            grad_local, mesh, [Shard(shard_dim)], run_check=False
        )

        opt = DTensorMuon(
            [param],
            lr=1e-3,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            tp_mode=tp_mode,  # type: ignore[arg-type]
        )
        opt.step()
        # Materialize the global parameter tensor and write from rank 0.
        full_after = param.full_tensor().detach().clone()
        if rank == 0:
            with open(out_path, "wb") as f:
                pickle.dump(full_after, f)
    finally:
        torch.distributed.destroy_process_group()


def _reference_single_rank_step(
    init_full: torch.Tensor,
    grad_full: torch.Tensor,
) -> torch.Tensor:
    w = nn.Parameter(init_full.detach().clone())
    w.grad = grad_full.clone()
    DTensorMuon(
        [w], lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False
    ).step()
    return w.data


@pytest.mark.parametrize("shard_dim", [0, 1])
@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise"])
def test_dtensor_muon_tp2_duplicated_matches_single_rank(
    tmp_path, shard_dim, tp_mode
):
    """TP=2 + duplicated/blockwise mode all-gathers the full gradient before
    Newton-Schulz, so the global parameter must equal the single-rank result
    bit-for-bit."""
    import multiprocessing as mp
    import pickle

    _seed()
    init_full = torch.randn(8, 16)
    grad_full = torch.randn(8, 16)

    out_path = tmp_path / "result.pkl"
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(
            target=_tp_worker,
            args=(rank, 2, init_full, grad_full, tp_mode, shard_dim, str(out_path)),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    with open(out_path, "rb") as f:
        tp_full = pickle.load(f)
    ref = _reference_single_rank_step(init_full, grad_full)
    torch.testing.assert_close(tp_full, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_dtensor_muon_tp2_distributed_matches_single_rank(tmp_path, shard_dim):
    """The distributed mode runs Newton-Schulz on the local shard with
    cross-rank communication. Some FP non-determinism is acceptable, but the
    global parameter should still be very close to the single-rank result."""
    import multiprocessing as mp
    import pickle

    _seed()
    init_full = torch.randn(8, 16)
    grad_full = torch.randn(8, 16)

    out_path = tmp_path / "result.pkl"
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(
            target=_tp_worker,
            args=(rank, 2, init_full, grad_full, "distributed", shard_dim, str(out_path)),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    with open(out_path, "rb") as f:
        tp_full = pickle.load(f)
    ref = _reference_single_rank_step(init_full, grad_full)
    # Distributed mode does its own all-reduces inside Newton-Schulz; allow
    # some FP slack vs the single-rank computation.
    torch.testing.assert_close(tp_full, ref, rtol=5e-4, atol=5e-4)
