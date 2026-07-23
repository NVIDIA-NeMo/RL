"""Unit tests for MultiLinearLoRA — modelled on
nemo_automodel/tests/unit_tests/_peft/test_lora.py for rigor.

Each upstream test that applies to the multi-adapter case has an
equivalent here (modulo the apply_lora_to_linear_modules / wildcard /
DoRA / dropout features we deliberately don't implement).  In addition,
we test multi-adapter-specific contracts:

- N stacked Parameters with the correct shapes
- per-row routing matches a manual reference for 2D and 3D inputs
- all-rows-pick-adapter-k matches a single LinearLoRA-equivalent built
  from adapter k's slice
- inactive adapters' slices get zero gradient
- bit-equivalence to frozen base when no routing is set
- monkey-patch path mirrors patch_linear_module's contract

The tests have no nemo_automodel / nemo_rl dependency so they run on a
plain torch CPU install — same expectation as upstream test_lora.py.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def base_linear():
    return nn.Linear(16, 32, bias=True)


@pytest.fixture
def base_linear_nobias():
    return nn.Linear(16, 32, bias=False)


@pytest.fixture
def mll(base_linear):
    return MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)


# ---------------------------------------------------------------------------
# Construction & init
# ---------------------------------------------------------------------------


def test_init_creates_n_stacked_adapters(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=5, dim=4, alpha=8)
    assert mll.n_adapters == 5
    assert mll.dim == 4
    assert mll.scale == pytest.approx(8 / 4)
    assert mll.lora_A.shape == (5, 4, 16)
    assert mll.lora_B.shape == (5, 32, 4)
    assert isinstance(mll.lora_A, nn.Parameter)
    assert isinstance(mll.lora_B, nn.Parameter)


def test_init_lora_B_is_zero(mll):
    assert torch.all(mll.lora_B == 0)


def test_init_lora_A_xavier_is_nonzero(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8, lora_A_init_method="xavier")
    # Each adapter's lora_A slice should be non-zero after xavier init.
    for i in range(mll.n_adapters):
        assert not torch.all(mll.lora_A[i] == 0)


def test_init_lora_A_kaiming(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8, lora_A_init_method="uniform")
    for i in range(mll.n_adapters):
        assert not torch.all(mll.lora_A[i] == 0)


def test_init_with_explicit_lora_dtype(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8, lora_dtype=torch.bfloat16)
    assert mll.lora_A.dtype == torch.bfloat16
    assert mll.lora_B.dtype == torch.bfloat16


def test_init_copies_weight_and_bias(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)
    assert torch.allclose(mll.weight, base_linear.weight)
    assert torch.allclose(mll.bias, base_linear.bias)


def test_init_no_bias(base_linear_nobias):
    mll = MultiLinearLoRA(base_linear_nobias, n_adapters=3, dim=4, alpha=8)
    assert mll.bias is None
    assert torch.allclose(mll.weight, base_linear_nobias.weight)


def test_init_rejects_non_linear():
    with pytest.raises(AssertionError):
        MultiLinearLoRA(nn.Conv2d(3, 3, 3), n_adapters=2, dim=4, alpha=8)


def test_init_rejects_zero_adapters(base_linear):
    with pytest.raises(AssertionError):
        MultiLinearLoRA(base_linear, n_adapters=0, dim=4, alpha=8)


def test_init_single_adapter_works(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=1, dim=4, alpha=8)
    assert mll.lora_A.shape == (1, 4, 16)


# ---------------------------------------------------------------------------
# Trainability / freeze (mirrors upstream test_lora_layers_are_trainable)
# ---------------------------------------------------------------------------


def test_base_weight_is_frozen(mll):
    assert mll.weight.requires_grad is False


def test_base_bias_is_frozen(mll):
    assert mll.bias.requires_grad is False


def test_lora_A_lora_B_are_trainable(mll):
    assert mll.lora_A.requires_grad is True
    assert mll.lora_B.requires_grad is True


# ---------------------------------------------------------------------------
# No-routing forward — bit-equivalent to frozen base
# ---------------------------------------------------------------------------


def test_no_routing_2d_equals_frozen_base(base_linear, mll):
    x = torch.randn(5, 16)
    y_base = F.linear(x, base_linear.weight, base_linear.bias)
    y_mll = mll(x)
    assert torch.allclose(y_mll, y_base)


def test_no_routing_3d_equals_frozen_base(base_linear, mll):
    x = torch.randn(5, 7, 16)
    y_base = F.linear(x, base_linear.weight, base_linear.bias)
    y_mll = mll(x)
    assert torch.allclose(y_mll, y_base)


def test_no_routing_after_clear(mll):
    x = torch.randn(5, 16)
    mll.set_routing(torch.tensor([0, 1, 2, 0, 1], dtype=torch.long))
    _ = mll(x)
    mll.clear_routing()
    assert mll.adapter_ids is None
    # And forward returns to frozen-base behaviour.
    y_with_clear = mll(x)
    y_expected = F.linear(x, mll.weight, mll.bias)
    assert torch.allclose(y_with_clear, y_expected)


# ---------------------------------------------------------------------------
# set_routing input validation
# ---------------------------------------------------------------------------


def test_set_routing_requires_long_tensor(mll):
    with pytest.raises(TypeError):
        mll.set_routing(torch.tensor([0, 1, 2], dtype=torch.float32))
    with pytest.raises(TypeError):
        mll.set_routing([0, 1, 2])


def test_set_routing_accepts_long(mll):
    ids = torch.tensor([0, 1], dtype=torch.long)
    mll.set_routing(ids)
    assert mll.adapter_ids is ids


# ---------------------------------------------------------------------------
# Routed forward — manual reference
# ---------------------------------------------------------------------------


def _manual_routed_forward_2d(mll, base_w, base_b, x, ids):
    """Brute-force per-row forward to compare against the bmm path."""
    out = torch.empty(x.shape[0], base_w.shape[0], dtype=x.dtype)
    for b, aid in enumerate(ids.tolist()):
        base_out = F.linear(x[b], base_w, base_b)
        A = mll.lora_A[aid]
        B = mll.lora_B[aid]
        lora = (B @ (A @ x[b])) * mll.scale
        out[b] = base_out + lora
    return out


def _manual_routed_forward_3d(mll, base_w, base_b, x, ids):
    out = torch.empty(x.shape[0], x.shape[1], base_w.shape[0], dtype=x.dtype)
    for b, aid in enumerate(ids.tolist()):
        base_out = F.linear(x[b], base_w, base_b)
        A = mll.lora_A[aid]
        B = mll.lora_B[aid]
        # x[b] : [S, in_f] ; A : [dim, in_f] -> Ax : [S, dim] ; B : [out_f, dim] -> [S, out_f]
        lora = (x[b] @ A.T @ B.T) * mll.scale
        out[b] = base_out + lora
    return out


def _give_lora_real_values(mll):
    """Break the lora_B=0 identity so routed forward actually differs from base."""
    with torch.no_grad():
        mll.lora_A.normal_(0, 0.1)
        mll.lora_B.normal_(0, 0.1)


def test_routed_2d_matches_manual_reference(mll):
    _give_lora_real_values(mll)
    x = torch.randn(5, 16)
    ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    mll.set_routing(ids)
    expected = _manual_routed_forward_2d(mll, mll.weight, mll.bias, x, ids)
    got = mll(x)
    assert torch.allclose(got, expected, atol=1e-5)


def test_routed_3d_matches_manual_reference(mll):
    _give_lora_real_values(mll)
    x = torch.randn(5, 7, 16)
    ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    mll.set_routing(ids)
    expected = _manual_routed_forward_3d(mll, mll.weight, mll.bias, x, ids)
    got = mll(x)
    assert torch.allclose(got, expected, atol=1e-5)


def test_routed_forward_4d_input_raises(mll):
    _give_lora_real_values(mll)
    mll.set_routing(torch.tensor([0, 1, 2], dtype=torch.long))
    with pytest.raises(ValueError, match="2D or 3D"):
        mll(torch.randn(3, 4, 5, 16))


def test_routed_forward_out_of_range_id_raises(mll):
    _give_lora_real_values(mll)
    # set_routing now validates up-front (raises ValueError before forward).
    with pytest.raises(ValueError, match="adapter_ids out of range"):
        mll.set_routing(torch.tensor([0, 5], dtype=torch.long))  # 5 >= n_adapters=3


def test_routed_forward_id_count_mismatch_falls_back_to_slot0(mll):
    """ids/rows mismatch is the MoE expert-dispatch signature (router-scattered
    token counts). Contract (adapter.py bug-11 guard): do NOT raise — fall back
    to slot 0's LoRA for all rows. Per-token correctness on experts is provided
    by moe_routing.py, which re-seeds aligned ids before each expert forward.
    (This test previously expected a RuntimeError; stale since the guard.)"""
    _give_lora_real_values(mll)
    x = torch.randn(5, 16)

    mll.set_routing(torch.tensor([0, 1, 2], dtype=torch.long))  # 3 ids != 5 rows
    out_mismatch = mll(x)

    mll.set_routing(torch.zeros(5, dtype=torch.long))  # all rows -> slot 0
    out_slot0 = mll(x)

    assert torch.allclose(out_mismatch, out_slot0, atol=1e-6), (
        f"mismatch fallback != slot-0 path: max diff "
        f"{(out_mismatch - out_slot0).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Equivalence to a single LinearLoRA-equivalent path
# ---------------------------------------------------------------------------


def test_all_rows_pick_adapter_k_matches_single_adapter_path(base_linear):
    """If every row routes to adapter k, MLL.forward must equal a hand-built
    single-adapter LoRA path using adapter k's slice."""
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)
    _give_lora_real_values(mll)

    x = torch.randn(8, 7, 16)
    for k in range(mll.n_adapters):
        ids = torch.full((8,), k, dtype=torch.long)
        mll.set_routing(ids)
        got = mll(x)

        Ak = mll.lora_A[k]  # [dim, in_f]
        Bk = mll.lora_B[k]  # [out_f, dim]
        base_out = F.linear(x, base_linear.weight, base_linear.bias)
        lora = (x @ Ak.T @ Bk.T) * mll.scale
        expected = base_out + lora

        assert torch.allclose(got, expected, atol=1e-5), f"adapter {k} divergence"


def test_homogeneous_routing_uses_exact_single_kernel_and_gradients(base_linear):
    """A homogeneous microbatch must be byte-identical to direct LinearLoRA math.

    This pins both forward and backward. It specifically prevents the per-row
    gather/bmm or expert index_select/index_copy paths from perturbing BF16
    gradients when all rows/tokens belong to one adapter.
    """
    base = base_linear.to(torch.bfloat16)
    mll = MultiLinearLoRA(
        base, n_adapters=3, dim=4, alpha=8, lora_dtype=torch.bfloat16
    )
    _give_lora_real_values(mll)
    x_multi = torch.randn(1, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.full((1,), 1, dtype=torch.long)
    mll.set_routing(ids)

    y_multi = mll(x_multi)
    y_multi.float().sum().backward()
    grad_A_multi = mll.lora_A.grad[1].clone()
    grad_B_multi = mll.lora_B.grad[1].clone()
    grad_x_multi = x_multi.grad.clone()

    # Independent direct two-F.linear graph using the same exact bytes.
    A = mll.lora_A[1].detach().clone().requires_grad_(True)
    B = mll.lora_B[1].detach().clone().requires_grad_(True)
    x_single = x_multi.detach().clone().requires_grad_(True)
    y_single = F.linear(x_single, mll.weight, mll.bias) + F.linear(
        F.linear(x_single, A) * mll.scale, B
    )
    y_single.float().sum().backward()

    assert torch.equal(y_multi, y_single)
    assert torch.equal(grad_A_multi, A.grad)
    assert torch.equal(grad_B_multi, B.grad)
    assert torch.equal(grad_x_multi, x_single.grad)
    assert torch.count_nonzero(mll.lora_A.grad[0]) == 0
    assert torch.count_nonzero(mll.lora_A.grad[2]) == 0
    assert torch.count_nonzero(mll.lora_B.grad[0]) == 0
    assert torch.count_nonzero(mll.lora_B.grad[2]) == 0


def test_mixed_contiguous_blocks_match_independent_single_graphs(base_linear):
    """Mixed dense batches execute exact standalone F.linear per row block."""
    base = base_linear.to(torch.bfloat16)
    mll = MultiLinearLoRA(
        base, n_adapters=2, dim=4, alpha=8, lora_dtype=torch.bfloat16
    )
    _give_lora_real_values(mll)
    x_multi = torch.randn(4, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mll.set_routing(ids)

    y_multi = mll(x_multi)
    y_multi.float().sum().backward()
    grad_A_multi = mll.lora_A.grad.clone()
    grad_B_multi = mll.lora_B.grad.clone()

    A0 = mll.lora_A[0].detach().clone().requires_grad_(True)
    B0 = mll.lora_B[0].detach().clone().requires_grad_(True)
    A1 = mll.lora_A[1].detach().clone().requires_grad_(True)
    B1 = mll.lora_B[1].detach().clone().requires_grad_(True)
    x_single = x_multi.detach().clone().requires_grad_(True)
    base_out = F.linear(x_single, mll.weight, mll.bias)
    y0 = base_out[0:2] + F.linear(F.linear(x_single[0:2], A0) * mll.scale, B0)
    y1 = base_out[2:4] + F.linear(F.linear(x_single[2:4], A1) * mll.scale, B1)
    y_single = torch.cat([y0, y1], dim=0)
    y_single.float().sum().backward()

    assert torch.equal(y_multi, y_single)
    assert torch.equal(grad_A_multi[0], A0.grad)
    assert torch.equal(grad_A_multi[1], A1.grad)
    assert torch.equal(grad_B_multi[0], B0.grad)
    assert torch.equal(grad_B_multi[1], B1.grad)
    assert torch.equal(x_multi.grad, x_single.grad)


def test_different_adapters_produce_different_outputs(base_linear):
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)
    _give_lora_real_values(mll)
    x = torch.randn(2, 16)

    mll.set_routing(torch.zeros(2, dtype=torch.long))
    out0 = mll(x).clone()
    mll.set_routing(torch.ones(2, dtype=torch.long))
    out1 = mll(x).clone()
    mll.set_routing(torch.full((2,), 2, dtype=torch.long))
    out2 = mll(x).clone()

    assert not torch.allclose(out0, out1)
    assert not torch.allclose(out1, out2)
    assert not torch.allclose(out0, out2)


# ---------------------------------------------------------------------------
# Scaling — alpha/dim
# ---------------------------------------------------------------------------


def test_scale_equals_alpha_over_dim(base_linear):
    for dim, alpha in [(4, 8), (8, 32), (16, 16)]:
        mll = MultiLinearLoRA(base_linear, n_adapters=2, dim=dim, alpha=alpha)
        assert mll.scale == pytest.approx(alpha / dim)


def test_doubling_alpha_doubles_lora_contribution(base_linear):
    base = base_linear
    mll1 = MultiLinearLoRA(base, n_adapters=2, dim=4, alpha=8)
    mll2 = MultiLinearLoRA(base, n_adapters=2, dim=4, alpha=16)
    with torch.no_grad():
        mll1.lora_A.normal_(0, 0.1)
        mll1.lora_B.normal_(0, 0.1)
        mll2.lora_A.copy_(mll1.lora_A)
        mll2.lora_B.copy_(mll1.lora_B)

    x = torch.randn(4, 16)
    ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    mll1.set_routing(ids)
    mll2.set_routing(ids)

    base_out = F.linear(x, base.weight, base.bias)
    lora1 = mll1(x) - base_out
    lora2 = mll2(x) - base_out

    assert torch.allclose(2 * lora1, lora2, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward — mirrors upstream test_backward_pass + adds per-adapter checks
# ---------------------------------------------------------------------------


def test_backward_produces_finite_gradients(mll):
    _give_lora_real_values(mll)
    x = torch.randn(5, 7, 16, requires_grad=True)
    mll.set_routing(torch.tensor([0, 1, 2, 0, 1], dtype=torch.long))
    loss = mll(x).sum()
    loss.backward()

    assert mll.lora_A.grad is not None and mll.lora_B.grad is not None
    assert torch.isfinite(mll.lora_A.grad).all()
    assert torch.isfinite(mll.lora_B.grad).all()
    # Base weight gradient must remain None or zero — it's frozen.
    assert mll.weight.grad is None


def test_inactive_adapter_slice_gets_zero_gradient(mll):
    """If no row picks adapter k, lora_A[k] and lora_B[k] must receive zero grad."""
    _give_lora_real_values(mll)
    x = torch.randn(4, 16)
    # Route only to adapters 0 and 2; adapter 1 is inactive.
    mll.set_routing(torch.tensor([0, 0, 2, 2], dtype=torch.long))
    mll(x).sum().backward()

    assert mll.lora_A.grad is not None
    assert torch.all(mll.lora_A.grad[1] == 0), "adapter 1's lora_A grad must be 0"
    assert torch.all(mll.lora_B.grad[1] == 0), "adapter 1's lora_B grad must be 0"
    # Active adapters got nonzero grad.
    assert not torch.all(mll.lora_A.grad[0] == 0)
    assert not torch.all(mll.lora_A.grad[2] == 0)


def test_optimizer_step_updates_lora_weights(mll):
    _give_lora_real_values(mll)
    A_before = mll.lora_A.detach().clone()
    B_before = mll.lora_B.detach().clone()
    weight_before = mll.weight.detach().clone()

    x = torch.randn(4, 16)
    mll.set_routing(torch.tensor([0, 1, 2, 0], dtype=torch.long))
    optim = torch.optim.SGD([mll.lora_A, mll.lora_B], lr=1e-2)
    optim.zero_grad()
    mll(x).sum().backward()
    optim.step()

    assert not torch.allclose(mll.lora_A, A_before)
    assert not torch.allclose(mll.lora_B, B_before)
    # Base weight unchanged — it's frozen.
    assert torch.allclose(mll.weight, weight_before)


# ---------------------------------------------------------------------------
# Forward output consistency (mirrors upstream test_forward_output_consistency)
# ---------------------------------------------------------------------------


def test_forward_output_shape_matches_base(base_linear, mll):
    _give_lora_real_values(mll)
    x = torch.randn(2, 16)
    mll.set_routing(torch.tensor([0, 1], dtype=torch.long))
    out_base = base_linear(x)
    out_mll = mll(x)
    assert out_base.shape == out_mll.shape
    assert not torch.allclose(out_base, out_mll), "MLL output must differ from base when LoRA is active"





# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_init():
    torch.manual_seed(123)
    a = MultiLinearLoRA(nn.Linear(16, 32), n_adapters=3, dim=4, alpha=8)
    torch.manual_seed(123)
    b = MultiLinearLoRA(nn.Linear(16, 32), n_adapters=3, dim=4, alpha=8)
    assert torch.allclose(a.lora_A, b.lora_A)
    assert torch.allclose(a.lora_B, b.lora_B)


def test_forward_deterministic(mll):
    _give_lora_real_values(mll)
    x = torch.randn(4, 16)
    mll.set_routing(torch.tensor([0, 1, 2, 0], dtype=torch.long))
    y1 = mll(x).clone()
    y2 = mll(x).clone()
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# Dtype propagation
# ---------------------------------------------------------------------------


def test_forward_preserves_dtype_in_bf16():
    base = nn.Linear(16, 32).to(torch.bfloat16)
    mll = MultiLinearLoRA(base, n_adapters=2, dim=4, alpha=8, lora_dtype=torch.bfloat16)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    mll.set_routing(torch.tensor([0, 1, 0, 1], dtype=torch.long))
    out = mll(x)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Worklog-derived behavioral contracts
# (gaps identified by reading /home/phuc/workspace/moe/worklogs/2026-05-*)
# ---------------------------------------------------------------------------


def test_set_routing_rejects_negative_id(mll):
    """set_routing must reject negative ids up-front, not at forward time."""
    with pytest.raises(ValueError, match="adapter_ids out of range"):
        mll.set_routing(torch.tensor([0, -1], dtype=torch.long))


def test_set_routing_accepts_empty_tensor(mll):
    """Empty routing is degenerate but legal — must not crash range validation."""
    ids = torch.tensor([], dtype=torch.long)
    mll.set_routing(ids)  # no rows → nothing to validate
    assert mll.adapter_ids is ids


def test_arithmetic_order_scale_between_A_and_B(base_linear):
    """Scale must be applied BETWEEN A and B (lora_B(lora_A(x) * scale)),
    matching upstream `LinearLoRA.forward`. In fp32 this is mathematically
    equivalent to `scale * lora_B(lora_A(x))`, but in bf16 the GEMM
    accumulation order differs and breaks bit-equivalence with single-LoRA.
    Pin the order with a manual reference. See worklog 2026-05-13/01.
    """
    mll = MultiLinearLoRA(base_linear, n_adapters=2, dim=4, alpha=8)
    _give_lora_real_values(mll)
    x = torch.randn(3, 16)
    ids = torch.tensor([0, 1, 0], dtype=torch.long)
    mll.set_routing(ids)

    expected = torch.empty(3, 32)
    for b, aid in enumerate(ids.tolist()):
        base_out = F.linear(x[b], base_linear.weight, base_linear.bias)
        A = mll.lora_A[aid]
        B = mll.lora_B[aid]
        # SCALE BETWEEN A AND B — not after B.
        Ax_scaled = (A @ x[b]) * mll.scale
        lora = B @ Ax_scaled
        expected[b] = base_out + lora
    got = mll(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_uniform_routing_matches_handbuilt_single_lora(base_linear):
    """Uniform routing (all rows → adapter k) must match a hand-built
    stock single-LoRA path using adapter k's slice. This is the
    bit-equivalence-with-single-LoRA contract for the uniform-router gate
    in worklog 2026-05-05/02.

    Tolerance is `atol=1e-6` (fp32) because the impl uses `bmm` while
    this reference uses `@`; both paths are mathematically identical but
    can differ by ULPs depending on cuBLAS heuristic choice. The point
    is to pin the OP ORDER (scale between A and B), not bit-identity.
    """
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)
    _give_lora_real_values(mll)
    x = torch.randn(6, 11, 16)  # 3D, non-trivial shape
    for k in range(mll.n_adapters):
        mll.set_routing(torch.full((6,), k, dtype=torch.long))
        got = mll(x)

        Ak = mll.lora_A[k]
        Bk = mll.lora_B[k]
        base_out = F.linear(x, base_linear.weight, base_linear.bias)
        Ax_scaled = (x @ Ak.T) * mll.scale
        lora = Ax_scaled @ Bk.T
        expected = base_out + lora
        assert torch.allclose(got, expected, atol=1e-6), f"adapter {k} forward mismatch"


def test_single_adapter_with_routing_matches_stock_linearlora(base_linear):
    """n_adapters=1 + routing=[0,0,...] must match a stock `nn.Linear` +
    LoRA pair. Pins the 'N==1 single-adapter' contract from worklog
    2026-05-05/HANDOFF §9. `allclose(atol=1e-6)` for the same bmm-vs-@
    ULP reason as the uniform-routing test.
    """
    mll = MultiLinearLoRA(base_linear, n_adapters=1, dim=4, alpha=8)
    _give_lora_real_values(mll)
    x = torch.randn(4, 16)
    mll.set_routing(torch.zeros(4, dtype=torch.long))
    got = mll(x)

    A0 = mll.lora_A[0]
    B0 = mll.lora_B[0]
    base_out = F.linear(x, base_linear.weight, base_linear.bias)
    Ax_scaled = (x @ A0.T) * mll.scale
    lora = Ax_scaled @ B0.T
    expected = base_out + lora
    assert torch.allclose(got, expected, atol=1e-6)


def test_inactive_adapter_optimizer_step_is_no_op(base_linear):
    """If adapter k is inactive for the backward pass, its lora_A[k] and
    lora_B[k] slices must be byte-identical before and after
    optimizer.step(). Worklog 2026-05-12/03 cites silent inactive-adapter
    drift as a real failure mode.
    """
    mll = MultiLinearLoRA(base_linear, n_adapters=3, dim=4, alpha=8)
    _give_lora_real_values(mll)
    A_before = {k: mll.lora_A[k].detach().clone() for k in range(mll.n_adapters)}
    B_before = {k: mll.lora_B[k].detach().clone() for k in range(mll.n_adapters)}

    # Route only to adapters 0 and 2; adapter 1 is inactive.
    mll.set_routing(torch.tensor([0, 0, 2, 2], dtype=torch.long))
    optim = torch.optim.SGD([mll.lora_A, mll.lora_B], lr=1e-2)
    optim.zero_grad()
    mll(torch.randn(4, 16)).sum().backward()
    optim.step()

    # Active adapters changed.
    assert not torch.equal(mll.lora_A[0], A_before[0])
    assert not torch.equal(mll.lora_A[2], A_before[2])
    # Inactive adapter (1) is byte-identical.
    assert torch.equal(mll.lora_A[1], A_before[1])
    assert torch.equal(mll.lora_B[1], B_before[1])


def test_repeated_init_in_same_process_is_deterministic(base_linear):
    """Two MultiLinearLoRA builds in the same process under the same
    explicit seed must produce identical lora_A. Pins the absence of any
    hidden class-level mutable init counter (worklog 2026-05-07/02 S3).
    """
    torch.manual_seed(2026)
    a = MultiLinearLoRA(nn.Linear(16, 32), n_adapters=3, dim=4, alpha=8)
    torch.manual_seed(2026)
    b = MultiLinearLoRA(nn.Linear(16, 32), n_adapters=3, dim=4, alpha=8)
    assert torch.equal(a.lora_A, b.lora_A)
    assert torch.equal(a.lora_B, b.lora_B)
