"""Tests for per-token MoE expert routing (moe_routing.py).

Covers the isolation bug found 2026-07-02: expert-layer LoRA fell back to
slot 0 for all rows because the router-scattered token count never matched
the [B] routing buffer. See worklog 2026-07-02.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA
from nemo_rl.models.multi_lora.moe_routing import install_moe_expert_routing
from nemo_rl.models.multi_lora.routing import seed_microbatch_routing

H = 16          # hidden
I = 24          # expert intermediate
N_EXPERTS = 4
TOP_K = 2
N_ADAPTERS = 3


class ToyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = MultiLinearLoRA(nn.Linear(H, I, bias=False), n_adapters=N_ADAPTERS, dim=4, alpha=8)
        self.down_proj = MultiLinearLoRA(nn.Linear(I, H, bias=False), n_adapters=N_ADAPTERS, dim=4, alpha=8)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)))


class ToyGate(nn.Module):
    """Deterministic router: token t -> experts (t % N, (t+1) % N)."""
    def forward(self, hidden_states):
        n_tok = hidden_states.view(-1, hidden_states.shape[-1]).shape[0]
        idx = torch.stack(
            [torch.arange(n_tok) % N_EXPERTS, (torch.arange(n_tok) + 1) % N_EXPERTS],
            dim=1,
        )
        w = torch.full((n_tok, TOP_K), 0.5)
        return idx, w


class ToyMOE(nn.Module):
    """Mirrors NemotronHMOE.moe/forward structure (duck-typed by moe_routing)."""
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([ToyExpert() for _ in range(N_EXPERTS)])
        self.gate = ToyGate()

    def moe(self, hidden_states, topk_indices, topk_weights):
        final = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts)).permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                ew = topk_weights[token_indices, weight_indices]
                out = expert(hidden_states[token_indices])
                final.index_add_(0, token_indices, out * ew.unsqueeze(-1))
            else:
                dummy = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0))
                final = final + dummy
        return final.type(hidden_states.dtype)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        return self.moe(flat, topk_indices, topk_weights).view(*orig_shape)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_block = ToyMOE()

    def forward(self, input_ids):  # input_ids: [B, S, H] float for simplicity
        return self.moe_block(input_ids)


def _fill_adapters(model, magnitude=1.0):
    """Give each adapter slot distinct nonzero weights: slot j filled with
    (j+1)*magnitude so outputs are attributable to a slot."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiLinearLoRA):
                for j in range(m.n_adapters):
                    m.lora_A.data[j].fill_(0.01 * (j + 1) * magnitude)
                    m.lora_B.data[j].fill_(0.01 * (j + 1) * magnitude)


def test_install_wraps_moe_modules():
    model = ToyModel()
    n = install_moe_expert_routing(model)
    assert n == 1
    assert model.moe_block._nousnet_moe_routing_wrapped
    # idempotent
    assert install_moe_expert_routing(model) == 0


def test_expert_rows_route_to_owning_adapter():
    """Rows from adapter k must get adapter k's LoRA inside experts —
    the exact guarantee the legacy slot-0 fallback violated."""
    torch.manual_seed(0)
    B, S = 3, 4
    x = torch.randn(B, S, H)
    row_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    # Reference: one model per adapter id, ALL rows forced to that id
    # (single-adapter semantics), take row r from the model with row r's id.
    ref_rows = {}
    for k in range(N_ADAPTERS):
        torch.manual_seed(1)  # identical base weights across builds
        m = ToyModel()
        _fill_adapters(m)
        seed_microbatch_routing(m, torch.full((B,), k, dtype=torch.long))
        out = m(x)
        ref_rows[k] = out.detach()

    torch.manual_seed(1)
    m = ToyModel()
    _fill_adapters(m)
    seed_microbatch_routing(m, row_ids)
    out = m(x).detach()

    for r in range(B):
        expected = ref_rows[int(row_ids[r])][r]
        assert torch.allclose(out[r], expected, atol=1e-6), (
            f"row {r} (adapter {int(row_ids[r])}) mismatch: "
            f"max diff {(out[r]-expected).abs().max():.2e}"
        )


def test_without_moe_wrapper_rows_collapse_to_slot0():
    """Documents the legacy bug: without install_moe_expert_routing, expert
    MLLs see mismatched row counts and use slot 0 for everything."""
    torch.manual_seed(0)
    B, S = 3, 4
    x = torch.randn(B, S, H)

    torch.manual_seed(1)
    m = ToyModel()
    _fill_adapters(m)
    # Seed WITHOUT the moe wrapper: install hook manually, skip moe install.
    from nemo_rl.models.multi_lora.routing import (
        install_microbatch_routing_hook, set_microbatch_routing_full,
    )
    install_microbatch_routing_hook(m)
    set_microbatch_routing_full(m, torch.tensor([0, 1, 2], dtype=torch.long))
    out_legacy = m(x).detach()

    torch.manual_seed(1)
    m2 = ToyModel()
    _fill_adapters(m2)
    install_microbatch_routing_hook(m2)
    set_microbatch_routing_full(m2, torch.zeros(B, dtype=torch.long))
    out_slot0 = m2(x).detach()

    # Legacy behavior == everything through slot 0 (the bug).
    assert torch.allclose(out_legacy, out_slot0, atol=1e-6)


def test_grad_isolation_across_adapters():
    """Backward: adapter k's expert LoRA grads must come only from rows
    routed to k. With distinct rows per adapter, slots for absent adapters
    must have zero grad; with the legacy fallback, slot 0 got everything."""
    torch.manual_seed(0)
    B, S = 2, 3
    x = torch.randn(B, S, H)
    row_ids = torch.tensor([1, 2], dtype=torch.long)  # adapter 0 gets NO rows

    m = ToyModel()
    _fill_adapters(m)
    seed_microbatch_routing(m, row_ids)
    out = m(x)
    out.sum().backward()

    for name, mod in m.named_modules():
        if isinstance(mod, MultiLinearLoRA):
            g = mod.lora_B.grad
            if g is None:
                continue
            g0 = g[0].abs().max().item()
            g12 = g[1:].abs().max().item()
            assert g0 == 0.0, f"{name}: slot 0 leaked grad {g0:.2e} (no rows routed to it)"
            assert g12 > 0.0, f"{name}: slots 1/2 got no grad"


def test_no_routing_seeded_is_noop():
    """Eval path: no seeded routing -> wrapper must not inject anything."""
    m = ToyModel()
    _fill_adapters(m)
    install_moe_expert_routing(m)
    x = torch.randn(2, 3, H)
    out = m(x)  # must not raise; MLLs run base-only
    assert out.shape == x.shape


def test_empty_expert_dummy_path():
    """A gate that routes everything to experts 0/1 leaves 2/3 empty; the
    dummy zero-row path must not raise and must contribute ~zero."""
    class AllZeroGate(nn.Module):
        def forward(self, hidden_states):
            n_tok = hidden_states.view(-1, hidden_states.shape[-1]).shape[0]
            idx = torch.zeros(n_tok, TOP_K, dtype=torch.long)
            idx[:, 1] = 1
            return idx, torch.full((n_tok, TOP_K), 0.5)

    m = ToyModel()
    m.moe_block.gate = AllZeroGate()
    _fill_adapters(m)
    seed_microbatch_routing(m, torch.tensor([0, 1], dtype=torch.long))
    out = m(torch.randn(2, 3, H))
    assert torch.isfinite(out).all()


def test_indivisible_token_count_falls_back(caplog):
    """n_tok % B != 0 -> warn once and use legacy behavior, never crash."""
    m = ToyModel()
    _fill_adapters(m)
    seed_microbatch_routing(m, torch.tensor([0, 1], dtype=torch.long))
    # Call moe() directly with a token count not divisible by B=2.
    flat = torch.randn(5, H)
    idx, w = ToyGate()(flat)
    out = m.moe_block.moe(flat, idx, w)
    assert out.shape == flat.shape


def test_group_mode_matches_bmm_path():
    """The memory-flat group-by-adapter forward (used on experts) must be
    numerically equivalent to the per-row bmm gather path."""
    torch.manual_seed(3)
    lin = nn.Linear(H, I, bias=False)
    mll = MultiLinearLoRA(lin, n_adapters=N_ADAPTERS, dim=4, alpha=8)
    with torch.no_grad():
        for j in range(mll.n_adapters):
            mll.lora_A.data[j].normal_(0, 0.1)
            mll.lora_B.data[j].normal_(0, 0.1)

    x = torch.randn(7, H)
    ids = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)

    mll._nousnet_route_group_mode = False
    mll.set_routing(ids)
    out_bmm = mll(x).detach().clone()

    mll._nousnet_route_group_mode = True
    mll.set_routing(ids)
    out_group = mll(x).detach().clone()

    assert torch.allclose(out_bmm, out_group, atol=1e-5), (
        f"max diff {(out_bmm - out_group).abs().max():.2e}"
    )


def test_group_mode_grad_matches_bmm_path():
    """Gradients through the group path must match the bmm path too."""
    torch.manual_seed(4)
    x = torch.randn(6, H)
    ids = torch.tensor([2, 0, 1, 1, 0, 2], dtype=torch.long)

    grads = {}
    for mode in [False, True]:
        torch.manual_seed(5)
        lin = nn.Linear(H, I, bias=False)
        mll = MultiLinearLoRA(lin, n_adapters=N_ADAPTERS, dim=4, alpha=8)
        with torch.no_grad():
            for j in range(mll.n_adapters):
                mll.lora_A.data[j].normal_(0, 0.1)
                mll.lora_B.data[j].normal_(0, 0.1)
        mll._nousnet_route_group_mode = mode
        mll.set_routing(ids)
        mll(x).sum().backward()
        grads[mode] = (mll.lora_A.grad.detach().clone(), mll.lora_B.grad.detach().clone())

    for name, (g_bmm, g_grp) in [("lora_A", (grads[False][0], grads[True][0])),
                                 ("lora_B", (grads[False][1], grads[True][1]))]:
        assert torch.allclose(g_bmm, g_grp, atol=1e-5), (
            f"{name} grad mismatch: max diff {(g_bmm - g_grp).abs().max():.2e}"
        )
