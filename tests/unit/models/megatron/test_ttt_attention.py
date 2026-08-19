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

"""Numerics tests for the two-part TTT draft attention.

The reference is a dense fp32 joint softmax over the concatenated trunk
(causal pass-1) and branch (same-anchor diagonal) key sets — the staircase
mask materialized — with native autograd. The kernel path must match it in
both outputs and all input gradients (including the cross-pass gradient
accumulation into the stashed pass-1 KV).
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.draft.eagle import (
    TTTDraftCoreAttention,
    TwoPartTTTAttention,
)

# The module under test is torch-only, but importing it goes through the
# draft package __init__, which pulls in megatron.core.
pytestmark = pytest.mark.mcore


def _flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


requires_flash_attn = pytest.mark.skipif(
    not _flash_attn_available(), reason="Requires CUDA and flash-attn"
)


def dense_two_part_reference(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Joint softmax over causal trunk + diagonal branch keys, materialized.

    Shapes follow TwoPartTTTAttention: q ``[B,S,Hq,D]``, k1/v1 ``[B,S,Hkv,D]``,
    kb/vb ``[B,S,Hkv,P,D]`` (P may be 0 for a pure causal pass).
    """
    batch, seqlen, num_q_heads, _ = q.shape
    group = num_q_heads // k1.shape[2]
    k1e = k1.repeat_interleave(group, dim=2)
    v1e = v1.repeat_interleave(group, dim=2)
    kbe = kb.repeat_interleave(group, dim=2)
    vbe = vb.repeat_interleave(group, dim=2)

    scores_trunk = torch.einsum("bihd,bjhd->bhij", q, k1e) * softmax_scale
    causal = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device))
    scores_trunk = scores_trunk.masked_fill(~causal, float("-inf"))
    scores_branch = torch.einsum("bihd,bihpd->bhip", q, kbe) * softmax_scale

    probs = torch.softmax(torch.cat([scores_trunk, scores_branch], dim=-1), dim=-1)
    out_trunk = torch.einsum("bhij,bjhd->bihd", probs[..., :seqlen], v1e)
    out_branch = torch.einsum("bhip,bihpd->bihd", probs[..., seqlen:], vbe)
    return out_trunk + out_branch


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    actual = actual.float()
    expected = expected.float()
    max_diff = (actual - expected).abs().max().item()
    scale = expected.abs().max().clamp(min=1e-6).item()
    assert max_diff <= 3e-2 + 3e-2 * scale, (
        f"{name}: max abs diff {max_diff:.4e} (ref scale {scale:.4e})"
    )


@requires_flash_attn
@pytest.mark.parametrize("num_kv_heads", [8, 4])  # MHA and GQA (group = 2)
def test_two_part_function_matches_dense(num_kv_heads):
    torch.manual_seed(0)
    batch, seqlen, num_q_heads, head_dim, num_branch = 2, 97, 8, 64, 2
    softmax_scale = head_dim**-0.5

    def make(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

    q = make(batch, seqlen, num_q_heads, head_dim)
    k1 = make(batch, seqlen, num_kv_heads, head_dim)
    v1 = make(batch, seqlen, num_kv_heads, head_dim)
    kb = make(batch, seqlen, num_kv_heads, num_branch, head_dim)
    vb = make(batch, seqlen, num_kv_heads, num_branch, head_dim)

    out = TwoPartTTTAttention.apply(q, k1, v1, kb, vb, softmax_scale)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k1, v1, kb, vb), dout)

    refs = [t.detach().float().requires_grad_() for t in (q, k1, v1, kb, vb)]
    out_ref = dense_two_part_reference(*refs, softmax_scale)
    ref_grads = torch.autograd.grad(out_ref, refs, dout.float())

    _assert_close(out, out_ref, "out")
    for grad, ref_grad, name in zip(
        grads, ref_grads, ("dq", "dk1", "dv1", "dkb", "dvb")
    ):
        _assert_close(grad, ref_grad, name)


@requires_flash_attn
def test_module_multi_pass_matches_dense_and_accumulates_trunk_grads():
    """Three TTT passes through the core-attention module vs the dense oracle.

    The pass-1 KV gradient must accumulate contributions from all passes
    (trunk reuse), which is the property the KV stash exists for.
    """
    torch.manual_seed(1)
    num_passes = 3
    seqlen, batch, num_q_heads, num_kv_heads, head_dim = 64, 2, 4, 2, 32
    softmax_scale = head_dim**-0.5

    module = TTTDraftCoreAttention(
        SimpleNamespace(
            attention_dropout=0.0, context_parallel_size=1, softmax_scale=None
        )
    )

    # sbhd layout, as handed to core_attention by MCore SelfAttention.
    qs = [
        torch.randn(
            seqlen, batch, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]
    ks = [
        torch.randn(
            seqlen, batch, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]
    vs = [
        torch.randn(
            seqlen, batch, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]

    outs = []
    for ttt_pass in range(1, num_passes + 1):
        module.begin_pass(ttt_pass)
        outs.append(module(qs[ttt_pass - 1], ks[ttt_pass - 1], vs[ttt_pass - 1], None))
    module.reset()

    dout = [torch.randn_like(o) for o in outs]
    grads = torch.autograd.grad(outs, qs + ks + vs, dout, allow_unused=False)

    # Dense reference on fp32 leaf copies (bshd layout).
    def to_bshd(t):
        return t.detach().float().transpose(0, 1).contiguous().requires_grad_()

    qs_ref = [to_bshd(t) for t in qs]
    ks_ref = [to_bshd(t) for t in ks]
    vs_ref = [to_bshd(t) for t in vs]
    outs_ref = []
    for ttt_pass in range(1, num_passes + 1):
        branch_k = ks_ref[1:ttt_pass]
        if branch_k:
            kb = torch.stack(branch_k, dim=3)
            vb = torch.stack(vs_ref[1:ttt_pass], dim=3)
        else:
            kb = ks_ref[0].new_zeros(batch, seqlen, num_kv_heads, 0, head_dim)
            vb = kb.clone()
        out_ref = dense_two_part_reference(
            qs_ref[ttt_pass - 1], ks_ref[0], vs_ref[0], kb, vb, softmax_scale
        )
        # bshd -> [S, B, H*D] to match the module output contract.
        outs_ref.append(
            out_ref.transpose(0, 1).reshape(seqlen, batch, num_q_heads * head_dim)
        )
    grads_ref = torch.autograd.grad(
        outs_ref, qs_ref + ks_ref + vs_ref, [d.float() for d in dout]
    )

    for out, out_ref in zip(outs, outs_ref):
        _assert_close(out, out_ref, "pass output")
    for grad, grad_ref, name in zip(
        grads,
        grads_ref,
        [f"dq{i}" for i in range(num_passes)]
        + [f"dk{i}" for i in range(num_passes)]
        + [f"dv{i}" for i in range(num_passes)],
    ):
        # Reference grads are bshd; module grads are sbhd.
        _assert_close(grad, grad_ref.transpose(0, 1), name)

    # The cross-pass property: pass-1 KV must receive gradient from passes 2/3
    # too — compare against a single-pass-only reference to prove they differ.
    single_pass_ref = torch.autograd.grad(
        dense_two_part_reference(
            to_bshd(qs[0]),
            (k_only := to_bshd(ks[0])),
            to_bshd(vs[0]),
            k_only.new_zeros(batch, seqlen, num_kv_heads, 0, head_dim),
            k_only.new_zeros(batch, seqlen, num_kv_heads, 0, head_dim),
            softmax_scale,
        )
        .transpose(0, 1)
        .reshape(seqlen, batch, num_q_heads * head_dim),
        k_only,
        dout[0].float(),
    )[0]
    assert not torch.allclose(
        grads[num_passes].float(), single_pass_ref.transpose(0, 1), atol=1e-3
    ), "pass-1 K grad shows no cross-pass contribution"


@requires_flash_attn
def test_two_part_matches_modelopt_multistep_mask_oracle():
    """External oracle: modelopt's own TTT mask must reproduce our attention.

    modelopt's ``set_multi_step_attention_mask`` (megatron_eagle.py) is the
    NVIDIA reference for EAGLE multi-step training: one dense attention over
    the K-fold concatenated sequence with a staircase mask. Its layout shifts
    the hidden stream down one row per pass, so its pass-(p) block row ``r``
    hosts our (pass p, anchor i = r - (p-1)) entry. Feeding the SAME per-pass
    q/k/v to both sides and comparing row-by-row pins our trunk/branch
    correspondence and the LSE merge against an implementation we did not
    write. RoPE is bypassed (raw q/k/v), so this checks mask semantics only.
    """
    from modelopt.torch.speculative.plugins.megatron_eagle import (
        set_multi_step_attention_mask,
    )

    torch.manual_seed(7)
    seqlen, num_heads, head_dim, num_passes = 8, 2, 16, 3
    softmax_scale = head_dim**-0.5
    device = "cuda"

    qs = [
        torch.randn(seqlen, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        for _ in range(num_passes)
    ]
    ks = [torch.randn_like(qs[0]) for _ in range(num_passes)]
    vs = [torch.randn_like(qs[0]) for _ in range(num_passes)]

    # --- Our side: per-pass two-part attention -> [S, 1, H*D] per pass.
    module = TTTDraftCoreAttention(
        SimpleNamespace(
            attention_dropout=0.0, context_parallel_size=1, softmax_scale=None
        )
    )
    ours = []
    for ttt_pass in range(1, num_passes + 1):
        module.begin_pass(ttt_pass)
        ours.append(
            module(qs[ttt_pass - 1], ks[ttt_pass - 1], vs[ttt_pass - 1], None).float()
        )
    module.reset()

    # --- Oracle side: modelopt runs S query rows per ttt step against the
    # KV cache of all previous blocks; set_multi_step_attention_mask(base, p)
    # returns the [S, (p+1)S] mask for block p's queries. Block p row r hosts
    # our (pass p+1, anchor i = r - p); rows with no valid anchor stay zero
    # and are never compared.
    def place_block(stream, block):
        placed = torch.zeros(
            seqlen, num_heads, head_dim, device=device, dtype=torch.float32
        )
        for row in range(block, seqlen):
            placed[row] = stream[block][row - block, 0].float()
        return placed

    # Base causal mask (True = masked) + modelopt's pass-1 edge adjustment
    # (their pass-1 stream carries rolled input_ids, so the mask is shifted
    # diagonally and the padding row/col is masked out).
    causal = torch.triu(
        torch.ones(seqlen, seqlen, dtype=torch.bool, device=device), diagonal=1
    )[None, None]
    base = causal.clone()
    base[:, :, :-1, :-1] = causal[:, :, 1:, 1:]
    base[:, :, -1, :] = True
    base[:, :, :, -1] = True

    for block in range(num_passes):
        qc = place_block(qs, block)
        kc = torch.cat([place_block(ks, b) for b in range(block + 1)], dim=0)
        vc = torch.cat([place_block(vs, b) for b in range(block + 1)], dim=0)
        mask = set_multi_step_attention_mask(base.clone(), block)[0, 0]
        assert mask.shape == (seqlen, (block + 1) * seqlen)

        scores = torch.einsum("ihd,jhd->hij", qc, kc) * softmax_scale
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        oracle = torch.einsum("hij,jhd->ihd", torch.softmax(scores, dim=-1), vc)

        # Compare on rows that are valid on both sides: modelopt unmasks the
        # block-p self-diagonal only for rows in [p, S-2].
        for row in range(block, seqlen - 1):
            anchor = row - block
            mine = ours[block][anchor, 0].view(num_heads, head_dim)
            ref = oracle[row]
            max_diff = (mine - ref).abs().max().item()
            assert max_diff < 2e-2, (
                f"pass {block + 1} anchor {anchor} (oracle row {row}): "
                f"max diff {max_diff:.4e}"
            )


def test_module_guards_do_not_require_gpu():
    module = TTTDraftCoreAttention(
        SimpleNamespace(
            attention_dropout=0.0, context_parallel_size=1, softmax_scale=None
        )
    )
    q = torch.randn(4, 1, 2, 8)

    with pytest.raises(RuntimeError, match="begin_pass"):
        module(q, q, q, None)

    module.begin_pass(1)
    with pytest.raises(RuntimeError, match="in order"):
        module.begin_pass(3)

    with pytest.raises(NotImplementedError, match="sequence packing"):
        module(q, q, q, None, packed_seq_params=object())

    with pytest.raises(ValueError, match="attention_mask"):
        module(q, q, q, torch.ones(1))


def test_module_rejects_attention_dropout():
    with pytest.raises(ValueError, match="dropout"):
        TTTDraftCoreAttention(
            SimpleNamespace(
                attention_dropout=0.1, context_parallel_size=1, softmax_scale=None
            )
        )
