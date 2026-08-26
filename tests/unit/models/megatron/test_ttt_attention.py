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
accumulation into the stashed pass-1 KV). Packed (THD) inputs must match the
per-subsequence references; the CP=2 ring is covered in
test_ttt_packing_cp.py.
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


def _attn_config() -> SimpleNamespace:
    return SimpleNamespace(softmax_scale=None)


def apply_two_part_bshd(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Drive the THD kernel entry with batched [B, S, ...] tensors.

    Flattens the batch into B equal-length packed subsequences (the same
    layout TTTDraftCoreAttention builds for unpacked input) and reshapes the
    output back; autograd flows through the views to the [B, S, ...] leaves.
    """
    batch, seqlen = q.shape[0], q.shape[1]
    cu_seqlens = torch.arange(0, batch + 1, device=q.device, dtype=torch.int32) * seqlen
    out = TwoPartTTTAttention.apply(
        q.reshape(batch * seqlen, *q.shape[2:]),
        k1.reshape(batch * seqlen, *k1.shape[2:]),
        v1.reshape(batch * seqlen, *v1.shape[2:]),
        kb.reshape(batch * seqlen, *kb.shape[2:]) if kb is not None else None,
        vb.reshape(batch * seqlen, *vb.shape[2:]) if vb is not None else None,
        cu_seqlens,
        seqlen,
        softmax_scale,
        None,
    )
    return out.view(batch, seqlen, *out.shape[1:])


def dense_two_part_reference(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Joint softmax over causal trunk + diagonal branch keys, materialized.

    Shapes: q ``[B,S,Hq,D]``, k1/v1 ``[B,S,Hkv,D]``, kb/vb ``[B,S,Hkv,P,D]``
    (P may be 0 for a pure causal pass).
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

    out = apply_two_part_bshd(q, k1, v1, kb, vb, softmax_scale)
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
def test_two_part_function_packed_matches_per_sequence():
    """A packed row of unequal subsequences vs each subsequence run alone.

    Both the varlen trunk (causal restart at cu boundaries) and the branch
    diagonals (purely index-local, so untouched by packing) must reproduce
    the per-subsequence dense oracle in outputs and all gradients.
    """
    torch.manual_seed(3)
    lengths = [48, 80, 32]
    num_q_heads, num_kv_heads, head_dim, num_branch = 8, 4, 64, 2
    softmax_scale = head_dim**-0.5
    total = sum(lengths)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )

    def make(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

    q = make(total, num_q_heads, head_dim)
    k1 = make(total, num_kv_heads, head_dim)
    v1 = make(total, num_kv_heads, head_dim)
    kb = make(total, num_kv_heads, num_branch, head_dim)
    vb = make(total, num_kv_heads, num_branch, head_dim)

    out = TwoPartTTTAttention.apply(
        q, k1, v1, kb, vb, cu, max(lengths), softmax_scale, None
    )
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k1, v1, kb, vb), dout)

    refs = [t.detach().float().requires_grad_() for t in (q, k1, v1, kb, vb)]
    ref_grads = [torch.zeros_like(r) for r in refs]
    for seq_idx, length in enumerate(lengths):
        start, end = int(cu[seq_idx]), int(cu[seq_idx + 1])
        seq_refs = [r[start:end].unsqueeze(0) for r in refs]
        out_ref = dense_two_part_reference(*seq_refs, softmax_scale)
        _assert_close(out[start:end], out_ref[0], f"out seq {seq_idx}")
        seq_grads = torch.autograd.grad(
            out_ref, refs, dout[start:end].unsqueeze(0).float(), retain_graph=True
        )
        for acc, g in zip(ref_grads, seq_grads):
            acc += g
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

    module = TTTDraftCoreAttention(_attn_config())

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
def test_module_packed_multi_pass_matches_unpacked():
    """Packed THD multi-pass module runs == per-subsequence unpacked runs.

    Feeds the module the exact tensors MCore hands it in THD mode ([T, H, D]
    with packed_seq_params) and checks outputs AND gradients against separate
    unpacked runs of each subsequence.
    """
    torch.manual_seed(5)
    num_passes = 3
    lengths = [40, 72]
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    total = sum(lengths)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=cu,
        max_seqlen_q=max(lengths),
    )

    qs = [
        torch.randn(
            total, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]
    ks = [
        torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]
    vs = [
        torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for _ in range(num_passes)
    ]

    module = TTTDraftCoreAttention(_attn_config())
    outs = []
    for ttt_pass in range(1, num_passes + 1):
        module.begin_pass(ttt_pass)
        outs.append(
            module(
                qs[ttt_pass - 1],
                ks[ttt_pass - 1],
                vs[ttt_pass - 1],
                None,
                packed_seq_params=packed_seq_params,
            )
        )
    module.reset()
    assert outs[0].shape == (total, num_q_heads * head_dim)
    douts = [torch.randn_like(o) for o in outs]
    grads = torch.autograd.grad(outs, qs + ks + vs, douts)

    # Reference: run each subsequence through its own module, unpacked
    # ([L, 1, H, D] sbhd), and scatter outputs/grads back into packed rows.
    outs_ref = [torch.zeros_like(o) for o in outs]
    grads_ref = [torch.zeros_like(t) for t in qs + ks + vs]
    for seq_idx, length in enumerate(lengths):
        start, end = int(cu[seq_idx]), int(cu[seq_idx + 1])
        seq_leaves = [
            t[start:end].detach().clone().requires_grad_() for t in qs + ks + vs
        ]
        seq_qs = seq_leaves[:num_passes]
        seq_ks = seq_leaves[num_passes : 2 * num_passes]
        seq_vs = seq_leaves[2 * num_passes :]
        ref_module = TTTDraftCoreAttention(_attn_config())
        seq_outs = []
        for ttt_pass in range(1, num_passes + 1):
            ref_module.begin_pass(ttt_pass)
            seq_outs.append(
                ref_module(
                    seq_qs[ttt_pass - 1].unsqueeze(1),
                    seq_ks[ttt_pass - 1].unsqueeze(1),
                    seq_vs[ttt_pass - 1].unsqueeze(1),
                    None,
                )
            )
        ref_module.reset()
        seq_grads = torch.autograd.grad(
            seq_outs,
            seq_leaves,
            [d[start:end].unsqueeze(1) for d in douts],
        )
        for out_ref, seq_out in zip(outs_ref, seq_outs):
            out_ref[start:end] = seq_out.squeeze(1)
        for acc, g in zip(grads_ref, seq_grads):
            acc[start:end] = g

    for out, out_ref in zip(outs, outs_ref):
        _assert_close(out, out_ref, "packed pass output")
    names = (
        [f"dq{i}" for i in range(num_passes)]
        + [f"dk{i}" for i in range(num_passes)]
        + [f"dv{i}" for i in range(num_passes)]
    )
    for grad, grad_ref, name in zip(grads, grads_ref, names):
        _assert_close(grad, grad_ref, f"packed {name}")


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
    module = TTTDraftCoreAttention(_attn_config())
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
    module = TTTDraftCoreAttention(_attn_config())
    q = torch.randn(4, 1, 2, 8)

    with pytest.raises(RuntimeError, match="begin_pass"):
        module(q, q, q, None)

    module.begin_pass(1)
    with pytest.raises(RuntimeError, match="in order"):
        module.begin_pass(3)

    with pytest.raises(NotImplementedError, match="thd"):
        module(q, q, q, None, packed_seq_params=SimpleNamespace(qkv_format="bshd"))

    # CP without packing has no zigzag metadata to ring over.
    module.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2))
    with pytest.raises(NotImplementedError, match="sequence packing"):
        module(q, q, q, None)

    with pytest.raises(ValueError, match="attention_mask"):
        module(q, q, q, torch.ones(1))

    with pytest.raises(ValueError, match="attention_bias"):
        module(q, q, q, None, attention_bias=torch.ones(1))


def test_module_rejects_attention_dropout():
    with pytest.raises(ValueError, match="dropout"):
        TTTDraftCoreAttention(
            SimpleNamespace(softmax_scale=None, attention_dropout=0.1)
        )
