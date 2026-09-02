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

"""Numeric equivalence tests for the DFlash block draft attention.

The custom two-part (bucketed dense FA + varlen FA, joint-LSE merged)
attention must match a dense fp32 softmax over the explicit staircase mask —
forward and gradients — including the edge cases: empty trunk (anchor 0),
anchor below one chunk (no dense part), and anchors spread over multiple
chunk buckets.
"""

import pytest
import torch

pytestmark = pytest.mark.mcore

from nemo_rl.models.megatron.draft.dflash import (  # noqa: E402
    block_draft_attention,
)
from nemo_rl.models.megatron.draft.utils import (  # noqa: E402
    _deinterleave_qkv,
    _interleave_qkv,
)

requires_gpu_flash = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flash-attn"
)


def _dense_reference(q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len, scale):
    """Per-block fp32 softmax over [trunk[:p]; own block] (bidirectional)."""
    num_blocks, block_width, num_q_heads, _ = q.shape
    num_kv_heads = k_own.shape[2]
    group = num_q_heads // num_kv_heads
    outs = []
    for block in range(num_blocks):
        row = int(block_row[block])
        prefix = int(vis_len[block])
        keys = torch.cat([trunk_k[row, :prefix], k_own[block]], dim=0).float()
        vals = torch.cat([trunk_v[row, :prefix], v_own[block]], dim=0).float()
        keys = keys.repeat_interleave(group, dim=1)
        vals = vals.repeat_interleave(group, dim=1)
        scores = torch.einsum("whd,lhd->hwl", q[block].float(), keys) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hwl,lhd->whd", probs, vals))
    return torch.stack(outs)


def _make_inputs(vis_len_values, *, block_width=4, seq_len=64, batch=2, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    vis_len = torch.tensor(vis_len_values, device=device, dtype=torch.int64)
    num_blocks = vis_len.shape[0]
    block_row = torch.arange(num_blocks, device=device) % batch

    def leaf(*shape):
        return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)

    q = leaf(num_blocks, block_width, num_q_heads, head_dim)
    k_own = leaf(num_blocks, block_width, num_kv_heads, head_dim)
    v_own = leaf(num_blocks, block_width, num_kv_heads, head_dim)
    trunk_k = leaf(batch, seq_len, num_kv_heads, head_dim)
    trunk_v = leaf(batch, seq_len, num_kv_heads, head_dim)
    return q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len


@requires_gpu_flash
@pytest.mark.parametrize(
    "vis_len_values,chunk",
    [
        # Multi-bucket: full chunks + partial remainders (chunk=16, S=64).
        ([5, 17, 33, 48, 63, 21], 16),
        # Everything below one chunk: dense part A never runs.
        ([0, 1, 7, 15], 16),
        # Exactly on chunk boundaries: empty partial remainders.
        ([16, 32, 48, 0], 16),
    ],
)
def test_block_attention_matches_dense_reference(vis_len_values, chunk):
    q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len = _make_inputs(vis_len_values)
    scale = q.shape[-1] ** -0.5

    out = block_draft_attention(
        q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len, chunk=chunk
    )
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    grads = [t.grad.clone() for t in (q, k_own, v_own, trunk_k, trunk_v)]
    for t in (q, k_own, v_own, trunk_k, trunk_v):
        t.grad = None

    ref_inputs = [
        t.detach().clone().requires_grad_(True)
        for t in (q, k_own, v_own, trunk_k, trunk_v)
    ]
    ref_out = _dense_reference(*ref_inputs, block_row, vis_len, scale)
    ref_out.backward(grad_out.float())
    ref_grads = [t.grad.clone() for t in ref_inputs]

    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    names = ["q", "k_own", "v_own", "trunk_k", "trunk_v"]
    for name, grad, ref_grad in zip(names, grads, ref_grads):
        # The reference leaves are bf16, so their grads come back bf16;
        # compare values in fp32.
        torch.testing.assert_close(
            grad.float(),
            ref_grad.float(),
            atol=5e-2,
            rtol=5e-2,
            msg=lambda m, name=name: f"grad mismatch in {name}: {m}",
        )


@requires_gpu_flash
def test_block_attention_trunk_isolation_across_rows():
    """A block must only read its own batch row's trunk."""
    q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len = _make_inputs(
        [32, 32], batch=2, seed=3
    )
    out = block_draft_attention(
        q, k_own, v_own, trunk_k, trunk_v, block_row, vis_len, chunk=16
    )
    # Perturb row 1's trunk; block 0 (row 0) must be unaffected.
    trunk_k2 = trunk_k.detach().clone()
    trunk_k2[1] += 1.0
    out2 = block_draft_attention(
        q,
        k_own,
        v_own,
        trunk_k2.requires_grad_(True),
        trunk_v,
        block_row,
        vis_len,
        chunk=16,
    )
    torch.testing.assert_close(out[0], out2[0])
    assert not torch.allclose(out[1], out2[1])


class _SliceConfig:
    """Minimal config stub for the interleave helpers."""

    def __init__(self, heads, groups, head_dim, hidden):
        self.num_attention_heads = heads
        self.num_query_groups = groups
        self.kv_channels = head_dim
        self.hidden_size = hidden


def test_trunk_kv_weight_slice_matches_deinterleave():
    """The grouped-view K/V slice used by _project_trunk_kv equals the
    canonical de-interleave of Megatron's fused qkv layout."""
    heads, groups, head_dim, hidden = 8, 2, 16, 64
    config = _SliceConfig(heads, groups, head_dim, hidden)
    q_w = torch.randn(heads * head_dim, hidden)
    k_w = torch.randn(groups * head_dim, hidden)
    v_w = torch.randn(groups * head_dim, hidden)
    fused = _interleave_qkv(q_w, k_w, v_w, config)

    # The slice arithmetic from DFlashDraftModel._project_trunk_kv.
    heads_per_group = heads // groups
    grouped = fused.view(groups, (heads_per_group + 2) * head_dim, -1)
    k_slice = grouped[
        :, heads_per_group * head_dim : (heads_per_group + 1) * head_dim
    ].reshape(groups * head_dim, -1)
    v_slice = grouped[:, (heads_per_group + 1) * head_dim :].reshape(
        groups * head_dim, -1
    )

    q_ref, k_ref, v_ref = _deinterleave_qkv(fused, config)
    torch.testing.assert_close(k_slice, k_ref)
    torch.testing.assert_close(v_slice, v_ref)
    torch.testing.assert_close(q_ref, q_w)


@requires_gpu_flash
def test_block_draft_model_forward_backward(tmp_path):
    """End-to-end module smoke: taps -> trunk KV -> block stream -> logits."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel

    created_process_group = False
    try:
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{tmp_path / 'mcore_pg_init'}",
            )
            created_process_group = True
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(7)

        hidden, target_hidden, vocab = 64, 96, 128
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden,
            ffn_hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            normalization="RMSNorm",
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
        )
        config.vocab_size = vocab
        config.draft_vocab_size = vocab
        config.apply_rope_fusion = False
        config.rotary_base = 10000
        config.gradient_accumulation_fusion = False

        gamma = 3
        shared_kwargs = dict(
            config=config,
            gamma=gamma,
            mask_token_id=vocab - 1,
            num_aux_hidden_states=3,
            target_hidden_size=target_hidden,
            trunk_chunk=8,
        )
        model = DFlashDraftModel(**shared_kwargs).cuda()

        seq_len, batch = 32, 2
        taps = torch.randn(
            seq_len, batch, 3 * target_hidden, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        embeds = torch.randn(
            seq_len, batch, hidden, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        anchors = torch.tensor([[5, 10, 20], [0, 15, 29]], device="cuda")
        anchor_valid = torch.ones_like(anchors, dtype=torch.bool)
        # The policy's LM head and frozen mask-token embed row, passed
        # detached (the draft owns neither; official DFlash contract).
        lm_head_weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
        mask_embedding = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

        out = model(
            taps=taps,
            input_embeds=embeds,
            anchors=anchors,
            anchor_valid=anchor_valid,
            lm_head_weight=lm_head_weight,
            mask_embedding=mask_embedding,
        )
        logits = out
        expected_width = gamma + 1
        loss = logits.float().sum()
        assert logits.shape == (batch, anchors.shape[1], expected_width, vocab)
        loss.backward()

        assert torch.isfinite(taps.grad).all()
        assert torch.isfinite(embeds.grad).all()
        assert torch.isfinite(model.fc.weight.grad).all()
        qkv_grad = model.decoder.layers[0].self_attention.linear_qkv.weight.grad
        assert qkv_grad is not None and torch.isfinite(qkv_grad).all()
    finally:
        parallel_state.destroy_model_parallel()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()
