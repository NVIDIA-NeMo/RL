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

"""Numeric equivalence tests for the DFlash/DSpark block draft attention.

The custom two-part (bucketed dense FA + varlen FA, joint-LSE merged)
attention must match a dense fp32 softmax over the explicit staircase mask —
forward and gradients — including the edge cases: empty trunk (anchor 0),
anchor below one chunk (no dense part), anchors spread over multiple chunk
buckets, unequal packed subsequences, and the CP=2 zigzag ring.
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.mcore

from nemo_rl.models.megatron.draft.dflash import BlockDraftAttention  # noqa: E402
from nemo_rl.models.megatron.draft.utils import (  # noqa: E402
    _deinterleave_qkv,
    _interleave_qkv,
)


def block_draft_attention(*args, chunk, window=0, cp_group=None):
    """Test-local wrapper over ``BlockDraftAttention.apply`` (defaulted scale).

    ``args`` = (q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu).
    """
    scale = args[0].shape[-1] ** -0.5
    return BlockDraftAttention.apply(*args, chunk, window, scale, cp_group)


requires_gpu_flash = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flash-attn"
)
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires 2 GPUs"
)
requires_4_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="requires 4 GPUs"
)

# CP sizes for the ring-parity spawn tests. CP=2 is degenerate in one way —
# every rank is simultaneously the first AND last rank — so only CP=4
# exercises the interior-rank paths: halo neighbors on both sides, ring steps
# with genuinely earlier AND later chunks in flight. The paired chunk value
# keeps the dense-bucket path alive at CP=4 (the local segments shrink below
# the CP=2 chunk).
CP_RING_CASES = [
    pytest.param(2, 8, id="cp2"),
    pytest.param(4, 2, id="cp4", marks=requires_4_gpus),
]
CP_SIZES = [
    pytest.param(2, id="cp2"),
    pytest.param(4, id="cp4", marks=requires_4_gpus),
]


def _dense_reference(
    q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu, scale, window=0
):
    """Per-block fp32 softmax over [trunk[seq, :p]; own block] (bidirectional).

    ``window > 0`` restricts each query to the causal band
    ``0 <= q_pos - k_pos <= window - 1`` (vLLM resolves sliding draft layers
    as causal), which also makes block-internal attention lower-triangular.
    """
    num_blocks = q.shape[0]
    block_width = q.shape[1]
    num_q_heads = q.shape[2]
    num_kv_heads = k_own.shape[2]
    group = num_q_heads // num_kv_heads
    outs = []
    for block in range(num_blocks):
        start = int(cu[int(block_seq[block])])
        prefix = int(vis_len[block])
        keys = torch.cat([trunk_k[start : start + prefix], k_own[block]], dim=0).float()
        vals = torch.cat([trunk_v[start : start + prefix], v_own[block]], dim=0).float()
        keys = keys.repeat_interleave(group, dim=1)
        vals = vals.repeat_interleave(group, dim=1)
        scores = torch.einsum("whd,lhd->hwl", q[block].float(), keys) * scale
        if window:
            offsets = torch.arange(block_width, device=q.device)
            q_pos = prefix + offsets
            k_pos = torch.cat([torch.arange(prefix, device=q.device), q_pos])
            dist_qk = q_pos[:, None] - k_pos[None, :]
            visible = (dist_qk >= 0) & (dist_qk <= window - 1)
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hwl,lhd->whd", probs, vals))
    return torch.stack(outs)


def _make_inputs(
    vis_len_values, *, block_width=4, seq_lens=(64, 64), block_seq_values=None, seed=0
):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    vis_len = torch.tensor(vis_len_values, device=device, dtype=torch.int64)
    num_blocks = vis_len.shape[0]
    if block_seq_values is None:
        block_seq = torch.arange(num_blocks, device=device) % len(seq_lens)
    else:
        block_seq = torch.tensor(block_seq_values, device=device, dtype=torch.int64)
    cu = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.long)
    cu[1:] = torch.cumsum(torch.tensor(seq_lens, device=device), dim=0)

    def leaf(*shape):
        return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)

    q = leaf(num_blocks, block_width, num_q_heads, head_dim)
    k_own = leaf(num_blocks, block_width, num_kv_heads, head_dim)
    v_own = leaf(num_blocks, block_width, num_kv_heads, head_dim)
    trunk_k = leaf(int(cu[-1]), num_kv_heads, head_dim)
    trunk_v = leaf(int(cu[-1]), num_kv_heads, head_dim)
    return q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu


def _compare_against_dense(inputs, chunk, cp_group=None, window=0):
    q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu = inputs
    scale = q.shape[-1] ** -0.5

    out = block_draft_attention(
        q,
        k_own,
        v_own,
        trunk_k,
        trunk_v,
        block_seq,
        vis_len,
        cu,
        chunk=chunk,
        window=window,
        cp_group=cp_group,
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
    ref_out = _dense_reference(*ref_inputs, block_seq, vis_len, cu, scale, window)
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
    _compare_against_dense(_make_inputs(vis_len_values), chunk)


@requires_gpu_flash
def test_block_attention_packed_unequal_subsequences():
    """A THD pack of unequal subsequences; anchors near each seq's end."""
    _compare_against_dense(
        _make_inputs(
            [0, 7, 23, 3, 39, 12],
            seq_lens=(24, 40),
            block_seq_values=[0, 0, 0, 1, 1, 1],
            seed=5,
        ),
        chunk=16,
    )


@requires_gpu_flash
@pytest.mark.parametrize(
    "window,vis_len_values",
    [
        # Anchors inside, straddling, and far beyond the window.
        (24, [0, 5, 23, 24, 30, 47, 63]),
        # Window below one chunk.
        (8, [7, 32, 63]),
        # Degenerate window < block width: block-internal pairs clip too.
        (3, [16, 40]),
    ],
)
def test_block_attention_sliding_matches_dense_reference(window, vis_len_values):
    _compare_against_dense(
        _make_inputs(vis_len_values, seed=11), chunk=16, window=window
    )


@requires_gpu_flash
def test_block_attention_sliding_packed_unequal_subsequences():
    _compare_against_dense(
        _make_inputs(
            [0, 7, 23, 3, 39, 12],
            seq_lens=(24, 40),
            block_seq_values=[0, 0, 0, 1, 1, 1],
            seed=6,
        ),
        chunk=16,
        window=16,
    )


@requires_gpu_flash
def test_block_attention_sliding_blocks_are_causal():
    """Sliding layers are causal: a later own key must not affect earlier slots."""
    inputs = _make_inputs([16, 40], seed=13)
    q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu = inputs
    out = block_draft_attention(*inputs, chunk=16, window=24)
    k_own2 = k_own.detach().clone()
    k_own2[:, -1] += 1.0
    out2 = block_draft_attention(
        q,
        k_own2.requires_grad_(True),
        v_own,
        trunk_k,
        trunk_v,
        block_seq,
        vis_len,
        cu,
        chunk=16,
        window=24,
    )
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])
    assert not torch.allclose(out[:, -1], out2[:, -1])


@requires_gpu_flash
def test_block_attention_trunk_isolation_across_rows():
    """A block must only read its own subsequence's trunk."""
    inputs = _make_inputs([32, 32], block_seq_values=[0, 1], seed=3)
    q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu = inputs
    out = block_draft_attention(
        q, k_own, v_own, trunk_k, trunk_v, block_seq, vis_len, cu, chunk=16
    )
    # Perturb seq 1's trunk; block 0 (seq 0) must be unaffected.
    trunk_k2 = trunk_k.detach().clone()
    trunk_k2[int(cu[1]) :] += 1.0
    out2 = block_draft_attention(
        q,
        k_own,
        v_own,
        trunk_k2.requires_grad_(True),
        trunk_v,
        block_seq,
        vis_len,
        cu,
        chunk=16,
    )
    torch.testing.assert_close(out[0], out2[0])
    assert not torch.allclose(out[1], out2[1])


def _run_block_attention_cp_rank(
    rank: int, world_size: int, init_file: str, chunk: int, window: int = 0
) -> None:
    """CP ring vs the single-rank global run of the SAME inputs.

    Blocks are assigned to the rank owning their anchor's zigzag chunk. The
    ring's outputs must match the global run's rows for the owned blocks, and
    — because dK/dV travel one full loop back to their owner — the local
    trunk-shard gradients must equal the global gradient's zigzag slice
    (contributions from EVERY rank's blocks included).

    ``window > 0`` runs the sliding path instead: trunk K/V is all-gathered
    to global order for the one windowed call, and trunk grads come home via
    reduce-scatter, so the same zigzag-slice comparison applies.
    """
    import torch.distributed as dist

    from nemo_rl.algorithms.loss.utils import packed_zigzag_token_coords

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    cp_group = dist.new_group(list(range(world_size)))

    seq_lens = (16, 24)  # multiples of 2*cp for cp in {2, 4}
    vis_values = [0, 5, 9, 15, 3, 11, 17, 23]
    seq_values = [0, 0, 0, 0, 1, 1, 1, 1]
    inputs = _make_inputs(
        vis_values, seq_lens=seq_lens, block_seq_values=seq_values, seed=17
    )
    q_g, k_own_g, v_own_g, trunk_k_g, trunk_v_g, block_seq, vis_len, cu = inputs

    # ---- Global reference (identical on every rank; same seed).
    out_ref = block_draft_attention(
        q_g,
        k_own_g,
        v_own_g,
        trunk_k_g,
        trunk_v_g,
        block_seq,
        vis_len,
        cu,
        chunk=chunk,
        window=window,
    )
    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    ref_grads = {
        "q": q_g.grad.clone(),
        "k_own": k_own_g.grad.clone(),
        "v_own": v_own_g.grad.clone(),
        "trunk_k": trunk_k_g.grad.clone(),
        "trunk_v": trunk_v_g.grad.clone(),
    }

    # ---- CP ring: zigzag trunk shard + owned blocks.
    cu_local = cu // world_size
    seq_index, pos = packed_zigzag_token_coords(cu.cpu(), rank, world_size)
    global_row = (cu[:-1].cpu()[seq_index] + pos).cuda()
    half = ((cu_local[1:] - cu_local[:-1]) // 2)[block_seq]
    chunk_idx = vis_len // half
    owner = torch.minimum(chunk_idx, 2 * world_size - 1 - chunk_idx)
    mine = owner == rank
    assert int(mine.sum()) > 0, f"rank {rank} owns no blocks; adjust the test data"

    def local_leaf(t, index):
        return t.detach()[index].clone().requires_grad_(True)

    q_l = local_leaf(q_g, mine)
    k_own_l = local_leaf(k_own_g, mine)
    v_own_l = local_leaf(v_own_g, mine)
    trunk_k_l = local_leaf(trunk_k_g, global_row)
    trunk_v_l = local_leaf(trunk_v_g, global_row)

    out_local = block_draft_attention(
        q_l,
        k_own_l,
        v_own_l,
        trunk_k_l,
        trunk_v_l,
        block_seq[mine],
        vis_len[mine],
        cu_local,
        chunk=chunk,
        window=window,
        cp_group=cp_group,
    )
    torch.testing.assert_close(
        out_local.float(), out_ref.detach()[mine].float(), atol=2e-2, rtol=2e-2
    )
    out_local.backward(grad_out[mine])

    torch.testing.assert_close(
        q_l.grad.float(), ref_grads["q"][mine].float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        k_own_l.grad.float(), ref_grads["k_own"][mine].float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        v_own_l.grad.float(), ref_grads["v_own"][mine].float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        trunk_k_l.grad.float(),
        ref_grads["trunk_k"][global_row].float(),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        trunk_v_l.grad.float(),
        ref_grads["trunk_v"][global_row].float(),
        atol=5e-2,
        rtol=5e-2,
    )

    dist.barrier()
    dist.destroy_process_group()


@requires_2_gpus
@requires_gpu_flash
@pytest.mark.parametrize("world_size,chunk", CP_RING_CASES)
def test_block_attention_cp_matches_single_rank(tmp_path, world_size, chunk):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / f"block_cp{world_size}_init")
    procs = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_run_block_attention_cp_rank,
            args=(rank, world_size, init_file, chunk),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"


# window=5 straddles zigzag chunk boundaries (halves are 4/6 at cp2, 2/3 at
# cp4); window=64 covers every trunk position, isolating the causal own-block
# handling from the windowing.
CP_SLIDING_CASES = [
    pytest.param(2, 5, id="cp2-w5"),
    pytest.param(2, 64, id="cp2-w64"),
    pytest.param(4, 5, id="cp4-w5", marks=requires_4_gpus),
]


@requires_2_gpus
@requires_gpu_flash
@pytest.mark.parametrize("world_size,window", CP_SLIDING_CASES)
def test_block_attention_sliding_cp_matches_single_rank(tmp_path, world_size, window):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / f"block_swa_cp{world_size}_w{window}_init")
    procs = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_run_block_attention_cp_rank,
            args=(rank, world_size, init_file, 8, window),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"


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
@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("layer_windows", [None, [16, 0]], ids=["full", "sliding"])
def test_block_draft_model_forward_backward(method, layer_windows, tmp_path):
    """End-to-end module smoke: taps -> trunk KV -> block stream -> logits."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel
    from nemo_rl.models.megatron.draft.dspark import DSparkDraftModel

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
            layer_windows=layer_windows,
        )
        if method == "dflash":
            model = DFlashDraftModel(**shared_kwargs).cuda()
        else:
            model = DSparkDraftModel(markov_rank=8, **shared_kwargs).cuda()

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

        method_kwargs = {}
        if method == "dspark":
            method_kwargs["input_ids"] = torch.randint(
                0, vocab, (batch, seq_len), device="cuda"
            )
        out = model(
            taps=taps,
            input_embeds=embeds,
            anchors=anchors,
            anchor_valid=anchor_valid,
            lm_head_weight=lm_head_weight,
            mask_embedding=mask_embedding,
            **method_kwargs,
        )
        if method == "dflash":
            logits = out
            expected_width = gamma + 1
            loss = logits.float().sum()
        else:
            # Markov-biased logits + confidence prediction, both from forward.
            logits, confidence_pred = out
            expected_width = gamma
            assert confidence_pred.shape == (batch, anchors.shape[1], gamma)
            loss = logits.float().sum() + confidence_pred.sum()
        assert logits.shape == (batch, anchors.shape[1], expected_width, vocab)
        loss.backward()

        assert torch.isfinite(taps.grad).all()
        assert torch.isfinite(embeds.grad).all()
        assert torch.isfinite(model.fc.weight.grad).all()
        qkv_grad = model.decoder.layers[0].self_attention.linear_qkv.weight.grad
        assert qkv_grad is not None and torch.isfinite(qkv_grad).all()
        if method == "dspark":
            assert model.markov_w1.weight.grad is not None
            assert model.markov_w2.weight.grad is not None
            assert model.confidence_head.weight.grad is not None
    finally:
        parallel_state.destroy_model_parallel()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()


def test_resolve_layer_windows():
    """Checkpoint layer_types/sliding_window -> per-layer window list."""
    from nemo_rl.models.megatron.draft.utils import _resolve_layer_windows

    ckpt_35b = {
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "sliding_window": 4096,
    }
    assert _resolve_layer_windows(ckpt_35b, 6) == [4096] * 5 + [0]
    assert _resolve_layer_windows({}, 3) == [0, 0, 0]
    all_full = {"layer_types": ["full_attention"] * 2, "sliding_window": None}
    assert _resolve_layer_windows(all_full, 2) == [0, 0]
    with pytest.raises(ValueError):
        _resolve_layer_windows({"layer_types": ["sliding_attention"]}, 1)
    with pytest.raises(NotImplementedError):
        _resolve_layer_windows(
            {"layer_types": ["linear_attention"], "sliding_window": 8}, 1
        )


def _make_draft_config():
    from megatron.core.transformer import TransformerConfig

    hidden, vocab = 64, 128
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
    return config


def _make_block_model(method, config, gamma=3, target_hidden=96, layer_windows=None):
    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel
    from nemo_rl.models.megatron.draft.dspark import DSparkDraftModel

    shared_kwargs = dict(
        config=config,
        gamma=gamma,
        mask_token_id=config.vocab_size - 1,
        num_aux_hidden_states=3,
        target_hidden_size=target_hidden,
        trunk_chunk=8,
        layer_windows=layer_windows,
    )
    if method == "dflash":
        return DFlashDraftModel(**shared_kwargs).cuda()
    return DSparkDraftModel(markov_rank=8, **shared_kwargs).cuda()


@requires_gpu_flash
@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("layer_windows", [None, [16, 0]], ids=["full", "sliding"])
def test_block_draft_model_packed_matches_unpacked(method, layer_windows, tmp_path):
    """Packed THD forward (flat blocks) == unpacked forward on the same data."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

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

        config = _make_draft_config()
        hidden, vocab, target_hidden, gamma = 64, 128, 96, 3
        model = _make_block_model(
            method, config, gamma=gamma, layer_windows=layer_windows
        )

        seq_len, batch = 32, 2
        taps = torch.randn(
            seq_len, batch, 3 * target_hidden, device="cuda", dtype=torch.bfloat16
        )
        embeds = torch.randn(
            seq_len, batch, hidden, device="cuda", dtype=torch.bfloat16
        )
        anchors = torch.tensor([[5, 10, 20], [0, 15, 29]], device="cuda")
        anchor_valid = torch.ones_like(anchors, dtype=torch.bool)
        lm_head_weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
        mask_embedding = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        input_ids = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        method_kwargs = {"input_ids": input_ids} if method == "dspark" else {}
        with torch.no_grad():
            out_ref = model(
                taps=taps,
                input_embeds=embeds,
                anchors=anchors,
                anchor_valid=anchor_valid,
                lm_head_weight=lm_head_weight,
                mask_embedding=mask_embedding,
                **method_kwargs,
            )

        # Pack the two equal-length rows batch-major (the layout the unpacked
        # path normalizes to) and rerun with flat block coords.
        cu = torch.tensor([0, seq_len, 2 * seq_len], device="cuda", dtype=torch.int32)
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
        )
        taps_packed = taps.permute(1, 0, 2).reshape(batch * seq_len, 1, -1)
        embeds_packed = embeds.permute(1, 0, 2).reshape(batch * seq_len, 1, -1)
        block_seq = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda")
        anchors_flat = anchors.reshape(-1)
        with torch.no_grad():
            out_packed = model(
                taps=taps_packed,
                input_embeds=embeds_packed,
                anchors=anchors_flat,
                anchor_valid=anchor_valid.reshape(-1),
                lm_head_weight=lm_head_weight,
                mask_embedding=mask_embedding,
                packed_seq_params=psp,
                block_seq_idx=block_seq,
                **method_kwargs,
            )

        if method == "dflash":
            torch.testing.assert_close(
                out_packed.float(), out_ref.reshape(*out_packed.shape).float()
            )
        else:
            logits_ref, conf_ref = out_ref
            logits_packed, conf_packed = out_packed
            torch.testing.assert_close(
                logits_packed.float(), logits_ref.reshape(*logits_packed.shape).float()
            )
            torch.testing.assert_close(
                conf_packed, conf_ref.reshape(*conf_packed.shape)
            )
    finally:
        parallel_state.destroy_model_parallel()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _run_block_model_cp_rank(rank: int, world_size: int, init_file: str) -> None:
    """CP packed model + DSpark loss vs the CP=1 global run of the SAME data.

    Phase 1 initializes model parallel with CP=1 (every rank computes the
    identical global packed forward + loss over ALL blocks). Phase 2
    re-initializes with CP=world_size, rebuilds the model from the same
    seeds, feeds each rank its zigzag shard and OWNED blocks, and checks the
    local logits/confidence rows and the CP-summed loss against the global
    reference — covering the trunk ring, the anchor-owner block placement,
    the loss's successor-halo teacher gather, and the CP metric reduction.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from nemo_rl.algorithms.loss.loss_functions import DSparkBlockLossFn
    from nemo_rl.algorithms.loss.utils import (
        compute_block_draft_slot_valid_counts,
        packed_zigzag_token_coords,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )

    hidden, vocab, target_hidden, gamma = 64, 128, 96, 3
    # Multiples of 2*cp for cp in {2, 4}; at cp=4 the smallest chunk (16/8=2)
    # still covers the gamma-1=2 successor halo exactly.
    lengths = [16, 24]
    seq_len = max(lengths)
    total = sum(lengths)
    cu_cpu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(), dtype=torch.int32
    )
    torch.manual_seed(11)
    taps_g = torch.randn(total, 1, 3 * target_hidden, dtype=torch.bfloat16).cuda()
    embeds_g = torch.randn(total, 1, hidden, dtype=torch.bfloat16).cuda()
    teacher_g = torch.randn(total, vocab, dtype=torch.bfloat16).cuda()
    lm_head_weight = torch.randn(vocab, hidden, dtype=torch.bfloat16).cuda()
    mask_embedding = torch.randn(hidden, dtype=torch.bfloat16).cuda()
    input_ids = torch.randint(0, vocab, (2, seq_len)).cuda()
    token_mask = torch.ones(2, seq_len).cuda()
    token_mask[0, lengths[0] :] = 0.0  # seq 0 is shorter than the padded S
    sample_mask = torch.ones(2).cuda()

    anchors = torch.tensor([[0, 5, 9, 15], [3, 11, 17, 23]]).cuda()
    anchor_valid = torch.ones_like(anchors, dtype=torch.bool)
    block_seq_all = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).cuda()
    anchors_all = anchors.reshape(-1)
    valid_all = anchor_valid.reshape(-1)
    counts = compute_block_draft_slot_valid_counts(
        token_mask, sample_mask, anchors, anchor_valid, gamma=gamma
    )

    def make_psp(cu):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
        )

    def run_loss(logits, confidence, seq_idx, anchors_f, valid_f, cu_local, cp_group):
        data = {
            "draft_anchor_positions": anchors_f,
            "draft_anchor_valid": valid_f,
            "draft_block_seq_idx": seq_idx,
            "draft_packed_local_cu_seqlens": cu_local,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "input_ids": input_ids,
            "draft_confidence_pred": confidence,
        }
        loss_fn = DSparkBlockLossFn(context_parallel_group=cp_group)
        teacher_local = (
            teacher_g if cp_group is None else teacher_g[global_row]
        ).unsqueeze(0)
        return loss_fn(
            teacher_logits=teacher_local,
            student_block_logits=logits,
            data=data,
            global_valid_seqs=None,
            global_valid_toks=torch.tensor(1.0).cuda(),
            global_draft_pass_counts=counts.cuda(),
        )

    # ---- Phase 1: CP=1 (DP=2) global reference, identical on both ranks.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(7)
    config = _make_draft_config()
    model = _make_block_model("dspark", config, gamma=gamma)
    global_row = None
    with torch.no_grad():
        logits_ref, conf_ref = model(
            taps=taps_g,
            input_embeds=embeds_g,
            anchors=anchors_all,
            anchor_valid=valid_all,
            lm_head_weight=lm_head_weight,
            mask_embedding=mask_embedding,
            input_ids=input_ids,
            packed_seq_params=make_psp(cu_cpu.cuda()),
            block_seq_idx=block_seq_all,
        )
        loss_ref, metrics_ref = run_loss(
            logits_ref,
            conf_ref,
            block_seq_all,
            anchors_all,
            valid_all,
            cu_cpu.cuda().long(),
            None,
        )
    del model
    dist.barrier()

    # ---- Phase 2: CP=2, same seeds -> same weights, zigzag-local inputs.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = cp_group.rank()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(7)
    config = _make_draft_config()
    model = _make_block_model("dspark", config, gamma=gamma)

    seq_index, pos = packed_zigzag_token_coords(cu_cpu, cp_rank, world_size)
    global_row = (cu_cpu[:-1].to(torch.long)[seq_index] + pos).cuda()
    cu_local = (cu_cpu.cuda().long()) // world_size
    half = ((cu_local[1:] - cu_local[:-1]) // 2)[block_seq_all]
    chunk_idx = anchors_all // half
    owner = torch.minimum(chunk_idx, 2 * world_size - 1 - chunk_idx)
    mine = owner == cp_rank
    assert int(mine.sum()) > 0

    with torch.no_grad():
        logits_loc, conf_loc = model(
            taps=taps_g[global_row],
            input_embeds=embeds_g[global_row],
            anchors=anchors_all[mine],
            anchor_valid=valid_all[mine],
            lm_head_weight=lm_head_weight,
            mask_embedding=mask_embedding,
            input_ids=input_ids,
            packed_seq_params=make_psp(cu_cpu.cuda()),
            block_seq_idx=block_seq_all[mine],
        )
        loss_loc, metrics_loc = run_loss(
            logits_loc,
            conf_loc,
            block_seq_all[mine],
            anchors_all[mine],
            valid_all[mine],
            cu_local,
            cp_group,
        )

    torch.testing.assert_close(
        logits_loc.float(), logits_ref[mine].float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        conf_loc.float(), conf_ref[mine].float(), atol=3e-2, rtol=3e-2
    )
    loss_sum = loss_loc.clone()
    dist.all_reduce(loss_sum, group=cp_group)
    torch.testing.assert_close(loss_sum, loss_ref, atol=2e-3, rtol=2e-3)
    # CP-reduced metrics must reproduce the global reference's.
    for key in ("dspark_ce_loss", "dspark_tv_loss", "dspark_confidence_loss"):
        assert abs(metrics_loc[key] - metrics_ref[key]) <= 2e-3 + 2e-3 * abs(
            metrics_ref[key]
        ), key

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@requires_2_gpus
@requires_gpu_flash
@pytest.mark.parametrize("world_size", CP_SIZES)
def test_block_model_cp_matches_global(tmp_path, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / f"block_model_cp{world_size}_init")
    procs = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_run_block_model_cp_rank, args=(rank, world_size, init_file)
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"


@requires_gpu_flash
def test_dspark_hf_checkpoint_roundtrip(tmp_path):
    """export -> safetensors -> load restores every tensor (markov +
    confidence heads included) with no missing/unexpected keys."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from safetensors.torch import save_file

    from nemo_rl.models.megatron.draft.dspark import DSparkDraftModel
    from nemo_rl.models.megatron.draft.utils import (
        export_block_draft_weights_to_hf,
        load_hf_weights_to_block_draft,
    )

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

        model = DSparkDraftModel(
            config=config,
            gamma=3,
            mask_token_id=vocab - 1,
            num_aux_hidden_states=3,
            target_hidden_size=target_hidden,
            trunk_chunk=8,
            markov_rank=8,
        ).cuda()

        hf_state = dict(export_block_draft_weights_to_hf(model))
        for key in (
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        ):
            assert key in hf_state, key
        ckpt_dir = tmp_path / "dspark_ckpt"
        ckpt_dir.mkdir()
        save_file(
            {k: v.contiguous().cpu() for k, v in hf_state.items()},
            str(ckpt_dir / "model.safetensors"),
        )

        reference = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if "_extra_state" not in k
        }
        with torch.no_grad():
            for p in model.parameters():
                p.normal_()
        missing, unexpected = load_hf_weights_to_block_draft(model, str(ckpt_dir))
        missing = [k for k in missing if "_extra_state" not in k]
        unexpected = [k for k in unexpected if "_extra_state" not in k]
        assert not missing and not unexpected, (missing, unexpected)
        restored = model.state_dict()
        for key, ref in reference.items():
            torch.testing.assert_close(restored[key], ref, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()
