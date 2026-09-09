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

"""Sequence packing + context parallelism for the TTT draft path.

Covers the pure helpers (zigzag coords, per-subsequence CP-aware roll), the
CP=2 zigzag-ring two-part attention against a single-rank global reference
(forward AND all gradients, including the dK/dV ring return-to-owner), and
the packed draft cross-entropy loss against the unpacked slice path.
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.algorithms.loss.utils import (
    packed_zigzag_token_coords,
    roll_packed_left_cp,
)

pytestmark = pytest.mark.mcore

requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires 2 GPUs (CP=2)"
)


def _flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def test_packed_zigzag_token_coords_manual():
    """cp=2, two subsequences of 8 and 16 tokens: hand-checked zigzag coords."""
    cu = torch.tensor([0, 8, 24], dtype=torch.int32)

    seq_index, pos = packed_zigzag_token_coords(cu, cp_rank=0, cp_size=2)
    assert seq_index.tolist() == [0] * 4 + [1] * 8
    assert pos.tolist() == [0, 1, 6, 7] + [0, 1, 2, 3, 12, 13, 14, 15]

    seq_index, pos = packed_zigzag_token_coords(cu, cp_rank=1, cp_size=2)
    assert seq_index.tolist() == [0] * 4 + [1] * 8
    assert pos.tolist() == [2, 3, 4, 5] + [4, 5, 6, 7, 8, 9, 10, 11]

    # cp=1 degenerates to the identity layout.
    seq_index, pos = packed_zigzag_token_coords(cu, cp_rank=0, cp_size=1)
    assert seq_index.tolist() == [0] * 8 + [1] * 16
    assert pos.tolist() == list(range(8)) + list(range(16))

    # Every rank's coords tile the global row exactly once.
    covered = set()
    for rank in range(2):
        seq_index, pos = packed_zigzag_token_coords(cu, cp_rank=rank, cp_size=2)
        covered.update(
            (int(s), int(p)) for s, p in zip(seq_index.tolist(), pos.tolist())
        )
    assert covered == {(0, p) for p in range(8)} | {(1, p) for p in range(16)}


def _reference_packed_roll(tensor: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """Per-subsequence left shift with zero tail, on an UNSHARDED row."""
    out = torch.zeros_like(tensor)
    for i in range(cu.numel() - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        out[start : end - 1] = tensor[start + 1 : end]
    return out


def test_roll_packed_left_cp_single_rank():
    torch.manual_seed(0)
    cu = torch.tensor([0, 5, 12, 14], dtype=torch.int32)
    x = torch.randn(14, 3)
    rolled = roll_packed_left_cp(x, cu, None)
    torch.testing.assert_close(rolled, _reference_packed_roll(x, cu))


def _run_cp2_rank(rank: int, world_size: int, init_file: str) -> None:
    import torch.distributed as dist

    from nemo_rl.models.megatron.draft.eagle import TwoPartTTTAttention

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    cp_group = dist.group.WORLD
    device = torch.device("cuda", rank)

    # Identical global data on both ranks (seeded CPU generator).
    torch.manual_seed(11)
    lengths = [32, 64]  # multiples of 2*cp
    cu_global = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(), dtype=torch.int32
    ).to(device)
    total = sum(lengths)
    num_q_heads, num_kv_heads, head_dim, num_branch = 4, 2, 32, 2
    softmax_scale = head_dim**-0.5

    def make(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16).to(device)

    q_g = make(total, num_q_heads, head_dim)
    k1_g = make(total, num_kv_heads, head_dim)
    v1_g = make(total, num_kv_heads, head_dim)
    kb_g = make(total, num_kv_heads, num_branch, head_dim)
    vb_g = make(total, num_kv_heads, num_branch, head_dim)
    dout_g = make(total, num_q_heads, head_dim)

    # Local zigzag shard via the coords helper (also validates it end to end:
    # global_row maps each local row to its global packed position).
    seq_index, pos = packed_zigzag_token_coords(cu_global, rank, world_size)
    global_row = (cu_global[:-1].to(torch.long)[seq_index] + pos).to(device)
    cu_local = torch.div(cu_global, world_size, rounding_mode="floor").to(torch.int32)
    max_local = max(lengths) // world_size

    def tol(actual, expected, name):
        actual, expected = actual.float(), expected.float()
        max_diff = (actual - expected).abs().max().item()
        scale = expected.abs().max().clamp(min=1e-6).item()
        assert max_diff <= 3e-2 + 3e-2 * scale, (
            f"rank {rank} {name}: max abs diff {max_diff:.4e} (scale {scale:.4e})"
        )

    # --- roll_packed_left_cp: local roll+boundary exchange == global roll.
    x_g = torch.randn(total, 5, dtype=torch.float32).to(device)
    ref = _reference_packed_roll(x_g, cu_global)
    rolled_local = roll_packed_left_cp(x_g[global_row], cu_local, cp_group)
    torch.testing.assert_close(rolled_local, ref[global_row])

    # --- two-part attention, pass 1 (no branch) and pass >= 2 (branch).
    for has_branch in (False, True):
        leaves_g = [
            t.detach().clone().requires_grad_() for t in (q_g, k1_g, v1_g, kb_g, vb_g)
        ]
        out_ref = TwoPartTTTAttention.apply(
            leaves_g[0],
            leaves_g[1],
            leaves_g[2],
            leaves_g[3] if has_branch else None,
            leaves_g[4] if has_branch else None,
            cu_global,
            max(lengths),
            softmax_scale,
            None,
        )
        ref_inputs = leaves_g[: 5 if has_branch else 3]
        grads_ref = torch.autograd.grad(out_ref, ref_inputs, dout_g)

        leaves_l = [
            t[global_row].detach().clone().requires_grad_()
            for t in (q_g, k1_g, v1_g, kb_g, vb_g)
        ]
        out_local = TwoPartTTTAttention.apply(
            leaves_l[0],
            leaves_l[1],
            leaves_l[2],
            leaves_l[3] if has_branch else None,
            leaves_l[4] if has_branch else None,
            cu_local,
            max_local,
            softmax_scale,
            cp_group,
        )
        tol(out_local, out_ref[global_row], f"out(branch={has_branch})")
        local_inputs = leaves_l[: 5 if has_branch else 3]
        grads_local = torch.autograd.grad(out_local, local_inputs, dout_g[global_row])
        names = ("dq", "dk1", "dv1", "dkb", "dvb")
        for g_local, g_ref, name in zip(grads_local, grads_ref, names):
            tol(g_local, g_ref[global_row], f"{name}(branch={has_branch})")

    dist.barrier()
    dist.destroy_process_group()


@requires_2_gpus
@pytest.mark.skipif(not _flash_attn_available(), reason="Requires flash-attn")
def test_two_part_attention_cp2_matches_single_rank(tmp_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / "ttt_cp2_init")
    procs = []
    for rank in range(2):
        proc = ctx.Process(target=_run_cp2_rank, args=(rank, 2, init_file))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"


def test_draft_ce_loss_packed_matches_unpacked():
    """Packed-mode loss (teacher roll + coord-gathered masks) == slice mode.

    The same student leaves feed both paths (the packed row is a gathered
    view), so loss values AND student gradients must agree.
    """
    from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
    from nemo_rl.algorithms.loss.utils import compute_draft_pass_valid_counts

    torch.manual_seed(2)
    batch, seq_len, vocab, num_passes = 2, 24, 32, 2
    lengths = torch.tensor([20, 24])

    teacher = torch.randn(batch, seq_len, vocab)
    students = [
        torch.randn(batch, seq_len, vocab, requires_grad=True)
        for _ in range(num_passes)
    ]
    # Assistant span starts at 6 / 3; zero outside the valid lengths and at
    # position 0 (never a next-token target in practice, keeps it realistic).
    token_mask = torch.zeros(batch, seq_len)
    token_mask[0, 6:20] = 1.0
    token_mask[1, 3:24] = 1.0
    sample_mask = torch.ones(batch)
    counts = compute_draft_pass_valid_counts(
        token_mask, sample_mask, ttt_steps=num_passes
    )

    loss_fn = DraftCrossEntropyLossFn()
    unpacked_loss, unpacked_metrics = loss_fn(
        teacher_logits=teacher.detach(),
        student_logits_by_pass=list(students),
        data={"token_mask": token_mask, "sample_mask": sample_mask},
        global_valid_seqs=None,
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=counts,
    )
    unpacked_loss.backward()
    unpacked_grads = [s.grad.detach().clone() for s in students]
    for s in students:
        s.grad = None

    # Packed layout: concatenate the valid prefixes (cp=1, no padding).
    cu = torch.tensor([0, 20, 44], dtype=torch.int32)
    seq_index, pos_in_seq = packed_zigzag_token_coords(cu, cp_rank=0, cp_size=1)

    def pack(t):
        return torch.cat([t[0, :20], t[1, :24]], dim=0).unsqueeze(0)

    packed_students = [pack(s) for s in students]
    packed_loss, packed_metrics = loss_fn(
        teacher_logits=pack(teacher).detach(),
        student_logits_by_pass=packed_students,
        data={
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "input_lengths": lengths,
            "draft_packed_seq_index": seq_index,
            "draft_packed_pos_in_seq": pos_in_seq,
            "draft_packed_local_cu_seqlens": cu,
        },
        global_valid_seqs=None,
        global_valid_toks=torch.tensor(1.0),
        global_draft_pass_counts=counts,
    )
    torch.testing.assert_close(packed_loss, unpacked_loss, rtol=1e-5, atol=1e-6)
    for ttt_pass in range(1, num_passes + 1):
        assert packed_metrics[f"draft_loss_pass_{ttt_pass}"] == pytest.approx(
            unpacked_metrics[f"draft_loss_pass_{ttt_pass}"], rel=1e-4
        )
    packed_loss.backward()
    for s, ref in zip(students, unpacked_grads):
        torch.testing.assert_close(s.grad, ref, rtol=1e-5, atol=1e-6)


def _run_forward_ttt_packed_rank(rank: int, world_size: int, init_file: str) -> None:
    """Whole-module check: packed forward_ttt == per-subsequence unpacked runs.

    Exercises the pieces the operator-level tests cannot: the per-pass packed
    RoPE (positions restart per subsequence, offset d-1), the CP-aware packed
    embedding roll between passes, and packed_seq_params threading through
    modelopt's EagleModule into the core attention.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )

    hidden, vocab = 64, 128
    draft = _make_eagle_model(hidden, vocab, ttt_steps=3)

    lengths = [24, 40]
    total = sum(lengths)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )
    taps = torch.randn(total, 1, 3 * hidden, device="cuda", dtype=torch.bfloat16)
    embeds = torch.randn(total, 1, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        packed_logits = draft.forward_ttt(
            hidden_states=taps,
            input_embeds=embeds,
            packed_seq_params=packed_seq_params,
        )
        assert all(t.shape == (1, total, vocab) for t in packed_logits)

        for seq_idx, length in enumerate(lengths):
            start, end = int(cu[seq_idx]), int(cu[seq_idx + 1])
            ref_logits = draft.forward_ttt(
                hidden_states=taps[start:end],
                input_embeds=embeds[start:end],
            )
            for ttt_pass, (packed_pass, ref_pass) in enumerate(
                zip(packed_logits, ref_logits), start=1
            ):
                actual = packed_pass[0, start:end].float()
                expected = ref_pass[0].float()
                # Positions past the subsequence tail feed rolled-in zeros in
                # the packed run vs rolled-in zeros in the unpacked run — the
                # inputs match, so ALL rows must agree.
                max_diff = (actual - expected).abs().max().item()
                scale = expected.abs().max().clamp(min=1e-6).item()
                assert max_diff <= 3e-2 + 3e-2 * scale, (
                    f"seq {seq_idx} pass {ttt_pass}: max diff {max_diff:.4e} "
                    f"(scale {scale:.4e})"
                )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(not _flash_attn_available(), reason="Requires flash-attn")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires a GPU")
def test_forward_ttt_packed_matches_per_sequence(tmp_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / "ttt_fwd_packed_init")
    proc = ctx.Process(target=_run_forward_ttt_packed_rank, args=(0, 1, init_file))
    proc.start()
    proc.join(timeout=600)
    assert proc.exitcode == 0, f"forward_ttt packed worker exited with {proc.exitcode}"


def _make_eagle_model(hidden: int, vocab: int, ttt_steps: int):
    """Shared tiny EagleModel builder for the forward_ttt equivalence tests."""
    import torch.nn.functional as F
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.models.megatron.draft.eagle import EagleModel

    # force_reset_rng: the tracker survives destroy_model_parallel(), so a
    # second build in the same process would otherwise draw from leftover
    # RNG state and get different weights.
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    torch.manual_seed(7)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    config.transformer_layer_spec = None
    config.vocab_size = vocab
    config.draft_vocab_size = vocab
    config.seq_length = 256
    config.gradient_accumulation_fusion = False
    config.position_embedding_type = "rope"
    config.rotary_percent = 1.0
    config.rotary_base = 10000
    config.rope_scaling = False
    config.rope_scaling_factor = 8.0
    config.use_input_layernorm_in_first_layer = True
    config.use_last_layernorm = True
    config.use_aux_hidden_state = True
    config.eagle_aux_hidden_state_layer_ids = [0, 1, 2]
    config.parallel_draft_step = 1
    config.use_mtp_layernorm = None
    config.parallel_draft_heads_num_layers = None
    config.has_lm_head = True
    config.apply_rope_fusion = False
    model = EagleModel(config=config, ttt_steps=ttt_steps).cuda()
    # modelopt's EagleModule builds fc with init_method=(lambda w: None) —
    # deliberately UNinitialized (production always loads a checkpoint), so
    # its weight is whatever memory the allocator hands back and differs
    # between builds. Give it a deterministic value for build-vs-build
    # equivalence tests.
    fc_weight = model.eagle_module.fc.weight
    with torch.no_grad():
        fc_weight.copy_(
            torch.randn(
                fc_weight.shape, generator=torch.Generator().manual_seed(21)
            ).to(dtype=fc_weight.dtype, device=fc_weight.device)
            * 0.02
        )
    return model


def _run_forward_ttt_cp2_rank(rank: int, world_size: int, init_file: str) -> None:
    """CP=2 forward_ttt end-to-end vs the CP=1 global run of the SAME weights.

    Phase 1 initializes model parallel with CP=1 (world becomes DP=2, both
    ranks compute the identical global packed forward). Phase 2 re-initializes
    with CP=2, rebuilds the model from the same seeds, feeds each rank its
    zigzag shard, and checks the local per-pass logits against the global
    reference's zigzag slice — covering the ring attention, the CP rotary
    slicing, and the CP boundary exchange of the inter-pass embed roll inside
    the real decoder stack.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )

    hidden, vocab, ttt_steps = 64, 128, 3
    lengths = [24, 40]  # multiples of 2*cp = 4
    total = sum(lengths)
    cu_cpu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(), dtype=torch.int32
    )
    torch.manual_seed(11)
    taps_g = torch.randn(total, 1, 3 * hidden, dtype=torch.bfloat16).cuda()
    embeds_g = torch.randn(total, 1, hidden, dtype=torch.bfloat16).cuda()

    def make_psp(cu):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )

    # ---- Phase 1: CP=1 (DP=2) global reference, identical on both ranks.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    draft = _make_eagle_model(hidden, vocab, ttt_steps)
    with torch.no_grad():
        ref_logits = draft.forward_ttt(
            hidden_states=taps_g,
            input_embeds=embeds_g,
            packed_seq_params=make_psp(cu_cpu.cuda()),
        )
    ref_logits = [t.clone() for t in ref_logits]
    del draft
    dist.barrier()

    # ---- Phase 2: CP=2, same seeds -> same weights, zigzag-local inputs.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )
    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = cp_group.rank()
    draft = _make_eagle_model(hidden, vocab, ttt_steps)
    assert draft._ttt_attn_modules, "CP=2 must install the custom core attention"
    # Mirror build_draft_model's pg_collection attribute pass.
    from types import SimpleNamespace

    for module in draft.modules():
        if hasattr(module, "pg_collection"):
            module.pg_collection = SimpleNamespace(cp=cp_group)

    seq_index, pos = packed_zigzag_token_coords(cu_cpu, cp_rank, 2)
    global_row = (cu_cpu[:-1].to(torch.long)[seq_index] + pos).cuda()
    with torch.no_grad():
        local_logits = draft.forward_ttt(
            hidden_states=taps_g[global_row],
            input_embeds=embeds_g[global_row],
            packed_seq_params=make_psp(cu_cpu.cuda()),
        )

    for ttt_pass, (local_pass, ref_pass) in enumerate(
        zip(local_logits, ref_logits), start=1
    ):
        actual = local_pass[0].float()
        expected = ref_pass[0, global_row].float()
        max_diff = (actual - expected).abs().max().item()
        scale = expected.abs().max().clamp(min=1e-6).item()
        assert max_diff <= 3e-2 + 3e-2 * scale, (
            f"rank {cp_rank} pass {ttt_pass}: max diff {max_diff:.4e} "
            f"(scale {scale:.4e})"
        )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@requires_2_gpus
@pytest.mark.skipif(not _flash_attn_available(), reason="Requires flash-attn")
def test_forward_ttt_cp2_matches_global(tmp_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    ctx = mp.get_context("spawn")
    init_file = str(tmp_path / "ttt_fwd_cp2_init")
    procs = []
    for rank in range(2):
        proc = ctx.Process(target=_run_forward_ttt_cp2_rank, args=(rank, 2, init_file))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"
