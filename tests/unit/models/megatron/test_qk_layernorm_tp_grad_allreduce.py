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

"""qk-layernorm grads need a TP all-reduce; the draft's must be covered too.

The q/k layernorm weights are [head_dim] params replicated across TP, but each
rank's raw gradient only sums its LOCAL attention heads — without a TP SUM the
replicas silently diverge one optimizer step at a time. mcore performs that
SUM inside ``finalize_model_grads`` (name-matched on ``q_layernorm`` /
``k_layernorm``); the worker's split-step path calls
``_allreduce_non_tensor_model_parallel_grads`` explicitly for the same effect.

This test runs the block draft at TP=2 (attached under a ``draft_model.``
prefix, mirroring how the worker hangs the draft off the policy chunk),
asserts the qk grads really do diverge across ranks pre-reduce (the hazard),
and that the mcore call makes every replicated grad bitwise identical.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.mcore

requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires 2 GPUs (TP=2)"
)


def _run_rank(rank: int, world_size: int, init_file: str) -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_non_tensor_model_parallel_grads,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.algorithms.loss.loss_functions import BlockDraftLossFn
    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    # Replicated modules (fc / norms) initialize from the DEFAULT generator on
    # CPU; the spawned ranks must seed it identically or their replicas start
    # out different (production loads a checkpoint, so this is test-only).
    torch.manual_seed(7)
    tp_group = parallel_state.get_tensor_model_parallel_group()

    hidden, target_hidden, vocab = 64, 96, 128
    config = TransformerConfig(
        num_layers=1,
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
        tensor_model_parallel_size=2,
    )
    config.vocab_size = vocab
    config.draft_vocab_size = vocab
    config.apply_rope_fusion = False
    config.rotary_base = 10000
    config.gradient_accumulation_fusion = False

    gamma = 3
    draft = DFlashDraftModel(
        config=config,
        gamma=gamma,
        mask_token_id=vocab - 1,
        num_aux_hidden_states=3,
        target_hidden_size=target_hidden,
        trunk_chunk=8,
    ).cuda()

    # Mirror the worker attach: params reach finalize's named_parameters()
    # walk as ``draft_model.decoder...q_layernorm.weight``.
    container = torch.nn.Module()
    container.draft_model = draft
    container.ddp_config = SimpleNamespace(use_megatron_fsdp=False)

    # Bitwise-identical batch on both ranks (CPU generator), vocab-sharded
    # teacher/head per rank.
    seq_len, batch = 32, 1
    v_local = vocab // 2
    g = torch.Generator().manual_seed(7)
    taps = torch.randn(seq_len, batch, 3 * target_hidden, generator=g).to(
        device="cuda", dtype=torch.bfloat16
    )
    embeds = torch.randn(seq_len, batch, hidden, generator=g).to(
        device="cuda", dtype=torch.bfloat16
    )
    teacher_full = torch.randn(batch, seq_len, vocab, generator=g).to(torch.bfloat16)
    head_full = torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    mask_row = torch.randn(hidden, generator=g).to(device="cuda", dtype=torch.bfloat16)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    teacher = (
        teacher_full[..., tp_rank * v_local : (tp_rank + 1) * v_local]
        .contiguous()
        .cuda()
    )
    lm_head_w = (
        head_full[tp_rank * v_local : (tp_rank + 1) * v_local].contiguous().cuda()
    )

    anchors = torch.tensor([[5, 10, 20]], device="cuda")
    anchor_valid = torch.ones_like(anchors, dtype=torch.bool)
    token_mask = torch.ones(batch, seq_len, device="cuda")
    sample_mask = torch.ones(batch, device="cuda")

    logits = draft(
        taps=taps,
        input_embeds=embeds,
        anchors=anchors,
        anchor_valid=anchor_valid,
        lm_head_weight=lm_head_w,
        mask_embedding=mask_row,
    )
    loss_fn = BlockDraftLossFn(vocab_parallel_group=tp_group)
    loss, _ = loss_fn(
        teacher,
        logits[:, :, 1:, :],  # drop the dflash bonus slot
        {
            "draft_anchor_positions": anchors,
            "draft_anchor_valid": anchor_valid,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        },
        global_valid_seqs=None,
        global_valid_toks=torch.tensor(float(anchors.numel() * gamma), device="cuda"),
    )
    loss.backward()

    attention = draft.decoder.layers[0].self_attention
    replicated = {
        "q_layernorm": attention.q_layernorm.weight,
        "k_layernorm": attention.k_layernorm.weight,
        "fc": draft.fc.weight,
        "hidden_norm": draft.hidden_norm.weight,
        "final_norm": draft.decoder.final_layernorm.weight,
    }

    def grad_delta(param: torch.nn.Parameter) -> float:
        pair = [torch.empty_like(param.grad) for _ in range(2)]
        dist.all_gather(pair, param.grad.contiguous(), group=tp_group)
        return float((pair[0].float() - pair[1].float()).abs().max())

    # Precondition: without the reduce, the per-head-partial qk grads differ
    # across ranks (otherwise this test asserts nothing).
    assert grad_delta(replicated["q_layernorm"]) > 0.0, (
        "q_layernorm grads were already identical across TP ranks; the test "
        "inputs no longer exercise the hazard."
    )

    _allreduce_non_tensor_model_parallel_grads([container], config, tp_group)

    for name, param in replicated.items():
        delta = grad_delta(param)
        assert delta == 0.0, (
            f"{name} grad differs across TP ranks after the all-reduce "
            f"(max|delta|={delta})."
        )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@requires_2_gpus
def test_qk_layernorm_grads_equalize_across_tp(tmp_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    ctx = mp.get_context("spawn")
    procs = []
    init_file = str(tmp_path / "qk_pg_init")
    for rank in range(2):
        proc = ctx.Process(target=_run_rank, args=(rank, 2, init_file))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=600)
    for rank, proc in enumerate(procs):
        assert proc.exitcode == 0, f"rank {rank} exited with {proc.exitcode}"
