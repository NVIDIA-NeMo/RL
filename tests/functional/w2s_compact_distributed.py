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

"""W2S-OPD proxy support under TP2 x CP2.

    uv run torchrun --nproc_per_node=4 tests/functional/w2s_compact_distributed.py

Needs no Megatron install, so it also runs on CPU with gloo, where only
values are checked. Checks that the negative teacher's packed,
context-parallel proxy support equals a dense per-row oracle, and that the
student's loss on that support equals the dense oracle in value and gradient.
"""

import os

import torch
import torch.distributed as dist

from nemo_rl.algorithms.w2s_opd import (
    w2s_compact_proxy,
    w2s_packed_proxy_support,
    w2s_reverse_kl_from_support,
)
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank

ALPHA, TOP_K, VOCAB, PADDED_VOCAB = 0.75, 4, 11, 12


def dense_support(arms, heads, negative_logits):
    anchor, positive = [arm @ head.t() for arm, head in zip(arms, heads)]
    proxy = anchor + ALPHA * (positive - negative_logits)
    values, support = proxy[..., :VOCAB].topk(TOP_K, dim=-1)
    return support, values.log_softmax(-1)


def main():
    use_cuda = torch.cuda.is_available()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if use_cuda
        else torch.device("cpu")
    )
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "This validation requires exactly four ranks"
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp_rank, cp_rank = rank % 2, rank // 2
    tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
    width = PADDED_VOCAB // 2
    torch.manual_seed(917)
    heads = [torch.randn(PADDED_VOCAB, h, device=device) for h in (3, 5)]
    local_heads = tuple(h[tp_rank * width : (tp_rank + 1) * width] for h in heads)

    # Teacher side: two packed rows, spans padded to multiples of 2 x CP. The
    # negative teacher's own logits are sharded over both TP and CP.
    lengths, spans = [7, 5], [(0, 8), (8, 16)]
    rows = [torch.randn(sum(lengths), h, device=device) for h in (3, 5)]
    negative_full = torch.randn(1, 16, PADDED_VOCAB, device=device)
    negative_local = torch.cat(
        [
            _get_tokens_on_this_cp_rank(
                negative_full[:, start:end, tp_rank * width : (tp_rank + 1) * width],
                cp_rank,
                2,
                seq_dim=1,
            )
            for start, end in spans
        ],
        dim=1,
    )
    got = w2s_packed_proxy_support(
        tuple(rows),
        negative_local,
        local_heads,
        lengths=lengths,
        spans=spans,
        alpha=ALPHA,
        top_k=TOP_K,
        vocab_size=VOCAB,
        chunk_size=3,
        payload_dtype=torch.float32,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )
    offset = 0
    for (support, target), length, (start, _) in zip(got, lengths, spans):
        arms = (rows[0][offset : offset + length], rows[1][offset : offset + length])
        offset += length
        expected_support, expected_target = dense_support(
            arms, heads, negative_full[0, start : start + length]
        )
        assert torch.equal(support, expected_support), (rank, support, expected_support)
        torch.testing.assert_close(target, expected_target, atol=2e-6, rtol=2e-6)

    # Student side: a 7-token payload in an 8-token CP window (one pad slot).
    hidden = [torch.randn(2, 7, h, device=device) for h in (3, 5)]
    negative_dense = torch.randn(2, 7, PADDED_VOCAB, device=device)
    support, target = w2s_compact_proxy(
        (hidden[0], hidden[1]),
        (heads[0], heads[1]),
        negative_dense,
        alpha=ALPHA,
        top_k=TOP_K,
        vocab_size=VOCAB,
        chunk_size=5,
        vocab_parallel_group=None,
    )
    dense_student = torch.randn(2, 8, PADDED_VOCAB, device=device, requires_grad=True)
    p = dense_student[:, :7].gather(-1, support).softmax(-1)
    oracle = (p * (p.log() - target)).sum(-1)
    mask = torch.tensor(
        [[1.0, 0.0, 0.2, 1.0, 0.4, 0.0, 1.0], [0.0, 0.3, 1.0, 0.0, 1.0, 0.5, 0.7]],
        device=device,
    )
    (oracle * mask).sum().backward()

    def local_logits():
        return (
            _get_tokens_on_this_cp_rank(
                dense_student.detach()[..., tp_rank * width : (tp_rank + 1) * width],
                cp_rank,
                2,
                seq_dim=1,
            )
            .clone()
            .requires_grad_()
        )

    compact_logits = local_logits()
    compact = w2s_reverse_kl_from_support(
        compact_logits,
        support.to(torch.int32),
        target,
        chunk_size=3,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )
    torch.testing.assert_close(compact[:, :7], oracle, atol=6e-6, rtol=6e-6)
    # The raw CP gather sums gradients over its replicas (LossPostProcessor
    # divides by CP in production), so compare against CP x the oracle slice.
    # AllGatherCPTensor.backward stages its index through CUDA, so gradients
    # are checked only on GPU ranks.
    gradient_cases = ((compact_logits, compact),) if use_cuda else ()
    for logits, values in gradient_cases:
        (values[:, :7] * mask).sum().backward()
        expected = _get_tokens_on_this_cp_rank(
            dense_student.grad[..., tp_rank * width : (tp_rank + 1) * width],
            cp_rank,
            2,
            seq_dim=1,
        )
        torch.testing.assert_close(logits.grad, expected * 2, atol=6e-6, rtol=6e-6)
    dist.barrier()
    if rank == 0:
        checked = "values+gradients" if use_cuda else "values"
        print(f"W2S_COMPACT_TP2_CP2_PASS ({checked})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
