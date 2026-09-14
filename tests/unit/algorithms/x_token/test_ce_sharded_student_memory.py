# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run_tp2_cp2_ce_memory(
    rank: int,
    world_size: int,
    init_file: str,
):
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module
    from nemo_rl.algorithms.loss.loss_functions import (
        XTOKEN_STUDENT_LOSS_CHUNK_SIZE,
        CrossTokenizerDistillationLossFn,
    )
    from nemo_rl.distributed import model_utils
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        tp_size = cp_size = 2
        tp_index, cp_index = rank // cp_size, rank % cp_size
        tp_group = None
        for cp_coordinate in range(cp_size):
            group = dist.new_group(
                ranks=[
                    tp_coordinate * cp_size + cp_coordinate
                    for tp_coordinate in range(tp_size)
                ]
            )
            if cp_coordinate == cp_index:
                tp_group = group
        cp_group = None
        for tp_coordinate in range(tp_size):
            group = dist.new_group(
                ranks=[
                    tp_coordinate * cp_size + cp_coordinate
                    for cp_coordinate in range(cp_size)
                ]
            )
            if tp_coordinate == tp_index:
                cp_group = group
        assert tp_group is not None
        assert cp_group is not None

        torch.manual_seed(11)
        batch_size = 2
        sequence_length = 1028
        vocab_size = 12
        full_logits = torch.randn(batch_size, sequence_length, vocab_size)
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            dtype=torch.long,
        )
        token_mask = torch.ones(batch_size, sequence_length)
        token_mask[0, 257:269] = 0
        token_mask[0, -1] = 0
        sample_mask = torch.tensor([1.0, 0.0])
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "token_mask": token_mask,
                "sample_mask": sample_mask,
            }
        )
        global_valid_toks = (token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum()
        loss_fn = CrossTokenizerDistillationLossFn.__new__(
            CrossTokenizerDistillationLossFn
        )

        reference_logits = full_logits.clone().requires_grad_(True)
        reference_loss = loss_fn._compute_ce(
            reference_logits,
            data,
            global_valid_toks,
            tp_group=None,
            cp_group=None,
        )
        reference_loss.backward()
        reference_grad = reference_logits.grad.detach().clone()

        vocab_block = vocab_size // tp_size
        vocab_start = tp_index * vocab_block
        vocab_slice = slice(vocab_start, vocab_start + vocab_block)
        student_shard = _get_tokens_on_this_cp_rank(
            full_logits[:, :, vocab_slice],
            cp_rank=cp_index,
            cp_size=cp_size,
            seq_dim=1,
        ).contiguous()
        student_shard = student_shard.clone().requires_grad_(True)
        assert student_shard.shape[1] > XTOKEN_STUDENT_LOSS_CHUNK_SIZE

        original_cp_contiguous_gather = (
            loss_functions_module.allgather_cp_contiguous_tensor
        )
        original_full_softmax = loss_functions_module.vocab_parallel_full_log_softmax
        original_scalar_cp_gather = model_utils.allgather_cp_sharded_tensor
        had_vocab_gather = hasattr(
            loss_functions_module,
            "vocab_parallel_gather_logits",
        )
        original_vocab_gather = getattr(
            loss_functions_module,
            "vocab_parallel_gather_logits",
            None,
        )
        scalar_gathers = 0

        def reject_contiguous_rank3_gather(*_args, **_kwargs):
            raise AssertionError("CE attempted a rank-3 contiguous CP gather")

        def reject_vocab_gather(*_args, **_kwargs):
            raise AssertionError("CE attempted a rank-3 TP vocabulary gather")

        def require_scalar_cp_gather(tensor, group, seq_dim=1):
            nonlocal scalar_gathers
            assert tensor.ndim == 2
            scalar_gathers += 1
            return original_scalar_cp_gather(tensor, group, seq_dim)

        loss_functions_module.allgather_cp_contiguous_tensor = (
            reject_contiguous_rank3_gather
        )
        loss_functions_module.vocab_parallel_full_log_softmax = reject_vocab_gather
        loss_functions_module.vocab_parallel_gather_logits = reject_vocab_gather
        model_utils.allgather_cp_sharded_tensor = require_scalar_cp_gather
        try:
            sharded_loss = loss_fn._compute_ce(
                student_shard,
                data,
                global_valid_toks,
                student_logits_contig=student_shard,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        finally:
            loss_functions_module.allgather_cp_contiguous_tensor = (
                original_cp_contiguous_gather
            )
            loss_functions_module.vocab_parallel_full_log_softmax = (
                original_full_softmax
            )
            model_utils.allgather_cp_sharded_tensor = original_scalar_cp_gather
            if had_vocab_gather:
                loss_functions_module.vocab_parallel_gather_logits = (
                    original_vocab_gather
                )
            else:
                delattr(loss_functions_module, "vocab_parallel_gather_logits")

        assert scalar_gathers == 1
        torch.testing.assert_close(
            sharded_loss,
            reference_loss,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        (sharded_loss / float(cp_size)).backward()

        gathered_gradients = [torch.empty_like(student_shard.grad) for _ in range(4)]
        dist.all_gather(gathered_gradients, student_shard.grad.contiguous())
        reconstructed_grad = torch.zeros_like(reference_grad)
        sequence_chunk = sequence_length // (2 * cp_size)
        for shard_rank, shard_gradient in enumerate(gathered_gradients):
            shard_tp, shard_cp = shard_rank // cp_size, shard_rank % cp_size
            shard_vocab = slice(
                shard_tp * vocab_block,
                (shard_tp + 1) * vocab_block,
            )
            head, tail = torch.chunk(shard_gradient, chunks=2, dim=1)
            head_chunk = shard_cp
            tail_chunk = 2 * cp_size - shard_cp - 1
            reconstructed_grad[
                :,
                head_chunk * sequence_chunk : (head_chunk + 1) * sequence_chunk,
                shard_vocab,
            ] = head
            reconstructed_grad[
                :,
                tail_chunk * sequence_chunk : (tail_chunk + 1) * sequence_chunk,
                shard_vocab,
            ] = tail
        torch.testing.assert_close(
            reconstructed_grad,
            reference_grad,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="gloo backend unavailable",
)
def test_ce_tp2_cp2_matches_full_loss_and_gradient_without_rank3_gathers(tmp_path):
    """Masked CE is exact across TP2/CP2 and crosses the fixed chunk boundary."""
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _run_tp2_cp2_ce_memory,
        args=(4, os.fspath(init_file)),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_ce_tp2_cp2_matches_full_loss_and_gradient_without_rank3_gathers(
            Path(tmpdir)
        )
    print("CE TP2/CP2 MEMORY-BOUNDED PARITY PASSED")
