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

from tests.unit.algorithms.x_token.test_v6_loss_parity import _build_case


def _loss_config(fx):
    knobs = fx["knobs"]
    return {
        "temperature": knobs["temperature"],
        "vocab_topk": 8,
        "reverse_kl": knobs["reverse_kl"],
        "kl_loss_weight": 1.0,
        "ce_loss_scale": 1.0,
        "dynamic_loss_scaling": False,
        "kd_loss_mode": "sum",
        "normalize_teacher_by_vocab": False,
        "alpha": 1.0,
        "sum_weights_metric": None,
        "student_vocab_size": fx["v_s"],
        "teacher_vocab_sizes": [fx["v_t"]],
        "projection_matrix_paths": ["dummy.pt"],
        "teacher_weights": [1.0],
        "common_indices_from_subtoks": knobs["common_indices_from_subtoks"],
        "pseudo_target_paths": [knobs["pseudo_target_path"]],
        "reverse_pseudo_target_paths": [knobs["reverse_pseudo_target_path"]],
        "kl_chunk_shift": knobs["kl_chunk_shift"],
        "prefix_bidir_v3_position_0_kl": knobs["prefix_bidir_v3_position_0_kl"],
        "prefix_bidir_v3_loss_fn": knobs["prefix_bidir_v3_loss_fn"],
        "prefix_bidir_v3_last_pos_loss_fn": knobs["prefix_bidir_v3_last_pos_loss_fn"],
        "prefix_bidir_v3_jsd_beta": knobs["prefix_bidir_v3_jsd_beta"],
        "prefix_bidir_v3_mismatch_pos0_alpha": knobs[
            "prefix_bidir_v3_mismatch_pos0_alpha"
        ],
        "prefix_bidir_v3_mismatch_loss_beta": knobs[
            "prefix_bidir_v3_mismatch_loss_beta"
        ],
        "prefix_bidir_v3_noise_filter_topk": knobs["prefix_bidir_v3_noise_filter_topk"],
        "teacher_topk_ipc_k": 0,
        "teacher_topk_ipc_support_mode": "row_topk",
        "teacher_topk_ipc_keep_realized": False,
    }


def _run_tp2_cp2_v6_student_memory(
    rank: int,
    world_size: int,
    init_file: str,
    fixture_path: str,
):
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module
    from nemo_rl.algorithms.loss.loss_functions import (
        CrossTokenizerDistillationLossFn,
    )
    from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment

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

        fx = torch.load(fixture_path, weights_only=False)
        loss_fn = CrossTokenizerDistillationLossFn(_loss_config(fx))
        batch_size, student_seq_len, student_vocab_size = fx["student_logits"].shape
        teacher_seq_len = fx["teacher_logits"].shape[1]
        global_valid_chunks = torch.tensor(float(fx["global_valid_chunks"]))

        def alignment(student_ids, teacher_ids):
            return LocalizedAlignment(
                sample_mask=torch.ones(batch_size, dtype=torch.bool),
                pair_valid=fx["pair_valid"],
                student_input_ids=student_ids,
                teacher_input_ids=teacher_ids,
                student_spans=fx["s_spans"],
                teacher_spans=fx["t_spans"],
                num_chunks=fx["num_chunks"],
            )

        reference_logits = fx["student_logits"].clone().requires_grad_(True)
        reference_loss, reference_metrics = (
            loss_fn._compute_prefix_bidir_partition_kl_v3(
                0,
                reference_logits,
                fx["teacher_logits"],
                alignment(fx["student_ids"], fx["teacher_ids"]),
                teacher_vocab_size=fx["v_t"],
                tp_group=None,
                cp_group=None,
                global_valid_chunks=global_valid_chunks,
            )
        )
        reference_loss.backward()
        reference_grad = reference_logits.grad.detach().clone()

        student_block = student_seq_len // cp_size
        teacher_block = teacher_seq_len // cp_size
        vocab_block = student_vocab_size // tp_size
        student_start = cp_index * student_block
        teacher_start = cp_index * teacher_block
        vocab_start = tp_index * vocab_block
        student_shard = (
            fx["student_logits"][
                :,
                student_start : student_start + student_block,
                vocab_start : vocab_start + vocab_block,
            ]
            .clone()
            .requires_grad_(True)
        )
        teacher_shard = fx["teacher_logits"][
            :, teacher_start : teacher_start + teacher_block, :
        ].clone()

        original_cp_gather = loss_functions_module.allgather_cp_contiguous_tensor
        original_full_softmax = loss_functions_module.vocab_parallel_full_log_softmax
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

        def reject_rank3_cp_gather(tensor, group, seq_dim=1):
            nonlocal scalar_gathers
            if tensor.ndim == 3:
                raise AssertionError("v6 attempted a rank-3 CP gather")
            scalar_gathers += 1
            return original_cp_gather(tensor, group, seq_dim)

        def reject_vocab_gather(*_args, **_kwargs):
            raise AssertionError("v6 attempted a rank-3 TP vocabulary gather")

        loss_functions_module.allgather_cp_contiguous_tensor = reject_rank3_cp_gather
        loss_functions_module.vocab_parallel_full_log_softmax = reject_vocab_gather
        loss_functions_module.vocab_parallel_gather_logits = reject_vocab_gather
        try:
            sharded_loss, sharded_metrics = (
                loss_fn._compute_prefix_bidir_partition_kl_v3(
                    0,
                    student_shard,
                    teacher_shard,
                    alignment(
                        fx["student_ids"][
                            :, student_start : student_start + student_block
                        ].contiguous(),
                        fx["teacher_ids"][
                            :, teacher_start : teacher_start + teacher_block
                        ].contiguous(),
                    ),
                    teacher_vocab_size=fx["v_t"],
                    tp_group=tp_group,
                    cp_group=cp_group,
                    global_valid_chunks=global_valid_chunks,
                )
            )
        finally:
            loss_functions_module.allgather_cp_contiguous_tensor = original_cp_gather
            loss_functions_module.vocab_parallel_full_log_softmax = (
                original_full_softmax
            )
            if had_vocab_gather:
                loss_functions_module.vocab_parallel_gather_logits = (
                    original_vocab_gather
                )
            else:
                delattr(loss_functions_module, "vocab_parallel_gather_logits")

        assert scalar_gathers > 0
        torch.testing.assert_close(
            sharded_loss,
            reference_loss,
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        (sharded_loss / float(cp_size)).backward()

        gathered_gradients = [torch.empty_like(student_shard.grad) for _ in range(4)]
        dist.all_gather(gathered_gradients, student_shard.grad.contiguous())
        reconstructed_grad = torch.zeros_like(reference_grad)
        for shard_rank, shard_gradient in enumerate(gathered_gradients):
            shard_tp, shard_cp = shard_rank // cp_size, shard_rank % cp_size
            reconstructed_grad[
                :,
                shard_cp * student_block : (shard_cp + 1) * student_block,
                shard_tp * vocab_block : (shard_tp + 1) * vocab_block,
            ] = shard_gradient
        torch.testing.assert_close(
            reconstructed_grad,
            reference_grad,
            rtol=1.0e-4,
            atol=1.0e-4,
        )

        for metric_name in (
            "kl_common_per_chunk",
            "kl_partition_first_per_chunk",
            "kl_partition_last_per_chunk",
            "kl_mismatch_combined_per_chunk",
            "kl_mismatch_scaled_per_chunk",
            "top1_acc_per_chunk",
        ):
            assert sharded_metrics[metric_name] == pytest.approx(
                reference_metrics[metric_name], rel=1.0e-4, abs=1.0e-4
            )
        assert sharded_metrics["num_common_chunks"] == 66
        assert sharded_metrics["num_mismatch_chunks"] == 20
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="gloo backend unavailable",
)
def test_v6_tp2_cp2_is_exact_without_rank3_student_gathers(tmp_path):
    """Qualification v6 knobs preserve loss/grad/metrics across TP2/CP2."""
    fx = _build_case("with_mismatch", os.fspath(tmp_path))
    chunks = [(0, 1, 0, 1)] * 33 + [(1, 2, 1, 3), (3, 5, 4, 5)] * 5
    batch_size = fx["student_logits"].shape[0]
    fx["s_spans"] = torch.zeros(batch_size, len(chunks), 2, dtype=torch.long)
    fx["t_spans"] = torch.zeros(batch_size, len(chunks), 2, dtype=torch.long)
    fx["pair_valid"] = torch.ones(batch_size, len(chunks), dtype=torch.bool)
    fx["num_chunks"] = torch.full((batch_size,), len(chunks), dtype=torch.long)
    for batch_index in range(batch_size):
        for chunk_index, (
            student_start,
            student_end,
            teacher_start,
            teacher_end,
        ) in enumerate(chunks):
            fx["s_spans"][batch_index, chunk_index] = torch.tensor(
                [student_start, student_end]
            )
            fx["t_spans"][batch_index, chunk_index] = torch.tensor(
                [teacher_start, teacher_end]
            )
    fx["global_valid_chunks"] = float(batch_size * len(chunks))

    fixture_path = tmp_path / "fixture.pt"
    torch.save(fx, fixture_path)
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _run_tp2_cp2_v6_student_memory,
        args=(4, os.fspath(init_file), os.fspath(fixture_path)),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_v6_tp2_cp2_is_exact_without_rank3_student_gathers(Path(tmpdir))
    print("V6 TP2/CP2 MEMORY-BOUNDED PARITY PASSED")
