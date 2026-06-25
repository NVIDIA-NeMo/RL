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
"""Batch-grid helpers for cross-tokenizer off-policy distillation.

Teacher and student may run at different data-parallel degrees / microbatch
sizes while consuming the same global batch. These helpers validate that the
global batch tiles cleanly across both grids
(:func:`assert_teacher_student_batch_grid`) and pad an uneven validation batch
up to a tileable size (:func:`pad_distillation_val_batch`).
"""

from __future__ import annotations

from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def assert_teacher_student_batch_grid(
    *,
    global_batch_size: int,
    student_gbs: int,
    teacher_gbs: int,
    student_dp: int,
    teacher_dp: int,
    student_mbs: int,
    teacher_mbs: int,
) -> None:
    """Fail fast unless teacher and student share one global batch and both tile it.

    A cross-tokenizer teacher and student may run at different data-parallel
    degrees and microbatch sizes, but they consume the SAME global batch in the
    SAME global order (the teacher exports a global-batch-ordered per-sample
    logits list and the student slices it by its own DP/MBS). That requires both
    sides' ``train_global_batch_size`` to agree with the step's
    ``global_batch_size``, and each side to split that batch into whole
    per-DP-rank chunks and whole microbatches. Not specific to distillation —
    any xtoken teacher/student pairing needs it.
    """
    assert student_gbs == teacher_gbs == global_batch_size, (
        "student/teacher train_global_batch_size and the step global_batch_size "
        f"must all match, got student={student_gbs}, teacher={teacher_gbs}, "
        f"global_batch_size={global_batch_size}."
    )
    gbs = global_batch_size
    assert gbs % student_dp == 0, (
        f"global batch size ({gbs}) must be divisible by student "
        f"data_parallel size ({student_dp})."
    )
    assert gbs % teacher_dp == 0, (
        f"global batch size ({gbs}) must be divisible by teacher "
        f"data_parallel size ({teacher_dp})."
    )
    assert (gbs // student_dp) % student_mbs == 0, (
        f"student local batch (gbs/student_dp = {gbs // student_dp}) must be "
        f"divisible by student micro batch size ({student_mbs})."
    )
    assert (gbs // teacher_dp) % teacher_mbs == 0, (
        f"teacher local batch (gbs/teacher_dp = {gbs // teacher_dp}) must be "
        f"divisible by teacher micro batch size ({teacher_mbs})."
    )


def assert_xtoken_ipc_node_local(
    *,
    num_nodes: int,
    gpus_per_node: int,
    student_tp: int,
    student_cp: int,
    teacher_tp: int,
    teacher_cp: int,
    student_dp: int,
    teacher_dp: int,
) -> None:
    """Fail fast if the teacher->student logit transport would cross a node.

    Teacher logits reach the student via CUDA IPC (``cudaIpcOpenMemHandle``),
    which can only map a buffer created by a process on the SAME physical node.
    A single-node job is therefore always safe; a multi-node job is only safe
    when every student rank's required teacher shards live on its own node.

    """
    if num_nodes <= 1:
        return

    student_group = student_tp * student_cp
    teacher_group = teacher_tp * teacher_cp
    assert teacher_dp == student_dp, (
        "Multi-node xtoken distillation uses node-local CUDA IPC for the teacher "
        "logits, which requires teacher and student to share a data_parallel "
        f"degree (got teacher_dp={teacher_dp}, student_dp={student_dp}); "
        "otherwise a student rank must read teacher shards from another node. "
        "Use a single node, or matching DP, or a cross-node transport."
    )
    assert teacher_group == student_group, (
        "Multi-node xtoken distillation requires teacher and student to share "
        f"the same model-parallel group size tp*cp (got teacher={teacher_group}, "
        f"student={student_group}); differing sizes imply a non-colocated grid "
        "whose teacher shards are not node-local."
    )
    assert student_group <= gpus_per_node, (
        f"Multi-node xtoken distillation needs the model-parallel group tp*cp "
        f"({student_group}) to fit within one node ({gpus_per_node} GPUs); a "
        "group spanning nodes makes the teacher-logit IPC cross-node."
    )
    assert gpus_per_node % student_group == 0, (
        f"Multi-node xtoken distillation needs gpus_per_node ({gpus_per_node}) "
        f"to be a multiple of the model-parallel group tp*cp ({student_group}) "
        "so DP groups are node-aligned and never straddle a node boundary."
    )


def pad_distillation_val_batch(
    batch: BatchedDataDict[Any], target_size: int
) -> BatchedDataDict[Any]:
    """Pad every key of a validation batch up to ``target_size`` (batch axis).

    Validation uses ``drop_last=False``, so the final batch can be smaller
    than ``num_prompts_per_step`` and may not tile evenly across the teacher
    and student DP/MBS grids. Padding the whole batch symmetrically (student,
    teacher and alignment keys together) lets both sides run the existing
    even-split path with zero shared-code changes. The padded rows carry
    ``sample_mask == 0``, so they are excluded from the valid-sample counts in
    ``process_global_batch`` and contribute nothing to the loss or gradients.
    """
    current_size = batch.size
    if target_size == current_size:
        return batch
    assert target_size > current_size, (
        f"target_size ({target_size}) must be >= batch size ({current_size})."
    )
    pad = target_size - current_size

    padded: BatchedDataDict[Any] = BatchedDataDict()
    for key, value in batch.items():
        if torch.is_tensor(value):
            pad_rows = value[-1:].repeat(pad, *([1] * (value.dim() - 1)))
            padded[key] = torch.cat([value, pad_rows], dim=0)
        else:
            padded[key] = list(value) + [value[-1]] * pad
    padded["sample_mask"][current_size:] = 0
    return padded
