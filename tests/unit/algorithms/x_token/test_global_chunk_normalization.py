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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn


def test_cross_token_teacher_receives_its_global_chunk_count():
    loss_fn = object.__new__(CrossTokenizerDistillationLossFn)
    loss_fn.projection_matrix_paths = [None, None, "projection.pt"]
    loss_fn.teacher_vocab_sizes = [16, 16, 32]
    loss_fn._compute_prefix_bidir_partition_kl_v3 = MagicMock(
        return_value=(torch.tensor(1.5), {})
    )
    counts = {0: torch.tensor(11), 2: torch.tensor(29)}

    loss_fn._compute_teacher_kd(
        2,
        torch.zeros(1),
        MagicMock(),
        {2: torch.zeros(1)},
        {2: MagicMock()},
        torch.tensor(7),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
        global_valid_chunks_by_idx=counts,
    )

    assert (
        loss_fn._compute_prefix_bidir_partition_kl_v3.call_args.kwargs[
            "global_valid_chunks"
        ]
        is counts[2]
    )


def test_missing_cross_token_teacher_count_does_not_fall_back_to_microbatch():
    loss_fn = object.__new__(CrossTokenizerDistillationLossFn)
    loss_fn.projection_matrix_paths = ["projection.pt"]
    loss_fn.teacher_vocab_sizes = [32]

    with pytest.raises(KeyError):
        loss_fn._compute_teacher_kd(
            0,
            torch.zeros(1),
            MagicMock(),
            {0: torch.zeros(1)},
            {0: MagicMock()},
            torch.tensor(7),
            teacher_sparse_logits_by_idx={},
            tp_group=None,
            cp_group=None,
            global_valid_chunks_by_idx={},
        )
