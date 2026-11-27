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
"""X-Token's loss, gradients and metrics match goldens recorded before the shared base class.

Every case runs the real ``XTokenLossFn.__call__`` on CPU, with no faked
``_compute_teacher_kd``, for two teachers: one cross-tokenizer teacher with a
tiny projection and one teacher that shares the student's tokenizer (or two
same-vocab teachers, for the ``averaged_logits`` fast path). The goldens hold
the loss, the gradient of the student logits and every metric.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.cross_tokenizer.loss_utils import (
    LocalizedAlignment,
    localize_alignment,
)
from nemo_rl.algorithms.loss.cross_tokenizer.x_token import XTokenLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import cp_shift_next

GOLDEN_PATH = Path(__file__).parent / "data" / "xtoken_loss_goldens.json"
GOLDENS = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN_CASES = {case["name"]: case for case in GOLDENS["cases"]}
# 0.7 / 0.3 sum to 1, so no recorded case checks the division by sum(weights);
# 7 / 3 normalize to the same weights, so the recorded goldens still apply.
_UNNORMALIZED = copy.deepcopy(GOLDEN_CASES["averaged_logits_same_vocab"])
_UNNORMALIZED["config"]["teacher_weights"] = [7.0, 3.0]
GOLDEN_CASES["averaged_logits_same_vocab_unnormalized_weights"] = _UNNORMALIZED
# Metrics added with the shared base class; the goldens predate them.
NEW_METRICS = {"teacher_0/weighted_kl", "teacher_1/weighted_kl"}

_V_STUDENT = 6
_V_TEACHER = 5

_STUDENT_IDS = [[0, 3, 5, 1, 2, 4], [5, 2, 0, 4, 1, 0], [1, 1, 2, 3, 5, 0]]
_STUDENT_MASK = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
# The third sample is dropped.
_SAMPLE_MASK = [1.0, 1.0, 0.0]
_TEACHER_IDS = [[0, 2, 4, 1, 3], [4, 1, 0, 3, 0], [2, 2, 1, 0, 4]]
_TEACHER_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
# Unshifted chunk alignment, as the collator emits it.
_STUDENT_CHUNK_ID = [[0, 1, 1, 2, 3, 3], [0, 1, 2, 2, 3, -1], [0, 0, 1, 2, 3, 3]]
_TEACHER_CHUNK_ID = [[0, 1, 2, 3, 3], [0, 1, 2, 3, -1], [0, 1, 1, 2, 3]]
_PAIR_VALID = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
_PAIR_IS_CORRECT = [[1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]


def _sine_logits(shape: tuple[int, ...], *, phase: float, scale: float) -> torch.Tensor:
    """Deterministic logits that need no random number generator."""
    steps = torch.arange(math.prod(shape), dtype=torch.float32)
    return (scale * torch.sin(0.37 * steps + phase)).reshape(shape)


def _write_projection(tmp_path: Path) -> str:
    """Dense top-2 projection: s0 -> t0 and s5 -> t4 map 1-to-1, the rest split."""
    indices = torch.tensor(
        [[0, -1], [1, 2], [2, 3], [3, 1], [1, 3], [_V_TEACHER - 1, -1]],
        dtype=torch.long,
    )
    likelihoods = torch.tensor(
        [[1.0, 0.0], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.6, 0.4], [1.0, 0.0]]
    )
    path = tmp_path / "projection.pt"
    torch.save({"indices": indices, "likelihoods": likelihoods}, path)
    return str(path)


def _loss_config(case: dict[str, Any], projection_path: str) -> dict[str, Any]:
    cross_tokenizer = case["teachers"] == "cross_tokenizer_and_same_vocab"
    config: dict[str, Any] = {
        "gold_loss": False,
        "xtoken_loss": False,
        "temperature": 2.0,
        "vocab_topk": 4,
        "uncommon_topk": 8192,
        "reverse_kl": False,
        "exact_token_match_only": False,
        "kl_loss_weight": 1.0,
        "ce_loss_scale": 0.5,
        "dynamic_loss_scaling": False,
        "kd_loss_mode": "sum",
        "alpha": 1.0,
        "normalize_teacher_by_vocab": False,
        "student_vocab_size": _V_STUDENT,
        "projection_matrix_paths": [projection_path if cross_tokenizer else None, None],
        "teacher_vocab_sizes": [
            _V_TEACHER if cross_tokenizer else _V_STUDENT,
            _V_STUDENT,
        ],
        "teacher_weights": [0.7, 0.3],
        # Sum mode honors the per-teacher override (gold loss for teacher 0);
        # the other modes use the global P-KL setting.
        "teacher_gold_loss": [True if cross_tokenizer else None, None],
        "teacher_xtoken_loss": [None, None],
    }
    config.update(copy.deepcopy(case["config"]))
    return config


def _run_case(
    case: dict[str, Any], tmp_path: Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run ``XTokenLossFn.__call__`` at CP=1 on one case's fixture."""
    cross_tokenizer = case["teachers"] == "cross_tokenizer_and_same_vocab"
    loss_fn = XTokenLossFn(_loss_config(case, _write_projection(tmp_path)))

    input_ids = torch.tensor(_STUDENT_IDS)
    token_mask = torch.tensor(_STUDENT_MASK, dtype=torch.float32)
    sample_mask = torch.tensor(_SAMPLE_MASK)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "alignment_0_student_chunk_id": torch.tensor(_STUDENT_CHUNK_ID),
            "alignment_0_teacher_chunk_id": torch.tensor(_TEACHER_CHUNK_ID),
            "alignment_0_pair_valid": torch.tensor(_PAIR_VALID, dtype=torch.bool),
            "alignment_0_pair_is_correct": torch.tensor(
                _PAIR_IS_CORRECT, dtype=torch.bool
            ),
        }
    )
    student_shape = (len(_STUDENT_IDS), len(_STUDENT_IDS[0]), _V_STUDENT)
    student_logits = _sine_logits(student_shape, phase=0.0, scale=3.0)
    student_logits.requires_grad_(True)
    same_vocab_logits = _sine_logits(student_shape, phase=1.3, scale=4.0)

    # The shared student fields every alignment carries at CP=1.
    student_fields = {
        "sample_mask": sample_mask,
        "student_input_ids": input_ids,
        "student_token_mask": token_mask,
    }
    if cross_tokenizer:
        teacher_shape = (len(_TEACHER_IDS), len(_TEACHER_IDS[0]), _V_TEACHER)
        teacher_0_logits = _sine_logits(teacher_shape, phase=0.6, scale=4.0)
        # prepare_xtoken_cross_tokenizer_loss_input's CP=1 path.
        align_0 = localize_alignment(
            data, teacher_seq_len=teacher_shape[1], alignment_prefix="alignment_0_"
        )
        align_0.student_chunk_id = cp_shift_next(
            align_0.student_chunk_id, None, fill=-1
        )
        align_0.teacher_chunk_id = cp_shift_next(
            align_0.teacher_chunk_id, None, fill=-1
        )
        align_0.teacher_input_ids = torch.tensor(_TEACHER_IDS)
        align_0.teacher_token_mask = torch.tensor(_TEACHER_MASK, dtype=torch.float32)
        align_0.student_input_ids = input_ids
        align_0.student_token_mask = token_mask
    else:
        teacher_0_logits = _sine_logits(student_shape, phase=2.1, scale=4.0)
        align_0 = LocalizedAlignment(**student_fields)

    loss, metrics = loss_fn(
        data,
        sample_mask.sum(),
        (token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(),
        student_logits,
        student_logits,
        {0: teacher_0_logits, 1: same_vocab_logits},
        {0: align_0, 1: LocalizedAlignment(**student_fields)},
    )
    (gradient,) = torch.autograd.grad(loss, student_logits)
    return loss.detach(), gradient, metrics


@pytest.mark.parametrize("case_name", GOLDEN_CASES)
def test_xtoken_loss_matches_goldens(case_name: str, tmp_path: Path) -> None:
    """Loss, gradients and the recorded metrics match; only the new
    ``teacher_{i}/weighted_kl`` metrics are added, and they add up to the
    aggregated KD term."""
    case = GOLDEN_CASES[case_name]
    expected = case["expected"]

    loss, gradient, metrics = _run_case(case, tmp_path)

    torch.testing.assert_close(loss, torch.tensor(expected["loss"]))
    torch.testing.assert_close(gradient, torch.tensor(expected["gradient"]))
    assert metrics.keys() - expected["metrics"].keys() == NEW_METRICS
    assert expected["metrics"].keys() <= metrics.keys()
    for name, value in expected["metrics"].items():
        assert metrics[name] == pytest.approx(value, rel=1e-5, abs=1e-6), name
    weighted_sum = sum(metrics[name] for name in NEW_METRICS)
    assert weighted_sum == pytest.approx(metrics["kl_loss"], rel=1e-5, abs=1e-6)
