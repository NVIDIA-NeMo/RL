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
"""Packing must not change what a metric means.

``SequencePackingLossWrapper`` folds per-sequence metric dicts into one. Sums
are right for globally normalized metrics and wrong for extrema, and the
workers that consume this dict downstream already tell the two apart by the
``_min``/``_max`` suffix (megatron_value_worker.py:611 and four sibling sites).
"""

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.loss.wrapper import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _value_prepare_fn(logits, data, loss_fn=None, **kwargs):
    """Stand-in for megatron_value_worker._value_loss_prepare_fn.

    That function all-gathers across CP and shifts; neither applies here, so
    this keeps only the part the wrapper's contract depends on -- the key the
    loss is called with.
    """
    del loss_fn, kwargs
    return {"logits": logits}, data


def _batch(value_rows, sample_mask):
    values = torch.tensor(value_rows, dtype=torch.float32)
    data = BatchedDataDict(
        {
            "values": values.clone(),
            "returns": torch.zeros_like(values),
            "token_mask": torch.ones_like(values),
            "sample_mask": torch.tensor(sample_mask, dtype=torch.float32),
        }
    )
    return data, values


def _packed_and_unpacked(value_rows, sample_mask):
    """Run the same values both ways and return (unpacked_metrics, packed_metrics)."""
    loss_fn = MseValueLossFn(MseValueLossConfig())
    data, values = _batch(value_rows, sample_mask)
    logits = values.unsqueeze(-1)
    global_valid_seqs = data["sample_mask"].sum()
    global_valid_toks = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).sum()

    unpacked_loss, unpacked = loss_fn(
        logits, data, global_valid_seqs, global_valid_toks
    )

    seq_len = values.shape[1]
    cu_seqlens = torch.tensor(
        [i * seq_len for i in range(len(value_rows) + 1)], dtype=torch.int32
    )
    wrapper = SequencePackingLossWrapper(
        loss_fn=loss_fn,
        prepare_fn=_value_prepare_fn,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
    )
    packed_loss, packed = wrapper(
        logits.reshape(1, -1, 1), data, global_valid_seqs, global_valid_toks
    )

    # The loss must be untouched by anything here; if it moves, the test is
    # measuring the wrong thing.
    assert packed_loss.item() == pytest.approx(unpacked_loss.item(), abs=1e-6)
    return unpacked, packed


def test_packing_reports_the_true_value_range():
    """Summed extrema are not extrema -- and the reported minimum flips sign.

    Three sequences spanning -3..9. Summing the per-sequence minima gives
    -3 + 1 + 6 = 4, so ``critic/values_min`` reports a positive number for a
    critic whose predictions go negative. It is the diagnostic operators read
    to catch a value head drifting, and the error grows with packing density.
    """
    unpacked, packed = _packed_and_unpacked(
        [[-3.0, -1.0, 2.0], [1.0, 4.0, 2.0], [6.0, 9.0, 7.0]], [1, 1, 1]
    )

    assert unpacked["values_min"] == pytest.approx(-3.0)
    assert unpacked["values_max"] == pytest.approx(9.0)
    assert packed["values_min"] == pytest.approx(unpacked["values_min"])
    assert packed["values_max"] == pytest.approx(unpacked["values_max"])


def test_a_fully_masked_sequence_does_not_floor_the_reported_minimum():
    """The empty-mask sentinel must not be a plausible value.

    ``sample_mask`` is ``loss_multiplier``, which ``overlong_filtering`` zeroes
    per sample -- and under packing one filtered sample in a pack is enough. A
    0.0 sentinel would win the min against this all-positive critic and report
    0.0 instead of 3.0, so it has to be +/-inf and be skipped, as
    ClippedPGLossFn's already is.
    """
    unpacked, packed = _packed_and_unpacked(
        [[3.0, 5.0, 4.0], [6.0, 9.0, 7.0], [8.0, 8.5, 8.2]], [1, 1, 0]
    )

    assert unpacked["values_min"] == pytest.approx(3.0)
    assert packed["values_min"] == pytest.approx(3.0)
    assert packed["values_max"] == pytest.approx(9.0)


def test_globally_normalized_metrics_are_still_summed():
    """Only extrema changed: everything else must still add up across sequences."""
    unpacked, packed = _packed_and_unpacked(
        [[-3.0, -1.0, 2.0], [1.0, 4.0, 2.0], [6.0, 9.0, 7.0]], [1, 1, 1]
    )

    for key in ("values_mean", "returns_mean", "returns_sq_mean", "residual_sq_mean"):
        assert packed[key] == pytest.approx(unpacked[key], abs=1e-5), key


def test_an_all_masked_microbatch_reports_the_sentinel_fallback():
    """Every sequence filtered: there is no value range, and inf must not leak."""
    import numpy as np

    from nemo_rl.algorithms.ppo import _compute_critic_metrics

    _, packed = _packed_and_unpacked([[3.0, 5.0, 4.0], [6.0, 9.0, 7.0]], [0, 0])
    assert np.isinf(packed["values_min"])

    critic = _compute_critic_metrics(
        {
            "grad_norm": torch.tensor(0.0),
            "loss": torch.tensor(0.0),
            "all_mb_metrics": {"values_min": [packed["values_min"]]},
        }
    )
    assert critic["critic/values_min"] == pytest.approx(-1.0)
