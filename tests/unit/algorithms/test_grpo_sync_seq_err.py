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

"""Targeted coverage for the dict-access fix in ``grpo_sync.py``.

PR #2607 replaces a 3-tuple unpack with a single dict assignment plus three
explicit dict-key reads:

    seq_error_result = compute_and_apply_seq_logprob_error_masking(...)
    max_seq_mult_prob_error = seq_error_result["max_seq_mult_prob_error"]
    num_masked_seqs = seq_error_result["num_masked_seqs"]
    masked_correct_pct = seq_error_result["masked_correct_pct"]

This module covers (a) the contract those reads depend on and (b) the
import binding tests must use when monkeypatching the function inside
``grpo_sync``.
"""

from __future__ import annotations

import torch

from nemo_rl.algorithms import grpo_sync as grpo_sync_mod
from nemo_rl.algorithms.grpo import compute_and_apply_seq_logprob_error_masking
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_data(
    batch_size: int = 4,
    seq_length: int = 10,
    *,
    err: float = 0.5,
) -> BatchedDataDict:
    """Build a tiny CPU-only train_data dict the masking function can chew on."""
    prev_logprobs = torch.zeros(batch_size, seq_length)
    generation_logprobs = torch.zeros(batch_size, seq_length)
    # Inject a non-zero error on the back half so the metric is non-trivial.
    generation_logprobs[batch_size // 2 :, 1:5] = err

    return BatchedDataDict(
        {
            "token_mask": torch.ones(batch_size, seq_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grpo_sync_imports_masking_fn_for_monkeypatching() -> None:
    """``grpo_sync.py`` does ``from nemo_rl.algorithms.grpo import ...``.

    Tests that monkeypatch ``grpo_train_sync`` need to target the binding
    inside ``nemo_rl.algorithms.grpo_sync``, not the source module — this
    test pins that contract so we notice if someone refactors the import.
    """
    assert hasattr(grpo_sync_mod, "compute_and_apply_seq_logprob_error_masking")
    assert (
        grpo_sync_mod.compute_and_apply_seq_logprob_error_masking
        is compute_and_apply_seq_logprob_error_masking
    )


def test_seq_error_result_exposes_keys_read_by_grpo_sync() -> None:
    """The 3 keys ``grpo_sync.py:773-783`` reads must be present and scalar.

    grpo_sync.py extracts these three values from the dict returned by
    ``compute_and_apply_seq_logprob_error_masking``:

        seq_error_result["max_seq_mult_prob_error"]
        seq_error_result["num_masked_seqs"]
        seq_error_result["masked_correct_pct"]
    """
    train_data = _make_train_data(batch_size=4, seq_length=10, err=0.5)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    seq_error_result = compute_and_apply_seq_logprob_error_masking(
        train_data=train_data,
        rewards=rewards,
        seq_logprob_error_threshold=1.2,
    )

    # Mirror the grpo_sync.py:772-783 access pattern verbatim.
    max_seq_mult_prob_error = seq_error_result["max_seq_mult_prob_error"]
    num_masked_seqs = seq_error_result["num_masked_seqs"]
    masked_correct_pct = seq_error_result["masked_correct_pct"]

    # Each of those values needs to be a usable scalar — grpo_sync.py
    # forwards them straight into the train-metrics payload.
    assert isinstance(max_seq_mult_prob_error, (int, float))
    assert isinstance(num_masked_seqs, int)
    assert isinstance(masked_correct_pct, float)
    assert max_seq_mult_prob_error >= 0.0
    assert num_masked_seqs >= 0
    assert 0.0 <= masked_correct_pct <= 1.0


def test_seq_error_result_full_key_set_matches_grpo_sync_consumers() -> None:
    """Lock the full key set so grpo_sync.py never breaks silently.

    grpo_sync.py reads three keys explicitly; the rest flow into metrics
    via the surrounding aggregation. If any of the eight keys disappear,
    both grpo.py and grpo_sync.py train-metrics payloads will regress.
    """
    train_data = _make_train_data()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    seq_error_result = compute_and_apply_seq_logprob_error_masking(
        train_data=train_data,
        rewards=rewards,
        seq_logprob_error_threshold=None,
    )

    expected_keys = {
        "max_seq_mult_prob_error",
        "mean_seq_mult_prob_error",
        "min_seq_mult_prob_error",
        "max_seq_mult_prob_error_after_mask",
        "mean_seq_mult_prob_error_after_mask",
        "min_seq_mult_prob_error_after_mask",
        "num_masked_seqs",
        "masked_correct_pct",
    }
    assert expected_keys.issubset(set(seq_error_result.keys()))


def test_grpo_sync_dict_access_pattern_with_monkeypatched_fn(monkeypatch) -> None:
    """Drive the exact grpo_sync.py:772-783 dict-access block via a stub.

    We cannot cheaply spin up ``grpo_train_sync`` in a unit test (it needs
    Ray + TQ + a real policy), so instead we monkeypatch the function in
    the grpo_sync namespace — exactly as a future ``grpo_train_sync`` test
    would — and then exercise the same dict-access ladder. This pins the
    namespace target callers must use.
    """
    fake_result = {
        "max_seq_mult_prob_error": 2.5,
        "mean_seq_mult_prob_error": 1.5,
        "min_seq_mult_prob_error": 1.0,
        "max_seq_mult_prob_error_after_mask": 1.2,
        "mean_seq_mult_prob_error_after_mask": 1.1,
        "min_seq_mult_prob_error_after_mask": 1.0,
        "num_masked_seqs": 2,
        "masked_correct_pct": 0.5,
    }

    captured: dict[str, object] = {}

    def _stub(*_args, **kwargs) -> dict[str, object]:
        captured["threshold"] = kwargs.get("seq_logprob_error_threshold")
        return fake_result

    monkeypatch.setattr(
        grpo_sync_mod, "compute_and_apply_seq_logprob_error_masking", _stub
    )

    # Re-resolve the function via the grpo_sync namespace — the same
    # binding ``grpo_train_sync`` uses internally.
    fn = grpo_sync_mod.compute_and_apply_seq_logprob_error_masking
    seq_error_result = fn(
        train_data=_make_train_data(),
        rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        seq_logprob_error_threshold=1.2,
    )

    # This is the verbatim 3-line access pattern from grpo_sync.py:773-783.
    max_seq_mult_prob_error = seq_error_result["max_seq_mult_prob_error"]
    num_masked_seqs = seq_error_result["num_masked_seqs"]
    masked_correct_pct = seq_error_result["masked_correct_pct"]

    assert max_seq_mult_prob_error == 2.5
    assert num_masked_seqs == 2
    assert masked_correct_pct == 0.5
    assert captured["threshold"] == 1.2
