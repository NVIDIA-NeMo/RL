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
"""Unit tests for the pure multi-trace diagnostics used by the async GRPO loop."""

import math

import pytest
import torch

from nemo_rl.algorithms.multi_trace_metrics import (
    compute_multi_trace_diagnostics,
    finalize_sum_count_metrics,
    trace_index_bucket,
)


def _assert_all_finite(metrics: dict) -> None:
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), key
        assert math.isfinite(value), f"{key} = {value}"


def _small_batch():
    """4 traces x 6 positions. Column 0 is the shifted prompt column (ignored).

    row 0: rollout A / trace 0 / pre_compaction, 4 valid tokens, trains.
    row 1: rollout A / trace 1 / compaction_summary, 2 tokens, zeroed by the
           seq-logprob gate (pre-gate mask 1, post-gate mask 0).
    row 2: rollout A / trace 2 / post_compaction, env-masked (mask_sample).
    row 3: rollout B / trace 0 / empty dummy, no tokens, env-masked.
    """
    token_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    generation_logprobs = torch.zeros(4, 6)
    # |err| for row 0 valid tokens = 0.1, 0.2, 0.3, 0.4 (one per quartile).
    prev_logprobs = torch.tensor(
        [
            [0.0, -0.1, -0.2, -0.3, -0.4, 0.0],
            [0.0, -5.0, -5.0, 0.0, 0.0, 0.0],
            [0.0, -9.0, -9.0, -9.0, -9.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return dict(
        seq_mult_prob_error=torch.tensor([1.1, 3.0, 0.0, 0.0]),
        masked_by_seq_logprob_error=torch.tensor([False, True, False, False]),
        pre_seq_error_sample_loss_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        mask_sample=torch.tensor([False, False, True, True]),
        is_empty_rollout=torch.tensor([False, False, False, True]),
        trace_in_rollout_idx=torch.tensor([0, 1, 2, 0]),
        trace_kinds=["pre_compaction", "compaction_summary", "post_compaction", "empty"],
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        generation_logprobs=generation_logprobs,
        prev_logprobs=prev_logprobs,
        num_unpadded_traces=4,
    )


def _uncompacted_batch():
    """`_small_batch` with row 2 turned into an `uncompacted` trace that trains.

    row 2: rollout C / trace 0 / uncompacted, 4 valid tokens with |err| = 0.5
           each (mean exp|err| = e^0.5 < 2), not masked by anything.
    """
    b = _small_batch()
    b["trace_kinds"] = ["pre_compaction", "compaction_summary", "uncompacted", "empty"]
    b["trace_in_rollout_idx"] = torch.tensor([0, 1, 0, 0])
    b["mask_sample"] = torch.tensor([False, False, False, True])
    b["pre_seq_error_sample_loss_mask"] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    b["sample_mask"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    b["seq_mult_prob_error"] = torch.tensor([1.1, 3.0, math.exp(0.5), 0.0])
    b["prev_logprobs"][2] = torch.tensor([0.0, -0.5, -0.5, -0.5, -0.5, 0.0])
    return b


class TestTraceIndexBucket:
    def test_buckets(self):
        assert trace_index_bucket(0) == "0"
        assert trace_index_bucket(1) == "1"
        assert trace_index_bucket(2) == "2plus"
        assert trace_index_bucket(7) == "2plus"


class TestComputeMultiTraceDiagnostics:
    def test_masked_fraction_decomposition(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        _assert_all_finite(m)
        assert m["multi_trace/env_masked_trace_fraction"] == pytest.approx(0.5)
        assert m["multi_trace/empty_rollout_trace_fraction"] == pytest.approx(0.25)
        assert m["multi_trace/seq_logprob_masked_trace_fraction"] == pytest.approx(0.25)

    def test_seq_error_by_trace_index_uses_pre_gate_eligible_traces(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        p = "logprob_error/seq_mult_prob_error/by_trace_in_rollout_idx"
        f = "logprob_error/seq_masked_fraction/by_trace_in_rollout_idx"
        # bucket "0" = rows {0, 3}; row 3 (empty, pre-gate masked) is excluded.
        assert m[f"{p}/0/mean"] == pytest.approx(1.1)
        assert m[f"{f}/0"] == pytest.approx(0.0)
        # bucket "1" = row 1, gated out.
        assert m[f"{p}/1/mean"] == pytest.approx(3.0)
        assert m[f"{f}/1"] == pytest.approx(1.0)
        # bucket "2plus" = row 2, env-masked before the gate -> no data -> omitted.
        assert f"{p}/2plus/mean" not in m
        assert f"{f}/2plus" not in m

    def test_seq_error_by_segment_kind_omits_empty_buckets(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        p = "logprob_error/seq_mult_prob_error/by_segment_kind"
        f = "logprob_error/seq_masked_fraction/by_segment_kind"
        assert m[f"{p}/pre_compaction/mean"] == pytest.approx(1.1)
        assert m[f"{f}/pre_compaction"] == pytest.approx(0.0)
        assert m[f"{p}/compaction_summary/mean"] == pytest.approx(3.0)
        assert m[f"{f}/compaction_summary"] == pytest.approx(1.0)
        for kind in ("post_compaction", "empty"):
            assert f"{p}/{kind}/mean" not in m
            assert f"{f}/{kind}" not in m

    def test_trace_kind_count_and_valid_token_share(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        for kind in ("pre_compaction", "compaction_summary", "post_compaction", "empty"):
            assert m[f"multi_trace/trace_kind_count/{kind}"] == 1
        shares = {
            kind: m[f"multi_trace/trace_kind_valid_token_share/{kind}"]
            for kind in ("pre_compaction", "compaction_summary", "post_compaction", "empty")
        }
        # Only row 0 survives token_mask * sample_mask after the gate.
        assert shares["pre_compaction"] == pytest.approx(1.0)
        assert shares["compaction_summary"] == pytest.approx(0.0)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_position_quartiles(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        p = "logprob_error/token_abs_err/by_position_quartile"
        assert m[f"{p}/q1"] == pytest.approx(0.1)
        assert m[f"{p}/q2"] == pytest.approx(0.2)
        assert m[f"{p}/q3"] == pytest.approx(0.3)
        assert m[f"{p}/q4"] == pytest.approx(0.4)

    def test_position_quartiles_two_tokens_land_in_q2_and_q4(self):
        b = _small_batch()
        # Make row 1 the only training row with 2 valid tokens: k/N = 1/2, 2/2.
        b["sample_mask"] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        m = compute_multi_trace_diagnostics(**b)
        p = "logprob_error/token_abs_err/by_position_quartile"
        assert m[f"{p}/q2"] == pytest.approx(5.0)
        assert m[f"{p}/q4"] == pytest.approx(5.0)
        assert f"{p}/q1" not in m
        assert f"{p}/q3" not in m

    def test_padding_rows_are_ignored(self):
        b = _small_batch()
        padded = dict(b)
        for key in (
            "seq_mult_prob_error",
            "masked_by_seq_logprob_error",
            "pre_seq_error_sample_loss_mask",
            "token_mask",
            "sample_mask",
            "generation_logprobs",
            "prev_logprobs",
        ):
            t = b[key]
            padded[key] = torch.cat([t, t[[0, 0]]], dim=0)
        # Padding duplicates row 0 with sample_mask=0 (as the trainer does).
        padded["sample_mask"][4:] = 0
        # repeated_batch fields are never padded; they stay at 4 rows.
        m_padded = compute_multi_trace_diagnostics(**padded)
        m_unpadded = compute_multi_trace_diagnostics(**b)
        assert m_padded == m_unpadded

    def test_never_emits_nan_and_handles_no_valid_tokens(self):
        b = _small_batch()
        # Everything masked after the gate, seq error NaN on an eligible row.
        b["sample_mask"] = torch.zeros(4)
        b["seq_mult_prob_error"] = torch.tensor([float("nan"), 3.0, 0.0, 0.0])
        m = compute_multi_trace_diagnostics(**b)
        _assert_all_finite(m)
        # Row 0 stays gate-eligible (pre-gate mask 1) so its masked fraction
        # is reported, but its NaN error is not averaged into anything.
        assert "logprob_error/seq_mult_prob_error/by_trace_in_rollout_idx/0/mean" not in m
        assert m["logprob_error/seq_masked_fraction/by_trace_in_rollout_idx/0"] == 0.0
        assert not any(k.startswith("logprob_error/token_abs_err/") for k in m)
        assert not any(k.startswith("multi_trace/trace_kind_valid_token_share/") for k in m)
        assert m["multi_trace/trace_kind_count/pre_compaction"] == 1

    def test_missing_kinds_default_to_unknown_and_zero_rows_returns_empty(self):
        b = _small_batch()
        b["trace_kinds"] = ["pre_compaction", None]  # shorter than the batch
        m = compute_multi_trace_diagnostics(**b)
        assert m["multi_trace/trace_kind_count/unknown"] == 3
        b["num_unpadded_traces"] = 0
        assert compute_multi_trace_diagnostics(**b) == {}

    def test_accepts_python_lists_for_batch_fields(self):
        b = _small_batch()
        b["mask_sample"] = [False, False, True, True]
        b["is_empty_rollout"] = [False, False, False, True]
        b["trace_in_rollout_idx"] = [0, 1, 2, 0]
        assert compute_multi_trace_diagnostics(**b) == compute_multi_trace_diagnostics(
            **_small_batch()
        )


class TestLengthRobustSeqErrorMetrics:
    """`seq_{mean,max}_abs_err`, `seq_gen_tokens`, `alt_gate_masked_fraction`."""

    def test_mean_and_max_abs_err_and_gen_tokens_by_kind(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        _assert_all_finite(m)
        k = "by_segment_kind"
        # row 0: |err| = .1,.2,.3,.4 over 4 tokens; row 1: |err| = 5,5 over 2.
        assert m[f"logprob_error/seq_mean_abs_err/{k}/pre_compaction/mean"] == pytest.approx(0.25)
        assert m[f"logprob_error/seq_mean_abs_err/{k}/compaction_summary/mean"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_max_abs_err/{k}/pre_compaction/mean"] == pytest.approx(0.4)
        assert m[f"logprob_error/seq_max_abs_err/{k}/pre_compaction/max"] == pytest.approx(0.4)
        assert m[f"logprob_error/seq_max_abs_err/{k}/compaction_summary/mean"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_max_abs_err/{k}/compaction_summary/max"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_gen_tokens/{k}/pre_compaction/mean"] == pytest.approx(4.0)
        assert m[f"logprob_error/seq_gen_tokens/{k}/compaction_summary/mean"] == pytest.approx(2.0)
        # Not gate-eligible (env-masked / empty) -> omitted like the existing buckets.
        for kind in ("post_compaction", "empty"):
            assert not any(
                key.startswith("logprob_error/") and f"/{k}/{kind}" in key for key in m
            )

    def test_mean_and_max_abs_err_by_trace_index(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        i = "by_trace_in_rollout_idx"
        # bucket "0" = rows {0, 3}; row 3 (empty) is not eligible -> row 0 only.
        assert m[f"logprob_error/seq_mean_abs_err/{i}/0/mean"] == pytest.approx(0.25)
        assert m[f"logprob_error/seq_max_abs_err/{i}/0/mean"] == pytest.approx(0.4)
        assert m[f"logprob_error/seq_max_abs_err/{i}/0/max"] == pytest.approx(0.4)
        assert m[f"logprob_error/seq_gen_tokens/{i}/0/mean"] == pytest.approx(4.0)
        assert m[f"logprob_error/seq_mean_abs_err/{i}/1/mean"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_max_abs_err/{i}/1/mean"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_max_abs_err/{i}/1/max"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_gen_tokens/{i}/1/mean"] == pytest.approx(2.0)
        assert not any("/2plus" in key for key in m)

    def test_max_over_traces_differs_from_mean_in_shared_bucket(self):
        b = _small_batch()
        b["trace_in_rollout_idx"] = torch.tensor([0, 0, 2, 0])  # rows 0 and 1 share "0"
        m = compute_multi_trace_diagnostics(**b)
        i = "by_trace_in_rollout_idx"
        assert m[f"logprob_error/seq_mean_abs_err/{i}/0/mean"] == pytest.approx((0.25 + 5.0) / 2)
        assert m[f"logprob_error/seq_max_abs_err/{i}/0/mean"] == pytest.approx((0.4 + 5.0) / 2)
        assert m[f"logprob_error/seq_max_abs_err/{i}/0/max"] == pytest.approx(5.0)
        assert m[f"logprob_error/seq_gen_tokens/{i}/0/mean"] == pytest.approx(3.0)

    def test_alt_gate_masked_fraction(self):
        m = compute_multi_trace_diagnostics(**_small_batch(), seq_logprob_error_threshold=2.0)
        _assert_all_finite(m)
        a = "logprob_error/alt_gate_masked_fraction"
        # ln 2 ~ 0.693: row 0 (mean .25) passes, row 1 (mean 5) is masked.
        assert m[f"{a}/by_segment_kind/pre_compaction"] == pytest.approx(0.0)
        assert m[f"{a}/by_segment_kind/compaction_summary"] == pytest.approx(1.0)
        assert m[f"{a}/by_trace_in_rollout_idx/0"] == pytest.approx(0.0)
        assert m[f"{a}/by_trace_in_rollout_idx/1"] == pytest.approx(1.0)
        assert f"{a}/by_segment_kind/post_compaction" not in m
        assert f"{a}/by_segment_kind/empty" not in m
        # Jensen: never masks more than the live gate in any bucket.
        for key, value in m.items():
            if key.startswith(a + "/"):
                live = m["logprob_error/seq_masked_fraction/" + key[len(a) + 1 :]]
                assert value <= live + 1e-9, key

    def test_alt_gate_absent_without_usable_threshold(self):
        for thr in (None, 1.0, 0.5):
            m = compute_multi_trace_diagnostics(
                **_small_batch(), seq_logprob_error_threshold=thr
            )
            assert not any(k.startswith("logprob_error/alt_gate_masked_fraction/") for k in m)
            # The other length-robust keys do not depend on the threshold.
            assert "logprob_error/seq_mean_abs_err/by_segment_kind/pre_compaction/mean" in m
        # Defaults keep the existing call signature working unchanged.
        assert compute_multi_trace_diagnostics(**_small_batch()) == compute_multi_trace_diagnostics(
            **_small_batch(), seq_logprob_error_threshold=None, trace_rollout_ids=None
        )

    def test_alt_gate_is_length_robust_where_live_gate_is_not(self):
        # One 8-token trace with a single outlier |err| = 3: mean exp|err| =
        # (7 + e^3) / 8 ~ 3.39 > 2 -> the live gate masks it; mean |err| =
        # 0.375 < ln 2 -> the length-robust alternative does not.
        token_mask = torch.tensor([[0] + [1] * 8], dtype=torch.float32)
        prev = torch.zeros(1, 9)
        prev[0, 8] = -3.0
        live_err = (7 + math.exp(3.0)) / 8
        m = compute_multi_trace_diagnostics(
            seq_mult_prob_error=torch.tensor([live_err]),
            masked_by_seq_logprob_error=torch.tensor([True]),
            pre_seq_error_sample_loss_mask=torch.tensor([1.0]),
            mask_sample=[False],
            is_empty_rollout=[False],
            trace_in_rollout_idx=[0],
            trace_kinds=["uncompacted"],
            token_mask=token_mask,
            sample_mask=torch.tensor([0.0]),
            generation_logprobs=torch.zeros(1, 9),
            prev_logprobs=prev,
            num_unpadded_traces=1,
            seq_logprob_error_threshold=2.0,
        )
        _assert_all_finite(m)
        assert m["logprob_error/seq_masked_fraction/by_segment_kind/uncompacted"] == 1.0
        assert m["logprob_error/alt_gate_masked_fraction/by_segment_kind/uncompacted"] == 0.0
        assert m["logprob_error/seq_mean_abs_err/by_segment_kind/uncompacted/mean"] == pytest.approx(0.375)
        assert m["logprob_error/seq_max_abs_err/by_segment_kind/uncompacted/max"] == pytest.approx(3.0)
        assert m["logprob_error/seq_gen_tokens/by_segment_kind/uncompacted/mean"] == pytest.approx(8.0)

    def test_nan_logprobs_are_dropped_not_propagated(self):
        b = _small_batch()
        # A NaN at a NON-generated position of row 0 must change nothing.
        b["prev_logprobs"][0, 5] = float("nan")
        m = compute_multi_trace_diagnostics(**b, seq_logprob_error_threshold=2.0)
        _assert_all_finite(m)
        p = "logprob_error/seq_mean_abs_err/by_segment_kind/pre_compaction/mean"
        assert m[p] == pytest.approx(0.25)
        assert m["logprob_error/alt_gate_masked_fraction/by_segment_kind/pre_compaction"] == 0.0
        # A NaN on a generated token: the trace's mean/max drop out of the
        # averages (no other eligible trace in the bucket -> keys omitted), the
        # token count stays, and the alt gate counts it as masked like the live
        # gate would (`err <= thr` is False for NaN).
        b["prev_logprobs"][0, 2] = float("nan")
        m = compute_multi_trace_diagnostics(**b, seq_logprob_error_threshold=2.0)
        _assert_all_finite(m)
        assert p not in m
        assert "logprob_error/seq_max_abs_err/by_segment_kind/pre_compaction/mean" not in m
        assert "logprob_error/seq_max_abs_err/by_segment_kind/pre_compaction/max" not in m
        assert m["logprob_error/seq_gen_tokens/by_segment_kind/pre_compaction/mean"] == pytest.approx(4.0)
        assert m["logprob_error/alt_gate_masked_fraction/by_segment_kind/pre_compaction"] == 1.0


class TestTrainableTokenMetrics:
    """`multi_trace/trainable_tokens/*` and the per-rollout compaction split."""

    def test_totals_by_kind_and_rollups(self):
        m = compute_multi_trace_diagnostics(**_small_batch())
        _assert_all_finite(m)
        t = "multi_trace/trainable_tokens"
        # Only row 0 (pre_compaction, 4 tokens) survives token_mask * sample_mask.
        assert m[f"{t}/total"] == 4.0
        assert m[f"{t}/by_segment_kind/pre_compaction"] == 4.0
        assert m[f"{t}/by_segment_kind/compaction_summary"] == 0.0
        assert m[f"{t}/by_segment_kind/post_compaction"] == 0.0
        assert m[f"{t}/by_segment_kind/empty"] == 0.0
        assert m[f"{t}/compacted_rollouts"] == 4.0
        assert m[f"{t}/uncompacted_rollouts"] == 0.0
        assert m[f"{t}/compacted_fraction"] == pytest.approx(1.0)
        # No rollout ids -> no per-rollout keys.
        assert not any(
            k.startswith("multi_trace/rollouts/") or "_per_rollout/" in k for k in m
        )

    def test_per_rollout_classification(self):
        m = compute_multi_trace_diagnostics(**_small_batch(), trace_rollout_ids=[0, 0, 0, 1])
        _assert_all_finite(m)
        # Rollout 0 = rows {0,1,2} (compaction kinds) -> compacted;
        # rollout 1 = row 3 whose only trace is `empty` -> uncompacted.
        assert m["multi_trace/rollouts/compacted_count"] == 1.0
        assert m["multi_trace/rollouts/uncompacted_count"] == 1.0
        assert m["multi_trace/trainable_tokens_per_rollout/compacted/mean"] == pytest.approx(4.0)
        assert m["multi_trace/trainable_tokens_per_rollout/uncompacted/mean"] == pytest.approx(0.0)
        assert m["multi_trace/traces_per_rollout/compacted/mean"] == pytest.approx(3.0)
        assert m["multi_trace/traces_per_rollout/uncompacted/mean"] == pytest.approx(1.0)

    def test_padded_rollout_ids_and_list_input_agree(self):
        # The trainer pads trace_rollout_ids with duplicate rows; only the first
        # num_unpadded_traces entries count. List and tensor inputs agree.
        m_list = compute_multi_trace_diagnostics(**_small_batch(), trace_rollout_ids=[0, 0, 0, 1])
        m_padded = compute_multi_trace_diagnostics(
            **_small_batch(), trace_rollout_ids=torch.tensor([0, 0, 0, 1, 0, 0])
        )
        assert m_list == m_padded
        assert "multi_trace/rollouts/compacted_count" in m_list

    def test_uncompacted_kind_rollup_and_mixed_rollouts(self):
        m = compute_multi_trace_diagnostics(
            **_uncompacted_batch(), trace_rollout_ids=[0, 0, 1, 2], seq_logprob_error_threshold=2.0
        )
        _assert_all_finite(m)
        t = "multi_trace/trainable_tokens"
        assert m[f"{t}/total"] == 8.0
        assert m[f"{t}/by_segment_kind/pre_compaction"] == 4.0
        assert m[f"{t}/by_segment_kind/uncompacted"] == 4.0
        assert m[f"{t}/compacted_rollouts"] == 4.0
        assert m[f"{t}/uncompacted_rollouts"] == 4.0
        assert m[f"{t}/compacted_fraction"] == pytest.approx(0.5)
        # Rollout 0 = rows {0,1} compacted; rollout 1 = row 2 (uncompacted);
        # rollout 2 = row 3 (empty) -> two uncompacted rollouts, 4 and 0 tokens.
        assert m["multi_trace/rollouts/compacted_count"] == 1.0
        assert m["multi_trace/rollouts/uncompacted_count"] == 2.0
        assert m["multi_trace/trainable_tokens_per_rollout/compacted/mean"] == pytest.approx(4.0)
        assert m["multi_trace/trainable_tokens_per_rollout/uncompacted/mean"] == pytest.approx(2.0)
        assert m["multi_trace/traces_per_rollout/compacted/mean"] == pytest.approx(2.0)
        assert m["multi_trace/traces_per_rollout/uncompacted/mean"] == pytest.approx(1.0)
        # The uncompacted trace also shows up in the length-robust buckets.
        assert m["logprob_error/seq_mean_abs_err/by_segment_kind/uncompacted/mean"] == pytest.approx(0.5)
        assert m["logprob_error/seq_max_abs_err/by_segment_kind/uncompacted/max"] == pytest.approx(0.5)
        assert m["logprob_error/seq_gen_tokens/by_segment_kind/uncompacted/mean"] == pytest.approx(4.0)
        assert m["logprob_error/alt_gate_masked_fraction/by_segment_kind/uncompacted"] == 0.0
        assert m["logprob_error/seq_masked_fraction/by_segment_kind/uncompacted"] == 0.0

    def test_subagent_tokens_in_total_but_in_neither_rollup(self):
        b = _uncompacted_batch()
        b["trace_kinds"] = ["pre_compaction", "compaction_summary", "subagent", "empty"]
        m = compute_multi_trace_diagnostics(**b, trace_rollout_ids=[0, 0, 0, 1])
        _assert_all_finite(m)
        t = "multi_trace/trainable_tokens"
        assert m[f"{t}/total"] == 8.0
        assert m[f"{t}/by_segment_kind/subagent"] == 4.0
        assert m[f"{t}/compacted_rollouts"] == 4.0
        assert m[f"{t}/uncompacted_rollouts"] == 0.0
        assert m[f"{t}/compacted_fraction"] == pytest.approx(0.5)
        # Per rollout the subagent trace belongs to a rollout that compacted.
        assert m["multi_trace/rollouts/compacted_count"] == 1.0
        assert m["multi_trace/trainable_tokens_per_rollout/compacted/mean"] == pytest.approx(8.0)
        assert m["multi_trace/traces_per_rollout/compacted/mean"] == pytest.approx(3.0)

    def test_no_valid_tokens_omits_fraction_and_keeps_counts(self):
        b = _small_batch()
        b["sample_mask"] = torch.zeros(4)
        m = compute_multi_trace_diagnostics(**b, trace_rollout_ids=[0, 0, 0, 1])
        _assert_all_finite(m)
        t = "multi_trace/trainable_tokens"
        assert m[f"{t}/total"] == 0.0
        assert m[f"{t}/by_segment_kind/pre_compaction"] == 0.0
        assert m[f"{t}/compacted_rollouts"] == 0.0
        assert m[f"{t}/uncompacted_rollouts"] == 0.0
        assert f"{t}/compacted_fraction" not in m
        assert m["multi_trace/rollouts/compacted_count"] == 1.0
        assert m["multi_trace/trainable_tokens_per_rollout/compacted/mean"] == 0.0
        assert m["multi_trace/traces_per_rollout/compacted/mean"] == pytest.approx(3.0)

    def test_empty_rollout_class_reports_zero_count_and_no_mean(self):
        m = compute_multi_trace_diagnostics(**_small_batch(), trace_rollout_ids=[0, 0, 0, 0])
        _assert_all_finite(m)
        assert m["multi_trace/rollouts/compacted_count"] == 1.0
        assert m["multi_trace/rollouts/uncompacted_count"] == 0.0
        assert "multi_trace/trainable_tokens_per_rollout/uncompacted/mean" not in m
        assert "multi_trace/traces_per_rollout/uncompacted/mean" not in m
        assert m["multi_trace/traces_per_rollout/compacted/mean"] == pytest.approx(4.0)

    def test_unknown_kind_is_uncompacted_per_rollout_but_in_neither_rollup(self):
        b = _small_batch()
        b["trace_kinds"] = ["pre_compaction", "compaction_summary", "post_compaction", None]
        m = compute_multi_trace_diagnostics(**b, trace_rollout_ids=[0, 0, 0, 1])
        assert m["multi_trace/trainable_tokens/by_segment_kind/unknown"] == 0.0
        assert m["multi_trace/rollouts/uncompacted_count"] == 1.0
        assert m["multi_trace/trainable_tokens/uncompacted_rollouts"] == 0.0


class TestFinalizeSumCountMetrics:
    def test_means_and_rates(self):
        aggregated = {
            "rollouts/count": 4,
            "reward/by_num_compactions/0/sum": 3.0,
            "reward/by_num_compactions/0/count": 4,
            "reward/by_num_compactions/1/sum": 0.0,
            "reward/by_num_compactions/1/count": 0,
            "termination/completed/count": 3,
            "termination/agent_timeout/count": 1,
            "reward/by_termination/completed/sum": 3.0,
            "reward/by_termination/completed/count": 3,
            "mask_sample/by_kind/agent_timeout/count": 1,
            "format/think_tag_violation/count": 2,
            "format/think_tag_violation/by_trace_in_rollout_idx/0/count": 2,
            "reward/masked_resolved/count": 1,
            "compaction/summary_gen_tokens/sum": 3000,
            "compaction/summary_gen_tokens/count": 2,
            "compaction/summary_gen_tokens/max": 2000,
            "total_reward/mean": 0.75,
            "turns_per_trace/max": 7,
            "turns_per_sample/histogram": ["not-a-number"],
        }
        out = finalize_sum_count_metrics(aggregated)

        # /sum -> /mean when count > 0; /sum never survives; /count is kept.
        assert not any(k.endswith("/sum") for k in out)
        assert out["reward/by_num_compactions/0/mean"] == pytest.approx(0.75)
        assert out["reward/by_num_compactions/0/count"] == 4
        assert "reward/by_num_compactions/1/mean" not in out
        assert out["reward/by_num_compactions/1/count"] == 0
        assert out["reward/by_termination/completed/mean"] == pytest.approx(1.0)
        assert out["compaction/summary_gen_tokens/mean"] == pytest.approx(1500.0)
        assert out["compaction/summary_gen_tokens/max"] == 2000

        # Rollout-level event counts get a /rate over rollouts/count.
        assert out["termination/completed/rate"] == pytest.approx(0.75)
        assert out["termination/agent_timeout/rate"] == pytest.approx(0.25)
        assert out["mask_sample/by_kind/agent_timeout/rate"] == pytest.approx(0.25)
        assert out["format/think_tag_violation/rate"] == pytest.approx(0.5)
        assert out["reward/masked_resolved/rate"] == pytest.approx(0.25)
        # ... but per-trace and reward-split counts do not.
        assert "reward/by_termination/completed/rate" not in out
        assert "format/think_tag_violation/by_trace_in_rollout_idx/0/rate" not in out
        assert "compaction/summary_gen_tokens/rate" not in out

        # Everything else passes through untouched.
        assert out["total_reward/mean"] == 0.75
        assert out["turns_per_trace/max"] == 7
        assert out["turns_per_sample/histogram"] == ["not-a-number"]
        assert out["rollouts/count"] == 4
        _assert_all_finite({k: v for k, v in out.items() if k != "turns_per_sample/histogram"})

    def test_no_rollout_count_means_no_rates(self):
        out = finalize_sum_count_metrics(
            {
                "termination/completed/count": 3,
                "reward/masked/sum": 2.0,
                "reward/masked/count": 2,
            }
        )
        assert out == {
            "termination/completed/count": 3,
            "reward/masked/count": 2,
            "reward/masked/mean": 1.0,
        }

    def test_does_not_mutate_input_and_ignores_orphan_sum(self):
        aggregated = {"rollouts/count": 2, "orphan/sum": 5.0}
        out = finalize_sum_count_metrics(aggregated)
        assert aggregated == {"rollouts/count": 2, "orphan/sum": 5.0}
        assert out == {"rollouts/count": 2}
