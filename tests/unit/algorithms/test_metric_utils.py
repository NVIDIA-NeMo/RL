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

"""Unit tests for SetupTimingMetrics + print_setup_timing_summary."""

from __future__ import annotations

import pytest

from nemo_rl.algorithms.metric_utils import (
    SetupTimingMetrics,
    normalize_filter_aware_mb_metrics,
    print_setup_timing_summary,
)


class TestNormalizeFilterAwareMbMetrics:
    def test_uses_actual_counts_across_uneven_microbatches(self):
        metrics = {
            "global_valid_toks": [5.0, 5.0],
            "probs_ratio": [4.0 / 5.0, 3.0 / 5.0],
            "token_mult_prob_error": [6.0 / 5.0, 4.0 / 5.0],
            "sampling_importance_ratio": [9.0 / 5.0, 3.0 / 5.0],
            "is_oob_ratio": [1.0 / 5.0, 1.0 / 5.0],
            "_actor_valid_toks": [2.0, 1.0],
            "_prev_valid_toks": [3.0, 2.0],
            "_sampling_importance_ratio_valid_toks": [3.0, 2.0],
            "_is_oob_valid_toks": [3.0, 2.0],
            "unrelated": [1.0, 2.0],
        }

        out = normalize_filter_aware_mb_metrics(metrics)

        assert out["probs_ratio"] == pytest.approx([7.0 / 3.0])
        assert out["token_mult_prob_error"] == pytest.approx([2.0])
        assert out["sampling_importance_ratio"] == pytest.approx([12.0 / 5.0])
        assert out["is_oob_ratio"] == pytest.approx([2.0 / 5.0])
        assert out["unrelated"] == [1.0, 2.0]
        assert not any(key.startswith("_") for key in out)
        assert metrics["_actor_valid_toks"] == [2.0, 1.0]

    def test_zero_actual_tokens_returns_zero(self):
        out = normalize_filter_aware_mb_metrics(
            {
                "global_valid_toks": [4.0],
                "probs_ratio": [0.0],
                "_actor_valid_toks": [0.0],
            }
        )
        assert out["probs_ratio"] == [0.0]

    def test_requires_original_denominator(self):
        with pytest.raises(ValueError, match="global_valid_toks"):
            normalize_filter_aware_mb_metrics(
                {"probs_ratio": [0.5], "_actor_valid_toks": [1.0]}
            )


class TestPrintSetupTimingSummary:
    """print_setup_timing_summary's code paths and assertions."""

    @staticmethod
    def _common_setup(**overrides) -> SetupTimingMetrics:
        base = {
            "policy_init_time_s": 20.0,
            "other_setup_time_s": 1.0,
            "total_setup_time_s": 25.0,
        }
        base.update(overrides)
        return SetupTimingMetrics(**base)

    def test_gym_on_prints_reserve_load_split(self, capsys):
        """Gym-on (a reserved address) renders the '(reserve X.Xs + load Y.Ys)' suffix."""
        metrics = self._common_setup(
            generation_init_time_s=15.0,
            generation_init_reserve_time_s=3.0,
            generation_init_load_time_s=12.0,
        )
        print_setup_timing_summary(metrics)
        out = capsys.readouterr().out
        assert "Generation init: 15.0s (reserve 3.0s + load 12.0s)" in out

    def test_gym_off_prints_plain_generation_init(self, capsys):
        """Gym-off renders only the top-level generation_init_time_s."""
        metrics = self._common_setup(generation_init_time_s=15.0)
        print_setup_timing_summary(metrics)
        out = capsys.readouterr().out
        assert "Generation init: 15.0s\n" in out
        # no reserve/load suffix on this path.
        assert "reserve" not in out
        assert "load" not in out

    def test_asserts_generation_init_time_populated(self):
        """Every driver must populate generation_init_time_s."""
        metrics = self._common_setup()
        with pytest.raises(AssertionError):
            print_setup_timing_summary(metrics)

    def test_optional_nemo_gym_and_teacher_lines(self, capsys):
        """nemo_gym_init_time_s and teacher_init_time_s only print when populated."""
        metrics = self._common_setup(
            generation_init_time_s=15.0,
            nemo_gym_init_time_s=8.0,
            teacher_init_time_s=6.0,
        )
        print_setup_timing_summary(metrics)
        out = capsys.readouterr().out
        assert "NeMo-Gym init: 8.0s" in out
        assert "Teacher init: 6.0s" in out

    def test_value_init_line_only_on_a_ppo_run(self, capsys):
        """Without it the summary does not add up: the critic's time is missing
        between Policy init and Total setup."""
        metrics = self._common_setup(generation_init_time_s=15.0)
        print_setup_timing_summary(metrics)
        assert "Value init" not in capsys.readouterr().out

        metrics = self._common_setup(generation_init_time_s=15.0, value_init_time_s=7.0)
        print_setup_timing_summary(metrics)
        assert "Value init: 7.0s" in capsys.readouterr().out


class TestSetupTimingMetricsToDict:
    """to_metrics_dict serializes into a dict for Logger.log_metrics."""

    def test_drops_none_fields(self):
        """Unset (None) fields are dropped."""
        metrics = SetupTimingMetrics(generation_init_time_s=1.5)
        d = metrics.to_metrics_dict()
        assert d == {"generation_init_time_s": 1.5}

    def test_zero_is_kept(self):
        """Zero survives the None-drop (the filter is 'is not None', not 'truthy')."""
        metrics = SetupTimingMetrics(generation_init_time_s=0.0, policy_init_time_s=0.0)
        d = metrics.to_metrics_dict()
        assert d == {"generation_init_time_s": 0.0, "policy_init_time_s": 0.0}

    def test_extras_merged_into_top_level(self):
        """extras dict entries appear as top-level keys, not nested."""
        metrics = SetupTimingMetrics(generation_init_time_s=1.0)
        metrics.extras["vllm_nccl_sparse_init_time_s"] = 2.5
        d = metrics.to_metrics_dict()
        assert d == {
            "generation_init_time_s": 1.0,
            "vllm_nccl_sparse_init_time_s": 2.5,
        }
        # extras itself is not exposed as a nested key.
        assert "extras" not in d

    def test_reserve_load_split_serialized(self):
        """Reserve/load split fields are included when populated."""
        metrics = SetupTimingMetrics(
            generation_init_time_s=15.0,
            generation_init_reserve_time_s=3.0,
            generation_init_load_time_s=12.0,
        )
        d = metrics.to_metrics_dict()
        assert d["generation_init_reserve_time_s"] == 3.0
        assert d["generation_init_load_time_s"] == 12.0
