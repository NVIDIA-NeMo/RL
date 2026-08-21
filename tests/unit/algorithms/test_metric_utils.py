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
    print_setup_timing_summary,
)


class TestPrintSetupTimingSummary:
    """print_setup_timing_summary has three code paths with assertions."""

    @staticmethod
    def _common_setup(**overrides) -> SetupTimingMetrics:
        base = {
            "policy_init_time_s": 20.0,
            "other_setup_time_s": 1.0,
            "total_setup_time_s": 25.0,
        }
        base.update(overrides)
        return SetupTimingMetrics(**base)

    def test_sc_gym_on_prints_reserve_load_split(self, capsys):
        """SC + gym-on renders the '(reserve X.Xs + load Y.Ys)' suffix."""
        metrics = self._common_setup(
            generation_init_time_s=15.0,
            generation_init_reserve_time_s=3.0,
            generation_init_load_time_s=12.0,
        )
        print_setup_timing_summary(metrics)
        out = capsys.readouterr().out
        assert "Generation init: 15.0s (reserve 3.0s + load 12.0s)" in out

    def test_sc_gym_off_prints_plain_generation_init(self, capsys):
        """SC + gym-off renders only the top-level generation_init_time_s."""
        metrics = self._common_setup(generation_init_time_s=15.0)
        print_setup_timing_summary(metrics)
        out = capsys.readouterr().out
        assert "Generation init: 15.0s\n" in out
        # no reserve/load suffix on this path.
        assert "reserve" not in out
        assert "load" not in out

    def test_sc_gym_off_asserts_generation_init_time_populated(self):
        """SC path with gen_init_time_key=None must have generation_init_time_s set."""
        metrics = self._common_setup()
        with pytest.raises(AssertionError):
            print_setup_timing_summary(metrics)

    def test_grpo_uses_backend_specific_key(self, capsys):
        """grpo.py path reads the field named by gen_init_time_key."""
        metrics = self._common_setup(vllm_init_time_s=15.0)
        print_setup_timing_summary(metrics, gen_init_time_key="vllm_init_time_s")
        out = capsys.readouterr().out
        assert "Generation init: 15.0s" in out
        assert "reserve" not in out

    def test_grpo_asserts_generation_init_time_unset(self):
        """grpo.py path forbids generation_init_time_s from being populated."""
        metrics = self._common_setup(
            generation_init_time_s=15.0,
            vllm_init_time_s=15.0,
        )
        with pytest.raises(AssertionError):
            print_setup_timing_summary(metrics, gen_init_time_key="vllm_init_time_s")

    def test_reserve_load_split_takes_precedence_over_gen_key(self, capsys):
        """If reserve_time_s is set, the SC+gym-on branch wins even if a key is passed."""
        metrics = self._common_setup(
            generation_init_time_s=15.0,
            generation_init_reserve_time_s=3.0,
            generation_init_load_time_s=12.0,
        )
        print_setup_timing_summary(metrics, gen_init_time_key="vllm_init_time_s")
        out = capsys.readouterr().out
        assert "Generation init: 15.0s (reserve 3.0s + load 12.0s)" in out

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
