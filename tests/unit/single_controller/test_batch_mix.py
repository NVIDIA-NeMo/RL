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
"""Per-step batch composition by agent family (batch_mix/* metrics and chart)."""

from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.single_controller_utils.batch_mix import (
    UNKNOWN_AGENT,
    BatchMixHistory,
    agent_name_from_env_info,
    agent_slices,
    batch_mix_metrics,
)
from nemo_rl.utils.logger import Logger

SWE = "swe_rebench_opencode_sandboxed_agent"
TB = "terminal_bench_2_1_opencode_sandboxed_agent"
SEARCH = "browsecomp_sandboxed_agent"
GDPVAL = "gdpval_train"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (SWE, ("swe_rebench_opencode", "swe")),
        ("swemer_oml_opencode_sandboxed_agent", ("swemer_oml_opencode", "swe")),
        (TB, ("terminal_bench_2_1_opencode", "tb")),
        (SEARCH, ("browsecomp", "search")),
        (GDPVAL, ("gdpval_train", "gdpval")),
        (UNKNOWN_AGENT, (UNKNOWN_AGENT, "other")),
    ],
)
def test_agent_slices_maps_our_agents_to_families(name, expected):
    assert agent_slices(name) == expected


def test_agent_name_from_env_info_reads_agent_ref():
    assert agent_name_from_env_info({"agent_ref": {"type": "x", "name": TB}}) == TB
    assert agent_name_from_env_info({"agent_ref": SEARCH}) == SEARCH
    assert agent_name_from_env_info({"agent_ref": {"name": ""}}) == UNKNOWN_AGENT
    assert agent_name_from_env_info({}) == UNKNOWN_AGENT
    assert agent_name_from_env_info(None) == UNKNOWN_AGENT


def test_batch_mix_metrics_counts_and_shares():
    names = [SWE] * 4 + [TB] * 2 + [SEARCH] + [GDPVAL]
    metrics = batch_mix_metrics(names)

    assert metrics["batch_mix/groups"] == 8
    assert metrics["batch_mix/family/swe/groups"] == 4
    assert metrics["batch_mix/family/swe/frac"] == pytest.approx(0.5)
    assert metrics["batch_mix/family/tb/frac"] == pytest.approx(0.25)
    assert metrics["batch_mix/agent/browsecomp/frac"] == pytest.approx(0.125)
    family_fracs = [v for k, v in metrics.items() if k.endswith("/frac") and "/family/" in k]
    assert sum(family_fracs) == pytest.approx(1.0)
    # Families absent from the step are absent from its metrics.
    assert "batch_mix/family/other/frac" not in metrics


def test_batch_mix_metrics_empty_step_logs_nothing():
    assert batch_mix_metrics([]) == {}


def test_history_skips_empty_steps_and_keeps_the_last_max_steps():
    history = BatchMixHistory(max_steps=3)
    history.add(1, [SWE])
    history.add(2, [])
    for step in range(3, 7):
        history.add(step, [SWE, TB])

    assert history.steps == [4, 5, 6]
    assert history.family_counts == [{"swe": 1, "tb": 1}] * 3


def test_history_figure_is_a_100_percent_stacked_bar_per_step():
    pytest.importorskip("plotly")
    history = BatchMixHistory()
    assert history.figure() is None
    history.add(10, [SWE, SWE, SWE, SEARCH])
    history.add(11, [TB, GDPVAL])

    fig = history.figure()

    assert fig.layout.barmode == "stack"
    # Families in FAMILY_ORDER, only those present.
    assert [trace.name for trace in fig.data] == [
        "SWE",
        "Terminal-Bench",
        "Search",
        "GDPval",
    ]
    for step_index in range(2):
        total = sum(trace.y[step_index] for trace in fig.data)
        assert total == pytest.approx(100.0)
    swe = fig.data[0]
    assert list(swe.x) == [10, 11]
    assert list(swe.y) == pytest.approx([75.0, 0.0])
    assert [list(c) for c in swe.customdata] == [[3, 4], [0, 2]]


def test_logger_sends_plotly_figures_to_every_backend():
    logger = Logger.__new__(Logger)
    backends = [MagicMock(), MagicMock()]
    logger.loggers = backends
    figure = object()

    logger.log_plotly(figure, step=7, name="train/batch_mix/chart")

    for backend in backends:
        backend.log_plotly.assert_called_once_with(figure, 7, "train/batch_mix/chart")
