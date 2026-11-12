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
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def report() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "skills/rollout-perf/rollout_perf_report.py"
    )
    spec = importlib.util.spec_from_file_location("rollout_perf_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_sample_percentile_stays_inside_observed_range(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    run = {report.GEN: {"1": 10, "2": 20}}
    report.compare([run], [run])
    output = capsys.readouterr().out
    assert "p90 19.0 s, max 20.0 s" in output
    assert "not pooled run tails" in output


def test_missing_steps_are_disclosed(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    report.compare(
        [{report.GEN: {"1": 10, "2": 1000}}, {report.GEN: {"1": 10}}],
        [{report.GEN: {"1": 5, "2": 500}}],
    )
    output = capsys.readouterr().out
    assert "unequal step coverage" in output
    assert "base [2, 1], treat [2]" in output
    assert "paired steps: 1" in output
    assert "0.500" in output


def test_same_count_different_steps_are_disclosed(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    report.compare([{report.GEN: {"1": 10}}], [{report.GEN: {"2": 5}}])
    output = capsys.readouterr().out
    assert "unequal step coverage" in output
    assert "no steps with generation timing" in output


def test_async_comparison_uses_exposed_time(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    key = report.PHASES["exposed generation (async)"]
    report.compare([{key: {"1": 10}}], [{key: {"1": 5}}])
    output = capsys.readouterr().out
    assert "exposed generation (non-overlapped) ratio treat/base median 0.500" in output
    assert "no steps with generation timing" not in output


def test_different_timing_semantics_are_not_paired(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    key = report.PHASES["exposed generation (async)"]
    report.compare([{report.GEN: {"1": 10}}], [{key: {"1": 5}}])
    output = capsys.readouterr().out
    assert "no steps with generation timing" in output
    assert "ratio treat/base median" not in output


def test_engine_dumps_do_not_supply_other_engines_settings(
    report: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "driver.log"
    path.write_text(
        "Initializing a V1 LLM engine (v0.29.0) with config: "
        "enforce_eager=False, enable_chunked_prefill=False\n"
        "LLM Args: enable_chunked_prefill=True, max_num_tokens=32768\n"
    )
    parsed = report.scan_log(path)
    assert parsed["trtllm"]["enable_chunked_prefill"] == "True"
    assert report.normalize(parsed)["chunked_prefill"] == ("on", "trtllm")


def test_config_echo_is_not_an_engine_anchor(
    report: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "driver.log"
    path.write_text(
        "MasterConfig(server_args=ServerArgs(disable_cuda_graph=True))\n"
        "unrelated requested config: disable_cuda_graph=True\n"
    )
    assert "sglang" not in report.scan_log(path)


@pytest.mark.parametrize(
    ("line", "engine", "field", "value"),
    [
        ("ServerArgs(disable_cuda_graph=True)", "sglang", "cuda_graphs", "off"),
        ("LLM Args: enable_block_reuse=False", "trtllm", "prefix_reuse", "off"),
        (
            "Initialized persistent inference engine\n[graph 1/4] [512]",
            "megatron",
            "cuda_graphs",
            "on (4 graphs, up to 512 tokens)",
        ),
        (
            "Initializing a V1 LLM engine (v0.29.0) with config: enforce_eager=True",
            "vllm",
            "cuda_graphs",
            "off",
        ),
    ],
)
def test_backend_setting_normalization(
    report: ModuleType, tmp_path: Path, line: str, engine: str, field: str, value: str
) -> None:
    path = tmp_path / "driver.log"
    path.write_text(line)
    assert report.normalize(report.scan_log(path))[field] == (value, engine)


def test_missing_timing_does_not_claim_a_bottleneck(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    report.diagnose({}, {}, None)
    assert "cannot diagnose" in capsys.readouterr().out


def test_zero_baseline_does_not_divide_by_zero(
    report: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    report.compare([{report.GEN: {"1": 0}}], [{report.GEN: {"1": 5}}])
    output = capsys.readouterr().out
    assert "(n/a)" in output
    assert "ratio treat/base median" not in output
