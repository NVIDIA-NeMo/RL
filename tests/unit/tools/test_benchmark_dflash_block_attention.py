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

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor

pytestmark = [
    pytest.mark.mcore,
    pytest.mark.filterwarnings("error:flex_attention called without torch\\.compile"),
]

_BENCHMARK_PATH = (
    Path(__file__).parents[3] / "tools/benchmark_dflash_block_attention.py"
)


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_dflash_block_attention", _BENCHMARK_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_benchmark_plan(benchmark: ModuleType) -> Any:
    return benchmark._build_plan(
        batch_size=1,
        sequence_length=32,
        anchors_per_sample=1,
        block_size=16,
        device=torch.device("cpu"),
    )


def test_benchmark_inputs_use_public_dflash_attention_geometry() -> None:
    benchmark = _load_benchmark()
    inputs = benchmark._make_inputs(
        _small_benchmark_plan(benchmark), device=torch.device("cpu")
    )

    assert [tuple(tensor.shape) for tensor in inputs] == [
        (1, 32, 8, 128),
        (1, 32, 8, 128),
        (1, 16, 32, 128),
        (1, 16, 8, 128),
        (1, 16, 8, 128),
    ]


def test_default_train_step_calls_production_block_only_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_benchmark()
    plan = _small_benchmark_plan(benchmark)
    inputs = benchmark._make_inputs(plan, device=torch.device("cpu"))
    calls = 0

    production_attention = benchmark.dflash_block_only_attention

    def record_block_only_attention(**kwargs: Any) -> Tensor:
        nonlocal calls
        calls += 1
        return production_attention(**kwargs)

    monkeypatch.setattr(
        benchmark, "dflash_block_only_attention", record_block_only_attention
    )
    benchmark._train_step(plan, inputs)

    assert calls == 1
    assert all(tensor.grad is not None for tensor in inputs)


def test_default_correctness_calls_production_block_only_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_benchmark()
    plan = _small_benchmark_plan(benchmark)
    calls = 0
    production_attention = benchmark.dflash_block_only_attention

    def record_block_only_attention(**kwargs: Any) -> Tensor:
        nonlocal calls
        calls += 1
        return production_attention(**kwargs)

    def run_without_cuda_timing(_device: torch.device, operation: Any) -> float:
        operation()
        return 0.0

    monkeypatch.setattr(benchmark, "_build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(benchmark, "_time_call", run_without_cuda_timing)
    monkeypatch.setattr(
        benchmark, "dflash_block_only_attention", record_block_only_attention
    )
    record = benchmark._correctness_comparison(device=torch.device("cpu"), iterations=1)

    assert calls == 2
    assert record["kind"] == "block_only_forward_correctness_comparison"
    assert record["attention_path"] == "dflash_block_only_attention"
