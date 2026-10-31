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

"""What NeMo-RL turns on inside vLLM when native tracing is requested."""

import sys
import types

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import (
    _maybe_enable_vllm_native_tracing,
)

ENDPOINT = "http://collector:4317"


@pytest.fixture
def engine_args(monkeypatch):
    """Stand in for vLLM's EngineArgs, supporting both tracing knobs.

    The real class is only reachable with vLLM installed, and the function
    under test only ever reads which parameter names exist.
    """

    class EngineArgs:
        def __init__(
            self,
            otlp_traces_endpoint=None,
            collect_detailed_traces=None,
        ):
            pass

    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = EngineArgs
    engine = types.ModuleType("vllm.engine")
    engine.arg_utils = arg_utils
    vllm = types.ModuleType("vllm")
    vllm.engine = engine

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    return EngineArgs


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_VLLM_NATIVE_TRACING", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ENDPOINT)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)


def test_detailed_traces_stay_off_even_when_the_engine_supports_them(
    engine_args, tracing_on
):
    """The expensive knob is not ours to turn on for the user.

    vLLM calls collect_detailed_traces "possibly costly and or blocking": it
    times every request inside the engine, so it slows generation rather than
    only adding spans. Native tracing is already one span per request; this
    would add engine overhead on top.
    """
    llm_kwargs: dict = {}

    _maybe_enable_vllm_native_tracing(llm_kwargs)

    assert llm_kwargs["otlp_traces_endpoint"] == ENDPOINT
    assert "collect_detailed_traces" not in llm_kwargs


def test_a_caller_can_still_ask_for_detailed_traces(engine_args, tracing_on):
    llm_kwargs: dict = {"collect_detailed_traces": ["model"]}

    _maybe_enable_vllm_native_tracing(llm_kwargs)

    assert llm_kwargs["collect_detailed_traces"] == ["model"]


def test_nothing_is_set_when_telemetry_is_off(engine_args, tracing_on, monkeypatch):
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "0")
    llm_kwargs: dict = {}

    _maybe_enable_vllm_native_tracing(llm_kwargs)

    assert llm_kwargs == {}
