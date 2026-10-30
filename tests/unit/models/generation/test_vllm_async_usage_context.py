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

"""The async vLLM engine must be created with the LLM-class usage context.

vLLM resolves ``max_num_batched_tokens``/``max_num_seqs`` from the usage
context when the user leaves them unset. ``AsyncLLM.from_engine_args`` defaults
to ``ENGINE_CONTEXT``, which has no entry in vLLM's defaults table and falls
back to 2048 batched tokens, while ``vllm.LLM`` declares ``LLM_CLASS`` and gets
a hardware-aware default. These tests pin the async worker to ``LLM_CLASS`` so
the two engines agree.
"""

import logging
import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest


class FakeUsageContext(Enum):
    ENGINE_CONTEXT = "ENGINE_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"


@pytest.fixture
def fake_vllm(monkeypatch):
    """Stub the vLLM modules ``_create_engine`` imports, recording the call."""
    recorded = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncLLM:
        @classmethod
        def from_engine_args(cls, engine_args, **kwargs):
            recorded["engine_args"] = engine_args
            recorded["kwargs"] = kwargs
            return SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(
                        max_num_batched_tokens=16384, max_num_seqs=1024
                    )
                )
            )

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": types.ModuleType("vllm.engine.arg_utils"),
        "vllm.usage": types.ModuleType("vllm.usage"),
        "vllm.usage.usage_lib": types.ModuleType("vllm.usage.usage_lib"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": types.ModuleType("vllm.v1.engine.async_llm"),
        "vllm.v1.metrics": types.ModuleType("vllm.v1.metrics"),
        "vllm.v1.metrics.loggers": types.ModuleType("vllm.v1.metrics.loggers"),
    }
    modules["vllm.config"].CompilationConfig = object

    class FakeSchedulerConfig:
        DEFAULT_MAX_NUM_SEQS = 128
        DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048

    modules["vllm.config"].SchedulerConfig = FakeSchedulerConfig
    modules["vllm.engine.arg_utils"].AsyncEngineArgs = FakeAsyncEngineArgs
    modules["vllm.usage.usage_lib"].UsageContext = FakeUsageContext
    modules["vllm.v1.engine.async_llm"].AsyncLLM = FakeAsyncLLM
    modules["vllm.v1.metrics.loggers"].PrometheusStatLogger = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return recorded


_WORKER_LOGGER = "nemo_rl.models.generation.vllm.vllm_worker_async"


def _worker(vllm_kwargs=None):
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {"max_model_len": 4096},
        "vllm_kwargs": vllm_kwargs or {},
    }
    return worker


def test_async_engine_declares_the_llm_class_usage_context(fake_vllm):
    worker = _worker()
    worker._create_engine({"model": "test-model"})

    assert fake_vllm["kwargs"]["usage_context"] is FakeUsageContext.LLM_CLASS
    assert fake_vllm["kwargs"]["usage_context"] is not FakeUsageContext.ENGINE_CONTEXT


def test_async_engine_does_not_inject_a_token_budget(fake_vllm):
    """vLLM only applies its own floors/caps to a value it chose itself."""
    worker = _worker()
    worker._create_engine({"model": "test-model", "max_model_len": 4096})

    # The only kwarg _create_engine adds is the max_num_seqs pin; the token
    # budget is left for vLLM to resolve from the usage context.
    assert set(fake_vllm["engine_args"].kwargs) == {
        "model",
        "max_model_len",
        "max_num_seqs",
    }
    assert "max_num_batched_tokens" not in fake_vllm["engine_args"].kwargs


def test_max_num_seqs_is_pinned_to_the_previous_default(fake_vllm):
    """The usage context must move the token budget only, not concurrency."""
    worker = _worker()
    worker._create_engine({"model": "test-model"})

    assert fake_vllm["engine_args"].kwargs["max_num_seqs"] == 128


def test_explicit_max_num_seqs_is_not_pinned(fake_vllm):
    worker = _worker(vllm_kwargs={"max_num_seqs": 512})
    worker._create_engine({"model": "test-model", "max_num_seqs": 512})

    assert fake_vllm["engine_args"].kwargs["max_num_seqs"] == 512


def test_explicit_user_batch_size_is_passed_through(fake_vllm):
    worker = _worker(vllm_kwargs={"max_num_batched_tokens": 32768})
    worker._create_engine({"model": "test-model", "max_num_batched_tokens": 32768})

    assert fake_vllm["engine_args"].kwargs["max_num_batched_tokens"] == 32768
    assert fake_vllm["kwargs"]["usage_context"] is FakeUsageContext.LLM_CLASS


def test_performance_mode_skips_the_pin(fake_vllm):
    """vLLM doubles both knobs there, but only the ones it chose itself."""
    worker = _worker(vllm_kwargs={"performance_mode": "throughput"})
    worker._create_engine({"model": "test-model", "performance_mode": "throughput"})

    assert "max_num_seqs" not in fake_vllm["engine_args"].kwargs


def _log_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if "vLLM async engine scheduler" in record.getMessage()
    ]


def test_effective_batching_config_is_logged(fake_vllm, caplog):
    caplog.set_level(logging.INFO, logger=_WORKER_LOGGER)
    worker = _worker()
    worker._create_engine({"model": "test-model"})

    messages = _log_messages(caplog)
    assert len(messages) == 1, caplog.records
    assert "max_num_batched_tokens=16384" in messages[0]


@pytest.mark.parametrize(
    ("vllm_kwargs", "expected_source"),
    [
        pytest.param({}, "pinned by NeMo RL", id="pinned"),
        pytest.param(
            {"max_num_seqs": 512}, "from vllm_kwargs", id="explicit_user_value"
        ),
        pytest.param(
            {"performance_mode": "throughput"},
            "vLLM default for the LLM_CLASS usage context",
            id="left_to_vllm",
        ),
    ],
)
def test_max_num_seqs_source_is_labelled_correctly(
    fake_vllm, caplog, vllm_kwargs, expected_source
):
    """128 is NeMo RL's pin, not a vLLM default; the log must not conflate them."""
    caplog.set_level(logging.INFO, logger=_WORKER_LOGGER)
    worker = _worker(vllm_kwargs=vllm_kwargs)
    worker._create_engine({"model": "test-model", **vllm_kwargs})

    message = _log_messages(caplog)[0]
    seqs_part = message.split("max_num_seqs=")[1]
    assert expected_source in seqs_part, message


@pytest.mark.vllm
def test_real_async_llm_accepts_a_usage_context():
    """Pin the kwarg name against the installed vLLM, which the stubs cannot."""
    import inspect

    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    parameters = inspect.signature(AsyncLLM.from_engine_args).parameters
    assert "usage_context" in parameters
    # The default is what this change exists to override.
    assert parameters["usage_context"].default is UsageContext.ENGINE_CONTEXT
    assert hasattr(UsageContext, "LLM_CLASS")


@pytest.mark.vllm
def test_real_scheduler_config_default_max_num_seqs_is_what_we_pin():
    from vllm.config import SchedulerConfig

    assert SchedulerConfig.DEFAULT_MAX_NUM_SEQS == 128
