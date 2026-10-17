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

"""Tests for policy.generation.vllm_cfg.thinking_token_budget."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_rl.models.generation.vllm import vllm_worker as vllm_worker_mod
from nemo_rl.models.generation.vllm.config import (
    VLLM_THINKING_TOKEN_BUDGET_UNLIMITED,
    VLLM_USE_V2_MODEL_RUNNER_ENV_VAR,
    check_http_server_reasoning_parser,
    resolve_reasoning_parser,
    resolve_thinking_token_budget,
    thinking_token_budget_sampling_kwargs,
)

_CONFIG_LOGGER = "nemo_rl.models.generation.vllm.config"

BASE_GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "max_new_tokens": 64,
    "stop_token_ids": None,
    "stop_strings": None,
}


class RecordingSamplingParams:
    """Stand-in for ``vllm.SamplingParams`` that records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _v1_model_runner(monkeypatch):
    """Pin the runner off so finite-budget cases are not warned about."""
    monkeypatch.setenv(VLLM_USE_V2_MODEL_RUNNER_ENV_VAR, "0")


def _config(vllm_kwargs=None, **vllm_cfg_overrides):
    """A policy.generation config with the given vllm_cfg overrides."""
    return dict(
        BASE_GENERATION_CONFIG,
        vllm_cfg={"max_model_len": 4096, **vllm_cfg_overrides},
        vllm_kwargs=vllm_kwargs or {},
    )


def _build_sampling_params(config):
    from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    worker = SimpleNamespace(
        cfg=config,
        SamplingParams=RecordingSamplingParams,
        _extra_sampling_kwargs=thinking_token_budget_sampling_kwargs(config),
    )
    return BaseVllmGenerationWorker._build_sampling_params(
        worker, greedy=False, stop_strings=None
    )


# --------------------------------------------------------------- resolution


def test_unset_budget_resolves_to_none():
    assert resolve_thinking_token_budget(_config()) is None
    assert thinking_token_budget_sampling_kwargs(_config()) == {}


def test_explicit_none_resolves_to_none():
    cfg = _config(thinking_token_budget=None)
    assert resolve_thinking_token_budget(cfg) is None
    assert thinking_token_budget_sampling_kwargs(cfg) == {}


def test_budget_with_a_reasoning_parser_resolves():
    cfg = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    assert resolve_thinking_token_budget(cfg) == 2048
    assert thinking_token_budget_sampling_kwargs(cfg) == {"thinking_token_budget": 2048}


def test_zero_is_a_valid_budget():
    """vLLM's domain is non-negative; 0 means "no reasoning tokens"."""
    cfg = _config(thinking_token_budget=0, reasoning_parser="deepseek_r1")
    assert resolve_thinking_token_budget(cfg) == 0


def test_unlimited_does_not_require_a_reasoning_parser():
    """vLLM normalizes -1 to "no cap", so it never reaches the parser check."""
    cfg = _config(thinking_token_budget=VLLM_THINKING_TOKEN_BUDGET_UNLIMITED)
    assert resolve_thinking_token_budget(cfg) == VLLM_THINKING_TOKEN_BUDGET_UNLIMITED


def test_budget_without_a_reasoning_parser_is_rejected():
    cfg = _config(thinking_token_budget=2048)
    with pytest.raises(ValueError, match="no reasoning parser is configured"):
        resolve_thinking_token_budget(cfg)


def test_reasoning_parser_plugin_alone_is_not_enough():
    cfg = _config(thinking_token_budget=2048, reasoning_parser_plugin="/tmp/p.py")
    with pytest.raises(ValueError, match="no reasoning parser is configured"):
        resolve_thinking_token_budget(cfg)


@pytest.mark.parametrize(
    "budget",
    [True, False, 2048.0, "2048", -2],
    ids=["true", "false", "float", "str", "below_unlimited"],
)
def test_values_outside_the_vllm_domain_are_rejected(budget):
    cfg = _config(thinking_token_budget=budget, reasoning_parser="deepseek_r1")
    with pytest.raises(ValueError, match="thinking_token_budget"):
        resolve_thinking_token_budget(cfg)


# ------------------------------------------------------------- passthrough


def test_unset_budget_adds_no_sampling_param():
    assert "thinking_token_budget" not in _build_sampling_params(_config()).kwargs


def test_set_budget_reaches_sampling_params():
    cfg = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    assert _build_sampling_params(cfg).kwargs["thinking_token_budget"] == 2048


def test_other_sampling_params_are_unchanged_by_the_budget():
    without = _build_sampling_params(_config()).kwargs
    with_budget = _build_sampling_params(
        _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    ).kwargs
    assert with_budget.pop("thinking_token_budget") == 2048
    assert with_budget == without


def test_reasoning_parser_is_forwarded_to_the_engine():
    """The engine kwarg, not the sampling param, is what enables the budget."""
    from nemo_rl.models.generation.vllm import vllm_worker

    llm_kwargs: dict = {}
    config = _config(
        reasoning_parser="deepseek_r1", reasoning_parser_plugin="/tmp/parser.py"
    )
    vllm_worker.BaseVllmGenerationWorker._apply_reasoning_engine_kwargs(
        config, llm_kwargs
    )
    assert llm_kwargs == {
        "reasoning_parser": "deepseek_r1",
        "reasoning_parser_plugin": "/tmp/parser.py",
    }


def test_reasoning_parser_plugin_is_not_forwarded_on_its_own():
    """Existing configs that only set the plugin must keep their behavior."""
    from nemo_rl.models.generation.vllm import vllm_worker

    llm_kwargs: dict = {}
    vllm_worker.BaseVllmGenerationWorker._apply_reasoning_engine_kwargs(
        _config(reasoning_parser_plugin="/tmp/parser.py"), llm_kwargs
    )
    assert llm_kwargs == {}


# ------------------------------------------------- parser resolution sources


def test_no_reasoning_parser_resolves_to_none():
    assert resolve_reasoning_parser(_config()) is None


def test_reasoning_parser_from_vllm_cfg():
    assert (
        resolve_reasoning_parser(_config(reasoning_parser="deepseek_r1"))
        == "deepseek_r1"
    )


def test_reasoning_parser_from_vllm_kwargs_is_accepted():
    """vllm_kwargs is splatted into EngineArgs, so it already reaches vLLM."""
    config = _config(vllm_kwargs={"reasoning_parser": "deepseek_r1"})
    assert resolve_reasoning_parser(config) == "deepseek_r1"


def test_budget_is_satisfied_by_a_vllm_kwargs_parser():
    config = _config(
        thinking_token_budget=2048, vllm_kwargs={"reasoning_parser": "deepseek_r1"}
    )
    assert resolve_thinking_token_budget(config) == 2048


def test_matching_parsers_in_both_places_are_fine():
    config = _config(
        reasoning_parser="deepseek_r1",
        vllm_kwargs={"reasoning_parser": "deepseek_r1"},
    )
    assert resolve_reasoning_parser(config) == "deepseek_r1"


def test_conflicting_parsers_are_rejected():
    """Never silently pick a winner for the same EngineArgs field."""
    config = _config(
        reasoning_parser="deepseek_r1", vllm_kwargs={"reasoning_parser": "qwen3"}
    )
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_reasoning_parser(config)


def test_a_vllm_kwargs_parser_is_forwarded_unchanged():
    from nemo_rl.models.generation.vllm import vllm_worker

    llm_kwargs: dict = {"reasoning_parser": "deepseek_r1"}
    vllm_worker.BaseVllmGenerationWorker._apply_reasoning_engine_kwargs(
        _config(vllm_kwargs={"reasoning_parser": "deepseek_r1"}), llm_kwargs
    )
    assert llm_kwargs == {"reasoning_parser": "deepseek_r1"}


# --------------------------------------------------- HTTP serving cross-check


def test_serving_parser_agreeing_with_the_engine_is_quiet(caplog):
    caplog.set_level(logging.WARNING, logger=_CONFIG_LOGGER)
    config = _config(
        reasoning_parser="deepseek_r1",
        expose_http_server=True,
        http_server_serving_chat_kwargs={"reasoning_parser": "deepseek_r1"},
    )
    assert check_http_server_reasoning_parser(config) is None
    assert caplog.records == []


def test_serving_parser_disagreeing_with_the_engine_warns(caplog):
    caplog.set_level(logging.WARNING, logger=_CONFIG_LOGGER)
    config = _config(
        reasoning_parser="deepseek_r1",
        expose_http_server=True,
        http_server_serving_chat_kwargs={"reasoning_parser": "qwen3"},
    )

    assert check_http_server_reasoning_parser(config) == "qwen3"

    assert any(
        "differs from the engine" in record.getMessage() for record in caplog.records
    ), caplog.records


def test_serving_parser_is_not_checked_without_the_http_server(caplog):
    caplog.set_level(logging.WARNING, logger=_CONFIG_LOGGER)
    config = _config(
        reasoning_parser="deepseek_r1",
        http_server_serving_chat_kwargs={"reasoning_parser": "qwen3"},
    )
    assert check_http_server_reasoning_parser(config) is None
    assert caplog.records == []


# ------------------------------------------------- V2 runner / tokenizer


def test_v2_model_runner_enabled_is_rejected(monkeypatch):
    monkeypatch.setenv(VLLM_USE_V2_MODEL_RUNNER_ENV_VAR, "1")
    config = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    with pytest.raises(ValueError, match="V2 model runner"):
        resolve_thinking_token_budget(config)


def test_unset_v2_model_runner_warns_but_resolves(monkeypatch, caplog):
    """Unset means vLLM picks per architecture; we cannot decide it here."""
    monkeypatch.delenv(VLLM_USE_V2_MODEL_RUNNER_ENV_VAR, raising=False)
    caplog.set_level(logging.WARNING, logger=_CONFIG_LOGGER)
    config = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")

    assert resolve_thinking_token_budget(config) == 2048
    assert any(
        VLLM_USE_V2_MODEL_RUNNER_ENV_VAR in record.getMessage()
        for record in caplog.records
    ), caplog.records


def test_unlimited_budget_skips_the_v2_check(monkeypatch):
    monkeypatch.setenv(VLLM_USE_V2_MODEL_RUNNER_ENV_VAR, "1")
    config = _config(thinking_token_budget=VLLM_THINKING_TOKEN_BUDGET_UNLIMITED)
    assert resolve_thinking_token_budget(config) == VLLM_THINKING_TOKEN_BUDGET_UNLIMITED


def test_unparseable_v2_env_var_is_left_to_vllm(monkeypatch):
    monkeypatch.setenv(VLLM_USE_V2_MODEL_RUNNER_ENV_VAR, "maybe")
    config = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    assert resolve_thinking_token_budget(config) == 2048


def test_skip_tokenizer_init_with_a_budget_is_rejected():
    config = _config(
        thinking_token_budget=2048,
        reasoning_parser="deepseek_r1",
        skip_tokenizer_init=True,
    )
    with pytest.raises(ValueError, match="skip_tokenizer_init"):
        resolve_thinking_token_budget(config)


def test_skip_tokenizer_init_is_fine_without_a_budget():
    assert resolve_thinking_token_budget(_config(skip_tokenizer_init=True)) is None


# --------------------------------------------------------- worker wiring


def test_worker_init_assigns_the_extra_sampling_kwargs():
    """_build_sampling_params splats this attribute; _init_config must set it."""
    from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    config = _config(thinking_token_budget=2048, reasoning_parser="deepseek_r1")
    config["model_name"] = "test-model"
    config["vllm_cfg"].update(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "precision": "bfloat16",
        }
    )
    with patch.object(vllm_worker_mod, "_apply_vllm_patches"):
        worker._init_config(
            config,
            bundle_indices=None,
            fraction_of_gpus=1.0,
            seed=None,
            extra_env_vars=None,
        )

    assert worker._extra_sampling_kwargs == {"thinking_token_budget": 2048}


# ------------------------------------------------------------------ vLLM


@pytest.mark.vllm
@pytest.mark.parametrize("budget", [0, 512, VLLM_THINKING_TOKEN_BUDGET_UNLIMITED])
def test_real_vllm_sampling_params_accepts_the_budget(budget):
    """Pin the kwarg name and domain against the installed vLLM."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=1.0, thinking_token_budget=budget)
    # vLLM normalizes -1 ("unlimited") to None and keeps other values as-is.
    expected = None if budget == VLLM_THINKING_TOKEN_BUDGET_UNLIMITED else budget
    assert params.thinking_token_budget == expected


@pytest.mark.vllm
@pytest.mark.parametrize("budget", [True, 2048.0, -2])
def test_real_vllm_sampling_params_rejects_values_we_reject(budget):
    """Our config-time domain check must match vLLM's request-time one."""
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        SamplingParams(temperature=1.0, thinking_token_budget=budget)
