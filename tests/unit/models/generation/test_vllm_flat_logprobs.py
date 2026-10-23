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

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

# Compile the production constructor/generate method without Ray or GPU imports.
# Only the downstream engine iterator is fake; the option handling is unchanged.
G_ROOT = Path(__file__).resolve().parents[4]
G_TREE = ast.parse(
    (G_ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py").read_text()
)
G_CLIENT = next(
    n
    for n in G_TREE.body
    if isinstance(n, ast.ClassDef) and n.name == "_AsyncLLMHTTPClient"
)
G_CLIENT.body = [
    n
    for n in G_CLIENT.body
    if isinstance(n, ast.FunctionDef) and n.name in {"__init__", "generate"}
]
G_FUTURE = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
)
G_NAMESPACE = {"threading": threading}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[G_FUTURE, G_CLIENT], type_ignores=[])
        ),
        "<production-client>",
        "exec",
    ),
    G_NAMESPACE,
)
_AsyncLLMHTTPClient = G_NAMESPACE["_AsyncLLMHTTPClient"]

G_CONFIG = ast.parse((G_ROOT / "nemo_rl/models/generation/vllm/config.py").read_text())
G_VALIDATOR = next(
    n
    for n in G_CONFIG.body
    if isinstance(n, ast.FunctionDef) and n.name == "validate_vllm_async_actor_settings"
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[G_FUTURE, G_VALIDATOR], type_ignores=[])
        ),
        "<production-validator>",
        "exec",
    ),
    G_NAMESPACE,
)
validate_vllm_async_actor_settings = G_NAMESPACE["validate_vllm_async_actor_settings"]


def _client(*, flat_logprobs: bool) -> _AsyncLLMHTTPClient:
    engine_client = SimpleNamespace(
        model_config=None, renderer=None, input_processor=None, vllm_config=None
    )
    client = _AsyncLLMHTTPClient(
        engine_client, SimpleNamespace(), flat_logprobs=flat_logprobs
    )
    client._generate = lambda prompt, sampling_params, request_id, kwargs: (
        sampling_params
    )
    return client


@pytest.mark.parametrize(
    ("flat_logprobs", "logprobs", "prompt_logprobs", "expected"),
    [
        (True, 0, None, True),  # logprobs requested: store them flat
        (True, None, None, False),  # no logprobs requested: nothing to store
        (False, 0, None, False),  # off: vLLM's default per-token objects
        # Prompt logprobs requested: vLLM's -inf clamp cannot edit flat storage.
        (True, 0, 0, False),
    ],
)
def test_generate_sets_flat_logprobs(
    flat_logprobs, logprobs, prompt_logprobs, expected
):
    sampling_params = SimpleNamespace(
        logprobs=logprobs, prompt_logprobs=prompt_logprobs, flat_logprobs=False
    )

    _client(flat_logprobs=flat_logprobs).generate("prompt", sampling_params, "req-0")

    assert sampling_params.flat_logprobs is expected


@pytest.mark.parametrize(
    ("vllm_cfg", "match"),
    [
        (
            {"flat_logprobs": True, "async_engine": False, "expose_http_server": True},
            "flat_logprobs only applies",
        ),
        (
            {"flat_logprobs": True, "async_engine": True, "expose_http_server": False},
            "flat_logprobs only applies",
        ),
        (
            {"output_loop_yield_budget_ms": 20, "async_engine": False},
            "output_loop_yield_budget_ms requires async_engine",
        ),
        (
            {"output_loop_yield_budget_ms": 0, "async_engine": True},
            "output_loop_yield_budget_ms must be positive",
        ),
    ],
)
def test_async_actor_settings_reject_ignored_options(
    vllm_cfg: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_vllm_async_actor_settings({"vllm_cfg": vllm_cfg})


@pytest.mark.parametrize(
    "vllm_cfg",
    [
        {},
        {"flat_logprobs": False, "output_loop_yield_budget_ms": None},
        {
            "flat_logprobs": True,
            "output_loop_yield_budget_ms": 20,
            "async_engine": True,
            "expose_http_server": True,
        },
    ],
)
def test_async_actor_settings_accept_supported_options(vllm_cfg: dict) -> None:
    validate_vllm_async_actor_settings({"vllm_cfg": vllm_cfg})


def test_async_actor_settings_allow_omitted_vllm_cfg() -> None:
    validate_vllm_async_actor_settings({})
