# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import copy
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parents[3]
PLUGIN_ROOT = REPO_ROOT / "tools/external_rollout_vllm/nemo_rl_vllm_prefix_plugin"
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

core_utils_spec = importlib.util.spec_from_file_location(
    "_nemo_rl_openai_server_utils",
    REPO_ROOT / "nemo_rl/models/generation/openai_server_utils.py",
)
assert core_utils_spec is not None and core_utils_spec.loader is not None
core_utils = importlib.util.module_from_spec(core_utils_spec)
core_utils_spec.loader.exec_module(core_utils)
core_replace_prefix_tokens = core_utils.replace_prefix_tokens

from nemo_rl_vllm_prefix_plugin.capture_runtime import CaptureRuntime  # noqa: E402
from nemo_rl_vllm_prefix_plugin.prefix import (  # noqa: E402
    replace_prefix_tokens as plugin_replace_prefix_tokens,
)
from nemo_rl_vllm_prefix_plugin.renderer_patch import (  # noqa: E402
    install_renderer_patch,
)
from nemo_rl_vllm_prefix_plugin.route_override import (  # noqa: E402
    CHAT_COMPLETIONS_PATH,
    remove_existing_chat_completion_routes,
)


class _Tokenizer:
    eos_token_id = 2

    @staticmethod
    def decode(token_ids):
        return repr(token_ids)


class _ValidationError(Exception):
    def __init__(self, message, *, parameter=None, value=None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class _Request:
    def __init__(self, required_prefix_token_ids, max_completion_tokens=100):
        self.required_prefix_token_ids = required_prefix_token_ids
        self.ng_capture = None
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = None
        self.add_generation_prompt = True
        self.kv_transfer_params = None

    def model_copy(self, *, update):
        result = copy.copy(self)
        for name, value in update.items():
            setattr(result, name, value)
        return result


class _OnlineRenderer:
    def __init__(self, max_model_len=8):
        self.model_config = SimpleNamespace(max_model_len=max_model_len)
        self.renderer = SimpleNamespace(tokenizer=_Tokenizer())
        self.seen_max_tokens = []

    async def preprocess_chat(self, request, messages, **kwargs):
        del kwargs
        self.seen_max_tokens.append(request.max_completion_tokens)
        if request.kv_transfer_params and request.kv_transfer_params.get(
            "prompt_token_ids"
        ):
            token_ids = request.kv_transfer_params.pop("prompt_token_ids")
        elif request.add_generation_prompt:
            token_ids = [10, 100, 2, 20, 21]
        else:
            assert messages[-1]["role"] == "assistant"
            token_ids = [10, 100, 2]
        return messages, [{"prompt_token_ids": token_ids}]


class _CaptureRuntimeForRenderer:
    def __init__(self):
        self.recorded = []

    @staticmethod
    def resolve_prefix(admission, *, prefill_prompt_token_ids):
        assert admission["mode"] == "token_in"
        assert prefill_prompt_token_ids is None
        return [10, 200, 201, 2]

    def record_prompt(self, request, **kwargs):
        self.recorded.append((request, kwargs))


@pytest.mark.parametrize(
    ("model_prefix", "template_prefix", "template_tokens"),
    [
        ([], [10, 100, 2], [10, 100, 2, 20]),
        ([10, 200, 2], [10, 100, 2], [10, 100, 2, 20]),
        (
            [10, 200, 2, 30, 201, 2],
            [10, 100, 2, 30, 101, 2],
            [10, 100, 2, 30, 101, 2, 40],
        ),
    ],
)
def test_plugin_prefix_helper_matches_nemo_rl(
    model_prefix, template_prefix, template_tokens
):
    tokenizer = _Tokenizer()

    assert plugin_replace_prefix_tokens(
        tokenizer, model_prefix, template_prefix, template_tokens
    ) == core_replace_prefix_tokens(
        tokenizer, model_prefix, template_prefix, template_tokens
    )


@pytest.mark.asyncio
async def test_renderer_patch_splices_exact_prefix_and_clamps_output_length():
    renderer = _OnlineRenderer(max_model_len=8)
    assert install_renderer_patch(renderer, _ValidationError) is True
    assert install_renderer_patch(renderer, _ValidationError) is False
    request = _Request([10, 200, 201, 2])
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]

    _, engine_inputs = await renderer.preprocess_chat(
        request,
        messages,
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
    )

    assert engine_inputs[0]["prompt_token_ids"] == [10, 200, 201, 2, 20, 21]
    assert request.max_completion_tokens == 2
    assert renderer.seen_max_tokens == [1, 1]


@pytest.mark.asyncio
async def test_renderer_patch_leaves_prompt_unchanged_without_prefix():
    renderer = _OnlineRenderer(max_model_len=10)
    install_renderer_patch(renderer, _ValidationError)
    request = _Request(None, max_completion_tokens=8)

    _, engine_inputs = await renderer.preprocess_chat(
        request,
        [{"role": "user", "content": "hello"}],
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
    )

    assert engine_inputs[0]["prompt_token_ids"] == [10, 100, 2, 20, 21]
    assert request.max_completion_tokens == 5
    assert renderer.seen_max_tokens == [1]


@pytest.mark.asyncio
async def test_renderer_patch_reuses_pd_prefill_prompt_without_second_render():
    renderer = _OnlineRenderer(max_model_len=10)
    install_renderer_patch(renderer, _ValidationError)
    request = _Request([10, 200, 201, 2], max_completion_tokens=8)
    request.kv_transfer_params = {"prompt_token_ids": [10, 200, 201, 2, 20, 21]}

    _, engine_inputs = await renderer.preprocess_chat(
        request,
        [{"role": "user", "content": "second"}],
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
    )

    assert engine_inputs[0]["prompt_token_ids"] == [10, 200, 201, 2, 20, 21]
    assert renderer.seen_max_tokens == [1]


@pytest.mark.asyncio
async def test_renderer_patch_restores_requested_length_after_render_failure():
    renderer = _OnlineRenderer()

    async def fail(*args, **kwargs):
        del args, kwargs
        raise ValueError("render failed")

    renderer.preprocess_chat = fail
    install_renderer_patch(renderer, _ValidationError)
    request = _Request([1, 2], max_completion_tokens=42)

    with pytest.raises(ValueError, match="render failed"):
        await renderer.preprocess_chat(
            request,
            [],
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs=None,
        )

    assert request.max_completion_tokens == 42


@pytest.mark.asyncio
async def test_renderer_patch_resolves_ng_capture_and_records_engine_prompt():
    renderer = _OnlineRenderer(max_model_len=8)
    capture_runtime = _CaptureRuntimeForRenderer()
    install_renderer_patch(renderer, _ValidationError, capture_runtime)
    request = _Request(None)
    request.ng_capture = {"mode": "token_in", "prev_len": 4}

    _, engine_inputs = await renderer.preprocess_chat(
        request,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
    )

    assert engine_inputs[0]["prompt_token_ids"] == [10, 200, 201, 2, 20, 21]
    assert request.required_prefix_token_ids == [10, 200, 201, 2]
    assert capture_runtime.recorded == [
        (
            request,
            {
                "admission": request.ng_capture,
                "prefix_token_ids": [10, 200, 201, 2],
                "prompt_token_ids": [10, 200, 201, 2, 20, 21],
            },
        )
    ]


def test_plugin_metadata_registers_vllm_endpoint_plugin():
    metadata = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())

    assert metadata["project"]["dependencies"] == ["vllm==0.29.0"]
    assert metadata["project"]["version"] == "0.2.0"
    assert metadata["project"]["entry-points"]["vllm.endpoint_plugins"] == {
        "nemo_rl_prefix_api": (
            "nemo_rl_vllm_prefix_plugin.plugin:NeMoRLPrefixEndpointPlugin"
        )
    }


def test_plugin_removes_stock_chat_route_before_registering_override():
    get_route = SimpleNamespace(path=CHAT_COMPLETIONS_PATH, methods={"GET"})
    stock_post_route = SimpleNamespace(path=CHAT_COMPLETIONS_PATH, methods={"POST"})
    other_route = SimpleNamespace(path="/health", methods={"GET"})
    app = SimpleNamespace(
        router=SimpleNamespace(routes=[get_route, stock_post_route, other_route])
    )

    assert remove_existing_chat_completion_routes(app) == 1
    assert app.router.routes == [get_route, other_route]


def test_plugin_refuses_to_claim_override_without_stock_chat_route():
    app = SimpleNamespace(
        router=SimpleNamespace(
            routes=[SimpleNamespace(path="/health", methods={"GET"})]
        )
    )

    with pytest.raises(RuntimeError, match="could not find"):
        remove_existing_chat_completion_routes(app)


def test_automationbench_launcher_enables_prefix_token_round_trip():
    launcher = (
        REPO_ROOT
        / "tools/external_rollout_vllm/launch_automationbench_super_external_vllm.sh"
    ).read_text()

    assert "VLLM_PLUGINS=nemo_rl_prefix_api" in launcher
    assert 'PYTHONPATH="${PREFIX_PLUGIN_WHEEL}"' in launcher
    assert "NEMO_RL_VLLM_PREFIX_PLUGIN_REQUIRED=1" in launcher
    assert '"token_capture.enabled=true"' in launcher
    assert "request_prompt_and_generation_token_ids=false" in launcher
    assert "supply_prefix_token_ids=false" in launcher


def test_external_server_checks_prefix_capability_before_registration():
    runner = (REPO_ROOT / "tools/external_gym_vllm/run_in_allocation.sh").read_text()

    capability_check = runner.index("Verified NeMo RL token-capture API capability")
    backend_registration = runner.index('registry_add "${REPLICA_ID}" "${HEAD_IP}"')
    assert capability_check < backend_registration


class _CaptureBridge:
    def __init__(self):
        self.commits = []

    @staticmethod
    def fetch_prefix(staging_chain):
        assert staging_chain == ["call-0", "call-1"]
        return [10, 11, 12]

    def commit(self, body):
        self.commits.append(body)
        return {"disposition": "staged", "staging_key": "call-2"}


def test_capture_runtime_fetches_staged_prefix_and_commits_exact_tokens():
    runtime = CaptureRuntime()
    bridge = _CaptureBridge()
    runtime._bridge = bridge
    admission = {
        "mode": "token_in",
        "prev_len": 3,
        "required_prefix_token_ids": [],
        "staging_chain": ["call-0", "call-1"],
    }
    request = object()

    prefix = runtime.resolve_prefix(admission, prefill_prompt_token_ids=None)
    runtime.record_prompt(
        request,
        admission=admission,
        prefix_token_ids=prefix,
        prompt_token_ids=[10, 11, 12, 20],
    )
    content = {
        "choices": [
            {
                "message": {"content": "done"},
                "logprobs": {
                    "content": [
                        {"token": "token_id:30", "logprob": -0.1},
                        {"token": "31", "logprob": -0.2},
                    ]
                },
            }
        ]
    }

    result = runtime.finish(request, content)

    assert result["ng_commit_coords"] == {
        "disposition": "staged",
        "staging_key": "call-2",
    }
    assert "logprobs" not in result["choices"][0]
    assert bridge.commits == [
        {
            "admission": admission,
            "prefix_token_ids": [10, 11, 12],
            "prompt_token_ids": [10, 11, 12, 20],
            "generated_token_ids": [30, 31],
            "generated_logprobs": [-0.1, -0.2],
            "extras": None,
        }
    ]


def test_capture_runtime_reuses_pd_prefill_prompt_as_prefix():
    runtime = CaptureRuntime()
    admission = {
        "mode": "token_in",
        "prev_len": 3,
        "required_prefix_token_ids": [],
        "staging_chain": ["not-fetched"],
    }

    assert runtime.resolve_prefix(
        admission,
        prefill_prompt_token_ids=[10, 11, 12, 20, 21],
    ) == [10, 11, 12]
