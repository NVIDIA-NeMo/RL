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
"""Megatron's side of NeMo RL's chat engine in Dynamo's frontend.

The backend's checks run without Megatron-LM; the BOS rule and the render
worker need a Megatron-LM with ``ChatRenderer`` (NVIDIA/Megatron-LM#9579), and
the real render a cached Qwen3 tokenizer, and are skipped without them.
"""

import asyncio
import importlib.util
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.dynamo.megatron.chat_backend import (
    CHAT_BACKEND,
    MegatronCallRejected,
    MegatronRenderer,
    megatron_chat_spec,
)
from nemo_rl.models.generation.dynamo.render_pool import RenderedCall, _renderer_class

pytestmark = pytest.mark.mcore

_CHAT_RENDERER_MODULE = (
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server."
    "endpoints.chat_renderer"
)


def _has_chat_renderer() -> bool:
    try:
        return importlib.util.find_spec(_CHAT_RENDERER_MODULE) is not None
    except ModuleNotFoundError:
        return False


needs_chat_renderer = pytest.mark.skipif(
    not _has_chat_renderer(), reason="this Megatron-LM has no ChatRenderer"
)

GENERATION = {
    "model_name": "Qwen/Qwen3-0.6B",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "val_temperature": 0.0,
    "val_top_p": 1.0,
    "val_top_k": None,
    "stop_token_ids": None,
    "_dynamo_role": "decode",
    "mcore_generation_config": {
        "expose_http_server": True,
        "parsers": [],
        "max_model_len": 64,
        "tensor_model_parallel_size": 2,
    },
}


def test_the_chat_spec_keeps_only_what_the_frontend_renders_with():
    spec = megatron_chat_spec(GENERATION)

    assert spec == {
        "generation": {
            "model_name": "Qwen/Qwen3-0.6B",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": None,
            "val_temperature": 0.0,
            "val_top_p": 1.0,
            "val_top_k": None,
            "mcore_generation_config": {
                "expose_http_server": True,
                "parsers": [],
                "max_model_len": 64,
            },
        }
    }
    assert CHAT_BACKEND.serves_chat(spec["generation"])


def test_the_renderer_is_found_by_name():
    assert _renderer_class(CHAT_BACKEND.renderer) is MegatronRenderer


@pytest.mark.parametrize(
    ("body", "filled"),
    [
        # Omitted fields take policy.generation's, as the native server's defaults.
        pytest.param({}, {"temperature": 0.7, "top_p": 0.9, "top_k": -1}, id="unset"),
        pytest.param(
            {"temperature": 1.0, "top_p": 0.5, "top_k": 4}, {}, id="caller_set"
        ),
    ],
)
def test_sampling_fills_what_the_call_leaves_unset(body, filled):
    assert (
        CHAT_BACKEND.checked_sampling(body, on_policy=True, generation=GENERATION)
        == filled
    )


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param({"n": 2}, "n=2", id="n"),
        pytest.param({"stream": True}, "stream=true", id="stream"),
        pytest.param({"logprobs": True, "top_logprobs": 2}, "top_logprobs", id="top"),
    ],
)
def test_sampling_refuses_what_the_engines_cannot_serve(body, match):
    with pytest.raises(MegatronCallRejected, match=match) as info:
        CHAT_BACKEND.checked_sampling(body, on_policy=True, generation=GENERATION)
    refusal = CHAT_BACKEND.error_reply(info.value)
    assert refusal is not None and refusal.status == 400
    assert not refusal.acknowledge


def _rendered(**fields) -> RenderedCall:
    return RenderedCall(
        **{
            "template_token_ids": [1, 7, 2, 3, 4],
            "history_token_ids": [1, 7, 2],
            "required_prefix_token_ids": None,
            "admission": None,
            "serving_admission": None,
            "verify_retained_media": False,
            "multimodal": None,
            "request": {"model": "model"},
            "request_max_tokens": None,
            "reasoning_ended": None,
            "reasoning_parser_kwargs": None,
            **fields,
        }
    )


def _prepare(rendered: RenderedCall, **kwargs):
    return asyncio.run(
        CHAT_BACKEND.prepare_prompt(
            rendered,
            resolve_prefix=kwargs.pop("resolve_prefix", lambda admission: []),
            render_info={
                "turn_end_token_ids": [2],
                "max_model_len": kwargs.pop("max_model_len", 16),
                "bos_token_id": kwargs.pop("bos_token_id", None),
            },
            measure_only=kwargs.pop("measure_only", False),
        )
    )


@needs_chat_renderer
def test_the_prompt_is_spliced_and_clamped_as_natively():
    # The model's own turn 1 (5, 6, EOS) replaces the template's re-render.
    plan = _prepare(
        _rendered(required_prefix_token_ids=[1, 5, 6, 2], request_max_tokens=100)
    )

    assert plan.prompt_token_ids == [1, 5, 6, 2, 3, 4]
    assert plan.prefix_len == 4
    assert plan.max_tokens == 16 - 6


@needs_chat_renderer
def test_a_token_in_prefix_is_read_from_tq():
    admission = SimpleNamespace(mode="token_in")
    reads = []
    plan = _prepare(
        _rendered(admission=admission, serving_admission=admission),
        resolve_prefix=lambda serving: reads.append(serving) or [1, 9, 2],
    )
    assert reads == [admission]
    assert plan.prompt_token_ids == [1, 9, 2, 3, 4]


@needs_chat_renderer
@pytest.mark.parametrize(
    ("add_bos", "prompt"),
    [
        # The template's BOS (1) is dropped unless the call asks for add_BOS.
        pytest.param(None, [7, 2, 3, 4], id="default"),
        pytest.param(False, [7, 2, 3, 4], id="off"),
        pytest.param(True, [1, 7, 2, 3, 4], id="on"),
    ],
)
def test_the_bos_rule_runs_before_the_clamp(add_bos, prompt):
    request = {"model": "model"}
    if add_bos is not None:
        request["add_BOS"] = add_bos
    plan = _prepare(_rendered(request=request, request_max_tokens=100), bos_token_id=1)

    assert plan.prompt_token_ids == prompt
    # The budget is measured on the prompt the engine runs.
    assert plan.max_tokens == 16 - len(prompt)


@needs_chat_renderer
def test_the_bos_rule_runs_after_the_splice():
    # The template renders BOS; the model's own prefix ran without one, as the
    # rule served its last turn. The splice keeps the model's prefix, and the
    # rule then leaves it as it is.
    plan = _prepare(
        _rendered(required_prefix_token_ids=[5, 6, 2], request_max_tokens=100),
        bos_token_id=1,
    )
    assert plan.prompt_token_ids == [5, 6, 2, 3, 4]
    assert plan.prefix_len == 3


@needs_chat_renderer
def test_a_prefix_the_bos_rule_would_change_is_refused():
    # The model's prefix starts with BOS, yet the call does not ask for one.
    with pytest.raises(MegatronCallRejected, match="add_BOS") as info:
        _prepare(_rendered(required_prefix_token_ids=[1, 5, 6, 2]), bos_token_id=1)
    assert CHAT_BACKEND.error_reply(info.value).status == 400


@needs_chat_renderer
def test_a_prompt_that_fills_the_context_is_refused_with_gyms_overflow_text():
    with pytest.raises(MegatronCallRejected, match="maximum context length") as info:
        _prepare(_rendered(), max_model_len=5)
    refusal = CHAT_BACKEND.error_reply(info.value)
    assert refusal.status == 400 and refusal.acknowledge
    # Gym measures any length.
    assert _prepare(_rendered(), max_model_len=5, measure_only=True).max_tokens is None


def test_other_errors_are_server_errors():
    assert CHAT_BACKEND.error_reply(RuntimeError("engine died")) is None


# ---------------------------------------------------------------------------
# The render worker over a faked ChatRenderer: what it does with the render
# ---------------------------------------------------------------------------


class _FakeChat:
    """``ChatRenderer``'s render, scripted."""

    def __init__(self, **rendered):
        self.rendered = {
            "prompt_token_ids": [1, 7, 2, 3, 4],
            "history_token_ids": None,
            "options": SimpleNamespace(chat_template_kwargs={}),
            "template_messages": [{"role": "user", "content": "hi"}],
            "template_tools": None,
            "media_slots": [],
            "multi_modal_data": None,
            **rendered,
        }
        self.renders: list[tuple] = []
        self.tokenized: list[tuple] = []

    def render(self, body, *, with_history=False, add_bos=None):
        self.renders.append((with_history, add_bos))
        return SimpleNamespace(**self.rendered)

    def has_chat_template(self, options):
        return True

    def tokenize(self, messages, media_slots=(), **kwargs):
        self.tokenized.append((messages, kwargs["add_generation_prompt"]))
        return [1, 7, 2]


def _fake_renderer(chat: _FakeChat) -> MegatronRenderer:
    renderer = object.__new__(MegatronRenderer)
    renderer.chat = chat
    return renderer


def test_the_render_leaves_the_bos_rule_to_the_frontend():
    chat = _FakeChat()
    rendered = _fake_renderer(chat).render(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "add_BOS": True}
    )

    # Without a prefix, no history is rendered; the BOS rule comes later.
    assert chat.renders == [(False, None)]
    assert rendered.template_token_ids == [1, 7, 2, 3, 4]
    assert rendered.history_token_ids is None
    assert rendered.request == {"model": "m", "add_BOS": True}


def test_an_echoed_prefix_takes_chatrenderers_history():
    chat = _FakeChat(history_token_ids=[1, 7, 2])
    rendered = _fake_renderer(chat).render(
        {"model": "m", "messages": [], "required_prefix_token_ids": [1, 5, 2]}
    )

    assert chat.renders == [(True, None)]
    assert rendered.history_token_ids == [1, 7, 2]
    assert rendered.required_prefix_token_ids == [1, 5, 2]
    assert chat.tokenized == []


def test_a_prefix_without_an_assistant_turn_cuts_against_the_whole_conversation():
    # ChatRenderer renders no history without an assistant turn; the vLLM path
    # splices against the whole conversation then.
    chat = _FakeChat()
    rendered = _fake_renderer(chat).render(
        {"model": "m", "messages": [], "required_prefix_token_ids": [1, 5, 2]}
    )

    assert chat.tokenized == [(chat.rendered["template_messages"], False)]
    assert rendered.history_token_ids == [1, 7, 2]


@pytest.mark.parametrize(
    "media",
    [
        pytest.param({"multi_modal_data": {"image": [b"x"]}}, id="bytes"),
        pytest.param({"media_slots": [("__S0__", "image", 0)]}, id="slots"),
    ],
)
def test_media_is_refused_from_the_render(media):
    with pytest.raises(MegatronCallRejected, match="text chat only") as info:
        _fake_renderer(_FakeChat(**media)).render({"model": "m", "messages": []})
    assert CHAT_BACKEND.error_reply(info.value).status == 400


def test_a_malformed_request_is_refused_with_400():
    class _Malformed(_FakeChat):
        def render(self, body, **kwargs):
            raise ValueError("Missing 'messages' field")

    with pytest.raises(MegatronCallRejected, match="Missing 'messages'") as info:
        _fake_renderer(_Malformed()).render({"model": "m"})
    assert CHAT_BACKEND.error_reply(info.value).status == 400


# ---------------------------------------------------------------------------
# The render worker over Megatron-LM's ChatRenderer and a Qwen3 tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture
def renderer():
    if not _has_chat_renderer():
        pytest.skip("this Megatron-LM has no ChatRenderer")
    from huggingface_hub import snapshot_download

    try:
        model_dir = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as error:  # noqa: BLE001 - any cache miss skips
        pytest.skip(f"no cached Qwen/Qwen3-0.6B tokenizer: {error}")
    MegatronRenderer.prepare_process()
    return MegatronRenderer(megatron_chat_spec(GENERATION), model_dir)


def test_render_and_reply_are_megatrons_chat_endpoints(renderer):
    messages = [{"role": "user", "content": "Hi"}]
    rendered = renderer.render({"model": "m", "messages": messages, "max_tokens": 8})

    # The prompt Megatron-LM's endpoint tokenizes, before its BOS rule.
    assert rendered.template_token_ids == renderer.chat.tokenize(messages)
    assert rendered.history_token_ids is None and rendered.admission is None
    assert rendered.request_max_tokens == 8
    assert "messages" not in rendered.request
    info = renderer.info()
    assert info["max_model_len"] == 64
    # The turn ends the splice cuts by are the endpoint's EOS ids.
    eos = renderer.chat.tokenizer.eod
    assert info["turn_end_token_ids"] == sorted(renderer.chat.eos_token_ids())
    assert eos in info["turn_end_token_ids"]
    assert info["bos_token_id"] == renderer.chat.tokenizer.bos

    hello = renderer.chat.tokenizer.tokenize("Hello")
    reply = renderer.respond(
        {"model": "m", "logprobs": True},
        rendered.template_token_ids,
        {
            "token_ids": [*hello, eos],
            "finish_reason": "stop",
            "log_probs": [-0.5] * (len(hello) + 1),
        },
    )
    (choice,) = reply["choices"]
    # The EOS is dropped from the text, as the endpoint's remove_EOD does.
    assert choice["message"]["content"] == "Hello"
    assert choice["finish_reason"] == "stop"
    assert len(choice["logprobs"]["content"]) == len(hello) + 1
    assert reply["usage"]["prompt_tokens"] == len(rendered.template_token_ids)
    assert reply["usage"]["completion_tokens"] == len(hello) + 1


def test_an_echoed_prefix_renders_the_history_it_replaces(renderer):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
    ]
    rendered = renderer.render(
        {"model": "m", "messages": messages, "required_prefix_token_ids": [1, 2]}
    )

    expected = renderer.chat.tokenize(messages[:2], add_generation_prompt=False)
    assert rendered.history_token_ids == expected
    assert rendered.required_prefix_token_ids == [1, 2]


def test_a_media_call_is_refused(renderer):
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ],
            }
        ],
    }
    # A text model's template has no media token: refused before it reaches
    # an engine either way.
    with pytest.raises(MegatronCallRejected):
        renderer.render(body)
