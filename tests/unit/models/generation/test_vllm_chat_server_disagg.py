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

"""The engine's chat server on PD disaggregation legs: supplied ids, shadow check."""

import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("vllm")

from nemo_rl.models.generation.openai_server_utils import (  # noqa: E402
    encode_token_ids_b64,
)
from nemo_rl.models.generation.vllm import chat_server  # noqa: E402
from nemo_rl.models.generation.vllm.chat_server import (  # noqa: E402
    NeMoRLChatCompletionRequest,
    _gym_prefix_from_messages,
    _is_prefill_leg,
    _supplied_prompt_token_ids,
)

_MESSAGES = [
    {"role": "user", "content": "q"},
    {
        "role": "assistant",
        "content": "a",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
    },
    {"role": "user", "content": "q2"},
]


def test_server_side_leg_helpers():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
        },
        {"role": "user", "content": "q2"},
    ]
    assert _gym_prefix_from_messages(messages) == [1, 2, 3]
    assert _gym_prefix_from_messages([{"role": "user", "content": "q"}]) is None

    assert _supplied_prompt_token_ids(
        SimpleNamespace(prompt_token_ids_b64=encode_token_ids_b64([7, 8]))
    ) == [7, 8]
    assert _supplied_prompt_token_ids(
        SimpleNamespace(prompt_token_ids=[1], prompt_token_ids_b64=None)
    ) == [1]
    assert _supplied_prompt_token_ids(
        SimpleNamespace(kv_transfer_params={"prompt_token_ids": [2]})
    ) == [2]
    assert (
        _supplied_prompt_token_ids(
            SimpleNamespace(kv_transfer_params={"do_remote_prefill": True})
        )
        is None
    )

    assert _is_prefill_leg(
        SimpleNamespace(kv_transfer_params={"do_remote_decode": True})
    )
    assert not _is_prefill_leg(
        SimpleNamespace(kv_transfer_params={"do_remote_prefill": True})
    )
    assert not _is_prefill_leg(SimpleNamespace(kv_transfer_params=None))


def test_a_leg_with_supplied_ids_hands_them_to_vllm_and_derives_no_prefix():
    leg = NeMoRLChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": copy.deepcopy(_MESSAGES),
            "prompt_token_ids_b64": encode_token_ids_b64([7, 8, 9]),
            "kv_transfer_params": {"do_remote_decode": True},
            "template_reasoning_ended": False,
        }
    )
    # vLLM's renderer takes the ids from kv_transfer_params and skips templating.
    assert leg.kv_transfer_params == {
        "do_remote_decode": True,
        "prompt_token_ids": [7, 8, 9],
    }
    # Already spliced: Gym's token fields on the messages are not a prefix here.
    assert leg.required_prefix_token_ids is None
    assert leg.template_reasoning_ended is False

    plain = NeMoRLChatCompletionRequest.model_validate(
        {"model": "m", "messages": copy.deepcopy(_MESSAGES)}
    )
    assert plain.required_prefix_token_ids == [1, 2, 3]


class _ShadowRenderer(chat_server.NeMoRLOpenAIServingMixin):
    """The mixin over a scripted render, with no vLLM renderer underneath."""

    def __init__(self, recomputed=None, error=None):
        self.recomputed = recomputed
        self.error = error
        self.calls = []
        self._model_stop_token_ids = frozenset()
        self.renderer = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
        self.model_config = SimpleNamespace(max_model_len=64)

    async def _render_chat_call(self, request, messages, render_kwargs, *, admission):
        self.calls.append((request, messages, render_kwargs, admission))
        if self.error is not None:
            raise self.error
        return chat_server.RenderedChat(
            template_token_ids=list(self.recomputed),
            history_token_ids=None,
            # The splice itself is prepare_prompt's; this render has no prefix.
            required_prefix_token_ids=None,
            admission=None,
            serving_admission=None,
            verify_retained_media=False,
            request_max_tokens=None,
            reasoning_ended=None,
            conversation=[],
            engine_prompt={},
        )


def _leg_request(supplied):
    request = SimpleNamespace(
        kv_transfer_params={
            "do_remote_decode": True,
            "prompt_token_ids": list(supplied),
        }
    )
    request.model_copy = lambda update: SimpleNamespace(**update)
    return request


def test_shadow_check_recomputes_from_the_messages_and_logs_only_a_mismatch(
    caplog,
):
    caplog.set_level(logging.ERROR, logger=chat_server.__name__)
    render_kwargs = {"default_template": None, "parser": None}

    renderer = _ShadowRenderer(recomputed=[1, 2, 3, 4])
    asyncio.run(
        renderer._shadow_check_supplied_prompt_ids(
            _leg_request([1, 2, 3, 4]), _MESSAGES, [1, 2, 3, 4], render_kwargs
        )
    )

    [(shadow, messages, kwargs, admission)] = renderer.calls
    # The supplied ids are withheld so the shadow render really renders, and
    # it splices from the Gym fields the way an unassisted request would.
    assert shadow.kv_transfer_params == {"do_remote_decode": True}
    assert shadow.required_prefix_token_ids == [1, 2, 3]
    assert (
        shadow.prompt_token_ids,
        shadow.prompt_token_ids_b64,
        shadow.template_reasoning_ended,
        shadow.ng_capture,
    ) == (None, None, None, None)
    assert messages == _MESSAGES and messages is not _MESSAGES
    # No capture begins for the shadow render.
    assert kwargs == render_kwargs and admission is None
    assert caplog.records == []

    renderer = _ShadowRenderer(recomputed=[1, 2, 9, 4])
    asyncio.run(
        renderer._shadow_check_supplied_prompt_ids(
            _leg_request([1, 2, 3, 4]), copy.deepcopy(_MESSAGES), [1, 2, 3, 4], {}
        )
    )
    assert "SHADOW MISMATCH" in caplog.text
    assert "first divergence at 2" in caplog.text


def test_shadow_check_never_fails_serving(caplog):
    caplog.set_level(logging.ERROR, logger=chat_server.__name__)

    asyncio.run(
        _ShadowRenderer(
            error=RuntimeError("no template")
        )._shadow_check_supplied_prompt_ids(_leg_request([1]), _MESSAGES, [1], {})
    )

    assert "shadow recompute failed" in caplog.text
