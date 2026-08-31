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

"""Guards the context-length overflow contract with the NeMo Gym vLLM proxy.

When a prompt no longer fits the context window, Gym's vllm_model proxy ends
the rollout gracefully — it returns an empty completion with
finish_reason="length" rather than failing the sample. That recovery only
happens when our chat endpoint answers with HTTP 400 *and* a body Gym
recognises:

    # Gym responses_api_models/vllm_model/app.py
    is_out_of_context_length = e.status == 400 and (
        "context length" in result_content_str or "max_tokens" in result_content_str
    )

Both halves are easy to break from either side: returning 500 (an uncaught
ValueError), or rewording the message so the substring check misses. Either
regression is silent — the rollout just burns its retry budget and fails — so
the predicate is mirrored here rather than described in prose.
"""

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    CONTEXT_LENGTH_ERROR_MARKER,
    context_length_overflow_message,
    is_context_length_error,
)


def gym_recovers(status_code: int, body: str) -> bool:
    """Mirror of Gym's classifier in responses_api_models/vllm_model/app.py."""
    return status_code == 400 and (
        "context length" in body or "max_tokens" in body
    )


# Messages seen in production for the three ways vLLM reports an overflow.
# The first two are raised as plain ValueError, which is why the endpoint
# cannot key on VLLMValidationError alone.
RENDERER_MESSAGE = (
    "Input length (196905) exceeds model's maximum context length (196608)."
)
VLLM_SERVING_MESSAGE = (
    "This model's maximum context length is 32768 tokens. However, you "
    "requested 32818 tokens in the messages, Please reduce the length of "
    "the messages. None"
)
CLAMP_MESSAGE = context_length_overflow_message(196614, 196608)


@pytest.mark.parametrize(
    "message",
    [CLAMP_MESSAGE, RENDERER_MESSAGE, VLLM_SERVING_MESSAGE],
    ids=["clamp", "renderer", "vllm_serving"],
)
def test_overflow_errors_are_detected(message):
    assert is_context_length_error(ValueError(message))


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "top_logprobs must be set when requesting token information",
        "Expected all tensors to be on the same device",
    ],
)
def test_unrelated_errors_are_not_detected(message):
    """Unrelated ValueErrors must keep surfacing as 500s, not be masked as 400s."""
    assert not is_context_length_error(ValueError(message))


@pytest.mark.parametrize(
    "message",
    [CLAMP_MESSAGE, RENDERER_MESSAGE, VLLM_SERVING_MESSAGE],
    ids=["clamp", "renderer", "vllm_serving"],
)
def test_gym_recovers_from_the_response_we_send(message):
    """End of the contract: what we return must be what Gym acts on."""
    assert is_context_length_error(ValueError(message)), (
        "endpoint would let this escape as a 500"
    )
    # The endpoint answers 400 with the exception text as the body.
    assert gym_recovers(400, message)


def test_gym_would_not_recover_from_a_500():
    """The bug this contract exists to prevent: right body, wrong status."""
    assert not gym_recovers(500, CLAMP_MESSAGE)


def test_clamp_message_reports_both_lengths():
    """Operators need the actual numbers to tell overflow from a config error."""
    message = context_length_overflow_message(196614, 196608)
    assert "196614" in message and "196608" in message
    assert CONTEXT_LENGTH_ERROR_MARKER in message


def test_chat_endpoint_catches_plain_value_error_and_answers_400():
    """The endpoint must catch ValueError, not just VLLMValidationError.

    The helpers above cannot see this half of the contract: it lives in
    create_chat_completion, nested inside a Ray actor method. Narrowing the
    except clause is precisely the regression that used to send Gym a 500 —
    vLLM's renderer and the max-token clamp both raise a plain ValueError — so
    the handler is asserted against the source.
    """
    import ast
    from pathlib import Path

    from nemo_rl.models.generation.vllm import vllm_worker_async

    tree = ast.parse(Path(vllm_worker_async.__file__).read_text())

    handlers = [
        h
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if "VLLMValidationError"
        in {n.id for n in ast.walk(h.type) if isinstance(n, ast.Name)}
    ]
    assert handlers, "no handler for VLLMValidationError found"

    overflow_handlers = [
        h
        for h in handlers
        if "ValueError" in {n.id for n in ast.walk(h.type) if isinstance(n, ast.Name)}
    ]
    assert overflow_handlers, (
        "create_chat_completion must catch plain ValueError alongside "
        "VLLMValidationError; otherwise renderer/clamp overflows escape as "
        "HTTP 500 and Gym cannot recover the rollout"
    )

    returns_400 = any(
        kw.arg == "status_code" and getattr(kw.value, "value", None) == 400
        for h in overflow_handlers
        for call in ast.walk(h)
        if isinstance(call, ast.Call)
        for kw in call.keywords
    )
    assert returns_400, "the overflow handler must answer HTTP 400"
