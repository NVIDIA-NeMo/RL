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
"""NeMo RL's chat engine in Dynamo's frontend, over a fake render pool and router.

The render pool's own workers are covered by ``test_dynamo_render_pool.py``;
here they are replaced by what they return, so each test checks what the
engine does between the render and the reply.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("dynamo._core")
pytest.importorskip("vllm")

from dynamo.llm.exceptions import HttpError, InvalidArgument  # noqa: E402

from nemo_rl.models.generation.dynamo.vllm import chat_engine  # noqa: E402
from nemo_rl.models.generation.dynamo.vllm.render_pool import (  # noqa: E402
    RenderedCall,
)
from nemo_rl.models.generation.gym_call import (  # noqa: E402
    GYM_EXTRAS_ANNOTATION_PREFIX,
    GymCallResult,
)

EOS = 2
MAX_MODEL_LEN = 16
GENERATION = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "val_temperature": 0.0,
    "val_top_p": 1.0,
    "val_top_k": None,
    "stop_token_ids": [EOS],
    "vllm_cfg": {},
}
# Turn 2: the model's own turn 1 replaces the template's re-render of it.
PREFIX = [1, 5, 6, EOS]
HISTORY = [1, 7, EOS]
TEMPLATE = [*HISTORY, 3, 4]
SPLICED = [*PREFIX, 3, 4]
TOKEN_IN = {"mode": "token_in", "capture_key": "rollout-1", "model_call_id": "c2"}


def _admission(fields: dict) -> SimpleNamespace:
    """A parsed capture admission, as the render pool returns it."""
    return SimpleNamespace(**fields, model_dump=lambda mode="python": dict(fields))


def _rendered(**fields) -> RenderedCall:
    """What a render-pool worker returns for a call."""
    return RenderedCall(
        **{
            "template_token_ids": [1, 3, 4],
            "history_token_ids": None,
            "required_prefix_token_ids": None,
            "admission": None,
            "serving_admission": None,
            "verify_retained_media": False,
            "multimodal": None,
            "request": {"model": "model", "max_tokens": None},
            "request_max_tokens": None,
            "reasoning_ended": None,
            "reasoning_parser_kwargs": None,
            **fields,
        }
    )


class _Pool:
    def __init__(self, rendered: RenderedCall, reply: dict | None = None) -> None:
        self.rendered = rendered
        self.reply = reply or {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
        self.bodies: list[dict] = []
        self.responded: list[tuple] = []
        self.error: Exception | None = None

    async def info(self) -> dict:
        return {"turn_end_token_ids": [EOS], "max_model_len": MAX_MODEL_LEN}

    async def render(self, body: dict, *, measure: bool = False) -> RenderedCall:
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return self.rendered

    async def respond(self, request, prompt_token_ids, output, restored=None) -> dict:
        self.responded.append((request, prompt_token_ids, output))
        return self.reply


class _Item:
    """One annotated response from Dynamo's router."""

    def __init__(self, data) -> None:
        self._data = data

    def data(self):
        return self._data


class _Router:
    def __init__(self, engine_data=None, error=None) -> None:
        self.engine_data = engine_data or {
            "output": {"text": "ok", "token_ids": [10], "finish_reason": "stop"},
            "result": {"prompt_token_ids": [1, 3, 4], "generation_token_ids": [10]},
        }
        self.error = error
        self.requests: list[dict] = []

    async def generate(self, request, context=None):
        self.requests.append(request)

        async def stream():
            if isinstance(self.error, Exception):
                raise self.error
            if self.error is not None:
                # Dynamo's binding raises an engine's error frame as a ValueError.
                raise ValueError(self.error)
            yield _Item({"token_ids": [10], "engine_data": self.engine_data})

        return stream()


class _Capture:
    """The TQ prefix source: returns the staged prefix for a token_in admission."""

    def __init__(self) -> None:
        self.resolved: list[dict] = []

    def resolve_prefix(self, serving_admission) -> list[int]:
        self.resolved.append(serving_admission.model_dump(mode="json"))
        return PREFIX


def _call(extras: dict | None = None, **fields) -> dict:
    """A chat call as Dynamo's frontend hands it over: unset fields are None."""
    nvext = None
    if extras is not None:
        nvext = {"annotations": [GYM_EXTRAS_ANNOTATION_PREFIX + json.dumps(extras)]}
    return {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "u", "detail": None}},
                ],
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "n": None,
        "min_p": None,
        "presence_penalty": None,
        "logit_bias": None,
        "max_tokens": 20,
        "chat_template_args": None,
        "unsupported_fields": None,
        "nvext": nvext,
        **fields,
    }


def _serve(pool, router, request, capture=None) -> list[dict]:
    engine = chat_engine.NemoRLChatEngine(
        pool=pool,
        routed_engine=router,
        spec={"generation": GENERATION},
        capture=capture or _Capture(),
    )

    async def run():
        return [chunk async for chunk in engine.generate(request, None)]

    return asyncio.run(run())


def test_a_gym_call_is_rendered_spliced_and_routed_with_its_capture_plan() -> None:
    pool = _Pool(
        _rendered(
            template_token_ids=TEMPLATE,
            history_token_ids=HISTORY,
            required_prefix_token_ids=PREFIX,
            request={"model": "model", "max_tokens": 20, "max_completion_tokens": None},
            request_max_tokens=20,
            reasoning_ended=False,
        ),
        reply={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "thinking",
                        "tool_calls": [
                            {
                                "id": "call-0",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
        },
    )
    router = _Router()

    (chunk,) = _serve(
        pool,
        router,
        _call(
            {"required_prefix_token_ids": PREFIX},
            chat_template_args={"enable_thinking": True},
        ),
    )

    # The worker renders the call without Dynamo's None fields, with the
    # profile's exact sampling and Gym's extras as the native request fields.
    (body,) = pool.bodies
    assert body["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "u"},
    }
    assert {"n", "logit_bias", "nvext", "chat_template_args"}.isdisjoint(body)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.7, 0.9, -1)
    assert (body["min_p"], body["repetition_penalty"]) == (0.0, 1.0)
    assert (body["ng_capture"], body["required_prefix_token_ids"]) == (None, PREFIX)
    # The router gets the spliced prompt, the budget clamped to the context,
    # and the plan the engine's adapter serves the call by.
    (routed,) = router.requests
    assert routed["token_ids"] == SPLICED
    assert routed["stop_conditions"] == {"max_tokens": MAX_MODEL_LEN - len(SPLICED)}
    assert routed["annotations"] == [chat_engine.GYM_CALL_ANNOTATION]
    wire = routed["extra_args"][chat_engine.PLAN_KEY]
    plan = wire["prepared_prompt"]
    # The request as rendered; the engine sets the plan's clamped budget.
    assert wire["request"] == {
        "model": "model",
        "max_tokens": 20,
        "max_completion_tokens": None,
    }
    assert plan["max_tokens"] == MAX_MODEL_LEN - len(SPLICED)
    assert (plan["template_token_ids"], plan["splice"], plan["prefix_len"]) == (
        TEMPLATE,
        # Each side is cut at its turn end: PREFIX[:3] + TEMPLATE[2:].
        [3, 2],
        len(PREFIX),
    )
    assert (plan["admission"], wire["multimodal"], wire["reasoning_ended"]) == (
        None,
        None,
        False,
    )
    # The reply is vLLM's, built from the engine's output, with the result in nvext.
    ((_, prompt_token_ids, output),) = pool.responded
    assert (prompt_token_ids, output) == (SPLICED, router.engine_data["output"])
    (choice,) = chunk["choices"]
    assert choice["delta"]["reasoning_content"] == "thinking"
    assert choice["delta"]["tool_calls"][0]["index"] == 0
    assert choice["finish_reason"] == "tool_calls"
    assert chunk["nvext"] == {"nemo_rl": router.engine_data["result"]}
    assert chunk["usage"]["prompt_tokens"] == 6


def test_a_token_in_call_splices_the_prefix_staged_in_tq() -> None:
    pool = _Pool(
        _rendered(
            template_token_ids=TEMPLATE,
            history_token_ids=HISTORY,
            admission=_admission(TOKEN_IN),
            serving_admission=_admission(TOKEN_IN),
        )
    )
    router = _Router()
    capture = _Capture()

    _serve(pool, router, _call({"ng_capture": TOKEN_IN}), capture)

    assert pool.bodies[0]["ng_capture"] == TOKEN_IN
    assert capture.resolved == [TOKEN_IN]
    (routed,) = router.requests
    assert routed["token_ids"] == SPLICED
    plan = routed["extra_args"][chat_engine.PLAN_KEY]["prepared_prompt"]
    assert (plan["admission"], plan["serving_admission"], plan["prefix_len"]) == (
        TOKEN_IN,
        TOKEN_IN,
        len(PREFIX),
    )


def test_a_measure_call_returns_the_prompt_length_without_generating() -> None:
    pool = _Pool(
        _rendered(
            template_token_ids=TEMPLATE,
            history_token_ids=HISTORY,
            required_prefix_token_ids=PREFIX,
        )
    )
    router = _Router()

    (chunk,) = _serve(
        pool,
        router,
        _call({"required_prefix_token_ids": PREFIX, "ng_measure": True}),
    )

    # The length of the prompt the call would run, as spliced.
    assert chunk["nvext"] == {
        "nemo_rl": GymCallResult(
            prompt_token_count=len(SPLICED), ng_context_measured=True
        ).model_dump(exclude_none=True)
    }
    assert (router.requests, pool.responded) == ([], [])


@pytest.mark.parametrize(
    ("request_fields", "rendered", "error"),
    [
        ({"extras": None}, {}, "nemo_rl:gym: annotation"),
        ({"temperature": 0.5}, {}, "matches neither"),
        ({"presence_penalty": 0.5}, {}, "presence_penalty=0.5"),
        ({"n": 2}, {}, "n=1"),
        # The echoed prefix, spliced in, leaves no room for output.
        (
            {},
            {
                "template_token_ids": TEMPLATE,
                "history_token_ids": HISTORY,
                "required_prefix_token_ids": [1, *range(20, 20 + MAX_MODEL_LEN), EOS],
            },
            "maximum context length",
        ),
    ],
)
def test_gym_calls_nemo_rl_cannot_serve_are_refused_with_a_400(
    request_fields, rendered, error
) -> None:
    pool = _Pool(_rendered(**rendered))
    router = _Router()
    request_fields = {"extras": {}, **request_fields}

    with pytest.raises(HttpError, match=error) as refused:
        _serve(pool, router, _call(**request_fields))

    assert refused.value.code == 400
    assert router.requests == []


@pytest.mark.nemo_gym
@pytest.mark.parametrize("captured", [True, False])
@pytest.mark.parametrize("refused_by", ["render", "clamp", "engine"])
def test_a_call_refused_before_it_runs_acknowledges_its_capture_to_gym(
    captured, refused_by
) -> None:
    from nemo_gym.token_id_capture.external_capture import (
        split_dynamo_capture_refusal,
    )
    from vllm.exceptions import VLLMValidationError

    pool = _Pool(_rendered())
    router = _Router()
    if refused_by == "render":
        pool.error = VLLMValidationError("maximum context length is 16 tokens")
    elif refused_by == "clamp":
        # The prompt alone fills the context.
        pool.rendered = _rendered(template_token_ids=list(range(MAX_MODEL_LEN)))
    else:
        router.engine_data = {
            "refusal": {
                "status": 400,
                "message": "maximum context length is 16 tokens",
                "acknowledge": True,
            }
        }
    extras = {"ng_capture": TOKEN_IN} if captured else {}

    with pytest.raises(HttpError) as refused:
        _serve(pool, router, _call(extras))

    # Gym reads the acknowledgement off the message, as natively off the header.
    message, refused_call = split_dynamo_capture_refusal(refused.value.message)
    assert (refused.value.code, "maximum context length" in message) == (400, True)
    assert refused_call == (TOKEN_IN["model_call_id"] if captured else None)


@pytest.mark.parametrize(
    ("router", "status", "message"),
    [
        (
            _Router(error="BackendInvalidArgument: RuntimeError: engine died"),
            500,
            "engine died",
        ),
        # From Dynamo 1.7, the engine adapter's InvalidArgument (a RuntimeError it
        # wrapped, e.g. non-finite log-probs) is raised as itself, not a ValueError.
        (
            _Router(error=InvalidArgument("RuntimeError: non-finite log-prob (nan)")),
            500,
            "non-finite log-prob",
        ),
        # Refused before it ran, with the status the native server answers.
        (
            _Router(
                engine_data={
                    "refusal": {
                        "status": 503,
                        "message": "queue full",
                        "acknowledge": False,
                    }
                }
            ),
            503,
            "queue full",
        ),
    ],
    ids=["error", "engine-invalid-argument", "refusal"],
)
def test_an_engine_that_does_not_serve_the_call_fails_it(router, status, message):
    with pytest.raises(HttpError, match=message) as failed:
        _serve(_Pool(_rendered()), router, _call({}))

    assert failed.value.code == status


def test_dynamos_frontend_still_builds_its_entrypoint_by_the_name_nemo_rl_wraps():
    import dynamo.frontend.main as frontend_main

    # frontend.main() swaps this module global for one that adds NeMo RL's
    # chat_engine_factory; if Dynamo stops reading it there, Dynamo's own
    # preprocessor serves chat and every NeMo-Gym call is refused.
    assert "EntrypointArgs" in frontend_main.async_main.__code__.co_names
    # It swaps this one to serve NeMo-Gym's generation cuts from the frontend's
    # runtime; if Dynamo stops reading it there, every cut gets a 404.
    assert "DistributedRuntime" in frontend_main.async_main.__code__.co_names


def test_only_nemo_rls_engines_are_served() -> None:
    model = SimpleNamespace(
        runtime_config=lambda: {"runtime_data": {}},
        name=lambda: "other-model",
        local_dir=lambda: "/snapshots/other",
    )

    async def build():
        return await chat_engine.ChatEngineFactory().chat_engine_factory(
            0, model, _Router()
        )

    with pytest.raises(RuntimeError, match="without NeMo RL's chat settings"):
        asyncio.run(build())
