import asyncio
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.captured_chat_logprobs import (
    CapturedChatLogprobsMixin,
)


class _OrdinaryFormatter:
    def _create_chat_logprobs(self, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs


class _Serving(CapturedChatLogprobsMixin, _OrdinaryFormatter):
    pass


def test_ordinary_logprobs_delegate_without_event_loop() -> None:
    assert _Serving()._create_chat_logprobs(7, top_logprobs=0) == (
        (7,),
        {"top_logprobs": 0},
    )


@pytest.mark.asyncio
async def test_bypass_isolated_between_concurrent_requests() -> None:
    serving = _Serving()
    entered = asyncio.Event()
    released = asyncio.Event()

    async def captured() -> None:
        with serving._bypass_chat_logprobs(enabled=True):
            entered.set()
            await released.wait()
            assert serving._create_chat_logprobs(7) is None

    task = asyncio.create_task(captured())
    await entered.wait()
    try:
        assert serving._create_chat_logprobs(7) == ((7,), {})
        with serving._bypass_chat_logprobs(enabled=False):
            assert serving._create_chat_logprobs(8) == ((8,), {})
    finally:
        released.set()
        await task
    assert serving._create_chat_logprobs(7) == ((7,), {})


@pytest.mark.asyncio
async def test_nested_scope_other_instance_and_child_task_delegate() -> None:
    serving = _Serving()

    async def child() -> Any:
        return serving._create_chat_logprobs(7)

    with serving._bypass_chat_logprobs(enabled=True):
        assert serving._create_chat_logprobs(7) is None
        assert _Serving()._create_chat_logprobs(7) == ((7,), {})
        assert await asyncio.create_task(child()) == ((7,), {})
        assert await asyncio.to_thread(serving._create_chat_logprobs, 7) == ((7,), {})
        with serving._bypass_chat_logprobs(enabled=False):
            assert serving._create_chat_logprobs(7) == ((7,), {})
        assert serving._create_chat_logprobs(7) is None
    assert serving._create_chat_logprobs(7) == ((7,), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError("format failed"), asyncio.CancelledError()]
)
async def test_bypass_resets_on_error_or_cancellation(error: BaseException) -> None:
    serving = _Serving()
    with pytest.raises(type(error)):
        with serving._bypass_chat_logprobs(enabled=True):
            assert serving._create_chat_logprobs(7) is None
            raise error
    assert serving._create_chat_logprobs(7) == ((7,), {})


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_bypass_delegates_to_real_vllm_formatter() -> None:
    """Exercise the pinned vLLM builder without starting a generation engine."""
    chat = pytest.importorskip("vllm.entrypoints.openai.chat_completion.serving")
    logprob_type = pytest.importorskip("vllm.logprobs").Logprob

    class Serving(CapturedChatLogprobsMixin, chat.OpenAIServingChat):
        pass

    serving = Serving.__new__(Serving)
    serving.return_tokens_as_token_ids = True
    engine_logprobs = [{7: logprob_type(logprob=-0.5, rank=1, decoded_token="x")}]
    kwargs = {
        "token_ids": [7],
        "top_logprobs": engine_logprobs,
        "tokenizer": None,
        "num_output_top_logprobs": 0,
        "logprob_token_ids": None,
        "return_as_token_id": True,
    }
    ordinary = serving._create_chat_logprobs(**kwargs)
    assert ordinary.content[0].logprob == -0.5
    with serving._bypass_chat_logprobs(enabled=True):
        assert serving._create_chat_logprobs(**kwargs) is None
    assert serving._create_chat_logprobs(**kwargs) == ordinary
    assert engine_logprobs[0][7].logprob == -0.5
