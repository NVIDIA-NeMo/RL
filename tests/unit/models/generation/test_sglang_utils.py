import pytest

from nemo_rl.models.generation.sglang.utils import AsyncLoopThread


def test_async_loop_thread_run_returns_result():
    loop_thread = AsyncLoopThread()

    async def sample():
        return 42

    try:
        assert loop_thread.run(sample()) == 42
    finally:
        loop_thread.shutdown()


def test_async_loop_thread_run_when_stopped_raises():
    loop_thread = AsyncLoopThread()
    loop_thread.shutdown()

    async def sample():
        return 1

    with pytest.raises(RuntimeError, match="Event loop is not running"):
        coro = sample()
        try:
            loop_thread.run(coro)
        finally:
            coro.close()
