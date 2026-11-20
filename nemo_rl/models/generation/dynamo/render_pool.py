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

"""Render and answer NeMo-Gym chat calls in worker processes of Dynamo's frontend.

Rendering (template, tokenizer, the checkpoint's HF processor) and building the
reply (detokenized text through the backend's reasoning and tool parsers) are
CPU-bound Python; on the frontend's one event loop they would serialize every
call in flight. Each worker process builds a backend's :class:`RenderWorker`
(``dynamo/vllm/chat_backend.py``, ``dynamo/megatron/chat_backend.py``), which
renders and replies as that backend's native chat server does:

- ``render``: the native server's render step (the render, the capture
  decision, the history render the splice cuts against);
- ``respond``: the native server's non-streaming response builder over the
  engine's output.

Jobs queue FIFO in the pool; nothing renders on the event loop.

A worker imports its backend's libraries before it has the settings it is
built from, so the frontend can start the workers at launch, before any engine
has registered.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, TypedDict

LOGGER = logging.getLogger(__name__)

# Set in each worker process by _init_worker.
_WORKER: Optional["RenderWorker"] = None


class RestoredReply(TypedDict):
    """What the engine resumed a call from, after a checkpoint cut it.

    The engine swaps the frontend's prompt for the saved rows and generates
    only the rest, so the reply needs these to read as the whole call's.
    """

    # The prompt the engine ran: the capture prefix, then the saved rows.
    prompt_token_ids: list[int]
    # The saved rows' generated tokens, which the reply's text and usage count.
    token_ids: list[int]
    # The saved prefix's finish, when it had already ended the call:
    # ``_RestoredPrefixTerminal``'s fields without the prompt.
    terminal: Optional[dict[str, Any]]


@dataclass(frozen=True)
class RenderedCall:
    """What the frontend needs from a render to plan, route and serve a call.

    A :class:`~nemo_rl.models.generation.gym_prompt.RenderedPrompt`.
    """

    template_token_ids: list[int]
    # The render up to the last assistant turn, when a prefix will be spliced.
    history_token_ids: Optional[list[int]]
    # The request's own prefix (Gym's echo) after the capture decision.
    required_prefix_token_ids: Optional[list[int]]
    # Capture admissions: what is stored, and how the prompt is served.
    admission: Any
    serving_admission: Any
    verify_retained_media: bool
    # The rendered engine input's multimodal part (pickled), or None for text.
    multimodal: Optional[bytes]
    # The chat request as rendered (parsers may adjust it), without messages:
    # the engine builds its SamplingParams from it, as the native server does.
    request: dict[str, Any]
    # The output-token budget the request asked for, before the clamp.
    request_max_tokens: Optional[int]
    # What vLLM's chat serving hands the engine besides the prompt.
    reasoning_ended: Optional[bool]
    reasoning_parser_kwargs: Optional[dict[str, Any]]


class RenderWorker(Protocol):
    """What one render-pool worker process serves, for one generation backend.

    Built in the worker process from the engine's chat settings
    (``spec``, what the engine publishes at registration) and a local
    directory with the model's config and tokenizer files.
    """

    @staticmethod
    def prepare_process() -> None:
        """Import what the worker builds and serves with, before its settings arrive."""
        ...

    def render(self, body: dict[str, Any], *, measure: bool = False) -> RenderedCall:
        """Render a chat call: the template, the capture decision, the history."""
        ...

    def respond(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int],
        output: dict[str, Any],
        restored: Optional[RestoredReply] = None,
    ) -> dict[str, Any]:
        """The OpenAI chat response for the engine's ``output``."""
        ...

    def info(self) -> dict[str, Any]:
        """``turn_end_token_ids``, ``max_model_len``, ``build_seconds`` and backend extras.

        The chat engine hands it to ``ChatBackend.prepare_prompt``.
        """
        ...


def _renderer_class(renderer: str) -> type[RenderWorker]:
    """The render worker class named ``module:Class``."""
    module, _, name = renderer.partition(":")
    return getattr(importlib.import_module(module), name)


def _wait_for_settings(path: str) -> tuple[dict[str, Any], str]:
    """The settings :meth:`RenderPool.build` writes, once it has."""
    frontend = multiprocessing.parent_process()
    while not os.path.exists(path):
        # None will come once the pool is shut down or the frontend is gone.
        if not os.path.isdir(os.path.dirname(path)) or (
            frontend is not None and not frontend.is_alive()
        ):
            raise SystemExit("the render pool went away before it had settings")
        time.sleep(0.1)
    with open(path) as settings_file:
        settings = json.load(settings_file)
    return settings["spec"], settings["model_dir"]


def _init_worker(settings_path: str, renderer: str) -> None:
    global _WORKER
    worker_cls = _renderer_class(renderer)
    # Before the wait, so that only the build is left once the settings arrive.
    worker_cls.prepare_process()
    _WORKER = worker_cls(*_wait_for_settings(settings_path))


def _render(body: dict[str, Any], measure: bool) -> RenderedCall:
    assert _WORKER is not None
    return _WORKER.render(body, measure=measure)


def _respond(
    request: dict[str, Any],
    prompt_token_ids: list[int],
    output: dict[str, Any],
    restored: Optional[RestoredReply],
) -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER.respond(request, prompt_token_ids, output, restored)


def _info() -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER.info()


class RenderPool:
    """``workers`` processes that render and answer chat calls, FIFO.

    A worker imports its backend's libraries, then waits for the settings
    :meth:`build` gives the pool, so :meth:`start` can come first. :meth:`start`
    starts them all; without it the executor starts one for each job submitted
    while none is idle, so the first calls wait for them.
    """

    def __init__(self, *, workers: int, renderer: str):
        """``renderer`` is the backend's ``module:Class`` :class:`RenderWorker`."""
        self._workers = workers
        self._renderer = renderer
        # Where build() writes the settings; a restarted pool's workers read
        # them here too.
        self._settings_dir = tempfile.mkdtemp(prefix="nemorl_render_pool_")
        self._settings_path = os.path.join(self._settings_dir, "settings.json")
        self._built_at: Optional[float] = None
        self._executor = self._start()
        self._info: Optional[dict[str, Any]] = None

    def build(self, spec: dict[str, Any], model_dir: str) -> None:
        """Give the workers what they are built from.

        ``spec`` is the engine's chat settings (``chat_runtime_data``), and
        ``model_dir`` a local directory with the model's config and tokenizer
        files.
        """
        if self._built_at is not None:
            raise RuntimeError("the render pool already has its settings")
        with tempfile.NamedTemporaryFile(
            "w", dir=self._settings_dir, delete=False
        ) as settings_file:
            json.dump({"spec": spec, "model_dir": model_dir}, settings_file)
        # Renamed into place, so a worker never reads a partial file.
        os.replace(settings_file.name, self._settings_path)
        self._built_at = time.monotonic()

    def _start(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._workers,
            # Fork would copy the frontend's Rust runtime and event loop.
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._settings_path, self._renderer),
        )

    async def _submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        executor = self._executor
        try:
            return await asyncio.wrap_future(executor.submit(fn, *args))
        except BrokenProcessPool as error:
            if self._info is None:
                # No worker ever served a job: most often _init_worker failed (a
                # config error, which a restart cannot fix), so do not restart.
                raise RuntimeError(
                    "the render pool's workers died before serving a call; see "
                    "this log for an 'Exception in initializer' traceback or a "
                    "kill (e.g. OOM)"
                ) from error
            if self._executor is executor:
                # A worker died (e.g. OOM-killed). The calls in flight fail;
                # later calls get a fresh pool instead of this broken one.
                LOGGER.error("A render-pool worker died; restarting the pool")
                executor.shutdown(wait=False, cancel_futures=True)
                self._executor = self._start()
            raise

    async def start(self) -> None:
        """Start every worker now: one job each, submitted while none is idle.

        Returns when the first is ready, which takes :meth:`build`'s settings.
        """
        started = time.monotonic()
        infos = await asyncio.gather(
            *(self._submit(_info) for _ in range(self._workers))
        )
        self._info = infos[0]
        ready = time.monotonic()
        assert self._built_at is not None
        LOGGER.info(
            "The render pool's first of %d workers is ready %.1fs after they "
            "started and %.1fs after their settings arrived; it took %.1fs to "
            "build",
            self._workers,
            ready - started,
            ready - self._built_at,
            self._info["build_seconds"],
        )

    async def info(self) -> dict[str, Any]:
        """The model's turn-end ids and context length, from a started worker."""
        if self._info is None:
            self._info = await self._submit(_info)
        return self._info

    async def render(
        self, body: dict[str, Any], *, measure: bool = False
    ) -> RenderedCall:
        """Render a call; a measure call is counted past the context, as natively."""
        return await self._submit(_render, body, measure)

    async def respond(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int],
        output: dict[str, Any],
        restored: Optional[RestoredReply] = None,
    ) -> dict[str, Any]:
        """Build the reply; ``restored`` is the engine's ``RestoredReply``, if any."""
        return await self._submit(_respond, request, prompt_token_ids, output, restored)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self._settings_dir, ignore_errors=True)
