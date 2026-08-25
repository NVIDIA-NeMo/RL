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

"""A NeMo-RL-owned HTTP router in front of the vLLM generation fleet.

NeMo-Gym deterministically hashes each session to one policy endpoint from a list fixed
at process start, with no health input and no failover. A dead vLLM endpoint therefore
keeps receiving the sessions mapped to it for the rest of the run, and Gym retries a
refused connection in an uncapped loop with no HTTP timeout.

Rather than change Gym, hand it a single URL that NeMo-RL owns. Gym's
``VLLMModelConfig.base_url`` accepts one string, so deterministic session hashing has
one possible target and the routing decision moves here, next to the fleet health that
already knows which shards are serving.

Two properties make this safe to put in Gym's critical path:

* **The URL never changes.** The port is reserved once and passed in, so Ray recreating a
  restarted actor rebinds the same address. Gym is never reconfigured and never has to
  fail over -- which matters because failing over is exactly what it cannot do.
* **Every piece of state is built in __init__.** A restarted actor is immediately usable.
  This is the deliberate inverse of the NemoGym mistake, where the servers were started
  from a separate ``_spinup`` that Ray never re-runs.

Deliberately *not* a redirect. Handing Gym a 307 would put its socket back on a vLLM
endpoint directly, so a backend dying mid-request would drop it into the same uncapped
retry loop this exists to avoid.

One thing this trades away: Gym's deterministic session hashing is *sticky* -- a
session keeps its backend -- so per-request least-outstanding gives up prefix-cache
affinity across the turns of a multi-turn rollout. That is a real cost, not purely a
defect being fixed, and it is worth measuring before enabling this on a multi-turn
workload.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Any, Optional

import ray
from pydantic import BaseModel, PositiveFloat, PositiveInt, model_validator
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_ROUTER_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_ROUTER_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)

# Hop-by-hop headers are per-connection and must not be forwarded; passing Host through
# would also make the backend see the router's address.
_SKIPPED_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_SKIPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection"}
)
# Body chunk size for the streaming pass-through. Large enough that a long completion
# does not cost thousands of iterations, small enough not to buffer a whole response.
_STREAM_CHUNK_BYTES = 64 * 1024
# Only these endpoints consume vLLM sequence-scheduler slots. Tokenization and model
# discovery stay outside bounded admission so a short control call cannot sit behind
# thousands of long generations in the central FIFO.
_GENERATION_ENDPOINTS = frozenset({"/v1/chat/completions", "/v1/responses"})
# HTTP statuses NeMo-Gym retries internally (nemo_gym/openai_utils.py). For the
# rate-limit subset it raises its own retry ceiling on each attempt, so answering with
# one of these is an unbounded loop rather than a bounded one.
_GYM_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 520})


class GenerationRouterConfig(BaseModel, extra="allow"):
    """NeMo-RL-owned HTTP router placed in front of vLLM for NeMo-Gym."""

    # When true, NeMo-Gym receives the router's URL instead of the raw backend URLs.
    enabled: bool = False
    # Dedicated head-node range below Ray GCS (1200). In particular, do not use
    # 6000-6999: ray.sub reserves that whole band for sandbox Nginx/uWSGI on every
    # allocated node, including the Ray head where this actor is pinned.
    port_range_low: PositiveInt = DEFAULT_GENERATION_ROUTER_PORT_RANGE_LOW
    port_range_high: PositiveInt = DEFAULT_GENERATION_ROUTER_PORT_RANGE_HIGH
    # Router -> backend deadline, covering the whole generation.
    backend_timeout_s: PositiveFloat = 600.0
    # TCP handshake deadline, separate from the whole-generation deadline.
    connect_timeout_s: PositiveFloat = 5.0
    # Maximum forwarded generation requests per backend. ``/tokenize`` and
    # ``/v1/models`` bypass this queue. ``None`` preserves unbounded behavior for
    # workloads whose request-to-sequence ratio is not known. For the common n=1
    # generation request, match this to vLLM's max_num_seqs so excess work stays
    # centrally dispatchable instead of becoming pinned in one backend's queue.
    max_inflight_requests_per_backend: Optional[PositiveInt] = None
    # Status returned when no shard is eligible.
    no_healthy_backend_status: PositiveInt = 409

    @model_validator(mode="after")
    def _check_port_range(self) -> "GenerationRouterConfig":
        if self.port_range_low >= self.port_range_high:
            raise ValueError(
                f"async_rl.generation_router.port_range_low ({self.port_range_low}) must be "
                f"< port_range_high ({self.port_range_high}). Transposed, this surfaces "
                "at setup as 'ValueError: empty range for randrange()' from deep inside "
                "port allocation, far from the typo."
            )
        return self

    @model_validator(mode="after")
    def _check_connect_timeout_fits(self) -> "GenerationRouterConfig":
        if self.connect_timeout_s > self.backend_timeout_s:
            raise ValueError(
                f"async_rl.generation_router.connect_timeout_s ({self.connect_timeout_s}) "
                f"exceeds backend_timeout_s ({self.backend_timeout_s}), so the total "
                "deadline would expire before the handshake one could ever fire."
            )
        return self

    @model_validator(mode="after")
    def _check_status_is_not_retried_by_gym(self) -> "GenerationRouterConfig":
        if self.no_healthy_backend_status in _GYM_RETRY_STATUSES:
            raise ValueError(
                "async_rl.generation_router.no_healthy_backend_status="
                f"{self.no_healthy_backend_status} is a status NeMo-Gym retries "
                f"internally ({sorted(_GYM_RETRY_STATUSES)}). For the rate-limit codes "
                "Gym raises its own retry ceiling on each attempt, so returning one "
                "would make it retry forever -- exactly the hang the router exists to "
                "prevent. Use a 4xx outside that set, e.g. 409."
            )
        return self


@dataclass
class _AdmissionWaiter:
    """One request waiting for a backend slot on the router's event loop."""

    future: asyncio.Future[Optional[str]]
    enqueued_at_s: float
    queued: bool = True


class GenerationRouterImpl:
    """Routing logic and HTTP server, split out so it is testable without Ray."""

    def __init__(
        self,
        *,
        backend_urls: list[str],
        host: str,
        port: int,
        backend_timeout_s: float,
        connect_timeout_s: float,
        max_inflight_requests_per_backend: Optional[int],
        no_healthy_backend_status: int,
        health_managed: bool = False,
    ) -> None:
        if not backend_urls:
            raise ValueError("GenerationRouter requires at least one backend URL")
        self._all_backends = list(backend_urls)
        # Starts as every backend: a restarted router has no health history, and routing
        # to a shard that turns out to be dead is self-correcting on the next push.
        self._serving: list[str] = list(backend_urls)
        # Total in-flight work remains the least-outstanding routing signal. Generation
        # in-flight is separate because tokenize/model-discovery calls bypass bounded
        # admission and must not consume vLLM sequence-scheduler slots.
        self._inflight: dict[str, int] = {url: 0 for url in backend_urls}
        self._generation_inflight: dict[str, int] = {url: 0 for url in backend_urls}
        self._backend_failures: dict[str, int] = {url: 0 for url in backend_urls}
        self._host = host
        self._port = port
        self._backend_timeout_s = backend_timeout_s
        self._connect_timeout_s = connect_timeout_s
        if (
            max_inflight_requests_per_backend is not None
            and max_inflight_requests_per_backend < 1
        ):
            raise ValueError("max_inflight_requests_per_backend must be >= 1")
        self._max_inflight_requests_per_backend = max_inflight_requests_per_backend
        self._no_healthy_backend_status = no_healthy_backend_status
        # Whether a GenerationFleetHealth is driving set_serving_backends. It gates the
        # reflex drop in _handle: dropping a backend locally is only safe because a later
        # membership push puts it back. With no monitor nothing ever pushes, so the drop
        # would be permanent and a few transient blips would drain the fleet to nothing.
        self._health_managed = health_managed
        self._requests_total = 0
        self._no_backend_total = 0
        self._backend_error_total = 0
        # Admission is owned by the aiohttp loop. Membership pushes arrive on the Ray
        # actor thread and only schedule _dispatch_waiters onto this loop; they never
        # touch Futures from the wrong thread.
        self._admission_waiters: deque[_AdmissionWaiter] = deque()
        self._admission_queued = 0
        self._admission_queued_peak = 0
        self._admission_wait_s_total = 0.0
        self._admission_wait_s_max = 0.0
        self._max_generation_inflight_per_backend_observed = 0
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._socket: Any = None

    def base_url(self) -> str:
        """The single URL handed to NeMo-Gym. Stable for the life of the run.

        A method rather than a property so the Ray actor can expose it remotely.
        """
        return f"http://{self._host}:{self._port}/v1"

    def set_serving_backends(self, urls: list[str]) -> None:
        """Replace the eligible backend set.

        Takes the full set rather than a delta, so a missed update, a reordered one, or a
        restarted router all converge on the next push instead of needing sequence
        numbers and replay.
        """
        eligible = [url for url in urls if url in self._inflight]
        unknown = [url for url in urls if url not in self._inflight]
        if unknown:
            # A URL-normalisation divergence between the ports reserved before load and
            # the URLs the monitor reports after it would otherwise show up only as
            # permanent 409s, with nothing anywhere saying why.
            print(
                f"policy router: ignoring {len(unknown)} pushed URL(s) it does not "
                f"serve: {unknown}; known backends: {self._all_backends}",
                flush=True,
            )
        # Rebound rather than mutated: the server thread reads this reference without a
        # lock, and swapping it wholesale means a reader always sees a consistent list.
        self._serving = eligible
        self._schedule_waiter_dispatch()

    def _schedule_waiter_dispatch(self) -> None:
        """Wake admission waiters after a membership update from the actor thread."""
        loop = self._server_loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._dispatch_waiters)
        except RuntimeError:
            # The loop can close between is_closed() and call_soon_threadsafe() during
            # actor teardown. Pending handlers are cancelled as part of app cleanup.
            return

    def drain_backend_failures(self) -> dict[str, int]:
        """Hand over the per-backend failure counts and reset them.

        The router sees failures no liveness probe can -- a wedged engine answers
        ``is_alive`` from a healthy worker process. It holds no monitor reference by
        design, so instead of reporting, it counts, and the controller's probe tick
        drains these into ``GenerationFleetHealth.report_failure``.
        """
        counts = {url: n for url, n in self._backend_failures.items() if n}
        for url in counts:
            self._backend_failures[url] = 0
        return counts

    def metrics(self) -> dict[str, float]:
        # Every bounded request enters the FIFO before synchronous dispatch, so the
        # queued peak is an enqueue high-water mark (and reaches at least one even when
        # every request is admitted in the same event-loop turn).
        return {
            "router/requests_total": float(self._requests_total),
            "router/no_healthy_backend_total": float(self._no_backend_total),
            "router/backend_error_total": float(self._backend_error_total),
            "router/serving_backends": float(len(self._serving)),
            "router/admission_queued_requests": float(self._admission_queued),
            "router/admission_queued_requests_peak": float(self._admission_queued_peak),
            "router/admission_wait_s_total": self._admission_wait_s_total,
            "router/admission_wait_s_max": self._admission_wait_s_max,
            "router/inflight_requests": float(sum(self._inflight.values())),
            "router/generation_inflight_requests": float(
                sum(self._generation_inflight.values())
            ),
            "router/max_generation_inflight_per_backend_observed": float(
                self._max_generation_inflight_per_backend_observed
            ),
        }

    def _pick_backend(self, *, enforce_admission_limit: bool = True) -> Optional[str]:
        """Least-outstanding eligible backend with admission capacity, if any."""
        serving = self._serving
        if not serving:
            return None
        limit = (
            self._max_inflight_requests_per_backend if enforce_admission_limit else None
        )
        candidates = (
            serving
            if limit is None
            else [
                url for url in serving if self._generation_inflight.get(url, 0) < limit
            ]
        )
        if not candidates:
            return None
        return min(candidates, key=lambda url: (self._inflight.get(url, 0), url))

    def _reserve_backend(self, backend: str, *, is_generation: bool) -> None:
        inflight = self._inflight.get(backend, 0) + 1
        self._inflight[backend] = inflight
        if is_generation:
            generation_inflight = self._generation_inflight.get(backend, 0) + 1
            self._generation_inflight[backend] = generation_inflight
            self._max_generation_inflight_per_backend_observed = max(
                self._max_generation_inflight_per_backend_observed,
                generation_inflight,
            )

    def _release_backend(self, backend: str, *, is_generation: bool) -> None:
        self._inflight[backend] = max(0, self._inflight.get(backend, 0) - 1)
        if is_generation:
            self._generation_inflight[backend] = max(
                0, self._generation_inflight.get(backend, 0) - 1
            )
        self._dispatch_waiters()

    def _finish_queued_waiter(self, waiter: _AdmissionWaiter) -> None:
        """Mark a dequeued or cancelled waiter and update its queue metrics once."""
        if not waiter.queued:
            return
        waiter.queued = False
        self._admission_queued = max(0, self._admission_queued - 1)

    def _dispatch_waiters(self) -> None:
        """Admit FIFO waiters while the current serving set has free capacity."""
        while self._admission_waiters:
            waiter = self._admission_waiters[0]
            if not waiter.queued or waiter.future.cancelled():
                self._admission_waiters.popleft()
                self._finish_queued_waiter(waiter)
                continue

            if not self._serving:
                # Preserve the existing fail-fast semantics: once fleet health says
                # nothing is eligible, queued calls return the configured non-retried
                # status instead of waiting forever for a hypothetical future push.
                self._admission_waiters.popleft()
                self._finish_queued_waiter(waiter)
                waiter.future.set_result(None)
                continue

            backend = self._pick_backend()
            if backend is None:
                # At least one backend is serving, but every one is at its cap. A
                # release or membership push will schedule the next dispatch.
                return

            self._admission_waiters.popleft()
            self._finish_queued_waiter(waiter)
            waited_s = monotonic() - waiter.enqueued_at_s
            self._admission_wait_s_total += waited_s
            self._admission_wait_s_max = max(self._admission_wait_s_max, waited_s)
            # Reserve before waking the handler. Another request can run as soon as
            # set_result yields control, and must already observe this slot as occupied.
            self._reserve_backend(backend, is_generation=True)
            waiter.future.set_result(backend)

    async def _acquire_backend(
        self, *, enforce_admission_limit: bool = True
    ) -> Optional[str]:
        """Return a reserved backend, waiting centrally when all are at capacity."""
        if (
            self._max_inflight_requests_per_backend is None
            or not enforce_admission_limit
        ):
            backend = self._pick_backend(
                enforce_admission_limit=enforce_admission_limit
            )
            if backend is not None:
                self._reserve_backend(backend, is_generation=enforce_admission_limit)
            return backend

        loop = asyncio.get_running_loop()
        waiter = _AdmissionWaiter(
            future=loop.create_future(), enqueued_at_s=monotonic()
        )
        self._admission_waiters.append(waiter)
        self._admission_queued += 1
        self._admission_queued_peak = max(
            self._admission_queued_peak, self._admission_queued
        )
        self._dispatch_waiters()
        try:
            return await waiter.future
        except asyncio.CancelledError:
            if waiter.queued:
                # Awaiting a Future propagates task cancellation into that Future. Leave
                # the tombstone for the dispatcher to discard without a deque scan.
                waiter.future.cancel()
                self._finish_queued_waiter(waiter)
            elif waiter.future.done() and not waiter.future.cancelled():
                # Cancellation can land after dispatch reserved a slot but before this
                # task resumes from await. Reclaim it here: _handle never receives the
                # backend and therefore cannot reach its normal finally block.
                backend = waiter.future.result()
                if backend is not None:
                    self._release_backend(backend, is_generation=True)
            raise

    @staticmethod
    def _target_url(backend: str, path_qs: str) -> str:
        """Map an inbound path onto a backend.

        Backends are advertised as ``http://host:port/v1`` while inbound paths already
        carry their own prefix -- ``/v1/chat/completions`` for most calls, but bare
        ``/tokenize`` because Gym's ``create_tokenize`` strips ``/v1`` first. Stripping
        the suffix and appending the full path handles both.
        """
        return backend.removesuffix("/v1") + path_qs

    async def _handle(self, request: Any) -> Any:
        from aiohttp import ClientError, web

        self._requests_total += 1
        is_generation = request.path in _GENERATION_ENDPOINTS
        backend = await self._acquire_backend(enforce_admission_limit=is_generation)
        if backend is None:
            self._no_backend_total += 1
            # The status matters: NeMo-Gym retries 429/500/502/503/504/520, and for the
            # rate-limit codes it *raises its own retry ceiling* each time, so returning
            # one of those would spin forever. This code must stay outside that set.
            return web.json_response(
                {
                    "error": "no healthy generation backend",
                    "backends": self._all_backends,
                },
                status=self._no_healthy_backend_status,
            )

        try:
            return await self._forward(request, backend)
        except (TimeoutError, ClientError) as error:
            return self._on_backend_error(backend, error)
        finally:
            self._release_backend(backend, is_generation=is_generation)

    def _on_backend_error(self, backend: str, error: BaseException) -> Any:
        """Answer for a backend that failed, deliberately rather than by accident.

        Without this, aiohttp answers instead, and its choice of status decides whether
        the run survives. A wedged backend trips the client timeout, which aiohttp
        reports as **504** -- and 504 is in NeMo-Gym's rate-limit retry subset, where
        ``_request_with_retry`` raises its own ceiling on every attempt. That is an
        unbounded retry loop at ``backend_timeout_s`` per turn: exactly the hang
        ``_check_status_is_not_retried_by_gym`` exists to prevent, reintroduced through
        the error path that validator never covered.

        500 instead, because it is in Gym's *bounded* retry set: Gym re-sends this one
        HTTP call, the next _pick_backend lands on a healthy shard, and a multi-turn
        rollout keeps the turns it had already completed. The no-healthy-backend status
        (409) would be wrong here -- not retried, so it fails the whole rollout and every
        turn is redone from scratch by the row re-dispatch a layer up.
        """
        from aiohttp import web

        self._backend_error_total += 1
        self._backend_failures[backend] = self._backend_failures.get(backend, 0) + 1
        if self._health_managed:
            # Reflex: stop routing here until the next membership push re-adds it.
            # Rebound, not mutated -- same reason as set_serving_backends, and this runs
            # on the server thread while pushes arrive on the actor's.
            self._serving = [url for url in self._serving if url != backend]
        status = 500 if self._serving else self._no_healthy_backend_status
        return web.json_response(
            {
                "error": f"backend failed: {type(error).__name__}: {error}",
                "backend": backend,
            },
            status=status,
        )

    async def _forward(self, request: Any, backend: str) -> Any:
        from aiohttp import ClientTimeout, web

        session = request.app["session"]
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _SKIPPED_REQUEST_HEADERS
        }

        async with session.request(
            method=request.method,
            url=self._target_url(backend, request.rel_url.path_qs),
            headers=headers,
            data=request.content,
            # The timeout Gym's own client never sets. Without it a wedged backend holds
            # this request, and the rollout behind it, indefinitely.
            #
            # total must cover the whole generation: Gym pins stream=false, so no bytes
            # arrive until the completion finishes and an idle-read timeout would kill
            # long generations -- elapsed-total is the only wedge detector this hop can
            # have. The handshake is the opposite: a connect to a local vLLM either
            # completes in milliseconds or never will, so giving it the full budget just
            # means a black-holed SYN (node gone, no RST) parks the rollout for the
            # whole configured backend deadline.
            timeout=ClientTimeout(
                total=self._backend_timeout_s, sock_connect=self._connect_timeout_s
            ),
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in _SKIPPED_RESPONSE_HEADERS
                },
            )
            await response.prepare(request)
            # Streamed rather than buffered: a completion carrying per-token logprobs is
            # large, and this sits on every rollout's critical path.
            async for chunk in upstream.content.iter_chunked(_STREAM_CHUNK_BYTES):
                await response.write(chunk)
            await response.write_eof()
            return response

    def build_app(self) -> Any:
        """Build the aiohttp application serving Gym's endpoint surface."""
        from aiohttp import ClientSession, TCPConnector, web

        app = web.Application()

        async def _open_session(app_: Any) -> None:
            self._server_loop = asyncio.get_running_loop()
            # Explicit connector: aiohttp's default global limit=100 would form a second,
            # opaque queue below the router's per-backend admission queue. Keep the
            # connector unlimited; _acquire_backend is the one auditable place that
            # decides how much work each backend owns.
            app_["session"] = ClientSession(connector=TCPConnector(limit=0))

        async def _close_session(app_: Any) -> None:
            while self._admission_waiters:
                waiter = self._admission_waiters.popleft()
                self._finish_queued_waiter(waiter)
                if not waiter.future.done():
                    waiter.future.cancel()
            await app_["session"].close()
            self._server_loop = None

        app.on_startup.append(_open_session)
        app.on_cleanup.append(_close_session)

        # Exactly the calls NeMo-Gym's NeMoGymAsyncOpenAI makes. /tokenize is not under
        # /v1 because create_tokenize strips the suffix before appending.
        #
        # Deliberately an allowlist: this router's URL becomes Gym's *global*
        # policy_base_url, and some Gym envs point other surfaces at it -- speed_bench
        # scrapes GET /metrics, the claude-code agent POSTs /v1/messages. Those get 404
        # here where a raw vLLM URL answered, so run those envs with the router off.
        # Forwarding /metrics would be worse than refusing it: each shard keeps its own
        # counters, so a routed scrape returns one arbitrary shard's numbers as though
        # they were the fleet's.
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/models"):
            app.router.add_route("*", path, self._handle)
        app.router.add_route("*", "/tokenize", self._handle)
        return app

    def serve_in_background(self) -> None:
        """Run the HTTP server on a daemon thread with its own event loop.

        The socket is bound **here**, synchronously, before the thread starts. Bound
        inside the thread instead, a port conflict raises on a daemon thread nobody
        awaits: the actor stays alive, ``base_url()`` is a pure string format so it keeps
        resolving, and setup's "fail here rather than inside Gym" guard never notices.
        Gym is then handed a URL with no listener and retries the refused connection in
        an uncapped loop -- the exact wedge this router exists to prevent. Binding first
        turns that into a failed actor construction with the port in the traceback.

        Same shape as the vLLM workers handing their reserved socket to uvicorn. Restart
        stays correct: the replacement process rebinds the port its dead predecessor
        freed.
        """
        import socket

        from aiohttp import web

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(128)
        self._socket = sock

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            runner = web.AppRunner(self.build_app(), access_log=None)
            loop.run_until_complete(runner.setup())
            site = web.SockSite(runner, sock)
            loop.run_until_complete(site.start())
            print(f"policy router listening on {self.base_url()}", flush=True)
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name="policy-router", daemon=True)
        self._thread.start()

    def is_serving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=-1)  # pragma: no cover
class GenerationRouterActor(GenerationRouterImpl):
    """Ray actor wrapper. Everything it needs is built in ``__init__``.

    ``max_restarts=-1`` is only meaningful because of that: Ray recreates a restarted
    actor through ``__init__`` alone, so a class that starts its server from a separate
    method comes back permanently broken.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.serve_in_background()


def maybe_start_generation_router(
    *,
    backend_urls: list[Optional[str]],
    config: GenerationRouterConfig,
    health_managed: bool,
) -> tuple[list[Optional[str]], Optional[ray.actor.ActorHandle[GenerationRouterImpl]]]:
    """Optionally front generation backends with one stable NeMo-Gym URL.

    Args:
        backend_urls: Per-shard OpenAI server URLs reserved by generation.
        config: Validated ``async_rl.generation_router`` configuration.
        health_managed: Whether fleet health will push serving-set updates.

    Returns:
        A copy of the raw URLs and ``None`` when disabled; otherwise the router's
        single URL and the actor handle that the caller must retain for the run.

    Raises:
        ValueError: If routing is enabled but generation exposes no HTTP backend.
    """
    raw_backend_urls = list(backend_urls)
    if not config.enabled:
        return raw_backend_urls, None

    if not health_managed:
        # This is a supported performance-only mode: least-outstanding routing and
        # backend deadlines work, but no component can re-admit a quarantined shard.
        print(
            "⚠️  async_rl.generation_router.enabled=true with "
            "generation_fleet_health.enabled=false: the router will never receive "
            "a serving-set update, so it cannot route around a dead shard. "
            "Least-outstanding routing and backend deadlines remain active, but "
            "failover requires fleet health.",
            flush=True,
        )

    active_backend_urls = [url for url in raw_backend_urls if url]
    if not active_backend_urls:
        raise ValueError(
            "async_rl.generation_router.enabled=true requires generation backends "
            "that expose OpenAI-compatible servers; none were reported. This needs "
            "the vllm backend with async_engine and expose_http_server enabled."
        )

    # Reserve once and pass the port into the actor so a Ray restart rebinds the same
    # address. NeMo-Gym keeps this URL for the full run and never re-resolves it.
    port = _get_free_port_local(config.port_range_low, config.port_range_high)
    router = GenerationRouterActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        )
    ).remote(
        backend_urls=active_backend_urls,
        host=_get_node_ip_local(),
        port=port,
        backend_timeout_s=config.backend_timeout_s,
        connect_timeout_s=config.connect_timeout_s,
        max_inflight_requests_per_backend=(config.max_inflight_requests_per_backend),
        no_healthy_backend_status=config.no_healthy_backend_status,
        # Without health pushes, reflexively dropping a backend would retire it
        # permanently after one transient error.
        health_managed=health_managed,
    )
    # Resolve now so setup fails before Gym starts if actor construction or binding
    # failed. GenerationRouterImpl binds synchronously during actor initialization.
    base_url = ray.get(router.base_url.remote())
    print(
        f"📡 Policy router fronting {len(active_backend_urls)} backend(s) at {base_url}",
        flush=True,
    )
    return [base_url], router
