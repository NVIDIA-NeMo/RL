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

NeMo-Gym picks a policy endpoint by static round-robin over a list fixed at process
start, with no health input and no failover. A dead vLLM endpoint therefore keeps
receiving roughly 1/N of new rollouts for the rest of the run, and Gym retries a refused
connection in an uncapped loop with no HTTP timeout.

Rather than change Gym, hand it a single URL that NeMo-RL owns. Gym's
``VLLMModelConfig.base_url`` accepts one string, so its round-robin becomes a no-op and
the routing decision moves here, next to the fleet health that already knows which shards
are serving.

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

One thing this trades away: Gym's selection is *sticky* round-robin -- a session keeps its
backend -- so per-request least-outstanding gives up prefix-cache affinity across the
turns of a multi-turn rollout. That is a real cost, not purely a defect being fixed, and
it is worth measuring before enabling this on a multi-turn workload.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

import ray

# Hop-by-hop headers are per-connection and must not be forwarded; passing Host through
# would also make the backend see the router's address.
_SKIPPED_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_SKIPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection"}
)
# Body chunk size for the streaming pass-through. Large enough that a long completion
# does not cost thousands of iterations, small enough not to buffer a whole response.
_STREAM_CHUNK_BYTES = 64 * 1024


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
        no_healthy_backend_status: int,
        health_managed: bool = False,
    ) -> None:
        if not backend_urls:
            raise ValueError("GenerationRouter requires at least one backend URL")
        self._all_backends = list(backend_urls)
        # Starts as every backend: a restarted router has no health history, and routing
        # to a shard that turns out to be dead is self-correcting on the next push.
        self._serving: list[str] = list(backend_urls)
        self._inflight: dict[str, int] = {url: 0 for url in backend_urls}
        self._backend_failures: dict[str, int] = {url: 0 for url in backend_urls}
        self._host = host
        self._port = port
        self._backend_timeout_s = backend_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._no_healthy_backend_status = no_healthy_backend_status
        # Whether a GenerationFleetHealth is driving set_serving_backends. It gates the
        # reflex drop in _handle: dropping a backend locally is only safe because a later
        # membership push puts it back. With no monitor nothing ever pushes, so the drop
        # would be permanent and a few transient blips would drain the fleet to nothing.
        self._health_managed = health_managed
        self._requests_total = 0
        self._no_backend_total = 0
        self._backend_error_total = 0
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
        return {
            "router/requests_total": float(self._requests_total),
            "router/no_healthy_backend_total": float(self._no_backend_total),
            "router/backend_error_total": float(self._backend_error_total),
            "router/serving_backends": float(len(self._serving)),
        }

    def _pick_backend(self) -> Optional[str]:
        """Least-outstanding among eligible backends, or None if there are none."""
        serving = self._serving
        if not serving:
            return None
        return min(serving, key=lambda url: (self._inflight.get(url, 0), url))

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
        backend = self._pick_backend()
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

        self._inflight[backend] = self._inflight.get(backend, 0) + 1
        try:
            return await self._forward(request, backend)
        except (TimeoutError, ClientError) as error:
            return self._on_backend_error(backend, error)
        finally:
            self._inflight[backend] = max(0, self._inflight.get(backend, 0) - 1)

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
            # whole 600s.
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
            # Explicit connector: aiohttp's default is TCPConnector(limit=100), which
            # would cap the whole fleet's rollout traffic at 100 concurrent upstream
            # requests -- the exemplar config alone puts 32 prompts x 16 generations =
            # 512 in flight. Requests over the cap queue *inside this connector*, where
            # the wait silently burns the total timeout before the backend ever sees
            # them. Unlimited here sends the excess to vLLM's scheduler instead, where
            # queueing is visible as engine metrics rather than proxy latency.
            app_["session"] = ClientSession(connector=TCPConnector(limit=0))

        async def _close_session(app_: Any) -> None:
            await app_["session"].close()

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
