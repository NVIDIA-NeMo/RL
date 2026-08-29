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

"""The NeMo-RL-owned router that fronts the vLLM fleet for the NeMo-Gym path.

Gym picks a policy endpoint by static round-robin over a list fixed at process start,
never fails over, and retries a refused connection in an uncapped loop with no HTTP
timeout. Handing it one NeMo-RL-owned URL moves the routing decision here, next to the
fleet health that knows which shards are serving -- without changing Gym.

Most of these run a real aiohttp server against real backends. A proxy is precisely the
component that unit fakes flatter: header handling, streaming and status propagation only
misbehave over an actual socket.
"""

import asyncio

import pytest
from aiohttp import ClientSession, web

from nemo_rl.models.generation.generation_router import GenerationRouterImpl


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Backend:
    """A stand-in vLLM OpenAI server that records what it was asked."""

    def __init__(self, name: str, *, status: int = 200, body: bytes = b'{"ok":true}'):
        self.name = name
        self.status = status
        self.body = body
        self.requests: list[tuple[str, str, bytes]] = []
        self._runner: web.AppRunner | None = None
        self.port = _free_port()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def start(self) -> None:
        app = web.Application()

        async def _handle(request: web.Request) -> web.Response:
            self.requests.append((request.method, request.path, await request.read()))
            return web.Response(
                status=self.status,
                body=self.body,
                headers={"X-Served-By": self.name},
            )

        app.router.add_route("*", "/{tail:.*}", _handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


class _Harness:
    """Router bound to a real port, with a client for driving it."""

    def __init__(self, backends, **router_kwargs):
        self.backends = backends
        self.port = _free_port()
        self.router = GenerationRouterImpl(
            backend_urls=[b.url for b in backends],
            host="127.0.0.1",
            port=self.port,
            backend_timeout_s=router_kwargs.pop("backend_timeout_s", 5.0),
            connect_timeout_s=router_kwargs.pop("connect_timeout_s", 2.0),
            no_healthy_backend_status=router_kwargs.pop(
                "no_healthy_backend_status", 409
            ),
            # On by default here: the reflex drop is the behaviour under test in most of
            # these cases, and a real run only enables the router alongside fleet health.
            health_managed=router_kwargs.pop("health_managed", True),
            **router_kwargs,
        )
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "_Harness":
        for backend in self.backends:
            await backend.start()
        self._runner = web.AppRunner(self.router.build_app(), access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        for backend in self.backends:
            await backend.stop()

    async def call(self, path: str, *, method: str = "POST", body: bytes = b"{}"):
        async with ClientSession() as session:
            async with session.request(
                method, f"http://127.0.0.1:{self.port}{path}", data=body
            ) as response:
                return response.status, await response.read(), dict(response.headers)


class TestEndpointSurface:
    """Exactly the four calls NeMo-Gym's NeMoGymAsyncOpenAI makes."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/models",
            # Not under /v1: Gym's create_tokenize strips the suffix before appending.
            "/tokenize",
        ],
    )
    def test_each_gym_endpoint_reaches_a_backend(self, path):
        backend = _Backend("b0")

        async def _main():
            async with _Harness([backend]) as harness:
                status, _, headers = await harness.call(path)
                assert status == 200
                assert headers["X-Served-By"] == "b0"
                assert backend.requests[0][1] == path

        asyncio.run(_main())

    def test_the_request_body_is_forwarded_intact(self):
        backend = _Backend("b0")

        async def _main():
            async with _Harness([backend]) as harness:
                payload = b'{"model":"x","messages":[{"role":"user"}]}'
                await harness.call("/v1/chat/completions", body=payload)
                assert backend.requests[0][2] == payload

        asyncio.run(_main())

    def test_backend_status_is_propagated(self):
        """A 400 from vLLM is a data failure; masking it would break classification."""
        backend = _Backend("b0", status=400, body=b'{"message":"context length"}')

        async def _main():
            async with _Harness([backend]) as harness:
                status, body, _ = await harness.call("/v1/chat/completions")
                assert status == 400
                assert b"context length" in body

        asyncio.run(_main())

    def test_a_large_response_survives_the_streaming_path(self):
        """Completions carry per-token logprobs; the proxy must not truncate them."""
        payload = b"x" * (512 * 1024)
        backend = _Backend("b0", body=payload)

        async def _main():
            async with _Harness([backend]) as harness:
                _, body, _ = await harness.call("/v1/chat/completions")
                assert body == payload

        asyncio.run(_main())


class TestBackendSelection:
    def test_only_serving_backends_receive_traffic(self):
        alive, dead = _Backend("alive"), _Backend("dead")

        async def _main():
            async with _Harness([alive, dead]) as harness:
                harness.router.set_serving_backends([alive.url])
                for _ in range(5):
                    _, _, headers = await harness.call("/v1/chat/completions")
                    assert headers["X-Served-By"] == "alive"
                assert dead.requests == []

        asyncio.run(_main())

    def test_an_unknown_url_in_a_push_is_ignored(self):
        """A stale or malformed push must not invent a backend."""
        backend = _Backend("b0")
        harness = _Harness([backend])
        harness.router.set_serving_backends(["http://elsewhere:1/v1"])
        assert harness.router._pick_backend() is None

    def test_the_full_set_is_replaced_not_merged(self):
        """Pushes carry the whole set, so shrink-then-grow must both take effect."""
        first, second = _Backend("first"), _Backend("second")
        harness = _Harness([first, second])
        harness.router.set_serving_backends([first.url])
        assert harness.router._pick_backend() == first.url
        harness.router.set_serving_backends([first.url, second.url])
        assert harness.router._pick_backend() in {first.url, second.url}

    def test_a_restarted_router_serves_every_backend_until_told_otherwise(self):
        """No health history is better than no service; the next push corrects it."""
        first, second = _Backend("first"), _Backend("second")
        harness = _Harness([first, second])
        assert harness.router._pick_backend() is not None


class TestNoHealthyBackend:
    def test_the_status_stays_outside_gyms_retry_set(self):
        """This is the whole ballgame.

        NeMo-Gym retries 429/500/502/503/504/520, and for the rate-limit codes it raises
        its own retry ceiling on each attempt -- so answering with one of those would
        spin forever, recreating the hang this router exists to prevent.
        """
        gym_retry_codes = {429, 500, 502, 503, 504, 520}
        backend = _Backend("b0")

        async def _main():
            async with _Harness([backend]) as harness:
                harness.router.set_serving_backends([])
                status, body, _ = await harness.call("/v1/chat/completions")
                assert status == 409
                assert status not in gym_retry_codes
                assert b"no healthy generation backend" in body
                assert backend.requests == [], "nothing should have been dispatched"

        asyncio.run(_main())

    def test_it_is_counted(self):
        backend = _Backend("b0")

        async def _main():
            async with _Harness([backend]) as harness:
                harness.router.set_serving_backends([])
                await harness.call("/v1/chat/completions")
                assert (
                    harness.router.metrics()["router/no_healthy_backend_total"] == 1.0
                )

        asyncio.run(_main())


class TestUrlStability:
    def test_the_advertised_url_carries_the_v1_suffix_gym_expects(self):
        router = GenerationRouterImpl(
            backend_urls=["http://a:1/v1"],
            host="10.0.0.5",
            port=6000,
            backend_timeout_s=1.0,
            connect_timeout_s=1.0,
            no_healthy_backend_status=409,
        )
        assert router.base_url() == "http://10.0.0.5:6000/v1"

    def test_the_url_is_fixed_by_construction(self):
        """Ray recreates a restarted actor with the same args, so the port is stable.

        If the router picked a fresh free port on restart -- the way everything else in
        this codebase allocates ports -- Gym would hold a URL that no longer exists and
        could never recover, because it never re-resolves.
        """
        kwargs = dict(
            backend_urls=["http://a:1/v1"],
            host="10.0.0.5",
            port=6000,
            backend_timeout_s=1.0,
            connect_timeout_s=1.0,
            no_healthy_backend_status=409,
        )
        assert (
            GenerationRouterImpl(**kwargs).base_url()
            == GenerationRouterImpl(**kwargs).base_url()
        )

    @pytest.mark.parametrize(
        ("backend", "path", "expected"),
        [
            ("http://h:8/v1", "/v1/chat/completions", "http://h:8/v1/chat/completions"),
            ("http://h:8/v1", "/tokenize", "http://h:8/tokenize"),
            ("http://h:8/v1", "/v1/models?x=1", "http://h:8/v1/models?x=1"),
        ],
    )
    def test_paths_map_onto_the_backend_correctly(self, backend, path, expected):
        assert GenerationRouterImpl._target_url(backend, path) == expected

    def test_construction_requires_a_backend(self):
        with pytest.raises(ValueError, match="at least one backend"):
            GenerationRouterImpl(
                backend_urls=[],
                host="127.0.0.1",
                port=1,
                backend_timeout_s=1.0,
                connect_timeout_s=1.0,
                no_healthy_backend_status=409,
            )


class _HangingBackend(_Backend):
    """Accepts the connection and never answers -- a wedged vLLM engine.

    The failure the probe cannot see: the worker process is alive and answers is_alive,
    so only a real request reveals it.
    """

    def __init__(self, name: str, *, delay_s: float = 2.0):
        super().__init__(name)
        self.delay_s = delay_s

    async def start(self) -> None:
        app = web.Application()

        async def _handle(request: web.Request) -> web.Response:
            self.requests.append((request.method, request.path, await request.read()))
            await asyncio.sleep(self.delay_s)
            return web.Response(status=200, body=b"never gets here")

        app.router.add_route("*", "/{tail:.*}", _handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()


class TestBackendErrorHandling:
    """What the router answers when a backend fails, and why the status matters.

    Left to aiohttp, a wedged backend produces 504 -- which is in NeMo-Gym's rate-limit
    retry subset, where Gym raises its own retry ceiling on every attempt. That is an
    unbounded retry loop at backend_timeout_s per turn: exactly the hang the
    no_healthy_backend_status validator exists to prevent, arriving through the error
    path the validator never covered.
    """

    def test_a_wedged_backend_is_answered_500_and_never_504(self):
        wedged, healthy = _HangingBackend("wedged"), _Backend("healthy")

        async def _main():
            async with _Harness([wedged, healthy], backend_timeout_s=0.5) as harness:
                harness.router.set_serving_backends([wedged.url, healthy.url])
                # Force the wedged one to be picked: least-outstanding breaks a tie by
                # URL, and both are 127.0.0.1 on arbitrary ports.
                harness.router._inflight[healthy.url] = 1
                status, _, _ = await harness.call("/v1/chat/completions")
                assert status != 504, "504 puts Gym in an unbounded retry loop"
                # 500 is in Gym's *bounded* retry set, so Gym re-sends this one call and
                # the next pick lands on a healthy shard -- the rollout keeps its turns.
                assert status == 500

        asyncio.run(_main())

    def test_a_dead_backend_is_answered_500(self):
        dead, healthy = _Backend("dead"), _Backend("healthy")

        async def _main():
            # dead is never started, so connecting to it is refused.
            async with _Harness([healthy]) as harness:
                harness.router._all_backends.append(dead.url)
                harness.router._inflight[dead.url] = 0
                harness.router._backend_failures[dead.url] = 0
                harness.router.set_serving_backends([dead.url, healthy.url])
                # Force the dead one: least-outstanding breaks the 0-0 tie by URL.
                harness.router._inflight[healthy.url] = 1
                status, _, _ = await harness.call("/v1/chat/completions")
                assert status == 500

        asyncio.run(_main())

    def test_the_failing_backend_leaves_the_serving_set(self):
        """The reflex. Without it least-outstanding returns the corpse to inflight=0 and
        picks it for every subsequent request -- worse than the round-robin replaced."""
        wedged, healthy = _HangingBackend("wedged"), _Backend("healthy")

        async def _main():
            async with _Harness([wedged, healthy], backend_timeout_s=0.5) as harness:
                harness.router.set_serving_backends([wedged.url, healthy.url])
                # Force the wedged one to be picked first: least-outstanding breaks the
                # 0-0 tie by URL, and both are 127.0.0.1 on arbitrary ports.
                harness.router._inflight[healthy.url] = 1
                await harness.call("/v1/chat/completions")
                assert wedged.url not in harness.router._serving
                assert healthy.url in harness.router._serving, (
                    "the drop is surgical -- healthy backends keep serving"
                )

        asyncio.run(_main())

    def test_the_reflex_is_disarmed_without_fleet_health(self):
        """No monitor means no membership push, so a local drop would be permanent.

        A few transient blips would then drain the fleet to nothing with no way back.
        """
        wedged, healthy = _HangingBackend("wedged"), _Backend("healthy")

        async def _main():
            async with _Harness(
                [wedged, healthy], backend_timeout_s=0.5, health_managed=False
            ) as harness:
                harness.router.set_serving_backends([wedged.url])
                status, _, _ = await harness.call("/v1/chat/completions")
                assert status == 500, "still a deliberate status, just no drop"
                assert wedged.url in harness.router._serving

        asyncio.run(_main())

    def test_the_last_backend_failing_falls_back_to_the_no_healthy_status(self):
        """Once the drop empties the fleet, 500 would tell Gym to retry into nothing."""
        wedged = _HangingBackend("wedged")

        async def _main():
            async with _Harness([wedged], backend_timeout_s=0.5) as harness:
                status, _, _ = await harness.call("/v1/chat/completions")
                assert status == 409

        asyncio.run(_main())

    def test_failures_are_counted_per_backend_and_drained_once(self):
        """The bridge to the ledger: the router counts, the controller's tick reports."""
        wedged, healthy = _HangingBackend("wedged"), _Backend("healthy")

        async def _main():
            async with _Harness([wedged, healthy], backend_timeout_s=0.5) as harness:
                harness.router.set_serving_backends([wedged.url])
                await harness.call("/v1/chat/completions")
                drained = harness.router.drain_backend_outcomes()
                assert drained == {wedged.url: (0, 1)}, "(successes, failures)"
                assert harness.router.drain_backend_outcomes() == {}, "drain resets"

        asyncio.run(_main())

    def test_backend_errors_are_published_as_a_metric(self):
        wedged = _HangingBackend("wedged")

        async def _main():
            async with _Harness([wedged], backend_timeout_s=0.5) as harness:
                await harness.call("/v1/chat/completions")
                assert harness.router.metrics()["router/backend_error_total"] == 1.0

        asyncio.run(_main())


class TestPortBinding:
    def test_a_port_conflict_fails_loudly_at_construction(self):
        """Bound on the calling thread precisely so this raises where setup can see it.

        Bound inside the daemon thread instead, EADDRINUSE killed that thread while
        base_url() -- a pure string format -- kept handing Gym a URL nobody listened on,
        and Gym retried the refused connection in an uncapped loop.
        """
        import socket

        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        try:
            router = GenerationRouterImpl(
                backend_urls=["http://a:1/v1"],
                host="127.0.0.1",
                port=port,
                backend_timeout_s=1.0,
                connect_timeout_s=1.0,
                no_healthy_backend_status=409,
            )
            with pytest.raises(OSError):
                router.serve_in_background()
        finally:
            holder.close()


class TestUnknownUrlDiagnostic:
    def test_a_push_of_unknown_urls_is_reported(self, capsys):
        """Silent filtering would show up only as permanent 409s with no explanation."""
        router = GenerationRouterImpl(
            backend_urls=["http://a:1/v1"],
            host="127.0.0.1",
            port=6000,
            backend_timeout_s=1.0,
            connect_timeout_s=1.0,
            no_healthy_backend_status=409,
        )
        router.set_serving_backends(["http://a:1/v1", "http://typo:9/v1"])
        assert "http://typo:9/v1" in capsys.readouterr().out
