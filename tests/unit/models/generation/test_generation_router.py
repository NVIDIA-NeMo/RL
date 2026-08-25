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

Gym deterministically hashes each session to one policy endpoint from a list fixed at
process start, never fails over, and retries a refused connection in an uncapped loop
with no HTTP timeout. Handing it one NeMo-RL-owned URL moves the routing decision here,
next to the fleet health that knows which shards are serving -- without changing Gym.

Most of these run a real aiohttp server against real backends. A proxy is precisely the
component that unit fakes flatter: header handling, streaming and status propagation only
misbehave over an actual socket.
"""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession, web

from nemo_rl.models.generation import generation_router as router_module
from nemo_rl.models.generation.generation_router import (
    GenerationRouterConfig,
    GenerationRouterImpl,
    maybe_start_generation_router,
)


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
            max_inflight_requests_per_backend=router_kwargs.pop(
                "max_inflight_requests_per_backend", None
            ),
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


class _GatedBackend(_Backend):
    """Backend whose individual responses are released explicitly by a test."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.gates: list[asyncio.Future[None]] = []
        self._request_started = asyncio.Event()

    async def start(self) -> None:
        app = web.Application()

        async def _handle(request: web.Request) -> web.Response:
            body = await request.read()
            self.requests.append((request.method, request.path, body))
            gate = asyncio.get_running_loop().create_future()
            self.gates.append(gate)
            self._request_started.set()
            await gate
            return web.Response(
                status=200,
                body=body,
                headers={"X-Served-By": self.name},
            )

        app.router.add_route("*", "/{tail:.*}", _handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def wait_for_started(self, count: int) -> None:
        async def _wait() -> None:
            while len(self.requests) < count:
                self._request_started.clear()
                if len(self.requests) < count:
                    await self._request_started.wait()

        await asyncio.wait_for(_wait(), timeout=2.0)

    def release(self, index: int) -> None:
        if not self.gates[index].done():
            self.gates[index].set_result(None)

    def release_all(self) -> None:
        for gate in self.gates:
            if not gate.done():
                gate.set_result(None)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def _wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), timeout=2.0)


class TestRouterStartup:
    def test_default_range_uses_the_dedicated_head_node_carveout(self) -> None:
        config = GenerationRouterConfig()

        assert (config.port_range_low, config.port_range_high) == (1100, 1200)
        assert config.max_inflight_requests_per_backend is None

    def test_admission_cap_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            GenerationRouterConfig(max_inflight_requests_per_backend=0)

    def test_disabled_returns_an_unmodified_copy_of_raw_urls(self) -> None:
        raw_urls = ["http://a:1/v1", None, "http://b:2/v1"]

        routed_urls, handle = maybe_start_generation_router(
            backend_urls=raw_urls,
            config=GenerationRouterConfig(enabled=False),
            health_managed=False,
        )

        assert routed_urls == raw_urls
        assert routed_urls is not raw_urls
        assert raw_urls == ["http://a:1/v1", None, "http://b:2/v1"]
        assert handle is None

    def test_enabled_returns_one_router_url_and_retained_handle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        raw_urls = ["http://a:1/v1", None, "http://b:2/v1"]
        actor_handle = MagicMock()
        base_url_ref = object()
        actor_handle.base_url.remote.return_value = base_url_ref
        actor_class = MagicMock()
        actor_class.options.return_value.remote.return_value = actor_handle
        get_free_port = MagicMock(return_value=1107)
        monkeypatch.setattr(router_module, "GenerationRouterActor", actor_class)
        monkeypatch.setattr(router_module, "_get_free_port_local", get_free_port)
        monkeypatch.setattr(router_module, "_get_node_ip_local", lambda: "10.0.0.1")
        monkeypatch.setattr(
            router_module.ray,
            "get_runtime_context",
            lambda: SimpleNamespace(get_node_id=lambda: "a" * 56),
        )
        monkeypatch.setattr(
            router_module.ray,
            "get",
            lambda value: "http://10.0.0.1:1107/v1" if value is base_url_ref else value,
        )

        config = GenerationRouterConfig(
            enabled=True, max_inflight_requests_per_backend=7
        )
        config_before = config.model_dump()
        routed_urls, handle = maybe_start_generation_router(
            backend_urls=raw_urls,
            config=config,
            health_managed=False,
        )

        assert routed_urls == ["http://10.0.0.1:1107/v1"]
        assert handle is actor_handle
        assert raw_urls == ["http://a:1/v1", None, "http://b:2/v1"]
        assert config.model_dump() == config_before
        assert "generation_fleet_health.enabled=false" in capsys.readouterr().out
        get_free_port.assert_called_once_with(1100, 1200)
        actor_class.options.return_value.remote.assert_called_once_with(
            backend_urls=["http://a:1/v1", "http://b:2/v1"],
            host="10.0.0.1",
            port=1107,
            backend_timeout_s=600.0,
            connect_timeout_s=5.0,
            max_inflight_requests_per_backend=7,
            no_healthy_backend_status=409,
            health_managed=False,
        )

    def test_enabled_requires_an_http_backend(self) -> None:
        with pytest.raises(ValueError, match="none were reported"):
            maybe_start_generation_router(
                backend_urls=[None],
                config=GenerationRouterConfig(enabled=True),
                health_managed=False,
            )


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


class TestBoundedAdmission:
    def test_the_cap_holds_excess_requests_centrally_in_fifo_order(self) -> None:
        backend = _GatedBackend("only")

        async def _main() -> None:
            async with _Harness(
                [backend], max_inflight_requests_per_backend=2
            ) as harness:
                tasks = [
                    asyncio.create_task(
                        harness.call("/v1/chat/completions", body=str(i).encode())
                    )
                    for i in range(2)
                ]
                await backend.wait_for_started(2)

                for i in range(2, 5):
                    tasks.append(
                        asyncio.create_task(
                            harness.call("/v1/chat/completions", body=str(i).encode())
                        )
                    )
                    expected_queued = i - 1
                    await _wait_until(
                        lambda: harness.router._admission_queued == expected_queued
                    )

                assert len(backend.requests) == 2
                assert max(harness.router._inflight.values()) == 2

                backend.release(0)
                await backend.wait_for_started(3)
                assert backend.requests[2][2] == b"2"
                backend.release(1)
                await backend.wait_for_started(4)
                assert backend.requests[3][2] == b"3"
                backend.release(2)
                await backend.wait_for_started(5)
                assert backend.requests[4][2] == b"4"

                backend.release_all()
                await asyncio.gather(*tasks)
                metrics = harness.router.metrics()
                assert metrics["router/admission_queued_requests"] == 0.0
                assert metrics["router/admission_queued_requests_peak"] == 3.0
                assert (
                    metrics["router/max_generation_inflight_per_backend_observed"]
                    == 2.0
                )
                assert harness.router._inflight[backend.url] == 0

        asyncio.run(_main())

    def test_a_freed_backend_refills_while_another_backend_remains_busy(
        self,
    ) -> None:
        first, second = _GatedBackend("first"), _GatedBackend("second")

        async def _main() -> None:
            async with _Harness(
                [first, second], max_inflight_requests_per_backend=1
            ) as harness:
                tasks = [
                    asyncio.create_task(
                        harness.call("/v1/chat/completions", body=b"first")
                    ),
                    asyncio.create_task(
                        harness.call("/v1/chat/completions", body=b"second")
                    ),
                ]
                await asyncio.gather(
                    first.wait_for_started(1), second.wait_for_started(1)
                )
                third = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"third")
                )
                tasks.append(third)
                await _wait_until(lambda: harness.router._admission_queued == 1)

                first.release(0)
                await first.wait_for_started(2)
                assert first.requests[1][2] == b"third"
                assert len(second.requests) == 1, "second remains occupied"

                first.release_all()
                second.release_all()
                await asyncio.gather(*tasks)

        asyncio.run(_main())

    def test_membership_readd_from_another_thread_wakes_a_waiter(self) -> None:
        first, second = _GatedBackend("first"), _GatedBackend("second")

        async def _main() -> None:
            harness = _Harness([first, second], max_inflight_requests_per_backend=1)
            harness.router.set_serving_backends([first.url])
            async with harness:
                active = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"active")
                )
                await first.wait_for_started(1)
                queued = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"queued")
                )
                await _wait_until(lambda: harness.router._admission_queued == 1)

                await asyncio.to_thread(
                    harness.router.set_serving_backends, [first.url, second.url]
                )
                await second.wait_for_started(1)
                assert second.requests[0][2] == b"queued"

                first.release_all()
                second.release_all()
                await asyncio.gather(active, queued)

        asyncio.run(_main())

    def test_empty_membership_releases_queued_requests_with_409(self) -> None:
        backend = _GatedBackend("only")

        async def _main() -> None:
            async with _Harness(
                [backend], max_inflight_requests_per_backend=1
            ) as harness:
                active = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"active")
                )
                await backend.wait_for_started(1)
                queued = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"queued")
                )
                await _wait_until(lambda: harness.router._admission_queued == 1)

                await asyncio.to_thread(harness.router.set_serving_backends, [])
                status, body, _ = await asyncio.wait_for(queued, timeout=1.0)
                assert status == 409
                assert b"no healthy generation backend" in body
                assert len(backend.requests) == 1

                backend.release_all()
                await active

        asyncio.run(_main())

    @pytest.mark.parametrize("path", ["/tokenize", "/v1/models"])
    def test_short_control_calls_bypass_the_generation_queue(self, path: str) -> None:
        backend = _GatedBackend("only")

        async def _main() -> None:
            async with _Harness(
                [backend], max_inflight_requests_per_backend=1
            ) as harness:
                generation = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"generation")
                )
                await backend.wait_for_started(1)
                control = asyncio.create_task(harness.call(path, body=b"control"))
                await backend.wait_for_started(2)

                assert backend.requests[1][1] == path
                assert backend.requests[1][2] == b"control"
                assert harness.router._admission_queued == 0

                backend.release_all()
                await asyncio.gather(generation, control)

        asyncio.run(_main())

    @pytest.mark.parametrize("path", ["/tokenize", "/v1/models"])
    def test_control_work_does_not_consume_a_generation_slot(self, path: str) -> None:
        backend = _GatedBackend("only")

        async def _main() -> None:
            async with _Harness(
                [backend], max_inflight_requests_per_backend=1
            ) as harness:
                control = asyncio.create_task(harness.call(path, body=b"control"))
                await backend.wait_for_started(1)
                generation = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"generation")
                )
                await backend.wait_for_started(2)

                assert backend.requests[1][1] == "/v1/chat/completions"
                assert backend.requests[1][2] == b"generation"
                assert harness.router._admission_queued == 0
                assert harness.router._inflight[backend.url] == 2
                assert harness.router._generation_inflight[backend.url] == 1

                backend.release_all()
                await asyncio.gather(control, generation)

        asyncio.run(_main())

    def test_cancellation_never_leaks_a_reserved_slot(self) -> None:
        backend = _Backend("only")
        router = _Harness([backend], max_inflight_requests_per_backend=1).router

        async def _main() -> None:
            first = await router._acquire_backend()
            assert first == backend.url

            cancelled_while_queued = asyncio.create_task(router._acquire_backend())
            await _wait_until(lambda: router._admission_queued == 1)
            cancelled_while_queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_while_queued
            assert router._admission_queued == 0

            # Dispatch reserves the released slot synchronously, but the waiter task
            # has not resumed yet. Cancellation in this window must reclaim that slot.
            assigned_before_resume = asyncio.create_task(router._acquire_backend())
            await _wait_until(lambda: router._admission_queued == 1)
            router._release_backend(first, is_generation=True)
            assigned_before_resume.cancel()
            with pytest.raises(asyncio.CancelledError):
                await assigned_before_resume
            assert router._inflight[backend.url] == 0

            final = await asyncio.wait_for(router._acquire_backend(), timeout=1.0)
            assert final == backend.url
            router._release_backend(final, is_generation=True)
            assert router._inflight[backend.url] == 0

        asyncio.run(_main())

    def test_queue_wait_does_not_consume_the_backend_timeout(self) -> None:
        backend = _GatedBackend("only")

        async def _main() -> None:
            async with _Harness(
                [backend],
                max_inflight_requests_per_backend=1,
                backend_timeout_s=0.2,
            ) as harness:
                reserved = await harness.router._acquire_backend()
                assert reserved == backend.url
                request = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"request")
                )
                await _wait_until(lambda: harness.router._admission_queued == 1)

                # The request spends longer than backend_timeout_s in admission. That
                # wait must not burn its router->backend deadline before forwarding.
                await asyncio.sleep(0.25)
                assert not request.done()
                harness.router._release_backend(reserved, is_generation=True)
                await backend.wait_for_started(1)
                backend.release(0)

                status, _, _ = await request
                assert status == 200

        asyncio.run(_main())

    def test_backend_failure_handoff_respects_the_remaining_backend_cap(self) -> None:
        failing = _HangingBackend("failing", delay_s=1.0)
        healthy = _GatedBackend("healthy")

        async def _main() -> None:
            harness = _Harness(
                [failing, healthy],
                max_inflight_requests_per_backend=1,
                backend_timeout_s=0.5,
            )
            harness.router.set_serving_backends([failing.url])
            async with harness:
                failing_call = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"A")
                )
                await _wait_until(lambda: len(failing.requests) == 1)
                # Give A a head start so its timeout cannot race B's while we inspect
                # the handoff. Both still use the same production timeout setting.
                await asyncio.sleep(0.2)

                harness.router.set_serving_backends([failing.url, healthy.url])
                healthy_call = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"B")
                )
                await healthy.wait_for_started(1)
                queued_call = asyncio.create_task(
                    harness.call("/v1/chat/completions", body=b"C")
                )
                await _wait_until(lambda: harness.router._admission_queued == 1)

                failing_status, _, _ = await failing_call
                assert failing_status == 500
                assert failing.url not in harness.router._serving
                assert len(healthy.requests) == 1
                assert harness.router._admission_queued == 1

                healthy.release(0)
                await healthy.wait_for_started(2)
                assert healthy.requests[1][2] == b"C"
                healthy.release(1)
                await asyncio.gather(healthy_call, queued_call)

        asyncio.run(_main())


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
            max_inflight_requests_per_backend=None,
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
            max_inflight_requests_per_backend=None,
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
                max_inflight_requests_per_backend=None,
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
        picks it for every subsequent request -- worse than the sticky hash replaced."""
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
                drained = harness.router.drain_backend_failures()
                assert drained == {wedged.url: 1}
                assert harness.router.drain_backend_failures() == {}, "drain resets"

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
                max_inflight_requests_per_backend=None,
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
            max_inflight_requests_per_backend=None,
            no_healthy_backend_status=409,
        )
        router.set_serving_backends(["http://a:1/v1", "http://typo:9/v1"])
        assert "http://typo:9/v1" in capsys.readouterr().out
