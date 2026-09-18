# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real agent-to-resource transport, not a mocked ServerClient."""

import asyncio
from unittest.mock import AsyncMock

import aiohttp
from aiohttp import web
from omegaconf import OmegaConf
import pytest

from nemo_gym import server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient


def test_mutation_budget_reaches_real_http_endpoint(monkeypatch):
    async def check():
        received = []

        async def endpoint(request):
            received.append(await request.json())
            return web.json_response({"env_id": "test-session"})

        app = web.Application()
        app.router.add_post("/seed_session", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            client = ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
                global_config_dict=OmegaConf.create(
                    {
                        "resource": {
                            "resources_servers": {
                                "test": {
                                    "host": "127.0.0.1",
                                    "port": port,
                                }
                            }
                        }
                    }
                ),
            )
            async with aiohttp.ClientSession() as session:
                monkeypatch.setattr(
                    server_utils, "get_global_aiohttp_client", lambda: session
                )
                response = await asyncio.wait_for(
                    client.post(
                        server_name="resource",
                        url_path="/seed_session",
                        json={"env_id": "test-session"},
                        max_connection_attempts=1,
                    ),
                    timeout=2,
                )
                assert response.status == 200
                assert await response.json() == {"env_id": "test-session"}
                assert received == [{"env_id": "test-session"}]
        finally:
            await runner.cleanup()

    asyncio.run(check())


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ServerDisconnectedError("disconnect"),
        aiohttp.ClientOSError(104, "reset"),
        asyncio.TimeoutError("timeout"),
    ],
)
@pytest.mark.parametrize("attempts", [1, 2])
def test_explicit_mutation_budget_bounds_every_transport_failure(
    monkeypatch, error, attempts
):
    client = AsyncMock()
    client.request.side_effect = error
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())
    with pytest.raises(type(error)):
        asyncio.run(
            server_utils.request(
                "POST",
                "http://unused/step",
                _internal=True,
                _max_connection_retries=attempts,
            )
        )
    assert client.request.await_count == attempts


def test_invalid_aiohttp_keyword_does_not_retry_forever(monkeypatch):
    async def check():
        async with aiohttp.ClientSession() as session:
            monkeypatch.setattr(
                server_utils, "get_global_aiohttp_client", lambda: session
            )
            with pytest.raises(TypeError, match="unexpected keyword"):
                await asyncio.wait_for(
                    server_utils.request(
                        "POST",
                        "http://127.0.0.1:1/step",
                        _internal=True,
                        nonexistent_request_option=True,
                    ),
                    timeout=1,
                )

    asyncio.run(check())
