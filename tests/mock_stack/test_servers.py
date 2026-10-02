# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSource
from tests.mock_stack.components import CopyRefit, Generation, Policy
from tests.mock_stack.config import Prompt
from tests.mock_stack.servers import GenerationServer


@pytest.mark.asyncio
async def test_http_generation_stages_exact_prefix_before_reply():
    plane = NoOpDataPlaneClient()
    plane.register_partition("staging", list(STAGING_FIELDS), 32, ["finalize"])
    source = TQTokenSource(plane, staging_partition="staging")
    generation = Generation()
    CopyRefit(Policy(), generation).sync_weights()
    server = GenerationServer(
        generation, [Prompt(id="P1", turns=2, turn_seconds=[0.001])]
    )
    server.setup_token_capture(plane, "staging")
    server.set_rollout_weight_version(2)
    request = {
        "model": "cpu",
        "messages": [{"role": "user", "content": "P1"}],
        "ng_capture": {
            "rollout_id": "group_g0",
            "model_call_id": "one",
            "mode": "text",
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200, response.text
        coords = response.json()["ng_commit_coords"]
        assert coords["disposition"] == "staged"
        first = source.fetch([coords["staging_key"]])[0]
        assert first.weight_version == 2
        prefix = source.fetch_prefix_token_ids([coords["staging_key"]])
        server.set_rollout_weight_version(3)
        request["messages"].append(response.json()["choices"][0]["message"])
        request["messages"].append(
            {"role": "tool", "tool_call_id": "one", "content": "ok"}
        )
        request["ng_capture"] = {
            "rollout_id": "group_g0",
            "model_call_id": "two",
            "mode": "token_in",
            "parent_call_id": "one",
            "prev_len": len(prefix),
            "parent_chain_hash": first.chain_hash,
            "staging_chain": [coords["staging_key"]],
        }
        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200, response.text
        second = source.fetch([response.json()["ng_commit_coords"]["staging_key"]])[0]
        assert second.prev_len == first.cum_len
        assert second.weight_version == 3
        assert (
            source.fetch_prefix_token_ids([first.staging_key, second.staging_key])[
                : len(prefix)
            ]
            == prefix
        )
        assert sum(first.token_mask_delta) == sum(second.token_mask_delta) == 4
        assert [event.turn for event in server.calls] == [1, 2]


@pytest.mark.asyncio
async def test_http_generation_rejects_requests_without_capture():
    server = GenerationServer(Generation(), [])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 422
