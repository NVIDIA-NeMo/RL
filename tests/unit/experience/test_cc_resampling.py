# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accepted-response retries through HTTP capture, RL decisions and publication."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management import (
    ContextHistoryConfig,
    ContextManagedResponsesClient,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.staging.records import RolloutManifest
from responses_api_agents.simple_agent_with_compaction.tests.test_client import (
    http_response,
)

from tests.unit.experience.test_cc_cpu_pipeline import pipeline  # noqa: F401
from tests.unit.experience.test_cc_dispatch import _env
from tests.unit.experience.test_logical_owner_finalization import gym_harness

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize(
    "compact,pipeline", [(False, True), (True, True)], indirect=["pipeline"]
)
def test_rejected_first_and_middle_responses_never_enter_selected_traces(
    pipeline: tuple,  # noqa: F811
    compact: bool,
) -> None:
    harness, plane, finalizer = pipeline
    env = _env(harness)

    async def exercise() -> None:
        transport = MagicMock(spec=ServerClient)

        async def post(**kwargs: Any) -> Any:
            assert kwargs["_retry"] is False
            response = harness.client.post(
                kwargs["url_path"],
                json=kwargs["json"].model_dump(mode="json"),
                headers=kwargs["headers"],
            )
            gym_harness.assert_clean(response)
            return http_response(response.json())

        transport.post = AsyncMock(side_effect=post)
        client = ContextManagedResponsesClient(
            server_client=transport,
            model_server=ModelServerRef(name="policy", type="responses_api_models"),
            logical_rollout_id="group_g0",
            initial_request=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            config=ContextHistoryConfig.model_validate(
                {
                    "enabled": compact,
                    "max_response_retries": 1,
                    "policy": {
                        "type": "recency",
                        "config": {
                            "reasoning": {"enabled": True, "keep_last_blocks": 0}
                        },
                    },
                    "schedule": {
                        "type": "turn_chunked_recency",
                        "actions_per_chunk": 10,
                    },
                }
            ),
        )
        accepted_tokens, rejected_tokens, rejected_ids = [], [], []
        accepted_output = []
        selected_ids = []
        for turn in range(20):
            attempts = 0

            async def select(response: Any) -> bool:
                nonlocal attempts
                attempts += 1
                if turn in (0, 10) and attempts == 1:
                    rejected_tokens.append(1000 + len(harness.worker_calls))
                    rejected_ids.append(response.id)
                    return False
                return True

            response = await client.create(select_response=select)
            selected_ids.append(response.id)
            accepted_output.extend(response.model_dump(mode="json")["output"])
            accepted_tokens.append(1000 + len(harness.worker_calls))
            if turn < 19:
                client.append_observation(
                    [{"role": "user", "content": f"observation {turn}"}]
                )
        client.finish(response)
        manifest = RolloutManifest.model_validate(
            await harness.ledger.manifest("group_g0")
        )
        assert len(manifest.records) == len(harness.worker_calls) == 22
        assert len(selected_ids) == 20
        assert not manifest.failures and not manifest.pending_call_ids
        by_id = {record.response_id: record for record in manifest.records}
        for index, response_id in enumerate(selected_ids):
            record = by_id[response_id]
            expected_parent = (
                None
                if index == 0 or (compact and index == 10)
                else by_id[selected_ids[index - 1]].model_call_id
            )
            assert record.parent_call_id == expected_parent
        assert all(response_id not in selected_ids for response_id in rejected_ids)
        # The retry after the middle rejection still proposes the accepted turn 10,
        # even across compaction; only the RL decision may turn it into a root.
        assert (
            harness.worker_calls[12][0].parent_call_id
            == by_id[selected_ids[9]].model_call_id
        )

        processed = await env._postprocess_receipt_mode(
            {"_ng_rollout_id": "group_g0"},
            {
                "reward": 1.0,
                "response": {"id": response.id, "output": accepted_output},
            },
        )
        assert processed["logical_selection"].response_ids == tuple(selected_ids)
        finalized = finalizer.finalize_group(
            "group",
            ["group_g0"],
            [processed["receipt"]],
            [1.0],
            mask_sample=[False],
            fallback_weight_version=7,
            prompt_idx=99,
            logical_selections=[processed["logical_selection"]],
            canonical_sample_ids=["group_g0"],
        )
        assert finalized.valid_row_count == finalized.meta.size == (2 if compact else 1)
        data = plane.get_samples(
            finalized.meta.sample_ids, "train", ["input_ids", "token_mask"]
        )
        trained, all_tokens = [], []
        for index in range(finalized.meta.size):
            tokens = data["input_ids"][index]
            trained.extend(tokens[data["token_mask"][index].bool()].tolist())
            all_tokens.extend(tokens.tolist())
        assert trained == accepted_tokens
        assert not set(rejected_tokens).intersection(all_tokens)
        assert (
            plane.list_sample_ids("staged") == []
        )  # Includes both discarded branches.

    asyncio.run(exercise())
