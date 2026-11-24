# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accepted-response retries through HTTP capture, RL decisions and publication."""

import asyncio

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.staging.records import RolloutManifest

from tests.unit.experience.test_cc_cpu_pipeline import pipeline  # noqa: F401
from tests.unit.experience.test_forest_dispatch import _env
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
        messages = list(gym_harness.HISTORY)
        accepted_tokens, rejected_tokens, rejected_ids = [], [], []
        accepted_output = []
        selected_ids = []
        for turn in range(20):
            if compact and turn == 10:
                messages = [{"role": "user", "content": "summary of previous work"}]
            if turn in (0, 10):
                rejected = gym_harness.assert_clean(harness.post("group_g0", messages))
                rejected_tokens.append(1000 + len(harness.worker_calls))
                rejected_ids.append(rejected["id"])
                # Retry the same accepted history, excluding the rejected output.
            response = gym_harness.assert_clean(harness.post("group_g0", messages))
            selected_ids.append(response["id"])
            accepted_output.extend(response["output"])
            accepted_tokens.append(1000 + len(harness.worker_calls))
            messages = (
                messages
                + response["output"]
                + [{"role": "user", "content": f"observation {turn}"}]
            )
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
        processed = await env._postprocess_receipt_mode(
            {"_ng_rollout_id": "group_g0"},
            {
                "reward": 1.0,
                "response": {"id": response["id"], "output": accepted_output},
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
