# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""CPU storage-double composition; this does not run live TransferQueue or vLLM.

Gym HTTP capture and worker staging feed the real TQ sink/source and unchanged
RL finalizer. Only model generation and the DataPlane storage boundary are doubles.
Requires the paired Gym checkout containing the segment-capture test harness.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tensordict import TensorDict

pytest.importorskip("nemo_gym", reason="requires the nemo_gym extra")

from responses_api_models.vllm_model.tests import test_segment_capture as gym_harness

from nemo_rl.data_plane.codec import stack_or_nest
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler

pytestmark = pytest.mark.nemo_gym


class MemoryDataPlane:
    """Store the sink's actual TensorDict rows and serve explicitly selected fields."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], TensorDict] = {}
        self.lose_ack = False
        self.write_count = 0
        self.delete_count = 0

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict],
    ) -> None:
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = fields[index : index + 1].clone()
            self.write_count += 1
        if self.lose_ack:
            raise OSError("write completed but acknowledgement lost")

    def get_samples(
        self, *, sample_ids: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        return TensorDict(
            {
                name: stack_or_nest(
                    [self.rows[partition_id, key][name][0] for key in sample_ids]
                )
                for name in select_fields
            },
            batch_size=[len(sample_ids)],
        )

    def clear_samples(self, **kwargs: Any) -> None:
        self.delete_count += 1
        raise AssertionError("finalize_rollout must retain staged evidence")


@pytest.fixture
def stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[Any, MemoryDataPlane, RolloutReassembler]]:
    data_plane = MemoryDataPlane()
    source = TQTokenSource(data_plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(data_plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
    )
    finalizer = RolloutReassembler(
        data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    yield harness, data_plane, finalizer
    harness.client.close()
    asyncio.run(harness.ledger.close())


def receipt_for(harness: Any, rollout_id: str, response_id: str | None = None) -> dict:
    manifest = harness.manifest(rollout_id)
    terminal = next(
        (
            record.model_call_id
            for record in manifest.records
            if record.response_id == response_id
        ),
        None,
    )
    return {
        "rollout_id": rollout_id,
        "manifest": [record.model_dump(mode="json") for record in manifest.records],
        "terminal_model_call_id": terminal,
        "terminal_selection": "declared",
        "capture_poisoned": bool(manifest.failures),
    }


def test_two_segments_round_trip_selected_chains_and_reject_corruption(
    stack: tuple,
) -> None:
    harness, data_plane, finalizer = stack
    receipts = {}
    for segment, selected_index in [("L_s0", 2), ("L_s1", 5)]:
        discarded = gym_harness.assert_clean(
            harness.post(segment, gym_harness.HISTORY, parent=None)
        )
        selected = gym_harness.assert_clean(
            harness.post(segment, gym_harness.HISTORY, parent=None)
        )
        child = gym_harness.assert_clean(
            harness.post(
                segment, gym_harness.HISTORY + selected["output"], parent=selected["id"]
            )
        )
        receipt = receipt_for(harness, segment, child["id"])
        receipts[segment] = receipt
        result = finalizer.finalize_rollout(segment, receipt, reward=1.0)
        assert result.valid, result.rejection_reason
        start = selected_index * 10
        assert result.token_ids == [
            start,
            start + 1,
            1000 + selected_index,
            start + 10,
            start + 11,
            1001 + selected_index,
        ]
        assert result.token_mask == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        assert result.logprobs == [0.0, 0.0, -0.25, 0.0, 0.0, -0.25]
        assert result.prompt_len == 2 and result.min_wv == result.max_wv == 7
        manifest = harness.manifest(segment)
        assert len(manifest.records) == 3 and manifest.failures == []
        assert {record.delta_len for record in manifest.records} == {3}
        discarded_record = next(
            record
            for record in manifest.records
            if record.response_id == discarded["id"]
        )
        assert (
            data_plane.rows["staged", discarded_record.staging_key]["token_ids_delta"][
                0, -1
            ].item()
            not in result.token_ids
        )
    assert data_plane.write_count == len(data_plane.rows) == 6
    selected_key = receipts["L_s0"]["manifest"][1]["staging_key"]
    data_plane.rows["staged", selected_key]["token_ids_delta"][0, 0] += 1
    rejected = finalizer.finalize_rollout("L_s0", receipts["L_s0"], reward=1.0)
    assert not rejected.valid and "digest" in rejected.rejection_reason
    assert rejected.token_ids == [] and data_plane.delete_count == 0
    assert len(data_plane.rows) == 6


@pytest.mark.parametrize("explicit", [False, True])
def test_lost_write_ack_fails_cc_and_retains_ordinary_poisoned_evidence(
    stack: tuple, explicit: bool
) -> None:
    harness, data_plane, finalizer = stack
    data_plane.lose_ack = True
    response = harness.post(
        "L_s0", gym_harness.HISTORY, **({"parent": None} if explicit else {})
    )
    gym_harness.assert_clean(response, 502 if explicit else 200)
    assert (
        len(harness.worker_calls) == data_plane.write_count == len(data_plane.rows) == 1
    )
    manifest = harness.manifest("L_s0")
    assert manifest.failures and manifest.records == []
    result = finalizer.finalize_rollout(
        "L_s0", receipt_for(harness, "L_s0"), reward=1.0
    )
    assert not result.valid and result.rejection_reason == "capture_poisoned"
    assert result.token_ids == [] and data_plane.delete_count == 0
    assert len(data_plane.rows) == 1
