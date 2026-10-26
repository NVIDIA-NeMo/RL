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

"""Real worker capture -> Gym extras -> TQ codecs -> Forest canonical publication.

Generation and storage are doubles; route extraction, commitments, verification,
owner collapse, publication and cleanup execute their production code.
"""

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.staging.records import RolloutReceipt

from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tests.unit.data_plane.token_capture_test_fixtures import _manifest
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    finalize,
)
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _FakeRequest,
    _served_content,
    _worker_with_capture,
)

pytestmark = pytest.mark.nemo_gym


class _RecordingSink(TQTokenSink):
    def __init__(self, data_plane: PublicationDataPlane) -> None:
        super().__init__(data_plane, staging_partition="staged")
        self.records = []

    def stage(self, record: Any) -> Any:
        self.records.append(record)
        return super().stage(record)


def _segment(
    worker: Any, sink: _RecordingSink, scope: str, *, child: bool = True, base: int = 10
) -> dict[str, Any]:
    first = len(sink.records)
    prefix = []
    parent_hash = None
    for ordinal in range(2 if child else 1):
        admission = {
            "capture_key": scope,
            "model_call_id": f"c{first + ordinal}",
            "parent_call_id": f"c{first}" if ordinal else None,
            "mode": "token_in" if ordinal else "text",
            "prev_len": len(prefix),
        }
        if ordinal:
            admission.update(
                required_prefix_token_ids=prefix, parent_chain_hash=parent_hash
            )
        request = _FakeRequest(ng_capture=admission, stream=False)
        prompt = prefix + [10, 11]
        VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, prompt)
        payload = _served_content([12], [-0.25])
        # Old prefix routes deliberately disagree everywhere. Only its last
        # token may be overwritten; the new decode tail remains a sentinel.
        values = (
            [base + 1, base + 2, -1]
            if not ordinal
            else [900, 901, base + 7, base + 3, base + 4, -1]
        )
        payload["choices"][0]["message"]["routed_experts"] = [
            [[value]] for value in values
        ]
        result = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, request, payload
        )
        assert result["ng_commit_coords"]["disposition"] == "staged"
        assert "routed_experts" not in result["choices"][0]["message"]
        assert "predecessor_tail_route" not in result["choices"][0]["message"]
        prefix = prompt + [12]
        parent_hash = result["ng_commit_coords"]["chain_hash"]
    manifest = [
        _manifest(record).model_copy(
            update={"response_id": f"{scope}/{record.model_call_id}"}
        )
        for record in sink.records
        if record.capture_key == scope
    ]
    ids = [f"{scope}/{record.model_call_id}" for record in sink.records[first:]]
    receipt = RolloutReceipt(
        capture_key=scope,
        manifest=manifest,
        terminal_model_call_id=manifest[-1].model_call_id,
        terminal_selection="declared",
    )
    return {
        "scope": scope,
        "ids": ids,
        "receipt": receipt.model_dump(),
        "manifest": SimpleNamespace(
            records=manifest,
            attempted_call_ids=[record.model_call_id for record in manifest],
            pending_call_ids=[],
        ),
    }


@pytest.fixture
def routes() -> tuple:
    data_plane = PublicationDataPlane()
    sink = _RecordingSink(data_plane)
    worker = _worker_with_capture(sink)
    finalizer = RolloutReassembler(
        data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=True,
    )
    return data_plane, sink, worker, finalizer


def test_forest_routes_patch_only_predecessor_tail_within_each_segment(
    routes: tuple,
) -> None:
    data_plane, sink, worker, finalizer = routes
    owners = [
        [
            _segment(worker, sink, "group_g0"),
            _segment(worker, sink, "group_g0", base=30),
        ],
        [_segment(worker, sink, "group_g1", child=False, base=50)],
    ]
    assert "predecessor_tail_route" not in sink.records[0].extras
    assert "predecessor_tail_route" not in sink.records[1].extras
    ordinary = finalizer.finalize_rollout("group_g0", owners[0][0]["receipt"], reward=1)
    assert ordinary.routed_experts[:, 0, 0].tolist() == [11, 12, 17, 13, 14, -1]
    result = finalize(finalizer, owners)
    assert result.valid_row_count == result.total_row_count == 3
    for key, expected in zip(
        result.meta.sample_ids,
        ([11, 12, 17, 13, 14, -1], [31, 32, 37, 33, 34, -1], [51, 52, -1]),
        strict=True,
    ):
        row = data_plane.rows["canonical", key]
        length = row["input_lengths"].item()
        assert row["routed_experts"][0, :length, 0, 0].tolist() == expected
    assert data_plane.events[-1][0] == "clear"
    assert not any(partition == "staged" for partition, _ in data_plane.rows)


@pytest.mark.parametrize("failure", ["boundary_tamper", "no_routes"])
def test_bad_committed_routes_invalidate_whole_owner(
    routes: tuple, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    data_plane, sink, worker, finalizer = routes
    first = _segment(worker, sink, "group_g0")
    extract = worker.token_capture.adapter.extract_extras
    if failure == "no_routes":
        monkeypatch.setattr(
            worker.token_capture.adapter, "extract_extras", lambda _: None
        )
    second = _segment(worker, sink, "group_g0", base=30)
    monkeypatch.setattr(worker.token_capture.adapter, "extract_extras", extract)
    healthy = _segment(worker, sink, "group_g1", child=False, base=50)
    if failure == "boundary_tamper":
        row = data_plane.rows["staged", second["manifest"].records[-1].staging_key]
        # The first route is the committed predecessor boundary. Changing only
        # this entry must invalidate the entire owner, including other roots.
        row["routed_experts"][0, 0, 0, 0] = 99
    result = finalize(finalizer, [[first, second], [healthy]])
    assert result.meta.sample_ids == ["group_g0_s0", "group_g0_s1", "group_g1_s0"]
    assert result.valid_row_count == 1
    assert len(result.reward_observations) == 1
    assert result.reward_observations[0].sample_id == "group_g1"
    assert data_plane.rows["canonical", "group_g0_s0"]["sample_mask"].item() == 0
    assert data_plane.rows["canonical", "group_g0_s1"]["sample_mask"].item() == 0
    assert data_plane.rows["canonical", "group_g1_s0"]["sample_mask"].item() == 1
    assert not any(partition == "staged" for partition, _ in data_plane.rows)


def test_forest_still_rejects_deferred_route_assembly(routes: tuple) -> None:
    _, sink, worker, finalizer = routes
    owner = _segment(worker, sink, "group_g0")
    finalizer._defer_routed_experts_to_policy = True
    with pytest.raises(ValueError, match="requires direct route assembly"):
        finalize(finalizer, [[owner]])
