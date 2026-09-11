"""Real worker capture -> Gym extras -> TQ codecs -> CC canonical publication.

Generation and storage are doubles; route extraction, commitments, verification,
owner collapse, publication and cleanup execute their production code.
"""

import json
from typing import Any

import pytest
import torch
from nemo_gym.token_id_capture.staging.records import RolloutReceipt

from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.experience.rollout_reassembler import RolloutReassembler, SegmentReceipt
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
) -> SegmentReceipt:
    first = len(sink.records)
    prefix = []
    parent_hash = None
    for ordinal in range(2 if child else 1):
        admission = {
            "rollout_id": scope,
            "model_call_id": f"c{ordinal}",
            "parent_call_id": "c0" if ordinal else None,
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
        for record in sink.records[first:]
    ]
    receipt = RolloutReceipt(
        rollout_id=scope,
        manifest=manifest,
        terminal_model_call_id=manifest[-1].model_call_id,
        terminal_selection="declared",
    )
    return SegmentReceipt(
        scope, receipt.model_dump(), tuple(item.response_id for item in manifest)
    )


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


def test_cc_routes_patch_only_predecessor_tail_within_each_segment(
    routes: tuple,
) -> None:
    data_plane, sink, worker, finalizer = routes
    owners = [
        [
            _segment(worker, sink, "group_g0_s0"),
            _segment(worker, sink, "group_g0_s1", base=30),
        ],
        [_segment(worker, sink, "group_g1_s0", child=False, base=50)],
    ]
    assert "predecessor_tail_route" not in sink.records[0].extras
    assert sink.records[1].extras["predecessor_tail_route"] == [[17]]
    ordinary = finalizer.finalize_rollout("group_g0_s0", owners[0][0].receipt, reward=1)
    assert ordinary.routed_experts[:, 0, 0].tolist() == [11, 12, -1, 13, 14, -1]
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


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "shape",
        "negative",
        "float",
        "overflow",
        "root",
        "tamper",
        "no_routes",
    ],
)
def test_bad_committed_tail_or_tampering_invalidates_whole_owner(
    routes: tuple, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    data_plane, sink, worker, finalizer = routes
    first = _segment(worker, sink, "group_g0_s0")
    extract = worker.token_capture.adapter.extract_extras

    def extras(payload: dict) -> dict | None:
        if failure == "no_routes":
            return None
        result = extract(payload)
        if failure == "root" and "predecessor_tail_route" not in result:
            result["predecessor_tail_route"] = [[1]]
        elif "predecessor_tail_route" in result:
            if failure == "missing":
                del result["predecessor_tail_route"]
            elif failure not in ("root", "tamper"):
                result["predecessor_tail_route"] = {
                    "shape": [[1, 2]],
                    "negative": [[-1]],
                    "float": [[1.0]],
                    "overflow": [[32768]],
                }[failure]
        return result

    monkeypatch.setattr(worker.token_capture.adapter, "extract_extras", extras)
    second = _segment(worker, sink, "group_g0_s1", base=30)
    monkeypatch.setattr(worker.token_capture.adapter, "extract_extras", extract)
    healthy = _segment(worker, sink, "group_g1_s0", child=False, base=50)
    if failure == "tamper":
        row = data_plane.rows["staged", "group_g0_s1/c1"]
        encoded = json.dumps(
            {"predecessor_tail_route": [[99]]}, separators=(",", ":")
        ).encode()
        row["extras_metadata_json"] = torch.tensor([list(encoded)], dtype=torch.uint8)
    result = finalize(finalizer, [[first, second], [healthy]])
    assert result.meta.sample_ids == ["group_g0_s0", "group_g1_s0"]
    assert result.valid_row_count == 1
    assert data_plane.rows["canonical", "group_g0_s0"]["sample_mask"].item() == 0
    assert data_plane.rows["canonical", "group_g1_s0"]["sample_mask"].item() == 1
    assert not any(partition == "staged" for partition, _ in data_plane.rows)


def test_cc_still_rejects_deferred_route_assembly(routes: tuple) -> None:
    _, sink, worker, finalizer = routes
    owner = _segment(worker, sink, "group_g0_s0")
    finalizer._defer_routed_experts_to_policy = True
    with pytest.raises(ValueError, match="requires direct route assembly"):
        finalize(finalizer, [[owner]])
