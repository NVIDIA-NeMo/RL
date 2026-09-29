"""CPU coverage of capture -> TQ -> eager/deferred boundary-route replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture  # noqa: E402
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    RolloutReceipt,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient  # noqa: E402
from nemo_rl.data_plane.interfaces import KVBatchMeta  # noqa: E402
from nemo_rl.data_plane.schema import (  # noqa: E402
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    ROUTED_EXPERTS_BOUNDARY_FIELD,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.experience.rollout_reassembler import RolloutReassembler  # noqa: E402
from nemo_rl.experience.route_plan import decode_route_plan  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts  # noqa: E402
from tests.unit.data_plane.token_capture_test_fixtures import _manifest  # noqa: E402

pytestmark = pytest.mark.nemo_gym
_STAGING = "boundary_staging"
_CANONICAL = "boundary_canonical"
_ROLLOUT = "boundary_g0"


class _Client(NoOpDataPlaneClient):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[list[str], list[str]]] = []

    def get_samples(
        self, sample_ids: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        self.reads.append((list(sample_ids), list(select_fields)))
        return super().get_samples(sample_ids, partition_id, select_fields)


class _Sink(TQTokenSink):
    def __init__(self, client: _Client) -> None:
        super().__init__(client, staging_partition=_STAGING)
        self.records: list[StagedCallRecord] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.records.append(record)
        return super().stage(record)


class _Worker(TQWorkerMixin):
    def __init__(self, client: _Client) -> None:
        self._dp_client = client
        self._route_fallback_counts = Counter()

    def _routed_experts_dimensions(self) -> tuple[int, int]:
        return 1, 2


def _client() -> _Client:
    client = _Client()
    client.register_partition(
        _STAGING,
        list(STAGING_FIELDS) + [ROUTED_EXPERTS_FIELD, ROUTED_EXPERTS_BOUNDARY_FIELD],
        64,
        ["finalize", "prev_lp", "train"],
    )
    client.register_partition(_CANONICAL, [], 64, ["train"])
    return client


def _stage_turn(
    client: _Client,
    *,
    call_id: str,
    prefix: list[int],
    parent: StagedCallRecord | None,
    generated: list[int],
    version: int,
    sentinel_boundary: bool = False,
    legacy: bool = False,
    route_dtype: torch.dtype = torch.int16,
) -> tuple[StagedCallRecord, torch.Tensor, list[int]]:
    prompt = prefix + ([10, 11] if parent is None else [20 + version])
    tokens = prompt + generated
    full_routes = (
        torch.arange(len(tokens) * 2, dtype=route_dtype).reshape(-1, 1, 2)
        + version * 10
    )
    full_routes[-1] = torch.tensor([[0, 1]], dtype=torch.int16)
    if sentinel_boundary:
        full_routes[len(prefix) - 1] = -1
    payload = {
        "prompt_token_ids": prompt,
        "choices": [
            {
                "message": {
                    "generation_token_ids": generated,
                    "generation_log_probs": [-0.5] * len(generated),
                    "routed_experts": encode_routed_experts(full_routes),
                }
            }
        ],
    }
    VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
        payload,
        prev_len=len(prefix),
        prompt_len=len(prompt),
        generated_len=len(generated),
    )
    if legacy:
        message = payload["choices"][0]["message"]
        message.pop("routed_experts_boundary", None)
        message.pop("routed_experts_boundary_index", None)
    sink = _Sink(client)
    capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: version, adapter=VLLMCaptureAdapter()
    )
    admission = CaptureAdmission(
        rollout_id=_ROLLOUT,
        model_call_id=call_id,
        parent_call_id=parent.model_call_id if parent else None,
        prev_len=len(prefix),
        mode="token_in" if parent else "text",
        required_prefix_token_ids=prefix,
        parent_chain_hash=parent.chain_hash if parent else None,
    )
    coords = capture.complete_call_from_response(
        capture.begin_call(admission, weight_version=version), payload
    )
    assert coords.disposition == "staged"
    return sink.records[0], full_routes, tokens


def _receipt(
    records: list[StagedCallRecord], *, terminal: str | None = None
) -> dict[str, Any]:
    return RolloutReceipt(
        rollout_id=_ROLLOUT,
        terminal_model_call_id=terminal or records[-1].model_call_id,
        manifest=[_manifest(record) for record in records],
        terminal_selection="declared",
    ).model_dump()


def _finalizer(client: _Client, *, deferred: bool) -> RolloutReassembler:
    return RolloutReassembler(
        client,
        partition_id=_CANONICAL,
        staging_partition=_STAGING,
        pad_token_id=0,
        max_seq_len=128,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=deferred,
    )


def _publish_and_fetch(
    client: _Client, receipt: dict[str, Any], *, deferred: bool
) -> tuple[BatchedDataDict[Any], _Worker]:
    client.reads.clear()
    result = _finalizer(client, deferred=deferred).finalize_group(
        "boundary",
        [_ROLLOUT],
        [receipt],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=0,
        prompt_idx=0,
    )
    assert result.meta is not None
    if deferred:
        assert all(
            ROUTED_EXPERTS_FIELD not in fields
            and ROUTED_EXPERTS_BOUNDARY_FIELD not in fields
            for _, fields in client.reads
        )
        plan = decode_route_plan(result.meta.tags[0][ROUTE_PLAN_TAG])
        assert plan.expected_token_length == result.meta.sequence_lengths[0]
        meta = replace(result.meta, extra_info={ROUTE_PASSTHROUGH_FLAG: True})
    else:
        assert client.list_sample_ids(_STAGING) == []
        meta = result.meta
    worker = _Worker(client)
    data = worker._fetch(meta, dp_aligned_seq_len=False)
    return data, worker


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("turns", [1, 2, 3])
@pytest.mark.parametrize("sentinel_boundary", [False, True])
@pytest.mark.parametrize("route_dtype", [torch.int8, torch.int16])
def test_boundary_replay_matches_ray_splicing(
    deferred: bool, turns: int, sentinel_boundary: bool, route_dtype: torch.dtype
) -> None:
    client = _client()
    records = []
    prefix: list[int] = []
    expected = torch.empty((0, 1, 2), dtype=torch.int16)
    for turn in range(turns):
        record, full, tokens = _stage_turn(
            client,
            call_id=f"c{turn}",
            prefix=prefix,
            parent=records[-1] if records else None,
            generated=[30 + 2 * turn, 31 + 2 * turn],
            version=turn + 4,
            sentinel_boundary=sentinel_boundary and turn > 0,
            route_dtype=route_dtype,
        )
        if prefix:
            expected[-1] = full[len(prefix) - 1]
        expected = torch.cat((expected, full[len(prefix) :]))
        prefix = tokens
        records.append(record)

    data, worker = _publish_and_fetch(client, _receipt(records), deferred=deferred)
    assert torch.equal(data[ROUTED_EXPERTS_FIELD][0], expected)
    assert data[ROUTED_EXPERTS_FIELD].dtype == torch.int16
    assert data["input_ids"][0].tolist() == prefix
    assert data["token_mask"][0].tolist() == [
        mask for record in records for mask in record.token_mask_delta
    ]
    assert data["generation_logprobs"][0].tolist() == [
        value for record in records for value in record.generation_log_probs_delta
    ]
    assert data["sample_mask"].tolist() == [1.0]
    assert not worker._route_fallback_counts
    if deferred:
        sidecar_reads = [
            keys
            for keys, fields in client.reads
            if fields == [ROUTED_EXPERTS_BOUNDARY_FIELD]
        ]
        assert sidecar_reads == (
            [[record.staging_key for record in records[1:]]] if turns > 1 else []
        )


@pytest.mark.parametrize("deferred", [False, True])
def test_only_selected_retry_child_repairs_parent(deferred: bool) -> None:
    client = _client()
    root, root_routes, prefix = _stage_turn(
        client,
        call_id="root",
        prefix=[],
        parent=None,
        generated=[12, 13],
        version=1,
    )
    abandoned, _, _ = _stage_turn(
        client,
        call_id="abandoned",
        prefix=prefix,
        parent=root,
        generated=[14, 15],
        version=2,
    )
    selected, selected_routes, _ = _stage_turn(
        client,
        call_id="selected",
        prefix=prefix,
        parent=root,
        generated=[16, 17],
        version=3,
    )
    data, _ = _publish_and_fetch(
        client,
        _receipt([root, selected, abandoned], terminal="selected"),
        deferred=deferred,
    )
    expected = torch.cat((root_routes[:-1], selected_routes[len(prefix) - 1 :]))
    assert torch.equal(data[ROUTED_EXPERTS_FIELD][0], expected)
    if deferred:
        source = TQTokenSource(client, staging_partition=_STAGING)
        fragment = source.fetch_for_finalization(
            [root.staging_key], include_route_fragments=True
        )[0].fragment
        assert fragment is not None
        assert torch.equal(fragment.routes, root_routes)  # No parent mutation.
        assert abandoned.staging_key not in [
            key
            for keys, fields in client.reads
            if fields == [ROUTED_EXPERTS_BOUNDARY_FIELD]
            for key in keys
        ]


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("damage", ["tamper", "missing"])
def test_boundary_failure_preserves_mode_failure_policy(
    deferred: bool, damage: str
) -> None:
    client = _client()
    root, _, prefix = _stage_turn(
        client,
        call_id="root",
        prefix=[],
        parent=None,
        generated=[12, 13],
        version=1,
    )
    child, _, _ = _stage_turn(
        client,
        call_id="child",
        prefix=prefix,
        parent=root,
        generated=[14, 15],
        version=2,
    )
    if damage == "tamper":
        client.put_samples(
            [child.staging_key],
            _STAGING,
            TensorDict(
                {
                    ROUTED_EXPERTS_BOUNDARY_FIELD: torch.full(
                        (1, 1, 1, 2), 999, dtype=torch.int16
                    )
                },
                batch_size=[1],
            ),
        )
    else:
        # Simulate loss of only the optional column, not the committed token row.
        client._partitions[_STAGING].rows[child.staging_key].pop(
            ROUTED_EXPERTS_BOUNDARY_FIELD
        )
    receipt = _receipt([root, child])
    if deferred:
        data, worker = _publish_and_fetch(client, receipt, deferred=True)
        assert data[ROUTED_EXPERTS_FIELD].eq(-1).all()
        assert data["sample_mask"].tolist() == [1.0]
        assert sum(worker._route_fallback_counts.values()) == 1
    else:
        row = _finalizer(client, deferred=False).finalize_rollout(
            _ROLLOUT, receipt, reward=1.0
        )
        assert not row.valid
        assert row.rejection_reason is not None


def test_boundary_sidecar_survives_data_plane_checkpoint(tmp_path: Path) -> None:
    client = _client()
    root, root_routes, prefix = _stage_turn(
        client,
        call_id="root",
        prefix=[],
        parent=None,
        generated=[12, 13],
        version=1,
    )
    child, full, _ = _stage_turn(
        client,
        call_id="child",
        prefix=prefix,
        parent=root,
        generated=[14, 15],
        version=2,
    )
    finalized = _finalizer(client, deferred=True).finalize_group(
        "boundary",
        [_ROLLOUT],
        [_receipt([root, child])],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=0,
        prompt_idx=0,
    )
    assert finalized.meta is not None
    client.save_checkpoint(
        tmp_path / "data_plane", metadata={"meta": asdict(finalized.meta)}
    )
    restored = _Client()
    metadata = restored.load_checkpoint(tmp_path / "data_plane")
    restored_meta = KVBatchMeta(**metadata["meta"])
    data = _Worker(restored)._fetch(
        replace(restored_meta, extra_info={ROUTE_PASSTHROUGH_FLAG: True}),
        dp_aligned_seq_len=False,
    )
    assert torch.equal(
        data[ROUTED_EXPERTS_FIELD][0],
        torch.cat((root_routes[:-1], full[len(prefix) - 1 :])),
    )


@pytest.mark.parametrize("deferred", [False, True])
def test_legacy_capture_is_readable_but_not_silently_repaired(
    deferred: bool, caplog
) -> None:
    client = _client()
    root, root_routes, prefix = _stage_turn(
        client,
        call_id="root",
        prefix=[],
        parent=None,
        generated=[12, 13],
        version=1,
    )
    child, full, _ = _stage_turn(
        client,
        call_id="legacy",
        prefix=prefix,
        parent=root,
        generated=[14, 15],
        version=2,
        legacy=True,
    )
    data, _ = _publish_and_fetch(client, _receipt([root, child]), deferred=deferred)
    assert torch.equal(
        data[ROUTED_EXPERTS_FIELD][0], torch.cat((root_routes, full[len(prefix) :]))
    )
    assert "Legacy captured call legacy has no boundary route" in caplog.text


@pytest.mark.parametrize("deferred", [False, True])
def test_empty_generation_links_keep_existing_lineage_rejection(deferred: bool) -> None:
    client = _client()
    root, _, prefix = _stage_turn(
        client,
        call_id="root",
        prefix=[],
        parent=None,
        generated=[12, 13],
        version=1,
    )
    empty, _, empty_prefix = _stage_turn(
        client,
        call_id="empty",
        prefix=prefix,
        parent=root,
        generated=[],
        version=2,
    )
    child, _, _ = _stage_turn(
        client,
        call_id="child",
        prefix=empty_prefix,
        parent=empty,
        generated=[14, 15],
        version=3,
    )
    assert "routed_experts_boundary" not in empty.extras
    row = _finalizer(client, deferred=deferred).finalize_rollout(
        _ROLLOUT, _receipt([root, empty, child]), reward=1.0
    )
    assert not row.valid
    assert row.rejection_reason is not None
    assert "empty_generation" in row.rejection_reason
    assert row.route_plan is None
