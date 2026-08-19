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

"""S4: BlackboxFinalizer against a live TQ simple backend.

Drives the S1 golden call sequences end to end: stage the fixture's delta
rows via TQTokenSink, hand the fixture receipt to the finalizer, and require
the published canonical rows to match the fixture's frozen training row.
Every rejection path (missing rows, digest corruption, poisoned receipts)
must yield a masked placeholder — always N rows — and the group publisher's
min/max weight versions and staging cleanup must hold.

Marked nemo_gym (run with ``--nemo-gym-only``): the finalizer delegates
rebuild semantics to Gym's staging package.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    StagedCallRecord,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
)
from nemo_rl.data_plane.schema import (  # noqa: E402
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin  # noqa: E402
from nemo_rl.experience.blackbox_finalizer import BlackboxFinalizer  # noqa: E402
from nemo_rl.experience.route_plan import decode_route_plan  # noqa: E402
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
    f32,
)

pytestmark = pytest.mark.nemo_gym

STAGING_PARTITION = "rollout_staging_fin_test"
CANONICAL_PARTITION = "rollout_data_fin_test"
PAD = 0


@pytest.fixture()
def partitions(tq_client):
    tq_client.register_partition(
        partition_id=STAGING_PARTITION,
        fields=list(STAGING_FIELDS),
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    tq_client.register_partition(
        partition_id=CANONICAL_PARTITION,
        fields=[
            "input_ids",
            "input_lengths",
            "generation_logprobs",
            "token_mask",
            "sample_mask",
            "prompt_ids_for_adv",
            "total_reward",
        ],
        num_samples=64,
        consumer_tasks=["train"],
    )
    yield
    tq_client.clear_samples(sample_ids=None, partition_id=STAGING_PARTITION)
    tq_client.clear_samples(sample_ids=None, partition_id=CANONICAL_PARTITION)


def _finalizer(tq_client, **overrides) -> BlackboxFinalizer:
    kwargs = dict(
        partition_id=CANONICAL_PARTITION,
        staging_partition=STAGING_PARTITION,
        pad_token_id=PAD,
        mixed_weight_version_policy="allow",
        min_valid_fraction_per_group=None,
    )
    kwargs.update(overrides)
    return BlackboxFinalizer(tq_client, **kwargs)


def _stage_fixture(tq_client, name: str, *, rollout_id: str | None = None):
    """Stage one golden fixture's rows (optionally re-keyed to rollout_id)
    and return (receipt_dict, expected LinearizedRow)."""
    records, receipt, row = build_fixture_artifacts(name, rollout_id=rollout_id)
    sink = TQTokenSink(tq_client, staging_partition=STAGING_PARTITION)
    for record in records:
        assert sink.stage(record).ok
    return receipt.model_dump(), row


def test_finalize_rollout_reproduces_the_golden_row(tq_client, partitions):
    receipt, expected = _stage_fixture(tq_client, "worked_example")
    finalizer = _finalizer(tq_client)
    row = finalizer.finalize_rollout("g7_r0", receipt, reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.token_ids == expected.token_ids
    assert row.token_mask == [f32(m) for m in expected.token_mask]
    assert row.logprobs == [f32(p) for p in expected.logprobs]
    assert row.prompt_len == expected.prompt_len
    # The worked example spans a single weight version (wv 4 throughout).
    assert (row.min_wv, row.max_wv) == (4, 4)


def test_finalize_rollout_rejections(tq_client, partitions):
    finalizer = _finalizer(tq_client)
    assert (
        finalizer.finalize_rollout("r", None, reward=0.0).rejection_reason
        == "missing_receipt"
    )

    receipt, _ = _stage_fixture(tq_client, "single_call", rollout_id="rej_a")
    poisoned = dict(receipt, capture_poisoned=True)
    assert (
        finalizer.finalize_rollout("rej_a", poisoned, reward=0.0).rejection_reason
        == "capture_poisoned"
    )
    empty = dict(receipt, manifest=[], terminal_model_call_id=None)
    assert (
        finalizer.finalize_rollout("rej_a", empty, reward=0.0).rejection_reason
        == "empty_manifest"
    )
    wrong_identity = finalizer.finalize_rollout("someone_else", receipt, reward=0.0)
    assert (wrong_identity.rejection_reason or "").startswith("identity_mismatch")

    # A manifest naming rows that were never staged.
    ghost = dict(receipt)
    ghost["manifest"] = [
        {**entry, "staging_key": "ghost/row"} for entry in receipt["manifest"]
    ]
    missing = finalizer.finalize_rollout("rej_a", ghost, reward=0.0)
    assert (missing.rejection_reason or "").startswith("missing_staging_row")

    # Digest corruption: break the manifest digest so recomputation misses.
    corrupted = dict(receipt)
    corrupted["manifest"] = [
        {**entry, "digest": "0" * 64} for entry in receipt["manifest"]
    ]
    bad = finalizer.finalize_rollout("rej_a", corrupted, reward=0.0)
    assert (bad.rejection_reason or "").startswith("digest_mismatch")


def test_mixed_weight_version_policy_reject(tq_client, partitions):
    receipt, _ = _stage_fixture(tq_client, "mixed_weight_versions", rollout_id="mix_r0")
    receipt["rollout_id"] = "mix_r0"
    allow_row = _finalizer(tq_client).finalize_rollout("mix_r0", receipt, reward=0.0)
    assert allow_row.valid
    assert allow_row.min_wv < allow_row.max_wv
    reject_row = _finalizer(
        tq_client, mixed_weight_version_policy="reject"
    ).finalize_rollout("mix_r0", receipt, reward=0.0)
    assert (reject_row.rejection_reason or "").startswith("mixed_weight_versions")


def _fetch_rows(tq_client, sample_ids):
    return tq_client.get_samples(
        sample_ids=sample_ids,
        partition_id=CANONICAL_PARTITION,
        select_fields=[
            "input_ids",
            "input_lengths",
            "generation_logprobs",
            "token_mask",
            "sample_mask",
            "prompt_ids_for_adv",
            "total_reward",
        ],
    )


def test_finalize_group_publishes_n_rows_with_placeholder(tq_client, partitions):
    group_id = "grp1"
    receipt, expected = _stage_fixture(
        tq_client, "worked_example", rollout_id=f"{group_id}_g0"
    )
    receipt["rollout_id"] = f"{group_id}_g0"
    # Mark this receipt's terminal as heuristically selected so the group
    # metric sees a mixed declared/heuristic population.
    receipt["terminal_selection"] = "heuristic"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]

    finalizer = _finalizer(tq_client)
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],  # second rollout lost its receipt -> placeholder
        [1.0, 0.0],
        fallback_weight_version=9,
    )
    assert not finalized.dropped
    assert finalized.meta is not None
    assert finalized.meta.sample_ids == rollout_ids
    # Group staleness comes from the valid rollout's calls (wv 4), not the fallback.
    assert (finalized.group_min_wv, finalized.group_max_wv) == (4, 4)
    assert finalized.metrics["finalize/invalid_row_rate"] == 0.5
    assert finalized.metrics["finalize/terminal_selection_heuristic_count"] == 1.0
    assert finalized.metrics["finalize/terminal_selection_heuristic_fraction"] == 0.5
    assert finalized.metrics["finalize/terminal_selection_declared_count"] == 0.0
    assert finalized.metrics["finalize/terminal_witness_disagreement_count"] == 0.0

    rows = _fetch_rows(tq_client, rollout_ids)
    sample_mask = torch.as_tensor(rows["sample_mask"]).flatten()
    assert sample_mask.tolist() == [1.0, 0.0]
    valid_len = len(expected.token_ids)
    input_ids = torch.as_tensor(rows["input_ids"][0]).flatten()
    assert input_ids[:valid_len].tolist() == expected.token_ids
    # Placeholder borrows the valid sibling's prompt for baseline grouping.
    prompt = expected.token_ids[: expected.prompt_len]
    adv_prompt_valid = torch.as_tensor(rows["prompt_ids_for_adv"][0]).flatten()
    adv_prompt_placeholder = torch.as_tensor(rows["prompt_ids_for_adv"][1]).flatten()
    assert adv_prompt_valid.tolist() == prompt
    assert adv_prompt_placeholder.tolist() == prompt
    placeholder_mask = torch.as_tensor(rows["token_mask"][1]).flatten()
    assert placeholder_mask.sum().item() == 0.0
    rewards = torch.as_tensor(rows["total_reward"]).flatten()
    assert rewards.tolist() == [1.0, 0.0]

    # The finalizer cleared its staged rows after publishing.
    with pytest.raises(KeyError):
        finalizer._source.fetch([receipt["manifest"][0]["staging_key"]])


def test_finalize_group_maps_physical_attempt_to_stable_canonical_id(
    tq_client, partitions
):
    group_id = "stable"
    physical_id = f"{group_id}_g0_aattempt"
    canonical_id = f"{group_id}_g0"
    receipt, _ = _stage_fixture(
        tq_client,
        "worked_example",
        rollout_id=physical_id,
    )
    receipt["rollout_id"] = physical_id

    finalized = _finalizer(tq_client).finalize_group(
        group_id,
        [physical_id],
        [receipt],
        [1.0],
        fallback_weight_version=4,
        canonical_sample_ids=[canonical_id],
    )

    assert finalized.meta is not None
    assert finalized.meta.sample_ids == [canonical_id]
    assert _fetch_rows(tq_client, [canonical_id])["input_ids"] is not None


def test_finalize_group_min_valid_fraction_drops(tq_client, partitions):
    group_id = "grp2"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    finalizer = _finalizer(tq_client, min_valid_fraction_per_group=0.5)
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [None, None],
        [0.0, 0.0],
        fallback_weight_version=3,
    )
    assert finalized.dropped
    assert finalized.meta is None
    assert (finalized.group_min_wv, finalized.group_max_wv) == (3, 3)
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        rows = _fetch_rows(tq_client, rollout_ids)
        assert not rows  # nothing published


# ---------------------------------------------------------------------------
# Router replay (R3): routed_experts rebuilt from staged extras and published
# ---------------------------------------------------------------------------

_R3_PARTITION = "rollout_data_fin_r3_test"
_R3_STAGING = "rollout_staging_fin_r3_test"


@pytest.fixture()
def r3_partitions(tq_client):
    from nemo_rl.data_plane.tq_token_sink import ROUTED_EXPERTS_FIELD

    tq_client.register_partition(
        partition_id=_R3_STAGING,
        fields=list(STAGING_FIELDS) + [ROUTED_EXPERTS_FIELD],
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    tq_client.register_partition(
        partition_id=_R3_PARTITION,
        fields=[
            "input_ids",
            "input_lengths",
            "generation_logprobs",
            "token_mask",
            "sample_mask",
            "prompt_ids_for_adv",
            "total_reward",
            "routed_experts",
        ],
        num_samples=64,
        consumer_tasks=["train"],
    )
    yield
    tq_client.clear_samples(sample_ids=None, partition_id=_R3_STAGING)
    tq_client.clear_samples(sample_ids=None, partition_id=_R3_PARTITION)


def _routes_for_delta(call_idx: int, n_tokens: int) -> list:
    """[n][L=2][K=2] rows, value = call*1000 + pos (recognizable per token)."""
    return [
        [[call_idx * 1000 + pos, call_idx * 1000 + pos + 500] for _ in range(2)]
        for pos in range(n_tokens)
    ]


def _record_with_routes(record: StagedCallRecord, routes: list) -> StagedCallRecord:
    extras = {"routed_experts": routes}
    extras_digest = compute_extras_digest(extras)
    digest = compute_staging_digest(
        schema_version=record.schema_version,
        digest_version=record.digest_version,
        extras_digest_version=record.extras_digest_version,
        rollout_id=record.rollout_id,
        model_call_id=record.model_call_id,
        parent_call_id=record.parent_call_id,
        mode=record.mode,
        prev_len=record.prev_len,
        delta_len=record.delta_len,
        cum_len=record.cum_len,
        weight_version=record.weight_version,
        token_ids_delta=record.token_ids_delta,
        token_mask_delta=record.token_mask_delta,
        generation_log_probs_delta=record.generation_log_probs_delta,
        extras_digest=extras_digest,
        chain_hash=record.chain_hash,
        cumulative_hash=record.cumulative_hash,
    )
    return StagedCallRecord.model_validate(
        record.model_dump()
        | {"extras": extras, "extras_digest": extras_digest, "digest": digest}
    )


def _receipt_with_staged_records(receipt, records):
    manifest_by_id = {record.model_call_id: record for record in receipt.manifest}
    return receipt.model_copy(
        update={
            "manifest": [
                manifest_by_id[record.model_call_id].model_copy(
                    update={
                        "digest": record.digest,
                        "extras_digest": record.extras_digest,
                    }
                )
                for record in records
            ]
        }
    )


def _stage_fixture_with_routes(tq_client, name: str, *, rollout_id: str):
    """Stage the golden fixture with per-call routed_experts extras attached.

    Returns (receipt_dict, expected LinearizedRow, routes_by_call).
    """
    records, receipt, row = build_fixture_artifacts(name, rollout_id=rollout_id)
    sink = TQTokenSink(tq_client, staging_partition=_R3_STAGING)
    routes_by_call = {}
    staged_records = []
    for idx, record in enumerate(records):
        routes = _routes_for_delta(idx, len(record.token_ids_delta))
        routes_by_call[record.model_call_id] = routes
        staged = _record_with_routes(record, routes)
        staged_records.append(staged)
        assert sink.stage(staged).ok
    receipt = _receipt_with_staged_records(receipt, staged_records)
    return receipt.model_dump(), row, routes_by_call


def _gym_linearize_supports_routes() -> bool:
    from nemo_gym.token_id_capture.staging.rebuild import LinearizedRow

    return "routed_experts" in getattr(LinearizedRow, "__dataclass_fields__", {})


@pytest.mark.skipif(
    not _gym_linearize_supports_routes(),
    reason="Gym pin predates LinearizedRow.routed_experts (Gym PR #2278 R3 follow-up)",
)
def test_finalize_group_publishes_routed_experts(tq_client, r3_partitions):
    group_id = "grpr3"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected, routes_by_call = _stage_fixture_with_routes(
        tq_client, "worked_example", rollout_id=rollout_ids[0]
    )

    finalizer = BlackboxFinalizer(
        tq_client,
        partition_id=_R3_PARTITION,
        staging_partition=_R3_STAGING,
        pad_token_id=PAD,
        mixed_weight_version_policy="allow",
        min_valid_fraction_per_group=None,
        router_replay_enabled=True,
    )
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],  # second rollout -> placeholder
        [1.0, 0.0],
        fallback_weight_version=9,
    )
    assert not finalized.dropped
    assert "routed_experts" in finalized.meta.fields
    assert finalized.metrics["finalize/routed_experts_row_coverage"] == 1.0
    assert finalized.metrics["finalize/routed_experts_sentinel_token_fraction"] == 0.0

    rows = tq_client.get_samples(
        sample_ids=rollout_ids,
        partition_id=_R3_PARTITION,
        select_fields=["routed_experts", "input_lengths"],
    )
    # Valid row: the delivered chain's staged extras, concatenated in chain
    # order (the golden fixture is a single linear chain).
    expected_routes = [
        row_routes
        for call_id in expected.call_ids
        for row_routes in routes_by_call[call_id]
    ]
    valid_len = len(expected.token_ids)
    assert len(expected_routes) == valid_len
    published = torch.as_tensor(rows["routed_experts"][0]).reshape(-1, 2, 2)
    assert published[:valid_len].tolist() == expected_routes
    # Placeholder row: all-sentinel (Megatron self-routes; sample_mask 0).
    placeholder = torch.as_tensor(rows["routed_experts"][1])
    assert bool(placeholder.eq(-1).all().item())


@pytest.mark.skipif(
    not _gym_linearize_supports_routes(),
    reason="Gym pin predates LinearizedRow.routed_experts (Gym PR #2278 R3 follow-up)",
)
def test_finalize_group_router_replay_without_routes_fails_loudly(
    tq_client, r3_partitions
):
    group_id = "grpr3b"
    rollout_id = f"{group_id}_g0"
    records, receipt, _ = build_fixture_artifacts(
        "worked_example", rollout_id=rollout_id
    )
    sink = TQTokenSink(tq_client, staging_partition=_R3_STAGING)
    for record in records:
        assert sink.stage(record).ok  # no extras staged

    finalizer = BlackboxFinalizer(
        tq_client,
        partition_id=_R3_PARTITION,
        staging_partition=_R3_STAGING,
        pad_token_id=PAD,
        mixed_weight_version_policy="allow",
        min_valid_fraction_per_group=None,
        router_replay_enabled=True,
    )
    with pytest.raises(RuntimeError, match="routed_experts"):
        finalizer.finalize_group(
            group_id,
            [rollout_id],
            [receipt.model_dump()],
            [1.0],
            fallback_weight_version=9,
        )


# ---------------------------------------------------------------------------
# Deferred router replay: canonical small rows + strict plans, worker assembly
# ---------------------------------------------------------------------------

_R3_DEFERRED_PARTITION = "rollout_data_fin_r3_deferred_test"
_R3_DEFERRED_STAGING = "rollout_staging_fin_r3_deferred_test"


@pytest.fixture()
def r3_deferred_partitions(tq_client):
    from nemo_rl.data_plane.tq_token_sink import ROUTED_EXPERTS_FIELD

    tq_client.register_partition(
        partition_id=_R3_DEFERRED_STAGING,
        fields=list(STAGING_FIELDS) + [ROUTED_EXPERTS_FIELD],
        num_samples=64,
        consumer_tasks=["finalize", "prev_lp", "train"],
    )
    tq_client.register_partition(
        partition_id=_R3_DEFERRED_PARTITION,
        fields=[
            "input_ids",
            "input_lengths",
            "generation_logprobs",
            "token_mask",
            "sample_mask",
            "prompt_ids_for_adv",
            "total_reward",
        ],
        num_samples=64,
        consumer_tasks=["train"],
    )
    yield
    tq_client.clear_samples(sample_ids=None, partition_id=_R3_DEFERRED_STAGING)
    tq_client.clear_samples(sample_ids=None, partition_id=_R3_DEFERRED_PARTITION)


def _stage_deferred_fixture(tq_client, *, rollout_id: str):
    records, receipt, row = build_fixture_artifacts(
        "worked_example", rollout_id=rollout_id
    )
    sink = TQTokenSink(tq_client, staging_partition=_R3_DEFERRED_STAGING)
    routes_by_call = {}
    staged_records = []
    for idx, record in enumerate(records):
        routes = _routes_for_delta(idx, len(record.token_ids_delta))
        routes_by_call[record.model_call_id] = routes
        staged = _record_with_routes(record, routes)
        staged_records.append(staged)
        assert sink.stage(staged).ok
    receipt = _receipt_with_staged_records(receipt, staged_records)
    return receipt.model_dump(), row, routes_by_call


class _DeferredRouteWorker(TQWorkerMixin):
    def __init__(self, client):
        self._dp_client = client
        self._route_fallback_counts = Counter()

    def _routed_experts_dimensions(self) -> tuple[int, int]:
        return 2, 2


def test_deferred_finalizer_publishes_plans_and_worker_replays_routes(
    tq_client, r3_deferred_partitions
):
    group_id = "grpr3deferred"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected, routes_by_call = _stage_deferred_fixture(
        tq_client, rollout_id=rollout_ids[0]
    )
    finalizer = BlackboxFinalizer(
        tq_client,
        partition_id=_R3_DEFERRED_PARTITION,
        staging_partition=_R3_DEFERRED_STAGING,
        pad_token_id=PAD,
        mixed_weight_version_policy="allow",
        min_valid_fraction_per_group=None,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=True,
    )

    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        fallback_weight_version=9,
    )

    assert finalized.meta is not None
    assert "routed_experts" not in finalized.meta.fields
    assert len(finalized.staging_keys) == len(receipt["manifest"])
    plans = [decode_route_plan(tag[ROUTE_PLAN_TAG]) for tag in finalized.meta.tags]
    assert plans[0].expected_token_length == len(expected.token_ids)
    assert set(plans[0].cleanup_staging_keys) == set(finalized.staging_keys)
    assert not plans[1].spans
    # Deferred finalization deliberately retains staging through consumption.
    source = TQTokenSource(tq_client, staging_partition=_R3_DEFERRED_STAGING)
    assert len(source.fetch_for_finalization(finalized.staging_keys)) == len(
        finalized.staging_keys
    )

    worker_meta = replace(
        finalized.meta,
        extra_info={ROUTE_PASSTHROUGH_FLAG: True},
        task_name="train",
    )
    materialized = _DeferredRouteWorker(tq_client)._fetch(
        worker_meta,
        dp_aligned_seq_len=False,
    )
    expected_routes = [
        route for call_id in expected.call_ids for route in routes_by_call[call_id]
    ]
    valid_len = len(expected.token_ids)
    assert materialized["routed_experts"][0, :valid_len].tolist() == expected_routes
    assert bool(materialized["routed_experts"][1].eq(-1).all())


@pytest.mark.parametrize("bad_routed_len", [-1, 999])
def test_deferred_finalizer_rejects_invalid_routed_len(
    tq_client, r3_deferred_partitions, bad_routed_len
):
    from dataclasses import replace as dataclass_replace

    rollout_id = "bad_route_len_g0"
    receipt, _, _ = _stage_deferred_fixture(tq_client, rollout_id=rollout_id)
    finalizer = BlackboxFinalizer(
        tq_client,
        partition_id=_R3_DEFERRED_PARTITION,
        staging_partition=_R3_DEFERRED_STAGING,
        pad_token_id=PAD,
        mixed_weight_version_policy="allow",
        min_valid_fraction_per_group=None,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=True,
    )
    fetched = finalizer._source.fetch_for_finalization(
        [record["staging_key"] for record in receipt["manifest"]]
    )
    fetched[0] = dataclass_replace(fetched[0], routed_len=bad_routed_len)

    class _InjectedSource:
        def fetch_for_finalization(self, staging_keys):
            del staging_keys
            return fetched

    finalizer._source = _InjectedSource()
    row = finalizer.finalize_rollout(rollout_id, receipt, reward=0.0)

    assert not row.valid
    assert (row.rejection_reason or "").startswith("routed_len_mismatch")
