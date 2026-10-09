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

"""S4: RolloutReassembler against a live TQ simple backend.

Drives the S1 golden call sequences end to end: stage the fixture's delta
rows via TQTokenSink, hand the fixture receipt to the finalizer, and require
the published canonical rows to match the fixture's frozen training row.
Every rejection path (missing rows, digest corruption, poisoned receipts)
must yield a masked placeholder — always N canonical rows — and the group
publisher's min/max weight versions and staging cleanup must hold. The
segment-row section at the bottom drives the multi-chain publish path
(token_capture.segment_rows) with a fake Gym linearizer.

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

from nemo_rl.data_plane.schema import (  # noqa: E402
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
)
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin  # noqa: E402
from nemo_rl.experience.legacy_rollout_metrics import (  # noqa: E402
    ROLLOUT_DEBUG_TAG,
    decode_rollout_debug_tag,
)
from nemo_rl.experience.rollout_reassembler import RolloutReassembler  # noqa: E402
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
            "mask_sample",
            "truncated",
        ],
        num_samples=64,
        consumer_tasks=["train"],
    )
    yield
    tq_client.clear_samples(sample_ids=None, partition_id=STAGING_PARTITION)
    tq_client.clear_samples(sample_ids=None, partition_id=CANONICAL_PARTITION)


def _finalizer(tq_client, **overrides) -> RolloutReassembler:
    kwargs = dict(
        partition_id=CANONICAL_PARTITION,
        staging_partition=STAGING_PARTITION,
        pad_token_id=PAD,
        # Well above every fixture's real row length (a handful of tokens),
        # so truncated defaults to False everywhere unless a test overrides
        # this to exercise the truncation-detection path deliberately.
        max_seq_len=4096,
    )
    kwargs.update(overrides)
    return RolloutReassembler(tq_client, **kwargs)


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
    rows = finalizer.finalize_rollout("g7_r0", receipt, reward=1.0)
    # Without segment_rows a rollout is exactly one row.
    assert len(rows) == 1
    row = rows[0]
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
        finalizer.finalize_rollout("r", None, reward=0.0)[0].rejection_reason
        == "missing_receipt"
    )

    receipt, _ = _stage_fixture(tq_client, "single_call", rollout_id="rej_a")
    poisoned = dict(receipt, capture_poisoned=True)
    assert (
        finalizer.finalize_rollout("rej_a", poisoned, reward=0.0)[0].rejection_reason
        == "capture_poisoned"
    )
    # Gym (since #2823) rejects an unpoisoned receipt with no terminal call, so
    # an empty manifest only reaches the finalizer poisoned, the way
    # ``nemo_gym.py`` emits it (``failure_reason="missing_terminal_row"``).
    unpoisoned_empty = dict(receipt, manifest=[], terminal_model_call_id=None)
    assert (
        finalizer.finalize_rollout(
            "rej_a", unpoisoned_empty, reward=0.0
        ).rejection_reason
        or ""
    ).startswith("invalid_receipt:")
    empty = dict(
        receipt,
        manifest=[],
        terminal_model_call_id=None,
        terminal_selection=None,
        capture_poisoned=True,
        failure_reason="missing_terminal_row",
    )
    assert (
        finalizer.finalize_rollout("rej_a", empty, reward=0.0)[0].rejection_reason
        == "rollout_failed:missing_terminal_row"
    )
    wrong_identity = finalizer.finalize_rollout("someone_else", receipt, reward=0.0)[0]
    assert (wrong_identity.rejection_reason or "").startswith("identity_mismatch")

    # A manifest naming rows that were never staged.
    ghost = dict(receipt)
    ghost["manifest"] = [
        {**entry, "staging_key": "ghost/row"} for entry in receipt["manifest"]
    ]
    missing = finalizer.finalize_rollout("rej_a", ghost, reward=0.0)[0]
    assert (missing.rejection_reason or "").startswith("missing_staging_row")

    # Digest corruption: break the manifest digest. Gym's verifier owns the
    # comparison, so the rejection surfaces through rebuild_failed.
    corrupted = dict(receipt)
    corrupted["manifest"] = [
        {**entry, "digest": "0" * 64} for entry in receipt["manifest"]
    ]
    bad = finalizer.finalize_rollout("rej_a", corrupted, reward=0.0)[0]
    assert (bad.rejection_reason or "").startswith("rebuild_failed:wrong_digest")


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
            "mask_sample",
            "truncated",
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

    # max_seq_len pinned to the valid row's real length so the finalizer's
    # own truncation computation (seq_len == max_seq_len) has something real
    # to detect: the valid row should read truncated, the placeholder
    # (seq_len floored to 1) should not.
    valid_len = len(expected.token_ids)
    finalizer = _finalizer(tq_client, max_seq_len=valid_len)
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],  # second rollout lost its receipt -> placeholder
        [1.0, 0.0],
        mask_sample=[True, False],
        fallback_weight_version=9,
        prompt_idx=17,
        loss_multiplier=0.25,
    )
    assert not finalized.dropped
    assert finalized.meta is not None
    assert finalized.meta.sample_ids == rollout_ids
    assert [tag["prompt_idx"] for tag in finalized.meta.tags] == [17, 17]
    # Group staleness comes from the valid rollout's calls (wv 4), not the fallback.
    assert (finalized.group_min_wv, finalized.group_max_wv) == (4, 4)
    assert finalized.metrics["finalize/invalid_row_rate"] == 0.5
    # Text-only rollouts never carry media.
    assert finalized.metrics["finalize/media_row_rate"] == 0.0
    assert finalized.metrics["finalize/terminal_selection_heuristic_count"] == 1.0
    assert finalized.metrics["finalize/terminal_selection_heuristic_fraction"] == 0.5
    assert finalized.metrics["finalize/terminal_selection_declared_count"] == 0.0
    assert finalized.metrics["finalize/terminal_witness_disagreement_count"] == 0.0
    assert finalized.canonical_output_tokens == sum(expected.token_mask)

    rows = _fetch_rows(tq_client, rollout_ids)
    sample_mask = torch.as_tensor(rows["sample_mask"]).flatten()
    assert sample_mask.tolist() == [0.25, 0.0]
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
    # mask_sample rides along unchanged from the dispatcher; truncated is
    # computed here from each row's rebuilt length against max_seq_len (the
    # valid row's real length, pinned above) -- the placeholder's length is
    # floored to 1 and never matches, so only the valid row reads truncated.
    assert torch.as_tensor(rows["mask_sample"]).flatten().tolist() == [True, False]
    assert torch.as_tensor(rows["truncated"]).flatten().tolist() == [True, False]

    # The finalizer cleared its staged rows after publishing.
    with pytest.raises(KeyError):
        finalizer._source.fetch([receipt["manifest"][0]["staging_key"]])


def test_finalize_group_skips_unset_terminal_selection(tq_client, partitions):
    group_id = "grp_unset"
    receipt, expected = _stage_fixture(
        tq_client, "worked_example", rollout_id=f"{group_id}_g0"
    )
    receipt["rollout_id"] = f"{group_id}_g0"
    assert receipt["terminal_selection"] == "declared"
    # A manifest that never parsed ran no attribution stage, so the receipt
    # carries terminal_selection=None (Gym #2823, pinned 9fc05c0f) rather than a method.
    unset = {
        "rollout_id": f"{group_id}_g1",
        "reward": 0.0,
        "terminal_model_call_id": None,
        "manifest": [],
        "capture_poisoned": True,
        "failure_reason": "invalid_manifest_row",
        "terminal_selection": None,
        "terminal_attribution_reason": None,
    }
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]

    finalizer = _finalizer(tq_client, max_seq_len=len(expected.token_ids))
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, unset],
        [1.0, 0.0],
        mask_sample=[True, False],
        fallback_weight_version=9,
        prompt_idx=3,
        loss_multiplier=1.0,
    )
    assert not finalized.dropped
    metrics = finalized.metrics
    # The unset receipt still counts toward the group-wide totals ...
    assert metrics["finalize/invalid_row_rate"] == 0.5
    assert metrics["finalize/capture_poisoned_rollouts"] == 1.0
    assert metrics["finalize/capture_failure_reason_rollout_failed_count"] == 1.0
    assert metrics["finalize/terminal_selection_declared_count"] == 1.0
    assert metrics["finalize/terminal_selection_declared_fraction"] == 0.5
    assert metrics["finalize/terminal_selection_heuristic_count"] == 0.0
    # ... but lands in no per-method bucket: None is not a method, so the
    # buckets sum to the attributed receipts only and no None key is emitted.
    assert not [key for key in metrics if "terminal_selection_None" in key]
    bucket_counts = [
        value
        for key, value in metrics.items()
        if key.startswith("finalize/terminal_selection_") and key.endswith("_count")
    ]
    assert sum(bucket_counts) == 1.0


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
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
        canonical_sample_ids=[canonical_id],
    )

    assert finalized.meta is not None
    assert finalized.meta.sample_ids == [canonical_id]
    assert _fetch_rows(tq_client, [canonical_id])["input_ids"] is not None


def test_finalize_group_reports_valid_and_total_row_counts(tq_client, partitions):
    """The finalizer reports validity; the controller owns replacement policy."""
    group_id = "grp2"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    finalizer = _finalizer(tq_client)
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [None, None],
        [0.0, 0.0],
        mask_sample=[False] * 2,
        fallback_weight_version=3,
        prompt_idx=17,
    )
    assert not finalized.dropped
    assert finalized.meta is not None
    assert finalized.valid_row_count == 0
    assert finalized.total_row_count == 2
    assert (finalized.group_min_wv, finalized.group_max_wv) == (3, 3)
    rows = _fetch_rows(tq_client, rollout_ids)
    sample_mask = torch.as_tensor(rows["sample_mask"]).flatten()
    assert sample_mask.tolist() == [0.0, 0.0]  # published as placeholders, not dropped


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
            "mask_sample",
            "truncated",
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


def test_finalize_group_publishes_routed_experts(tq_client, r3_partitions):
    group_id = "grpr3"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected, routes_by_call = _stage_fixture_with_routes(
        tq_client, "worked_example", rollout_id=rollout_ids[0]
    )

    finalizer = RolloutReassembler(
        tq_client,
        partition_id=_R3_PARTITION,
        staging_partition=_R3_STAGING,
        pad_token_id=PAD,
        max_seq_len=4096,
        router_replay_enabled=True,
    )
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],  # second rollout -> placeholder
        [1.0, 0.0],
        mask_sample=[False] * 2,
        fallback_weight_version=9,
        prompt_idx=17,
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

    finalizer = RolloutReassembler(
        tq_client,
        partition_id=_R3_PARTITION,
        staging_partition=_R3_STAGING,
        pad_token_id=PAD,
        max_seq_len=4096,
        router_replay_enabled=True,
    )
    with pytest.raises(RuntimeError, match="routed_experts"):
        finalizer.finalize_group(
            group_id,
            [rollout_id],
            [receipt.model_dump()],
            [1.0],
            mask_sample=[False],
            fallback_weight_version=9,
            prompt_idx=17,
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
            "mask_sample",
            "truncated",
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
    finalizer = RolloutReassembler(
        tq_client,
        partition_id=_R3_DEFERRED_PARTITION,
        staging_partition=_R3_DEFERRED_STAGING,
        pad_token_id=PAD,
        max_seq_len=4096,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=True,
    )

    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False] * 2,
        fallback_weight_version=9,
        prompt_idx=17,
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
    finalizer = RolloutReassembler(
        tq_client,
        partition_id=_R3_DEFERRED_PARTITION,
        staging_partition=_R3_DEFERRED_STAGING,
        pad_token_id=PAD,
        max_seq_len=4096,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=True,
    )
    fetched = finalizer._source.fetch_for_finalization(
        [record["staging_key"] for record in receipt["manifest"]]
    )
    fetched[0] = dataclass_replace(fetched[0], routed_len=bad_routed_len)

    class _InjectedSource:
        def fetch_for_finalization(
            self, staging_keys, *, include_route_fragments=False
        ):
            del staging_keys, include_route_fragments
            return fetched

    finalizer._source = _InjectedSource()
    row = finalizer.finalize_rollout(rollout_id, receipt, reward=0.0)[0]

    assert not row.valid
    assert (row.rejection_reason or "").startswith("routed_len_mismatch")


# ---------------------------------------------------------------------------
# Unified flow: direct and deferred share one plan and one executor
# ---------------------------------------------------------------------------


def _mode_finalizer(tq_client, *, deferred: bool) -> RolloutReassembler:
    return RolloutReassembler(
        tq_client,
        partition_id=_R3_DEFERRED_PARTITION,
        staging_partition=_R3_DEFERRED_STAGING,
        pad_token_id=PAD,
        max_seq_len=4096,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=deferred,
    )


def test_direct_and_deferred_build_identical_plans_and_tensors(
    tq_client, r3_deferred_partitions
):
    """Both modes construct byte-identical plans; the shared executor driven
    from the worker's inputs reproduces the direct-mode tensor exactly."""
    from nemo_rl.experience.route_assembly import execute_route_plan
    from nemo_rl.experience.route_plan import encode_route_plan

    rollout_id = "unified_g0"
    receipt, expected, _ = _stage_deferred_fixture(tq_client, rollout_id=rollout_id)

    direct_row = _mode_finalizer(tq_client, deferred=False).finalize_rollout(
        rollout_id, receipt, reward=1.0
    )[0]
    deferred_row = _mode_finalizer(tq_client, deferred=True).finalize_rollout(
        rollout_id, receipt, reward=1.0
    )[0]
    assert direct_row.valid, direct_row.rejection_reason
    assert deferred_row.valid, deferred_row.rejection_reason

    # Same canonical token row in both modes.
    assert direct_row.token_ids == deferred_row.token_ids == expected.token_ids
    assert direct_row.token_mask == deferred_row.token_mask
    assert direct_row.logprobs == deferred_row.logprobs

    # Byte-identical RouteAssemblyPlans.
    assert direct_row.route_plan is not None and deferred_row.route_plan is not None
    assert encode_route_plan(direct_row.route_plan) == encode_route_plan(
        deferred_row.route_plan
    )
    # The plan carries exactly the receipt-bound extras commitments.
    committed = {
        entry["staging_key"]: entry["extras_digest"] for entry in receipt["manifest"]
    }
    for span in deferred_row.route_plan.spans:
        assert span.extras_digest == committed[span.staging_key]

    # Executor equivalence: driving the shared executor with worker-side
    # fragments and the deferred plan reproduces the direct-mode tensor.
    source = TQTokenSource(tq_client, staging_partition=_R3_DEFERRED_STAGING)
    fetched = source.fetch_for_finalization(
        list(deferred_row.route_plan.cleanup_staging_keys),
        include_route_fragments=True,
    )
    fragments = {
        item.staging_key: item.fragment for item in fetched if item.fragment is not None
    }
    tensor, reason = execute_route_plan(
        deferred_row.route_plan,
        fragments,
        dims=(2, 2),
        canonical_len=len(expected.token_ids),
    )
    assert reason is None
    assert direct_row.routed_experts is not None
    assert torch.equal(tensor, direct_row.routed_experts)


def test_direct_extras_corruption_rejects_before_publication(
    tq_client, r3_deferred_partitions
):
    from dataclasses import replace as dataclass_replace

    rollout_id = "corrupt_direct_g0"
    receipt, _, _ = _stage_deferred_fixture(tq_client, rollout_id=rollout_id)
    finalizer = _mode_finalizer(tq_client, deferred=False)
    fetched = finalizer._source.fetch_for_finalization(
        [record["staging_key"] for record in receipt["manifest"]],
        include_route_fragments=True,
    )
    tampered = fetched[0].fragment.routes.clone()
    tampered[0, 0, 0] = 9999
    fetched[0] = dataclass_replace(
        fetched[0],
        fragment=dataclass_replace(fetched[0].fragment, routes=tampered),
    )

    class _InjectedSource:
        def fetch_for_finalization(
            self, staging_keys, *, include_route_fragments=False
        ):
            del staging_keys, include_route_fragments
            return fetched

    finalizer._source = _InjectedSource()
    row = finalizer.finalize_rollout(rollout_id, receipt, reward=0.0)[0]

    assert not row.valid
    assert row.rejection_reason == "route_assembly:fragment_integrity"


def test_deferred_chain_hash_corruption_rejects_the_row(
    tq_client, r3_deferred_partitions
):
    """The deferred path now recomputes parent-chain and cumulative hashes
    (previously skipped by RL's local metadata-only linearizer)."""
    from nemo_gym.token_id_capture.staging.digest import compute_chain_hash

    from tests.unit.data_plane.token_capture_test_fixtures import (
        _manifest,
        _record,
    )

    rollout_id = "corrupt_chain_g0"
    root = _record(
        rollout_id=rollout_id,
        model_call_id="c1",
        parent_call_id=None,
        prev_len=0,
        token_ids=[10, 11, 12, 13],
        token_mask=[0.0, 0.0, 1.0, 1.0],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        weight_version=4,
    )
    # The child's chain hash extends a fabricated parent chain, with all
    # digests self-consistent — only chain replay can catch it.
    child = _record(
        rollout_id=rollout_id,
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=root.cum_len,
        token_ids=[20, 21, 22],
        token_mask=[0.0, 1.0, 1.0],
        logprobs=[0.0, -0.3, -0.4],
        weight_version=4,
        parent_chain_hash=compute_chain_hash(None, [99, 98]),
        cumulative_prefix=root.token_ids_delta,
    )
    from nemo_gym.token_id_capture.staging.records import RolloutReceipt

    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        terminal_model_call_id="c2",
        manifest=[_manifest(root), _manifest(child)],
        terminal_selection="declared",
    )
    sink = TQTokenSink(tq_client, staging_partition=_R3_DEFERRED_STAGING)
    for record in (root, child):
        assert sink.stage(record).ok

    row = _mode_finalizer(tq_client, deferred=True).finalize_rollout(
        rollout_id, receipt.model_dump(), reward=0.0
    )[0]
    assert not row.valid
    assert (row.rejection_reason or "").startswith("rebuild_failed:chain_hash_mismatch")


# ---------------------------------------------------------------------------
# Segment rows (token_capture.segment_rows): one row per verified chain
# ---------------------------------------------------------------------------
#
# Gym's ``verify_and_linearize_all`` is replaced by a fake that returns the
# real terminal chain (still verified by ``verify_and_linearize`` against the
# staged rows) plus synthetic extra chains, so these tests pin the finalizer's
# row layout, ids, tags, replicated columns, cap and cleanup behaviour without
# depending on how Gym derives the extra chains.

import types  # noqa: E402

import nemo_gym.token_id_capture.staging.rebuild as _rebuild_mod  # noqa: E402

_CHAIN_FIELDS = (
    "rollout_id",
    "token_ids",
    "token_mask",
    "logprobs",
    "model_call_ids",
    "prompt_len",
    "weight_versions",
    "weight_version_spans",
    "link_spans",
    "extras_commitments",
)


def _chain_view(row, **overrides):
    """Duck-typed copy of a LinearizedRow with segment placement fields."""
    values = {name: getattr(row, name) for name in _CHAIN_FIELDS}
    values.update(
        terminal_model_call_id=row.model_call_ids[-1],
        chain_index=0,
        chain_kind="terminal",
        segment_index=0,
        boundary_parent_call_id=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _synthetic_segment(terminal, *, chain_index: int, kind: str, tokens: list[int]):
    """An extra chain that shares nothing with the terminal chain's tokens."""
    prompt_len = 2
    return _chain_view(
        terminal,
        token_ids=list(tokens),
        token_mask=[0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len),
        logprobs=[0.0] * prompt_len + [-0.5] * (len(tokens) - prompt_len),
        model_call_ids=[f"seg{chain_index}"],
        prompt_len=prompt_len,
        link_spans=[(f"seg{chain_index}", prompt_len, len(tokens) - prompt_len)],
        extras_commitments=[],
        terminal_model_call_id=f"seg{chain_index}",
        chain_index=chain_index,
        chain_kind=kind,
        segment_index=chain_index,
        boundary_parent_call_id=("c1" if kind == "compaction_segment" else None),
    )


def _install_fake_linearize_all(monkeypatch, *, extras_by_rollout, skipped=1):
    """Patch Gym's verify_and_linearize_all with a fake building on the real verifier.

    ``extras_by_rollout`` maps rollout_id -> list of (kind, tokens) extra
    chains appended after the (real) terminal chain.
    """
    calls: list[str] = []

    def fake(receipt, snapshots):
        calls.append(receipt.rollout_id)
        terminal = _rebuild_mod.verify_and_linearize(receipt, snapshots)
        rows = [_chain_view(terminal)]
        for idx, (kind, tokens) in enumerate(
            extras_by_rollout.get(receipt.rollout_id, []), start=1
        ):
            rows.append(
                _synthetic_segment(terminal, chain_index=idx, kind=kind, tokens=tokens)
            )
        return types.SimpleNamespace(
            rows=rows,
            skipped=[
                types.SimpleNamespace(root_call_id=f"amb{i}", reason="ambiguous_leaf")
                for i in range(skipped)
            ],
            num_roots=len(rows) + skipped,
            num_boundary_roots=sum(
                1 for row in rows if row.boundary_parent_call_id is not None
            ),
        )

    monkeypatch.setattr(_rebuild_mod, "verify_and_linearize_all", fake)
    return calls


def _segment_finalizer(tq_client, **overrides) -> RolloutReassembler:
    kwargs = dict(segment_rows_enabled=True, max_rows_per_rollout=3)
    kwargs.update(overrides)
    return _finalizer(tq_client, **kwargs)


def test_segment_rows_config_is_validated():
    with pytest.raises(ValueError, match="max_rows_per_rollout >= 2"):
        RolloutReassembler(
            object(),
            partition_id=CANONICAL_PARTITION,
            staging_partition=STAGING_PARTITION,
            pad_token_id=PAD,
            max_seq_len=16,
            segment_rows_enabled=True,
            max_rows_per_rollout=1,
        )


def test_finalize_rollout_returns_terminal_plus_segments(
    tq_client, partitions, monkeypatch
):
    rollout_id = "seg_r0"
    receipt, expected = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    calls = _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={
            rollout_id: [("compaction_segment", [30, 31, 32, 33, 34])]
        },
    )
    rows = _segment_finalizer(tq_client).finalize_rollout(
        rollout_id, receipt, reward=1.0
    )

    assert calls == [rollout_id]
    assert [row.trace_in_rollout_idx for row in rows] == [0, 1]
    terminal, segment = rows
    assert terminal.valid and segment.valid
    assert terminal.token_ids == expected.token_ids
    assert terminal.trace_kind == "terminal"
    assert segment.token_ids == [30, 31, 32, 33, 34]
    assert segment.trace_kind == "compaction_segment"
    assert segment.segment_index == 1
    assert segment.boundary_parent_call_id == "c1"
    assert segment.chain_index == 1
    # Cleanup ownership rides the canonical row only.
    assert set(terminal.staging_keys) == {
        entry["staging_key"] for entry in receipt["manifest"]
    }
    assert segment.staging_keys == []
    # Rollout-level diagnostics ride the canonical row.
    assert terminal.num_chains == 2
    assert terminal.chains_skipped_ambiguous == 1
    assert terminal.boundary_roots == 1
    assert terminal.segments_dropped_by_cap == 0
    # The shared reward is replicated.
    assert (terminal.reward, segment.reward) == (1.0, 1.0)


def test_finalize_rollout_caps_segment_rows(tq_client, partitions, monkeypatch):
    rollout_id = "cap_r0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={
            rollout_id: [
                ("compaction_segment", [40, 41, 42]),
                ("compaction_segment", [50, 51, 52, 53]),
                ("subagent", [60, 61, 62]),
            ]
        },
        skipped=0,
    )
    rows = _segment_finalizer(tq_client, max_rows_per_rollout=2).finalize_rollout(
        rollout_id, receipt, reward=0.0
    )
    assert len(rows) == 2
    # Gym order is kept: the first extra chain survives the cap.
    assert rows[1].token_ids == [40, 41, 42]
    assert rows[0].segments_dropped_by_cap == 2
    assert rows[0].num_chains == 2


def test_finalize_rollout_rejection_yields_single_placeholder_row(
    tq_client, partitions, monkeypatch
):
    rollout_id = "rej_seg"
    receipt, _ = _stage_fixture(tq_client, "single_call", rollout_id=rollout_id)
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_id: [("subagent", [70, 71, 72])]}
    )
    finalizer = _segment_finalizer(tq_client)
    assert finalizer.finalize_rollout(rollout_id, None, reward=0.0) == [
        finalizer.finalize_rollout(rollout_id, None, reward=0.0)[0]
    ]
    poisoned = dict(receipt, capture_poisoned=True)
    rows = finalizer.finalize_rollout(rollout_id, poisoned, reward=0.0)
    assert len(rows) == 1
    assert rows[0].rejection_reason == "capture_poisoned"


def test_finalize_rollout_falls_back_without_verify_and_linearize_all(
    tq_client, partitions, monkeypatch
):
    rollout_id = "nofn_r0"
    receipt, expected = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    monkeypatch.delattr(_rebuild_mod, "verify_and_linearize_all", raising=False)
    finalizer = _segment_finalizer(tq_client)
    with pytest.warns(RuntimeWarning, match="verify_and_linearize_all"):
        rows = finalizer.finalize_rollout(rollout_id, receipt, reward=1.0)
    assert len(rows) == 1
    assert rows[0].valid and rows[0].token_ids == expected.token_ids
    # Warned once per finalizer, not per rollout.
    receipt2, _ = _stage_fixture(tq_client, "worked_example", rollout_id="nofn_r1")
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert len(finalizer.finalize_rollout("nofn_r1", receipt2, reward=1.0)) == 1


def test_finalize_group_publishes_segment_rows_after_canonical_block(
    tq_client, partitions, monkeypatch
):
    group_id = "seggrp"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected = _stage_fixture(
        tq_client, "worked_example", rollout_id=rollout_ids[0]
    )
    segment_tokens = [30, 31, 32, 33, 34]
    _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={rollout_ids[0]: [("compaction_segment", segment_tokens)]},
    )
    # max_seq_len pinned to the segment row's length: only that row may read
    # truncated (the canonical row is a different length, the placeholder is 1).
    assert len(segment_tokens) != len(expected.token_ids)
    finalizer = _segment_finalizer(tq_client, max_seq_len=len(segment_tokens))

    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],  # rollout 1 lost its receipt -> placeholder, no extras
        [1.0, 0.0],
        mask_sample=[True, False],
        fallback_weight_version=9,
        prompt_idx=17,
        loss_multiplier=0.25,
    )

    assert not finalized.dropped
    assert finalized.meta is not None
    # Canonical block first (N rows under the canonical ids), extras after.
    extra_id = f"{group_id}_g0_t1"
    assert finalized.meta.sample_ids == rollout_ids + [extra_id]
    assert finalized.meta.sequence_lengths == [
        len(expected.token_ids),
        1,
        len(segment_tokens),
    ]
    # Rollout counts stay rollout counts; extras are reported separately.
    assert finalized.valid_row_count == 1
    assert finalized.total_row_count == 2
    assert finalized.extra_row_count == 1
    # Per-row placement tags.
    tags = finalized.meta.tags
    assert [tag["rollout_local_idx"] for tag in tags] == [0, 1, 0]
    assert [tag["trace_in_rollout_idx"] for tag in tags] == [0, 0, 1]
    assert [tag["trace_kind"] for tag in tags] == [
        "terminal",
        "placeholder",
        "compaction_segment",
    ]
    assert [tag["segment_index"] for tag in tags] == [0, 0, 1]
    assert [tag["prompt_idx"] for tag in tags] == [17, 17, 17]
    assert [tag["weight_version"] for tag in tags] == [4, 4, 4]
    # canonical_output_tokens covers every published row.
    assert finalized.canonical_output_tokens == sum(expected.token_mask) + 3
    # Metrics.
    m = finalized.metrics
    assert m["finalize/invalid_row_rate"] == 0.5
    assert m["finalize/rows_per_rollout_mean"] == 1.5
    assert m["finalize/rows_per_rollout_max"] == 2.0
    assert m["finalize/rollouts_with_segments"] == 1.0
    assert m["finalize/segment_rows"] == 1.0
    assert m["finalize/segment_rows_by_kind_compaction_segment"] == 1.0
    assert m["finalize/chains_skipped_ambiguous"] == 1.0
    assert m["finalize/boundary_roots"] == 1.0
    assert m["finalize/segments_dropped_by_cap"] == 0.0

    rows = _fetch_rows(tq_client, rollout_ids + [extra_id])
    # Extra row copies reward / mask_sample / sample_mask from its rollout.
    assert torch.as_tensor(rows["total_reward"]).flatten().tolist() == [1.0, 0.0, 1.0]
    assert torch.as_tensor(rows["mask_sample"]).flatten().tolist() == [
        True,
        False,
        True,
    ]
    assert torch.as_tensor(rows["sample_mask"]).flatten().tolist() == [0.25, 0.0, 0.25]
    # prompt_ids_for_adv is the same group prompt on every row.
    prompt = expected.token_ids[: expected.prompt_len]
    for r in range(3):
        assert torch.as_tensor(rows["prompt_ids_for_adv"][r]).flatten().tolist() == prompt
    # Tokens / masks / logprobs are per row.
    seg_ids = torch.as_tensor(rows["input_ids"][2]).flatten()
    assert seg_ids[: len(segment_tokens)].tolist() == segment_tokens
    seg_mask = torch.as_tensor(rows["token_mask"][2]).flatten()
    assert seg_mask[: len(segment_tokens)].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    seg_lp = torch.as_tensor(rows["generation_logprobs"][2]).flatten()
    assert seg_lp[2:5].tolist() == [-0.5, -0.5, -0.5]
    assert torch.as_tensor(rows["input_lengths"]).flatten().tolist() == [
        len(expected.token_ids),
        1,
        len(segment_tokens),
    ]
    # truncated is per row: only the segment row hits max_seq_len.
    assert torch.as_tensor(rows["truncated"]).flatten().tolist() == [
        False,
        False,
        True,
    ]

    # Staging cleared exactly once for the rollout's manifest (shared by both rows).
    with pytest.raises(KeyError):
        finalizer._source.fetch([receipt["manifest"][0]["staging_key"]])


def test_finalize_group_segment_rows_disabled_publishes_exactly_n_rows(
    tq_client, partitions, monkeypatch
):
    """With segment_rows off the fake is never consulted and the group is the
    classic N-row publish (same ids, same tensors as the single-chain path)."""
    group_id = "segoff"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected = _stage_fixture(
        tq_client, "worked_example", rollout_id=rollout_ids[0]
    )
    calls = _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={rollout_ids[0]: [("compaction_segment", [30, 31, 32])]},
    )
    finalizer = _finalizer(tq_client)  # segment rows disabled (default)
    finalized = finalizer.finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[True, False],
        fallback_weight_version=9,
        prompt_idx=17,
        loss_multiplier=0.25,
    )
    assert calls == []
    assert finalized.meta.sample_ids == rollout_ids
    assert finalized.extra_row_count == 0
    assert (finalized.valid_row_count, finalized.total_row_count) == (1, 2)
    assert finalized.metrics["finalize/rows_per_rollout_max"] == 1.0
    assert finalized.metrics["finalize/segment_rows"] == 0.0
    assert [tag["trace_kind"] for tag in finalized.meta.tags] == [
        "terminal",
        "placeholder",
    ]
    rows = _fetch_rows(tq_client, rollout_ids)
    valid_len = len(expected.token_ids)
    assert torch.as_tensor(rows["input_ids"][0]).flatten()[:valid_len].tolist() == (
        expected.token_ids
    )
    assert torch.as_tensor(rows["sample_mask"]).flatten().tolist() == [0.25, 0.0]
    assert torch.as_tensor(rows["total_reward"]).flatten().tolist() == [1.0, 0.0]
    assert torch.as_tensor(rows["mask_sample"]).flatten().tolist() == [True, False]
    assert finalized.canonical_output_tokens == sum(expected.token_mask)


def test_finalize_group_all_placeholders_publish_one_row_each_with_segments_on(
    tq_client, partitions, monkeypatch
):
    group_id = "segph"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1", f"{group_id}_g2"]
    _install_fake_linearize_all(monkeypatch, extras_by_rollout={})
    finalized = _segment_finalizer(tq_client).finalize_group(
        group_id,
        rollout_ids,
        [None, None, None],
        [0.0, 0.0, 0.0],
        mask_sample=[False] * 3,
        fallback_weight_version=3,
        prompt_idx=17,
    )
    assert finalized.meta.sample_ids == rollout_ids
    assert finalized.extra_row_count == 0
    assert (finalized.valid_row_count, finalized.total_row_count) == (0, 3)
    assert [tag["trace_kind"] for tag in finalized.meta.tags] == ["placeholder"] * 3
    assert [tag["rollout_local_idx"] for tag in finalized.meta.tags] == [0, 1, 2]
    assert finalized.metrics["finalize/rows_per_rollout_mean"] == 1.0


def test_finalize_group_segment_rows_cap_metric(tq_client, partitions, monkeypatch):
    group_id = "segcap"
    rollout_ids = [f"{group_id}_g0"]
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={
            rollout_ids[0]: [
                ("compaction_segment", [40, 41, 42]),
                ("subagent", [50, 51, 52]),
                ("subagent", [60, 61, 62]),
            ]
        },
        skipped=0,
    )
    finalized = _segment_finalizer(tq_client, max_rows_per_rollout=3).finalize_group(
        group_id,
        rollout_ids,
        [receipt],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
    )
    assert finalized.meta.sample_ids == [
        f"{group_id}_g0",
        f"{group_id}_g0_t1",
        f"{group_id}_g0_t2",
    ]
    assert finalized.extra_row_count == 2
    assert finalized.metrics["finalize/segments_dropped_by_cap"] == 1.0
    assert finalized.metrics["finalize/segment_rows_by_kind_compaction_segment"] == 1.0
    assert finalized.metrics["finalize/segment_rows_by_kind_subagent"] == 1.0
    assert [tag["trace_in_rollout_idx"] for tag in finalized.meta.tags] == [0, 1, 2]


def test_finalize_group_segment_rows_with_stable_canonical_ids(
    tq_client, partitions, monkeypatch
):
    """Physical attempt ids map to canonical ids; extras extend the canonical id."""
    group_id = "segstable"
    physical_id = f"{group_id}_g0_aattempt"
    canonical_id = f"{group_id}_g0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=physical_id)
    _install_fake_linearize_all(
        monkeypatch,
        extras_by_rollout={physical_id: [("compaction_segment", [30, 31, 32])]},
    )
    finalized = _segment_finalizer(tq_client).finalize_group(
        group_id,
        [physical_id],
        [receipt],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
        canonical_sample_ids=[canonical_id],
    )
    assert finalized.meta.sample_ids == [canonical_id, f"{canonical_id}_t1"]


# ── summary rows, Gym row order, rows_in_rollout ──────────────────────────
#
# A compaction yields three chains per boundary: the pre-compaction segment
# (compaction_segment), the summary call as a one-call chain
# (compaction_summary) and the post-compaction chain. Gym orders the chains on
# the terminal's compaction sequence first (segment order), then other roots.
# The finalizer keeps that order, drops summary rows first when they are not
# wanted, and applies the cap to what remains.


def _three_chain_extras():
    return [
        ("compaction_segment", [30, 31, 32, 33, 34]),
        ("compaction_summary", [40, 41, 42, 43]),
        ("subagent", [50, 51, 52]),
    ]


def test_finalize_rollout_keeps_gym_order_and_caps_after_it(
    tq_client, partitions, monkeypatch
):
    rollout_id = "order_r0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_id: _three_chain_extras()}, skipped=0
    )
    rows = _segment_finalizer(tq_client, max_rows_per_rollout=3).finalize_rollout(
        rollout_id, receipt, reward=1.0
    )
    # Default include_summary_rows=True: Gym's order, cap drops the trailing
    # subagent chain (not the summary).
    assert [row.trace_kind for row in rows] == [
        "terminal",
        "compaction_segment",
        "compaction_summary",
    ]
    assert [row.trace_in_rollout_idx for row in rows] == [0, 1, 2]
    assert rows[2].token_ids == [40, 41, 42, 43]
    assert rows[0].segments_dropped_by_cap == 1
    assert rows[0].segments_skipped_summary == 0


def test_finalize_rollout_skips_summary_rows_before_the_cap_when_disabled(
    tq_client, partitions, monkeypatch
):
    rollout_id = "nosum_r0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_id: _three_chain_extras()}, skipped=0
    )
    rows = _segment_finalizer(
        tq_client, max_rows_per_rollout=3, include_summary_rows=False
    ).finalize_rollout(rollout_id, receipt, reward=1.0)
    # The summary is dropped first, so the cap budget goes to the subagent chain.
    assert [row.trace_kind for row in rows] == ["terminal", "compaction_segment", "subagent"]
    assert [row.trace_in_rollout_idx for row in rows] == [0, 1, 2]
    assert rows[2].token_ids == [50, 51, 52]
    assert rows[0].segments_skipped_summary == 1
    assert rows[0].segments_dropped_by_cap == 0
    assert rows[0].num_chains == 3


def test_finalize_group_stamps_rows_in_rollout_and_summary_metrics(
    tq_client, partitions, monkeypatch
):
    group_id = "sumgrp"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_ids[0]: _three_chain_extras()}, skipped=0
    )
    finalized = _segment_finalizer(tq_client, max_rows_per_rollout=4).finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=4,
        prompt_idx=17,
    )
    assert finalized.meta.sample_ids == rollout_ids + [
        f"{rollout_ids[0]}_t1",
        f"{rollout_ids[0]}_t2",
        f"{rollout_ids[0]}_t3",
    ]
    tags = finalized.meta.tags
    assert [tag["trace_kind"] for tag in tags] == [
        "terminal",
        "placeholder",
        "compaction_segment",
        "compaction_summary",
        "subagent",
    ]
    # Every row of a rollout carries the rollout's total row count; the
    # placeholder rollout has exactly one row.
    assert [tag["rows_in_rollout"] for tag in tags] == [4, 1, 4, 4, 4]
    assert [tag["rollout_local_idx"] for tag in tags] == [0, 1, 0, 0, 0]
    m = finalized.metrics
    assert m["finalize/segment_rows"] == 3.0
    assert m["finalize/segment_rows_by_kind_compaction_segment"] == 1.0
    assert m["finalize/segment_rows_by_kind_compaction_summary"] == 1.0
    assert m["finalize/segment_rows_by_kind_subagent"] == 1.0
    assert m["finalize/segment_rows_skipped_summary"] == 0.0
    assert m["finalize/segments_dropped_by_cap"] == 0.0
    assert m["finalize/rows_per_rollout_max"] == 4.0


def test_finalize_group_counts_skipped_summary_rows(tq_client, partitions, monkeypatch):
    group_id = "sumoff"
    rollout_ids = [f"{group_id}_g0"]
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_ids[0]: _three_chain_extras()}, skipped=0
    )
    finalized = _segment_finalizer(
        tq_client, max_rows_per_rollout=4, include_summary_rows=False
    ).finalize_group(
        group_id,
        rollout_ids,
        [receipt],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
    )
    tags = finalized.meta.tags
    assert [tag["trace_kind"] for tag in tags] == ["terminal", "compaction_segment", "subagent"]
    assert [tag["rows_in_rollout"] for tag in tags] == [3, 3, 3]
    m = finalized.metrics
    assert m["finalize/segment_rows"] == 2.0
    assert m["finalize/segment_rows_skipped_summary"] == 1.0
    assert "finalize/segment_rows_by_kind_compaction_summary" not in m
    # Disabled segment rows never consult the switch: exactly one row, no tag
    # beyond rows_in_rollout == 1.
    plain = _finalizer(tq_client, include_summary_rows=False).finalize_group(
        group_id + "_plain",
        [f"{group_id}_plain_g0"],
        [None],
        [0.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
    )
    assert [tag["rows_in_rollout"] for tag in plain.meta.tags] == [1]
    assert plain.metrics["finalize/segment_rows_skipped_summary"] == 0.0


def _install_nan_subagent(monkeypatch, rollout_id: str) -> None:
    """One subagent segment row whose generation logprobs hold a NaN."""
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_id: [("subagent", [40, 41, 42, 43])]}
    )
    linearize_all = _rebuild_mod.verify_and_linearize_all

    def with_nan(receipt, snapshots):
        linearized = linearize_all(receipt, snapshots)
        linearized.rows[1].logprobs = [0.0, 0.0, float("nan"), -0.5]
        return linearized

    monkeypatch.setattr(_rebuild_mod, "verify_and_linearize_all", with_nan)


def test_finalize_group_force_masks_only_the_row_with_nan_logprobs(
    tq_client, partitions, monkeypatch
):
    group_id = "nangrp"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_nan_subagent(monkeypatch, rollout_ids[0])

    finalized = _segment_finalizer(tq_client).finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=9,
        prompt_idx=3,
        loss_multiplier=1.0,
    )

    extra_id = f"{group_id}_g0_t1"
    assert finalized.meta is not None
    assert finalized.meta.sample_ids == rollout_ids + [extra_id]
    rows = _fetch_rows(tq_client, rollout_ids + [extra_id])
    # Only the NaN row is masked (the echo path's per-trace forced mask).
    assert torch.as_tensor(rows["mask_sample"]).flatten().tolist() == [
        False,
        False,
        True,
    ]
    seg_lp = torch.as_tensor(rows["generation_logprobs"][2]).flatten()
    assert seg_lp[:4].tolist() == [0.0, 0.0, 0.0, -0.5]
    assert torch.isfinite(torch.as_tensor(rows["generation_logprobs"])).all()
    assert finalized.metrics["finalize/nan_logprob_rows_masked"] == 1.0


def test_finalize_rollout_raises_on_nan_logprobs_when_configured(
    tq_client, partitions, monkeypatch
):
    rollout_id = "nan_r0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    _install_nan_subagent(monkeypatch, rollout_id)
    finalizer = _segment_finalizer(tq_client, nan_generation_logprobs="raise")
    with pytest.raises(ValueError, match="NaN generation logprobs"):
        finalizer.finalize_rollout(rollout_id, receipt, reward=1.0)


def test_nan_generation_logprobs_policy_is_validated():
    with pytest.raises(ValueError, match="nan_generation_logprobs"):
        RolloutReassembler(
            object(),
            partition_id=CANONICAL_PARTITION,
            staging_partition=STAGING_PARTITION,
            pad_token_id=PAD,
            max_seq_len=16,
            nan_generation_logprobs="drop",
        )


# ── token-scoped reward penalties (capture-path parity with the echo path) ────


class _CharDecodeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _penalty_finalizer(tq_client, **penalty_config) -> RolloutReassembler:
    finalizer = _segment_finalizer(tq_client, reward_penalty_config=penalty_config)
    finalizer._tokenizer = _CharDecodeTokenizer()
    return finalizer


def _publish_with_subagent(tq_client, monkeypatch, group_id, subagent_tokens, **cfg):
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_ids[0]: [("subagent", subagent_tokens)]}
    )
    finalized = _penalty_finalizer(tq_client, **cfg).finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=9,
        prompt_idx=3,
    )
    rows = _fetch_rows(tq_client, rollout_ids + [f"{group_id}_g0_t1"])
    return finalized, torch.as_tensor(rows["total_reward"]).flatten().tolist()


def test_unwanted_token_in_a_subagent_generation_zeroes_every_row(
    tq_client, partitions, monkeypatch
):
    # Subagent chain: carry [40, 41], generation [99, 43] -> unwanted 99 generated.
    finalized, rewards = _publish_with_subagent(
        tq_client,
        monkeypatch,
        "pen_unw",
        [40, 41, 99, 43],
        penalize_unwanted_tokens=True,
        token_ids={"unwanted": [99]},
    )
    assert rewards == [0.0, 0.0, 0.0]
    assert finalized.metrics["finalize/penalty/unwanted_token_rate"] == 0.5


def test_unwanted_token_in_the_prompt_carry_is_not_penalized(
    tq_client, partitions, monkeypatch
):
    finalized, rewards = _publish_with_subagent(
        tq_client,
        monkeypatch,
        "pen_carry",
        [99, 41, 42, 43],
        penalize_unwanted_tokens=True,
        token_ids={"unwanted": [99]},
    )
    assert rewards == [1.0, 0.0, 1.0]
    assert finalized.metrics["finalize/penalty/unwanted_token_rate"] == 0.0


def test_think_tag_spelled_out_in_a_generation_is_penalized(
    tq_client, partitions, monkeypatch
):
    generation = [ord(c) for c in "<think>x"]
    finalized, rewards = _publish_with_subagent(
        tq_client,
        monkeypatch,
        "pen_str",
        [65, 66, *generation],
        penalize_malformed_think_tag=True,
        thinking_tags=["<think>", "</think>"],
    )
    assert rewards == [0.0, 0.0, 0.0]
    assert finalized.metrics["finalize/penalty/malformed_think_tag_rate"] == 0.5


def test_think_tag_token_counts_follow_the_prompt_thinking_mode(
    tq_client, partitions, monkeypatch
):
    # Prompt carry ends with an open tag (thinking on): the generation must
    # close it exactly once. This one never does -> violation.
    finalized, rewards = _publish_with_subagent(
        tq_client,
        monkeypatch,
        "pen_tok",
        [65, 500, 66, 67],
        penalize_malformed_think_tag=True,
        thinking_tags=["<think>", "</think>"],
        token_ids={"think_open": 500, "think_close": 501},
    )
    assert rewards == [0.0, 0.0, 0.0]
    finalized, rewards = _publish_with_subagent(
        tq_client,
        monkeypatch,
        "pen_tok_ok",
        [65, 500, 501, 67],
        penalize_malformed_think_tag=True,
        thinking_tags=["<think>", "</think>"],
        token_ids={"think_open": 500, "think_close": 501},
    )
    assert finalized.metrics["finalize/penalty/malformed_think_tag_rate"] == 0.0


def test_penalty_spans_that_do_not_tile_the_chain_reject_the_rollout(
    tq_client, partitions, monkeypatch
):
    rollout_id = "pen_span_r0"
    receipt, _ = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_id)
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_id: [("subagent", [40, 41, 42, 43])]}
    )
    linearize_all = _rebuild_mod.verify_and_linearize_all

    def short_spans(receipt, snapshots):
        linearized = linearize_all(receipt, snapshots)
        linearized.rows[1].link_spans = [("seg1", 1, 1)]
        return linearized

    monkeypatch.setattr(_rebuild_mod, "verify_and_linearize_all", short_spans)
    rows = _penalty_finalizer(
        tq_client, penalize_unwanted_tokens=True, token_ids={"unwanted": [99]}
    ).finalize_rollout(rollout_id, receipt, reward=1.0)
    assert len(rows) == 1 and not rows[0].valid
    assert rows[0].rejection_reason.startswith("penalty_spans:")


def test_finalize_group_reports_legacy_per_trace_metrics(
    tq_client, partitions, monkeypatch
):
    group_id = "legacy_m"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_ids[0]: [("subagent", [40, 41, 42, 43])]}
    )
    finalized = _segment_finalizer(tq_client).finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=9,
        prompt_idx=3,
    )
    m = finalized.metrics
    # One valid rollout (the other is a placeholder) with a terminal row and a
    # one-turn subagent row.
    assert m["finalize/legacy/traces_per_sample"] == 2.0
    assert m["finalize/legacy/subagent_traces_per_sample"] == 1.0
    assert m["finalize/legacy/gen_tokens_per_sample"] == sum(expected.token_mask) + 2
    assert m["finalize/legacy/total_tokens_per_sample"] == len(expected.token_ids) + 4
    assert m["finalize/legacy/turns_per_sample"] == len(expected.link_spans) + 1
    assert m["finalize/legacy/turns_per_trace"] == (len(expected.link_spans) + 1) / 2
    # No rollout_infos: no rollout_debug provenance on the rows.
    assert all(ROLLOUT_DEBUG_TAG not in tag for tag in finalized.meta.tags)


def test_finalize_group_stamps_rollout_debug_tags(tq_client, partitions, monkeypatch):
    group_id = "dbg_tag"
    rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
    receipt, expected = _stage_fixture(tq_client, "worked_example", rollout_id=rollout_ids[0])
    _install_fake_linearize_all(
        monkeypatch, extras_by_rollout={rollout_ids[0]: [("subagent", [40, 41, 42, 43])]}
    )
    infos = [{"agent_timed_out": False, "dataset_name": "swe_rebench"}, {}]
    finalized = _segment_finalizer(tq_client).finalize_group(
        group_id,
        rollout_ids,
        [receipt, None],
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=9,
        prompt_idx=3,
        rollout_infos=infos,
    )
    decoded = [decode_rollout_debug_tag(tag) for tag in finalized.meta.tags]
    # Canonical block (g0 terminal, g1 placeholder), then g0's subagent row.
    assert [d["rollout_local_idx"] for d in decoded] == [0, 1, 0]
    assert [d["trace_in_rollout_idx"] for d in decoded] == [0, 0, 1]
    kinds = [d["trace_metadata"]["kind"] for d in decoded]
    assert kinds == ["uncompacted", "empty", "subagent"]
    assert [d["is_empty_rollout"] for d in decoded] == [False, True, False]
    assert decoded[0]["rollout_info"] == infos[0] == decoded[2]["rollout_info"]
    assert decoded[1]["rollout_info"] == {}
    assert decoded[0]["trace_metadata"]["turns"] == len(expected.link_spans)
    assert decoded[0]["trace_metadata"]["gen_tokens"] == sum(expected.token_mask)
    assert decoded[2]["trace_metadata"]["turns"] == 1
    # Per-rollout generation sizes: g0 terminal + subagent rows, g1 rejected.
    assert finalized.rollout_gen_tokens == (
        sum(expected.token_mask) + decoded[2]["trace_metadata"]["gen_tokens"],
        0,
    )
    assert finalized.rollout_max_gen_tokens_per_turn[0] >= max(
        gen for _, _, gen in expected.link_spans
    )
    assert finalized.rollout_max_gen_tokens_per_turn[1] == 0
