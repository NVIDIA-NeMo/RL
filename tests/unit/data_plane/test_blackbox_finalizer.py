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

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.conformance.kit import (  # noqa: E402
    build_fixture_artifacts,
    f32,
    load_fixture,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
)
from nemo_rl.experience.blackbox_finalizer import BlackboxFinalizer  # noqa: E402

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
    fixture = load_fixture(name)
    if rollout_id is not None:
        fixture = dict(fixture)
        fixture["rollout_id"] = rollout_id
    records, _, receipt, row = build_fixture_artifacts(fixture)
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
    empty = dict(receipt, manifest=[], terminal_call_id=None)
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
