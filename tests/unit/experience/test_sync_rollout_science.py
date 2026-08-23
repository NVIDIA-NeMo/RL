# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nemo_rl.experience.sync_rollout_actor import (
    ACCEPTED_TOKEN_COUNT_KEY,
    APPLIED_DRAFT_VERSION_KEY,
    DRAFT_TOKEN_COUNT_KEY,
    ServingDraftVersionTracker,
)


def test_science_stamp_requires_published_serving_version() -> None:
    tracker = ServingDraftVersionTracker()

    with pytest.raises(RuntimeError, match="serving draft version is absent"):
        tracker.stamp(
            {
                ACCEPTED_TOKEN_COUNT_KEY: 6.0,
                DRAFT_TOKEN_COUNT_KEY: 10.0,
            }
        )


def test_science_stamp_propagates_counts_and_serving_version() -> None:
    tracker = ServingDraftVersionTracker()
    tracker.publish(4)
    metrics = {
        ACCEPTED_TOKEN_COUNT_KEY: 6.0,
        DRAFT_TOKEN_COUNT_KEY: 10.0,
    }

    stamped = tracker.stamp(metrics, expected_version=4)

    assert stamped == {**metrics, APPLIED_DRAFT_VERSION_KEY: 4}
    assert APPLIED_DRAFT_VERSION_KEY not in metrics


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_version_publication_rejects_nonintegral_or_negative_values(
    value: object,
) -> None:
    tracker = ServingDraftVersionTracker()

    with pytest.raises(ValueError, match="nonnegative integer"):
        tracker.publish(value)  # type: ignore[arg-type]


def test_version_publication_is_idempotent_and_rejects_stale_rollback() -> None:
    tracker = ServingDraftVersionTracker()
    tracker.publish(2)
    tracker.publish(2)

    with pytest.raises(RuntimeError, match="stale serving draft version"):
        tracker.publish(1)


def test_science_stamp_rejects_mixed_or_stale_batch_version() -> None:
    tracker = ServingDraftVersionTracker()
    tracker.publish(3)
    counts = {
        ACCEPTED_TOKEN_COUNT_KEY: 6.0,
        DRAFT_TOKEN_COUNT_KEY: 10.0,
    }

    with pytest.raises(ValueError, match="mixed serving draft version"):
        tracker.stamp({**counts, APPLIED_DRAFT_VERSION_KEY: 2})
    with pytest.raises(RuntimeError, match="stale selected rollout"):
        tracker.stamp(counts, expected_version=2)


@pytest.mark.parametrize(
    "metrics",
    [
        {ACCEPTED_TOKEN_COUNT_KEY: 1.0},
        {ACCEPTED_TOKEN_COUNT_KEY: -1.0, DRAFT_TOKEN_COUNT_KEY: 2.0},
        {ACCEPTED_TOKEN_COUNT_KEY: 3.0, DRAFT_TOKEN_COUNT_KEY: 2.0},
        {ACCEPTED_TOKEN_COUNT_KEY: True, DRAFT_TOKEN_COUNT_KEY: 2.0},
    ],
)
def test_science_stamp_rejects_absent_or_invalid_counts(
    metrics: dict[str, object],
) -> None:
    tracker = ServingDraftVersionTracker()
    tracker.publish(0)

    with pytest.raises(ValueError, match="accepted/draft token counts"):
        tracker.stamp(metrics)
