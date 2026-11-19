# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_rl.telemetry.vocabulary``.

The registry is process-global and every owner module fills it at import, so
each test here swaps in empty state rather than declaring into the real one.
"""

import pytest

from nemo_rl.telemetry import vocabulary
from nemo_rl.telemetry.vocabulary import (
    RecordedMetric,
    TeedMetric,
    as_scalar,
    register_recorded_metrics,
    register_teed_metrics,
    registry_key,
)


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Isolate each test from the rows the owner modules declared at import."""
    monkeypatch.setattr(vocabulary, "_REGISTERED", {})
    monkeypatch.setattr(vocabulary, "_RECORDED", {})
    monkeypatch.setattr(vocabulary, "_FROZEN", False)


def test_registry_key_strips_the_prefix_and_flattens_dots():
    assert registry_key("rl.reward.mean") == "reward_mean"
    # lens keys have to be identifiers, which is the whole point of the rule.
    assert registry_key("rl.reward.mean").isidentifier()


def test_registry_key_is_derived_not_typed():
    """A row names its series once; the key cannot disagree with it."""
    assert TeedMetric("reward", "rl.reward.mean").key == "reward_mean"


@pytest.mark.parametrize(
    "value,expected",
    [(1, 1.0), (2.5, 2.5), (True, None), (False, None), ("3", None), (None, None)],
)
def test_as_scalar_accepts_only_real_numbers(value, expected):
    """``bool`` is an ``int`` subclass, so it has to be rejected explicitly."""
    assert as_scalar(value) == expected


def test_a_duplicate_logger_key_is_rejected():
    """Silent otherwise: the second row would make the first unreachable."""
    register_teed_metrics([TeedMetric("reward", "rl.reward.mean")])

    with pytest.raises(ValueError, match="already teed"):
        register_teed_metrics([TeedMetric("reward", "rl.reward.other")])


def test_a_duplicate_series_name_is_rejected():
    """Silent otherwise: two rows would record against one instrument."""
    register_teed_metrics([TeedMetric("reward", "rl.reward.mean")])

    with pytest.raises(ValueError, match="already declared"):
        register_teed_metrics([TeedMetric("other_key", "rl.reward.mean")])


def test_declaring_the_same_row_twice_is_allowed():
    """A module can be imported more than once; an identical row is a no-op."""
    row = TeedMetric("reward", "rl.reward.mean")
    register_teed_metrics([row])
    register_teed_metrics([row])

    assert vocabulary.teed_metrics() == (row,)


def test_a_row_declared_after_freezing_raises():
    """lens takes the specs once, so a later row has no instrument.

    Without this it is recorded into the values and dropped by lens with a
    single warning, which is exactly the kind of quiet gap the registry exists
    to prevent.
    """
    register_teed_metrics([TeedMetric("reward", "rl.reward.mean")])
    vocabulary.freeze_metrics()

    with pytest.raises(RuntimeError, match="after the metric group was registered"):
        register_teed_metrics([TeedMetric("late", "rl.late.metric")])

    with pytest.raises(RuntimeError, match="after the metric group was registered"):
        register_recorded_metrics([RecordedMetric("rl.late.recorded")])


def test_recorded_rows_are_kept_apart_from_teed_rows():
    """They are declared separately and only meet in the spec list."""
    teed = TeedMetric("reward", "rl.reward.mean")
    recorded = RecordedMetric("rl.vllm.batch.duration", kind="histogram", unit="s")
    register_teed_metrics([teed])
    register_recorded_metrics([recorded])

    assert vocabulary.teed_metrics() == (teed,)
    assert vocabulary.recorded_metrics() == (recorded,)
    assert recorded.key == "vllm_batch_duration"
