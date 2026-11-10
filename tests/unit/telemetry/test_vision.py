# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_rl.telemetry.vision``.

The behaviour worth pinning is which of the three series survive a given batch
shape: the vision encoder has no span of its own, so these counts are the only
thing that makes a microbatch's duration interpretable, and a wrong count is
worse than a missing one.
"""

import sys
import types
from typing import Optional

import pytest

from nemo_rl.telemetry.vision import (
    RL_VISION_IMAGE_ROWS_METRIC,
    RL_VISION_IMAGES_METRIC,
    RL_VISION_ROWS_METRIC,
    RL_VISION_STAGE_ATTR,
    VISION_METRICS,
    VISION_STAGE_LOADER_BATCH,
    VISION_STAGE_TRAIN_BATCH,
    VISION_STAGE_TRAIN_MICROBATCH,
    record_vision_workload,
    traced_train_phase,
    vision_workload,
)


class FakePackedTensor:
    """Stands in for ``PackedTensor``'s one method this module reads.

    Not a subclass: the real constructor takes a tensor layout none of this
    needs, and the ``packed`` fixture makes it satisfy the ``isinstance`` guard
    anyway.
    """

    def __init__(self, counts: list[int]) -> None:
        self._counts = counts

    def logical_segment_counts_by_row(self) -> list[int]:
        return self._counts


@pytest.fixture(autouse=True)
def packed(monkeypatch):
    """Make ``FakePackedTensor`` pass the module's ``isinstance`` guard.

    Swaps the whole module in ``sys.modules`` rather than setting the attribute
    on the real one, because importing ``multimodal_utils`` pulls in torch and
    this suite is kept free of it — the same reason ``conftest`` parses
    ``algorithms/utils.py`` instead of importing it. ``vision_workload`` imports
    ``PackedTensor`` inside the call, so the stub is what it resolves.
    """
    stub = types.ModuleType("nemo_rl.data.multimodal_utils")
    stub.PackedTensor = FakePackedTensor
    monkeypatch.setitem(sys.modules, "nemo_rl.data.multimodal_utils", stub)


def workload(counts: Optional[list[int]], num_rows: int) -> Optional[dict[str, int]]:
    """``vision_workload`` over a batch whose ``pixel_values`` carries *counts*."""
    pixel_values = None if counts is None else FakePackedTensor(counts)
    return vision_workload({"pixel_values": pixel_values}, num_rows)


def test_row_aligned_batch_reports_all_three_series():
    assert workload([2, 0, 1], num_rows=3) == {
        RL_VISION_IMAGES_METRIC: 3,
        RL_VISION_IMAGE_ROWS_METRIC: 2,
        RL_VISION_ROWS_METRIC: 3,
    }


def test_packed_batch_reports_the_total_without_the_row_breakdown():
    """The case three of the five shipped Energon VLM recipes are in.

    Packing concatenates rows into one THD sequence, so the per-row counts stop
    corresponding to the batch's rows. The total still does, and dropping it
    too would leave those recipes reporting nothing at all.
    """
    assert workload([2, 0, 1], num_rows=1) == {RL_VISION_IMAGES_METRIC: 3}


def test_text_only_batch_is_a_zero_rather_than_a_gap():
    """A text-only microbatch is the baseline an image-heavy one is read against.

    Reporting it as absent would remove the comparison the whole series exists
    to support.
    """
    assert workload(None, num_rows=4) == {
        RL_VISION_IMAGES_METRIC: 0,
        RL_VISION_IMAGE_ROWS_METRIC: 0,
        RL_VISION_ROWS_METRIC: 4,
    }


def test_unreadable_pixel_values_emit_nothing():
    """A layout this cannot read is a gap, never a guess.

    A zero here would read as "this batch had no images", which is the one
    wrong answer that cannot be told apart from the truth downstream.
    """
    assert vision_workload({"pixel_values": object()}, 2) is None


class FakeSpan:
    """The span surface lens's ``safe_set_span_attributes`` touches."""

    def __init__(self) -> None:
        self.attributes: dict[str, int] = {}

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_attributes(self, attributes):
        self.attributes.update(attributes)


def test_record_stamps_the_span_and_the_metrics_from_one_count(monkeypatch):
    """Both sinks get the same numbers, so neither can drift from the other."""
    recorded: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.metrics.record_rl_metrics",
        lambda values, attributes=None: recorded.append((dict(values), attributes)),
    )
    span = FakeSpan()

    record_vision_workload(
        {"pixel_values": FakePackedTensor([1, 1])},
        2,
        stage=VISION_STAGE_TRAIN_MICROBATCH,
        span=span,
    )

    assert span.attributes[RL_VISION_IMAGES_METRIC] == 2
    assert recorded == [
        (
            {
                RL_VISION_IMAGES_METRIC: 2,
                RL_VISION_IMAGE_ROWS_METRIC: 2,
                RL_VISION_ROWS_METRIC: 2,
            },
            {RL_VISION_STAGE_ATTR: VISION_STAGE_TRAIN_MICROBATCH},
        )
    ]


def test_record_is_a_noop_on_an_unreadable_batch(monkeypatch):
    recorded: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.metrics.record_rl_metrics",
        lambda values, attributes=None: recorded.append((dict(values), attributes)),
    )
    span = FakeSpan()

    record_vision_workload(
        {"pixel_values": object()}, 2, stage=VISION_STAGE_LOADER_BATCH, span=span
    )

    assert not recorded
    assert not span.attributes


def test_record_without_a_span_still_records(monkeypatch):
    """The Energon loader passes a span that is None when its group is off."""
    recorded: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.metrics.record_rl_metrics",
        lambda values, attributes=None: recorded.append((dict(values), attributes)),
    )

    record_vision_workload(
        {"pixel_values": FakePackedTensor([3])},
        1,
        stage=VISION_STAGE_LOADER_BATCH,
        span=None,
    )

    assert recorded[0][1] == {RL_VISION_STAGE_ATTR: VISION_STAGE_LOADER_BATCH}


class FakeBatch(dict):
    """A ``BatchedDataDict`` stand-in: dict access plus ``.size``."""

    @property
    def size(self) -> int:
        return self["rows"]


def test_traced_train_phase_stamps_the_batch_it_was_called_with(monkeypatch):
    """The decorator is what carries coverage to every train entrypoint.

    Both backends' ``train`` and Megatron's ``train_microbatch`` go through it,
    so a batch it failed to read would silently blank the vision series on
    whichever path used it.
    """
    recorded: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.metrics.record_rl_metrics",
        lambda values, attributes=None: recorded.append((dict(values), attributes)),
    )
    seen = []

    class Worker:
        @traced_train_phase("rl.policy.train", stage=VISION_STAGE_TRAIN_BATCH)
        def train(self, data, loss_fn=None):
            seen.append((data, loss_fn))
            return "result"

    batch = FakeBatch(rows=2, pixel_values=FakePackedTensor([1, 0]))

    assert Worker().train(batch, loss_fn="fn") == "result"
    assert seen == [(batch, "fn")]
    assert recorded[0][0][RL_VISION_IMAGES_METRIC] == 1
    assert recorded[0][1] == {RL_VISION_STAGE_ATTR: VISION_STAGE_TRAIN_BATCH}


def test_traced_train_phase_accepts_the_batch_by_keyword(monkeypatch):
    """``train_microbatch_presharded`` dispatches ``data=`` rather than positionally."""
    recorded: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.metrics.record_rl_metrics",
        lambda values, attributes=None: recorded.append((dict(values), attributes)),
    )

    class Worker:
        @traced_train_phase(
            "rl.policy.train_microbatch", stage=VISION_STAGE_TRAIN_MICROBATCH
        )
        def train_microbatch(self, data):
            return None

    Worker().train_microbatch(data=FakeBatch(rows=1, pixel_values=None))

    assert recorded[0][1] == {RL_VISION_STAGE_ATTR: VISION_STAGE_TRAIN_MICROBATCH}


def test_traced_train_phase_keeps_a_var_keyword_signature():
    """What lets it stack above ``wrap_with_nvtx_name`` without breaking Ray.

    ``instrumentation._advertise_carrier`` leaves a callable alone when it
    already takes ``**kwargs``. If this wrapper ever narrowed its signature,
    Ray would start rejecting the trace carrier on every decorated actor
    method, and the failure would surface as a dispatch TypeError rather than
    as anything telemetry-shaped.
    """
    import inspect

    class Worker:
        @traced_train_phase("rl.policy.train", stage=VISION_STAGE_TRAIN_BATCH)
        def train(self, data):
            return None

    parameters = inspect.signature(Worker.train, follow_wrapped=False).parameters
    assert any(p.kind is p.VAR_KEYWORD for p in parameters.values())


def test_every_declared_series_is_one_vision_workload_can_produce():
    """A declared row with nothing writing it is an instrument that reports silence.

    The reverse direction is covered by ``vision_workload`` returning its keys
    by series name, which is what keeps the two from needing a mapping.
    """
    producible = set(workload([1], num_rows=1)) | set(workload(None, num_rows=1))
    assert {row.name for row in VISION_METRICS} == producible
