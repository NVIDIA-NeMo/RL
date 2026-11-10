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

"""How much vision work a batch carries, as spans attributes and metrics.

The vision encoder cannot be timed directly here. Both training backends make
one fused ``model(input_ids=..., **multimodal_data)`` call and leave the vision
tower to the upstream model, and the tower itself sits behind a different
attribute per family (``model.visual`` on Qwen-VL, ``vision_tower.vision_model``
on Gemma3, ``vision_model.radio_model`` on RADIO), so there is no call this
repository makes that contains the encoder and nothing else.

What is available is the *size* of the vision workload on each batch, which is
what makes encoder cost recoverable without a span on the encoder: with image
counts on the same span as the microbatch duration, a text-only microbatch is
the baseline that an image-heavy one is measured against. These are recorded
rather than teed, because both producers are worker-side and neither has a
``Logger`` — see ``metrics.record_rl_metrics``.

Nothing is emitted for a batch whose image layout cannot be read, so a family
this does not understand is a gap in the series rather than a wrong number in
it. Under sequence packing the image *total* is still readable while the per-row
breakdown is not, and the two halves are reported independently for that reason
-- see :func:`vision_workload`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, cast

from nemo_rl.telemetry.instrumentation import safe_set_span_attributes, umbrella_span
from nemo_rl.telemetry.span_groups import RLSpanGroup
from nemo_rl.telemetry.vocabulary import RecordedMetric, register_recorded_metrics

_F = TypeVar("_F", bound=Callable[..., Any])

#: Which producer a sample describes. One instrument per quantity with the stage
#: as an attribute, rather than one series per producer, because the question is
#: almost always how the two compare — whether the loader fed more images than
#: the step consumed — and that is a filter on one series, not a join of two.
RL_VISION_STAGE_ATTR = "rl.vision.stage"

#: One training microbatch, as one policy worker rank saw it. Emitted by the
#: split begin/microbatch/finish path, which SFT v2 and the single controller
#: drive.
VISION_STAGE_TRAIN_MICROBATCH = "train_microbatch"

#: One whole ``train()`` call on one rank, from the algorithms that hand the
#: worker a batch and let it do its own microbatching (GRPO, PPO, SFT v1).
#: Coarser than ``train_microbatch`` and never emitted alongside it for the
#: same work, so summing across stages would double count.
VISION_STAGE_TRAIN_BATCH = "train_batch"

#: One batch delivered by an Energon loader owner.
VISION_STAGE_LOADER_BATCH = "loader_batch"

RL_VISION_IMAGES_METRIC = "rl.vision.images"
RL_VISION_IMAGE_ROWS_METRIC = "rl.vision.image_rows"
RL_VISION_ROWS_METRIC = "rl.vision.rows"

VISION_METRICS = (
    RecordedMetric(
        RL_VISION_IMAGES_METRIC,
        unit="{image}",
        description=(
            "Images carried by one batch at the stage named by the "
            "rl.vision.stage attribute."
        ),
    ),
    RecordedMetric(
        RL_VISION_IMAGE_ROWS_METRIC,
        unit="{row}",
        description="Rows of that batch carrying at least one image.",
    ),
    RecordedMetric(
        # The denominator for the two above. Emitted alongside them rather than
        # left to be joined against a batch-size series, because the loader and
        # the policy worker count rows differently once packing is on and a
        # ratio built across that boundary would be wrong without saying so.
        RL_VISION_ROWS_METRIC,
        unit="{row}",
        description="Rows in that batch, whether or not they carry images.",
    ),
)
register_recorded_metrics(VISION_METRICS)

__all__ = [
    "RL_VISION_STAGE_ATTR",
    "RL_VISION_IMAGES_METRIC",
    "RL_VISION_IMAGE_ROWS_METRIC",
    "RL_VISION_ROWS_METRIC",
    "VISION_METRICS",
    "VISION_STAGE_LOADER_BATCH",
    "VISION_STAGE_TRAIN_BATCH",
    "VISION_STAGE_TRAIN_MICROBATCH",
    "vision_workload",
    "record_vision_workload",
    "traced_train_phase",
]


def vision_workload(batch: Any, num_rows: int) -> Optional[dict[str, int]]:
    """Images, and the row breakdown when rows still line up, in *batch*.

    Keyed by series name, so the result doubles as span attributes and as the
    ``record_rl_metrics`` payload without a second mapping between the two.
    ``None`` when the batch describes its images in a way this cannot read.

    Deliberately not ``data.multimodal_utils.image_counts_by_row``, which this
    otherwise mirrors: that helper returns ``None`` unless its per-row counts
    line up with the row count, because its caller masks against them and a
    misaligned count there would corrupt a batch. Nothing is masked here, and
    the alignment fails on exactly the configs most worth measuring — sequence
    packing concatenates rows into one THD sequence, so three of the five
    shipped Energon VLM recipes would report nothing at all. The total is
    well-defined either way, so it is reported either way, and only the per-row
    half is dropped when the rows no longer correspond.
    """
    # Imported here to keep this module free of torch: it is imported for its
    # metric declarations during telemetry startup, which happens before the
    # data stack is needed and in processes that may never load it.
    from nemo_rl.data.multimodal_utils import PackedTensor

    pixel_values = batch.get("pixel_values", None)
    if pixel_values is None:
        # A text-only batch genuinely carries no images; that is a zero rather
        # than an unreadable batch, and reporting it is what makes a text-only
        # microbatch usable as the baseline an image-heavy one is read against.
        return {
            RL_VISION_IMAGES_METRIC: 0,
            RL_VISION_IMAGE_ROWS_METRIC: 0,
            RL_VISION_ROWS_METRIC: num_rows,
        }
    if not isinstance(pixel_values, PackedTensor):
        return None
    counts = pixel_values.logical_segment_counts_by_row()
    workload = {RL_VISION_IMAGES_METRIC: sum(counts)}
    if len(counts) == num_rows:
        workload[RL_VISION_IMAGE_ROWS_METRIC] = sum(1 for count in counts if count)
        workload[RL_VISION_ROWS_METRIC] = num_rows
    return workload


def record_vision_workload(
    batch: Any, num_rows: int, *, stage: str, span: Optional[Any] = None
) -> None:
    """Record *batch*'s vision workload, and stamp it on *span* when given.

    Both sinks or neither: the span attributes are what make a duration
    comparable against the load that produced it, and the metrics are what make
    the load readable without sampling traces. A caller that wanted only one of
    them would be asking a question the other half already answers.
    """
    # Imported here rather than at module scope to keep this module's own
    # imports to ``vocabulary`` and ``instrumentation``. ``metrics`` imports
    # ``setup``, and this module is imported from the package ``__init__`` for
    # its declarations, so pulling that in would widen what every
    # ``import nemo_rl.telemetry`` costs.
    from nemo_rl.telemetry.metrics import record_rl_metrics

    workload = vision_workload(batch, num_rows)
    if workload is None:
        return
    # Deliberately not guarded: both call sites are already inside a span helper
    # that no-ops when its group is off, and record_rl_metrics no-ops when
    # nothing is exporting, so the cost on an untraced run is the count itself.
    safe_set_span_attributes(span, workload)
    record_rl_metrics(workload, {RL_VISION_STAGE_ATTR: stage})


def traced_train_phase(span_name: str, *, stage: str) -> Callable[[_F], _F]:
    """Span a worker's training entrypoint and stamp its batch's vision load.

    Stacks *above* ``wrap_with_nvtx_name`` rather than replacing it, so the
    boundaries an nsys profile shows and the ones lens shows stay the same
    ones. That order also keeps the Ray trace carrier working:
    ``instrumentation._advertise_carrier`` unwraps to the innermost
    ``__wrapped__``, and the NVTX wrapper sets none, so the unwrap stops at its
    bare ``**kwargs`` and the carrier is absorbed there instead of reaching a
    signature Ray would reject it against.

    ``data`` is named in the wrapper's signature rather than dug out of
    ``*args``: every entrypoint this decorates takes the batch first, and
    naming it keeps the dependency visible at the decorator instead of hidden
    behind an index. Callers pass it either way — ``train_presharded``
    positionally, ``train_microbatch_presharded`` by keyword.
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            with umbrella_span(RLSpanGroup.U_WORKER_TRAIN, span_name) as span:
                # Before the body, so a batch that fails still reports the
                # payload it failed on -- the one whose size is worth knowing.
                record_vision_workload(data, data.size, stage=stage, span=span)
                return func(self, data, *args, **kwargs)

        return cast(_F, wrapper)

    return decorator
