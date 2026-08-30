# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Lean per-op metrics decorator for ``DataPlaneClient``.

Wraps any ``DataPlaneClient`` and invokes a single user-provided
callback on each operation. Each event is a flat dict::

    {"op", "partition_id", "n_keys", "n_bytes", "wall_ms", "status"}

Plug wandb / file logging / debug print at the call site by passing
``on_event=<your function>``. ``snapshot()`` returns cumulative
totals **plus** live memory consumption: ``bytes_outstanding`` (sum of
bytes currently held in TQ, i.e. put minus cleared) and
``peak_bytes_outstanding`` (high-water mark over the run lifetime).

Every method here runs on the hot path of a transfer, so nothing traverses
a structure twice and nothing is allocated for a payload no callback reads.

``verify_tensor_hash=True`` adds an opt-in correctness check:
``torch.hash_tensor`` fingerprints recorded at put and re-checked at get,
so a tensor that changes between wire-in and wire-out is reported rather
than trained on. It reads every tensor byte again on both sides, so it is a
debugging tool, not a metric. See ``README.md`` for what it does and does
not catch.
"""

from __future__ import annotations

import logging
import zlib
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Collection, Literal, NamedTuple, TypedDict

EventStatus = Literal["ok", "error", "timeout"]


class DataPlaneEvent(TypedDict):
    op: str
    partition_id: str
    n_keys: int
    n_bytes: int
    wall_ms: float
    status: EventStatus


import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict, TensorDictBase

from nemo_rl.data_plane.codec import drain_codec_ms
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta

logger = logging.getLogger(__name__)


# Upper edges in ms for the latency histogram. Fixed buckets (rather than
# retained samples) keep memory O(1) per op and, crucially, make the counts
# *additive*: the 256 per-rank histograms sum into one cluster-wide
# distribution, which a mean or a per-rank percentile cannot do.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
)

# Ops that move payload, split by direction, for communication volume.
_WRITE_OPS = frozenset({"put"})
_READ_OPS = frozenset({"get", "get_data"})

# A corrupted wire usually corrupts every row of a batch, so the log is
# capped: the counter in ``HashStats`` carries the magnitude, and the first
# few lines carry the identity of what broke.
_MAX_HASH_MISMATCH_LOGS = 20

# Quantiles reported per op, each with the sample count it needs: enough for
# roughly four observations above the rank, or n >= 4 / (1 - q).
#
# The tail one is p90, not p99, because a step holds tens of calls, not
# thousands. A p99 needs ~100 samples before any observation lies above its
# rank at all, and below that it collapses onto the largest one -- measured
# over a lognormal-with-tail draw, a p99 off 58 calls equalled the maximum
# 80% of the time, which is ``max_ms`` under a more precise-sounding name.
# p90 off the same 58 never did on a smooth tail and 12% of the time on a
# bimodal one. A coarser quantile that is actually resolved beats a finer
# one that is not.
_QUANTILES = ((0.50, "p50_ms", 20), (0.90, "p90_ms", 40))


class _WriteScheme(NamedTuple):
    """How a field was reduced when it was written, so a read can replay it.

    A positional ``("scoped", n)`` / ``("rows", width)`` tuple carried two
    different quantities in one slot, discriminated by a magic string, under
    an annotation that said ``int``. Naming both makes each read site say
    which one it means.
    """

    batch_scoped: bool
    n_rows: int
    row_width: int


class _FieldDigest(NamedTuple):
    """One fingerprint per row, plus how far it can be trusted.

    ``batch_scoped`` means the values were derived from the whole batch's
    buffer, so they only reconcile against a read of that same batch. A
    shard of it computes a different buffer digest and must be reported
    unverified rather than as a mismatch.

    ``row_lens`` is how long each row was, which is the one thing still
    comparable when the read cannot reproduce the write's scheme.
    """

    per_row: list[int]
    batch_scoped: bool
    row_lens: tuple[int, ...]


# Same-width signed integer for each tensor element size, used to bitcast a
# leaf before hashing.
_INT_VIEW_BY_WIDTH = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _as_int_view(t: torch.Tensor) -> torch.Tensor:
    """Bitcast to a same-width integer type, or pass through.

    ``hash_tensor`` has no float8 kernel and raises there; viewing the bytes
    as integers sidesteps every dtype-specific kernel for free.
    """
    int_dtype = _INT_VIEW_BY_WIDTH.get(t.element_size())
    return t.view(int_dtype) if int_dtype is not None else t


def _as_list(sample_ids: Any) -> Any:
    """Materialize ``sample_ids`` once; ``None`` passes through.

    ``_run`` consumes its lambda and the accounting needs the same sequence
    afterwards, so a generator would be exhausted by the time it is indexed.
    """
    if sample_ids is None or isinstance(sample_ids, list):
        return sample_ids
    return list(sample_ids)


def _pop_partition_keys(
    store: dict[str, dict[str, Any]], partition_id: str, keys: list[str] | None
) -> list[Any]:
    """Drop ``keys`` from ``store[partition_id]``, returning what was removed.

    ``keys=None`` drops the whole partition. Shared by the byte accounting
    and the fingerprint store so their teardown cannot drift apart.
    """
    partition = store.get(partition_id)
    if partition is None:
        return []
    if keys is None:
        del store[partition_id]
        return list(partition.values())
    removed = [partition.pop(key) for key in keys if key in partition]
    if not partition:
        del store[partition_id]
    return removed


def _tensor_bytes(v: torch.Tensor) -> int:
    """Wire bytes of one tensor leaf, rectangular or nested.

    A nested tensor's ``nbytes`` dispatches through ``__torch_function__``;
    its packed values buffer answers the same question without dispatching,
    and every per-token field on this wire is nested.

    Two guards, because a wrong byte count is worse than a slow one:

    * ``_values`` is a bound *method* on every dense tensor, so the first
      check is on the type, not for absence — ``buf is None`` would be wrong.
    * A buffer holding more elements than the offsets describe means the
      tensor views a larger allocation (``torch.nested.narrow``), where the
      buffer overcounts. Nothing here builds one, but this is handed
      whatever a caller passes.
    """
    buf = getattr(v, "_values", None)
    if type(buf) is not torch.Tensor:
        return v.nbytes
    offsets = getattr(v, "_offsets", None)
    if offsets is not None and buf.shape[0] != int(offsets[-1]):
        return v.nbytes
    return buf.nbytes


def _dtype_salt(dtype: torch.dtype) -> int:
    """Salt distinguishing dtypes whose values reduce to the same words.

    ``crc32``, not the builtin ``hash()``: ``hash()`` of a ``str`` is salted
    per process, so the fingerprint would not survive being compared across
    ranks.
    """
    return zlib.crc32(str(dtype).encode())


def percentile_from_hist(hist: list[int], q: float) -> float:
    """Interpolated ``q``-quantile (0-1) from bucket counts.

    Linear interpolation inside the containing bucket. A value landing in
    the overflow bucket returns the top edge as a *lower bound* -- we know
    it exceeded 5 s but not by how much.
    """
    total = sum(hist)
    if total <= 0:
        return 0.0
    target = q * total
    cum = 0
    for i, count in enumerate(hist):
        if count and cum + count >= target:
            if i >= len(LATENCY_BUCKETS_MS):
                return LATENCY_BUCKETS_MS[-1]
            lo = 0.0 if i == 0 else LATENCY_BUCKETS_MS[i - 1]
            hi = LATENCY_BUCKETS_MS[i]
            return lo + (hi - lo) * ((target - cum) / count)
        cum += count
    return LATENCY_BUCKETS_MS[-1]


def _estimate_encoded_bytes(obj: Any, budget: list[int]) -> int:
    """Approximate msgpack-encoded size of a non-tensor object.

    TQ encodes non-tensors with msgpack (``serial_utils.batch_encode_into``),
    falling back to pickle/cloudpickle via ``Ext`` for unknown types. Getting
    the exact size means running that encoder, which would double the
    serialisation work on the hot path -- so this walks the structure and
    approximates instead. Container framing (1-5 bytes per element) is not
    modelled, so treat the result as a lower bound.

    ``budget`` bounds the walk to ``max_nodes`` container elements. Only the
    container branches charge it: a leaf cannot itself expand the walk.
    Containers stop iterating once it is exhausted -- summing a generator
    would otherwise keep walking every element while each recursive call
    returned 0, making the cost O(size) despite the budget.
    """
    if obj is None or isinstance(obj, bool):
        return 1
    if isinstance(obj, int):
        # msgpack packs small ints in a single byte; only wide values cost 9.
        if -32 <= obj < 128:
            return 1
        if -(2**15) <= obj < 2**16:
            return 3
        if -(2**31) <= obj < 2**32:
            return 5
        return 9
    if isinstance(obj, float):
        return 9
    if isinstance(obj, str):
        n = len(obj) if obj.isascii() else len(obj.encode("utf-8"))
        return n + (1 if n < 32 else 2 if n < 256 else 3 if n < 65536 else 5)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        n = len(obj)
        return n + (2 if n < 256 else 3 if n < 65536 else 5)
    if isinstance(obj, dict):
        n = len(obj)
        total = 1 if n < 16 else 3 if n < 65536 else 5
        for k, v in obj.items():
            if budget[0] <= 0:
                break
            budget[0] -= 1
            total += _estimate_encoded_bytes(k, budget)
            total += _estimate_encoded_bytes(v, budget)
        return total
    if isinstance(obj, (list, tuple, set)):
        n = len(obj)
        total = 1 if n < 16 else 3 if n < 65536 else 5
        for v in obj:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            total += _estimate_encoded_bytes(v, budget)
        return total
    if isinstance(obj, torch.Tensor):
        return _tensor_bytes(obj)
    # Unknown type -> pickle/cloudpickle Ext. Cheap proxy; the real size
    # would need an actual dumps(), which is what we are avoiding.
    return 64


# Rows sampled from a NonTensorStack to estimate its payload. The stack
# holds one Python object per batch element, so materialising it (``tolist``)
# and walking every row is O(batch) *per put*. Sampling assumes rows are
# exchangeable in size, which is only approximately true -- rollout rows
# differ in length by construction -- so this is a model, not a measurement.
_NONTENSOR_STACK_SAMPLES = 4


def _nontensor_stack_bytes(stack: NonTensorStack, budget: list[int]) -> int:
    """Extrapolate a ``NonTensorStack``'s payload from a strided row sample."""
    rows = getattr(stack, "tensordicts", None)
    if not rows:
        return _estimate_encoded_bytes(stack.tolist(), budget)
    n = len(rows)
    step = max(1, n // _NONTENSOR_STACK_SAMPLES)
    sampled = rows[::step][:_NONTENSOR_STACK_SAMPLES]
    sampled_bytes = 0
    for row in sampled:
        # Matched by type rather than ``getattr(row, "data", row)``: every
        # TensorDictBase carries a ``.data`` property of its own, so the
        # duck-typed form would silently hand a nested stack's tensor view to
        # the msgpack estimator instead of recursing into its payload.
        if isinstance(row, NonTensorData):
            sampled_bytes += _estimate_encoded_bytes(row.data, budget)
        elif isinstance(row, NonTensorStack):
            sampled_bytes += _nontensor_stack_bytes(row, budget)
        else:
            sampled_bytes += _estimate_encoded_bytes(row, budget)
    return sampled_bytes * n // len(sampled)


def _td_bytes(td: TensorDict | None, max_nodes: int = 10_000) -> int:
    """Payload bytes of a TensorDict, as the wire will see them.

    Tensor leaves count ``nbytes`` (see :func:`_tensor_bytes`), which is the
    size mooncake registers and sends.
    Non-tensor leaves are estimated with :func:`_estimate_encoded_bytes`,
    since TQ ships them over a separate msgpack path. Both kinds are counted
    in a single ``items()`` pass; ``keys()`` + ``get()`` would re-resolve
    every nested key from the root.

    ``leaves_only=True`` would hide the non-tensor entries entirely
    (``NonTensorData`` is not treated as a leaf), so this walks with
    ``leaves_only=False`` and skips container nodes itself.

    ``NonTensorData`` and ``NonTensorStack`` are matched by type rather than
    ``hasattr``, and the distinction matters: ``NonTensorData`` exposes BOTH
    ``.data`` and ``.tolist()``, and its ``.tolist()`` broadcasts the single
    stored object across the batch dim (a 64-row batch reported 20x the real
    payload).

    Aliased storage is counted per field: two keys viewing one buffer count
    twice, which is right for volume (both are serialised) and is what lets
    ``max_bytes_per_key_seen`` catch view-aliasing regressions.
    """
    if td is None:
        return 0
    budget = [max_nodes]
    total = 0
    for _, v in td.items(include_nested=True, leaves_only=False):
        if isinstance(v, torch.Tensor):
            total += _tensor_bytes(v)
        elif isinstance(v, NonTensorData):
            total += _estimate_encoded_bytes(v.data, budget)
        elif isinstance(v, NonTensorStack):
            # Checked before TensorDictBase: NonTensorStack subclasses
            # LazyStackedTensorDict but carries payload, so skipping it as a
            # container would drop those bytes entirely.
            total += _nontensor_stack_bytes(v, budget)
        elif isinstance(v, TensorDictBase):
            continue  # container; its leaves are visited separately
        else:
            total += _estimate_encoded_bytes(v, budget)
    return total


def _step_deltas(snap: dict[str, Any], prev: dict[str, Any]) -> dict[str, float]:
    """The three series both step-metric paths report, identically.

    Shared so the single-process and cluster views cannot drift on series
    names -- which is the whole point of the ``step/``/``now/`` convention
    they publish under.

    Write and read volume are deliberately not here. They were computed and
    then dropped by :func:`headline_series`, charted by nobody, while the
    breakdown table already carries per-op ``mb`` -- put's is the write
    volume and get's is the read volume, split finer than a global pair
    would be.
    """

    def _delta_s(field: str) -> float:
        """A millisecond accumulator differenced into the charted seconds.

        Every ``_s`` series goes through here so a new one cannot forget the
        conversion and chart milliseconds under a seconds name. Seconds
        because these sit beside ``timing/train/total_step_time``: a real
        step logged 78800.9 ms, which reads as noise against a 674 s clock.
        """
        return (snap[field] - prev.get(field, 0.0)) / 1e3

    return {
        "step/wall_s": _delta_s("total_wall_ms"),
        "step/comm_volume_mb": (
            snap["comm_volume_bytes"] - prev.get("comm_volume_bytes", 0)
        )
        / 1e6,
        "now/bytes_outstanding_mb": snap["bytes_outstanding"] / 1e6,
        "step/codec/pack_s": _delta_s("pack_ms"),
        "step/codec/unpack_s": _delta_s("unpack_ms"),
    }


def _row_len(digest: _FieldDigest, row: int) -> int:
    """How long ``row`` was, whichever scheme the digest used.

    A per-row digest stores the one width its uniform rows shared; a
    batch-scoped one stores every row's length.
    """
    if not digest.row_lens:
        return -1
    return digest.row_lens[row] if digest.batch_scoped else digest.row_lens[0]


def _op_step_stats(
    by_op: dict[str, Any], prev_ops: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """This step's per-op detail, keyed by op, from two snapshots.

    Shared by the single-process and cluster paths so the two cannot drift,
    and used for both the emitted percentages and the breakdown table -- one
    computation, so a chart and the table beside it can never disagree.

    ``max_ms`` comes from ``step_max_ms``, which the reader resets, rather
    than from the cumulative ``max_ms``: a maximum is not differenceable, so
    the cumulative one latches at the worst call ever seen and never comes
    back down. Ops with no calls this step are absent, not zero.
    """
    out: dict[str, dict[str, float]] = {}
    for op, st in by_op.items():
        prev_op = prev_ops.get(op, {})
        calls = st["calls"] - prev_op.get("calls", 0)
        if calls <= 0:
            continue
        op_ms = st["wall_ms"] - prev_op.get("wall_ms", 0.0)
        op_bytes = st["n_bytes"] - prev_op.get("n_bytes", 0)
        row: dict[str, float] = {
            "calls": calls,
            "wall_ms": op_ms,
            # Per call, which is the only form of this that describes the
            # wire rather than the shape of the run: ``wall_ms`` is summed
            # over concurrent processes and so scales with DP degree.
            "mean_ms": op_ms / calls,
            "max_ms": st.get("step_max_ms", 0.0),
            "mb": op_bytes / 1e6,
        }
        step_hist = [
            now - was
            for now, was in zip(
                st["latency_hist"],
                prev_op.get("latency_hist") or [0] * len(st["latency_hist"]),
            )
        ]
        row.update(_clamped_percentiles(step_hist, row["max_ms"]))
        out[op] = row
    return out


def _hash_deltas(hv: dict[str, int], prev_hv: dict[str, int]) -> dict[str, float]:
    """This step's hash-verification counters, or nothing if the guard is off.

    Shared by both step-metric paths. It was emitted only on the driver
    path, but ``_log_data_plane_metrics`` prefers the cluster path whenever
    the fan-out reaches more than one process -- which is every real run --
    so with ``verify_tensor_hash`` on, ``mismatches`` never reached the
    logger. A guard whose findings are not reported is not a guard.

    ``fields_skipped`` is here for the same reason it exists at all: a guard
    that quietly stops covering a field still reports zero mismatches, so
    the abstention count has to be visible beside the finding count.

    Args:
        hv: This step's cumulative ``hash_verify`` block.
        prev_hv: The previous step's, for differencing.

    Returns:
        ``step/hash/{counter}`` deltas, or ``{}`` when the guard never ran.
    """
    # ``guard_failures`` counts too: a guard that raised on the first put
    # records no rows, and gating on rows alone would make it look switched off.
    if not hv or not (hv.get("rows_recorded") or hv.get("guard_failures")):
        return {}
    deltas: dict[str, float] = {
        f"step/hash/{name}": hv[name] - prev_hv.get(name, 0)
        for name in (
            "rows_checked",
            "rows_recorded",
            "rows_unverified",
            "mismatches",
            "fields_skipped",
            "guard_failures",
        )
    }
    # Corruption of every row of every field in a step, repeated identically,
    # is not what a broken wire looks like -- it is what a broken guard looks
    # like. Both false alarms this check has produced had exactly this shape
    # (3584 mismatches against 1536 rows, unchanging), and both were the
    # guard's own bookkeeping. Say so rather than leaving a reader to decide
    # whether to believe a number that large.
    checked, bad = deltas["step/hash/rows_checked"], deltas["step/hash/mismatches"]
    if checked > 0 and bad >= checked:
        logger.warning(
            "data-plane hash: %d mismatches against %d rows checked this step. "
            "A rate that high is more likely a bug in the check than in the "
            "wire -- confirm against the per-sample lines before acting on it.",
            bad,
            checked,
        )
    return deltas


def _volume_mb(per_op: dict[str, dict[str, float]]) -> dict[str, float]:
    """Bytes each op moved this step, in MB, per op that moved any.

    ``comm_volume_mb`` is the total and hides the asymmetry that matters:
    on a real step ``get`` moved 20.8 MB against ``put``'s 2.7 MB, because
    every DP rank fetches its shard once for the logprob pass and again for
    the train pass. Those are separate transfers over the wire, not an
    accounting artifact, and the same is true of summing across processes --
    each rank pulls its own shard.

    Ops that carry no payload (``register``, ``clear``) are omitted rather
    than reported as zero, matching how the percentages treat an op that
    did not run.
    """
    return {
        f"step/volume_mb/by_op/{op}": row["mb"]
        for op, row in per_op.items()
        if row["mb"] > 0
    }


# Per-op detail lives under one namespace so it can be recognised by what it
# is rather than by what it is not. A deny-list of "middles that are not op
# tags" was a list against an open set: every later ``step/<x>/<field>``
# series -- queue depth, retry counts -- would have become a phantom row in
# the breakdown table beside put and get until someone remembered to extend
# the list.
_BY_OP = "step/by_op/"


# Namespaces that publish one value per op, and the table column each fills.
_BY_OP_NAMESPACES = {"percent_of_dataplane": "percent_of_dataplane", "volume_mb": "mb"}


def _op_series(by_op: dict[str, Any], prev_ops: dict[str, Any]) -> dict[str, float]:
    """Every per-op series for one step, from two snapshots.

    The two step-metric paths share this rather than each assembling the same
    keys: the helpers below exist so the single-process and cluster views
    cannot drift on series *names*, and duplicating the six lines that build
    those names one level up would have given the drift back.
    """
    per_op = _op_step_stats(by_op, prev_ops)
    metrics = _percent_of_dataplane(per_op)
    metrics.update(_volume_mb(per_op))
    for op, row in per_op.items():
        for field_name, value in row.items():
            if field_name != "mb":  # published once, under volume_mb/by_op
                metrics[f"{_BY_OP}{op}/{field_name}"] = value
    return metrics


def _percent_of_dataplane(per_op: dict[str, dict[str, float]]) -> dict[str, float]:
    """Where this step's data-plane time went, in percent.

    The name carries the denominator because that is the one thing a reader
    has to know before acting on the number: it is a percentage of *the
    data plane*, not of the step. ``by_op/put = 43`` reads "43% of the time
    spent inside the data plane went to put". Whether that time mattered at all against compute is a
    different question, answered by ``step/frac_of_step``, which divides by
    the step's own wall clock. A workload can be 43% put and still not be
    worth touching.

    ``by_op`` answers which call is expensive, and sums to 100 by
    construction.

    On the cluster path ``wall_ms`` is summed over processes that ran
    concurrently, so these are percentages of aggregate process-time rather
    than of elapsed time. That is the right denominator for "what should I
    optimise" and the wrong one for "what blocked the step".

    Args:
        per_op: Per-op step detail from :func:`_op_step_stats`.

    Returns:
        ``step/percent_of_dataplane/by_op/{op}`` in percent. Empty when no
        op ran.
    """
    total = sum(r["wall_ms"] for r in per_op.values())
    if total <= 0:
        return {}
    percent = {
        f"step/percent_of_dataplane/by_op/{op}": 100.0 * r["wall_ms"] / total
        for op, r in per_op.items()
    }
    return percent


def _clamped_percentiles(hist: list[int], max_ms: float) -> dict[str, float]:
    """Whichever of :data:`_QUANTILES` this sample can actually support.

    Two corrections, both needed wherever a percentile is taken off a coarse
    histogram. Each quantile is withheld until there are enough samples to
    resolve it: below that the interpolation returns bucket geometry rather
    than data -- one sample in (100, 250] yields a p50 of 175 whatever the
    call took. And the interpolation spreads a bucket's samples uniformly
    across it, so calls clustered low in a wide bucket read high, above the
    exact maximum measured beside them; the maximum is the tighter bound.

    Returns a dict rather than a fixed pair so a caller emits only what the
    data supports. An absent series says "not enough calls"; a zero would
    read as a measurement.
    """
    n = sum(hist)
    ceiling = max_ms if max_ms > 0 else float("inf")
    return {
        name: min(percentile_from_hist(hist, q), ceiling)
        for q, name, min_samples in _QUANTILES
        if n >= min_samples
    }


def _derive_op_metrics(by_op: dict[str, Any], total_wall_ms: float) -> None:
    """Fill in the derived per-op fields, in place.

    Shared by :meth:`MetricsDataPlaneClient.snapshot` and
    :func:`merge_snapshots` so a cluster-wide view is derived by exactly the
    same arithmetic as a single process -- percentiles off the (summed)
    histogram, the fit off the (summed) sufficient statistics. Nothing
    derived is ever averaged across processes.
    """
    for stats in by_op.values():
        calls = stats["calls"]
        wall_ms = stats["wall_ms"]
        stats["mean_ms"] = wall_ms / calls if calls else 0.0
        stats["mb_per_s"] = (
            (stats["n_bytes"] / 1e6) / (wall_ms / 1e3) if wall_ms else 0.0
        )
        stats["percent_of_total_ms"] = (
            100.0 * wall_ms / total_wall_ms if total_wall_ms else 0.0
        )
        hist = stats["latency_hist"]
        # Only what the sample supports; an absent key says "not enough
        # calls", which a zero would not.
        stats.update(_clamped_percentiles(hist, stats["max_ms"]))


# Snapshot fields that combine by summing, by taking a maximum, and the
# per-op ones of each kind. Everything else in a snapshot is derived and is
# recomputed from the merged totals rather than merged itself.
_SNAPSHOT_SUM = (
    "total_bytes",
    "total_keys",
    "total_ops",
    "total_wall_ms",
    "bytes_outstanding",
    "peak_bytes_outstanding",
    "n_keys_outstanding",
    "self_ms",
    "pack_ms",
    "unpack_ms",
)
_SNAPSHOT_MAX = ("max_bytes_per_key_seen", "last_put_bytes_per_key")
_OP_SUM = (
    "calls",
    "errors",
    "wall_ms",
    "n_bytes",
    "n_keys",
)
_OP_MAX = ("max_ms", "step_max_ms")


def merge_snapshots(snapshots: "list[dict[str, Any]]") -> dict[str, Any]:
    """Combine per-process snapshots into one cluster-wide view.

    This is what the accumulators were shaped for. Latency lives in fixed
    histogram buckets precisely so they *add*: summing 256 per-rank
    histograms gives the true cluster distribution, which averaging 256
    per-rank percentiles cannot. Everything derived — percentiles,
    throughput — is recomputed from the merged totals, never averaged.

    Counters sum. ``max_*`` fields take a maximum. ``peak_bytes_outstanding``
    is the one approximation: summing per-process peaks assumes they
    coincided, so it is an upper bound on true cluster peak occupancy.

    Args:
        snapshots: One :meth:`MetricsDataPlaneClient.snapshot` per process.

    Returns:
        A snapshot-shaped dict covering every process, plus ``n_processes``.
    """
    if not snapshots:
        return {}
    merged: dict[str, Any] = {k: 0 for k in _SNAPSHOT_SUM}
    merged.update({k: 0 for k in _SNAPSHOT_MAX})
    hashes = {
        k: 0
        for k in (
            "rows_recorded",
            "rows_checked",
            "rows_unverified",
            "mismatches",
            "fields_skipped",
            "guard_failures",
        )
    }
    by_op: dict[str, dict[str, Any]] = {}

    for snap in snapshots:
        for key in _SNAPSHOT_SUM:
            merged[key] += snap.get(key, 0)
        for key in _SNAPSHOT_MAX:
            merged[key] = max(merged[key], snap.get(key, 0))
        for key in hashes:
            hashes[key] += (snap.get("hash_verify") or {}).get(key, 0)
        for op, stats in (snap.get("by_op") or {}).items():
            acc = by_op.setdefault(
                op,
                {
                    **{k: 0 for k in _OP_SUM},
                    **{k: 0.0 for k in _OP_MAX},
                    "latency_hist": [0] * (len(LATENCY_BUCKETS_MS) + 1),
                },
            )
            for key in _OP_SUM:
                acc[key] += stats.get(key, 0)
            for key in _OP_MAX:
                acc[key] = max(acc[key], stats.get(key, 0.0))
            for i, count in enumerate(stats.get("latency_hist") or []):
                acc["latency_hist"][i] += count

    merged["by_op"] = by_op
    merged["hash_verify"] = hashes
    merged["n_processes"] = len(snapshots)
    _derive_op_metrics(by_op, merged["total_wall_ms"])
    merged["bytes_written"] = sum(by_op[o]["n_bytes"] for o in _WRITE_OPS if o in by_op)
    merged["bytes_read"] = sum(by_op[o]["n_bytes"] for o in _READ_OPS if o in by_op)
    merged["comm_volume_bytes"] = merged["bytes_written"] + merged["bytes_read"]
    return merged


def cluster_step_metrics(
    merged: dict[str, Any],
    prev: dict[str, Any],
    step_time_s: float,
    collect_ms: float = 0.0,
) -> dict[str, float]:
    """Per-step cluster metrics from two merged snapshots.

    The single-process equivalent of this lives on the client, which owns
    its own previous reading. A cluster has no such owner, so the caller
    holds ``prev`` and passes it back.

    ``observability_overhead_ms`` is the whole bill for measuring: every
    process's wrapper time plus ``collect_ms``, the fan-out that gathered
    the snapshots. The fan-out is the larger half; omitting it understates
    by an order of magnitude.

    Args:
        merged: Cluster-wide snapshot from :func:`merge_snapshots`.
        prev: The previous merged snapshot, for differencing.
        step_time_s: Step wall time, for ``frac_of_step``.
        collect_ms: Wall time the caller spent gathering and merging.
    """
    wall_ms = merged["total_wall_ms"] - prev.get("total_wall_ms", 0.0)
    overhead_ms = merged["self_ms"] - prev.get("self_ms", 0.0) + collect_ms
    n_procs = max(merged.get("n_processes", 1), 1)
    # step/ is a delta over this step; now/ is a level at this instant.
    # The unit alone does not distinguish them -- see README.md.
    metrics = _step_deltas(merged, prev)
    metrics.update(
        {
            # The one metric that says whether optimising the data plane is
            # worth anything: per-op percentages say where its time went, never
            # whether it mattered against compute. ``wall_ms`` sums
            # processes that ran concurrently, so dividing it by one step's
            # wall clock exceeds 1 whenever they overlapped (measured 1.054
            # across ten processes) -- correct arithmetic that reads as
            # "105% of the step". Per process it is bounded and answers the
            # question people actually ask of it.
            "step/frac_of_step": (
                (wall_ms / n_procs) / (step_time_s * 1e3) if step_time_s > 0 else 0.0
            ),
            "now/n_processes": n_procs,
            "step/self/overhead_ms": overhead_ms,
            "step/self/frac": overhead_ms / wall_ms if wall_ms > 0 else 0.0,
        }
    )
    metrics.update(
        _hash_deltas(merged.get("hash_verify") or {}, prev.get("hash_verify") or {})
    )
    metrics.update(_op_series(merged["by_op"], prev.get("by_op", {})))
    return metrics


# What goes on a chart. Everything else this module computes is per-op
# detail, which belongs in the breakdown table beside it: four ops times
# eight fields is 32 series saying one thing, and a dashboard of 32 lines
# does not answer "what is my bottleneck" -- a table sorted by time does.
# The full dict is still returned, so the table and the series are derived
# from one computation and cannot disagree.
_HEADLINE = (
    "step/wall_s",
    "step/frac_of_step",
    "step/comm_volume_mb",
    "now/bytes_outstanding_mb",
    "now/n_processes",
)
_HEADLINE_PREFIXES = (
    "step/percent_of_dataplane/",
    "step/volume_mb/",
    "step/hash/",
    "step/self/",
    "step/codec/",
)


def headline_series(metrics: dict[str, float]) -> dict[str, float]:
    """The subset of ``metrics`` worth a time series.

    Args:
        metrics: A flat dict from :func:`cluster_step_metrics` or
            :meth:`MetricsDataPlaneClient.get_step_metrics`.

    Returns:
        Totals, time percentages, and hash counters -- the per-op detail is
        dropped, since :func:`breakdown_table` presents it better.
    """
    return {
        k: v
        for k, v in metrics.items()
        if k in _HEADLINE or k.startswith(_HEADLINE_PREFIXES)
    }


# Per-op columns worth a row in the breakdown, in the order they read.
# ``p50_ms``/``p90_ms`` are present only above the sample gate, so a row
# carries None where a series was withheld rather than a zero that would
# read as a measurement.
_BREAKDOWN_COLUMNS = (
    "percent_of_dataplane",
    "calls",
    "wall_ms",
    "mean_ms",
    "max_ms",
    "p50_ms",
    "p90_ms",
    "mb",
)


def breakdown_table(
    metrics: dict[str, float],
) -> tuple[list[str], list[list[Any]]]:
    """Reshape the flat per-op series into one row per op.

    A stack of line charts answers "how did put's wall time trend"; the
    question this feeds is "where did this step's time go, across ops, at a
    glance" -- which is a table, and reading it off eight separate charts is
    the wrong tool. Rows are ordered by their share of data-plane time, so
    the
    bottleneck is the first line read.

    Built from the metrics dict that is logged rather than from the snapshot
    it came from, so the table and the series can never disagree: a value
    withheld from the series (a percentile below the sample gate, a fit that
    is not trustworthy) is absent from the table too.

    Args:
        metrics: A flat ``step/{op}/{field}`` dict from
            :meth:`MetricsDataPlaneClient.get_step_metrics` or
            :func:`cluster_step_metrics`.

    Returns:
        ``(columns, rows)`` for :meth:`Logger.log_table`.
    """
    per_op: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        parts = key.split("/")
        if len(parts) == 4 and parts[0] == "step" and parts[2] == "by_op":
            column = _BY_OP_NAMESPACES.get(parts[1])
            if column:
                per_op.setdefault(parts[3], {})[column] = value
        elif (
            len(parts) == 4
            and parts[:2] == ["step", "by_op"]
            and parts[3] in _BREAKDOWN_COLUMNS
        ):
            per_op.setdefault(parts[2], {})[parts[3]] = value
    rows = [
        [op, *(stats.get(col) for col in _BREAKDOWN_COLUMNS)]
        # By wall time, which orders identically to ``percent_of_dataplane`` (that is
        # wall time over a common total) and is present even when the
        # percentages are not -- a table built from a partial metrics dict
        # still reads worst-first.
        for op, stats in sorted(
            per_op.items(), key=lambda kv: -kv[1].get("wall_ms", 0.0)
        )
    ]
    return ["op", *_BREAKDOWN_COLUMNS], rows


def log_event(event: DataPlaneEvent) -> None:
    logger.info("data_plane_event: %s", event)


@dataclass
class OpStats:
    """Per-op-tag accumulation. ``calls``/``wall_ms`` count every status.

    ``n_bytes``/``n_keys`` count successful calls only, matching the
    cumulative totals — a failed transfer moved no payload, but the time
    it burned is still time the data plane cost the step.
    """

    calls: int = 0
    errors: int = 0
    wall_ms: float = 0.0
    n_bytes: int = 0
    n_keys: int = 0
    # Slowest single call, exact. The histogram below can only place a
    # call in a bucket, so at the handful of calls an op makes in one step
    # a percentile off it is bucket geometry rather than data -- a tail
    # quantile of one
    # sample in (10, 25] is always 10 + 15*0.99 = 24.85. This is the
    # per-step tail signal; the histogram is for the cumulative view.
    max_ms: float = 0.0
    # Same, but scoped to the current step: ``get_step_metrics`` zeroes it
    # each time it reports. Without this the per-step series is the lifetime
    # max, which is monotonic and goes flat the moment the worst call has
    # been seen -- the same defect as logging a cumulative percentile.
    step_max_ms: float = 0.0
    # Latency distribution over ALL statuses, matching calls/wall_ms: a
    # timeout is real tail latency the pipeline actually paid for.
    latency_hist: list[int] = field(
        default_factory=lambda: [0] * (len(LATENCY_BUCKETS_MS) + 1)
    )


@dataclass
class HashStats:
    """Wire-in / wire-out fingerprint reconciliation. All zero unless enabled.

    ``rows_unverified`` is as important as ``mismatches``: a run that reads
    back rows this process never wrote (the normal case for a consumer-side
    client, which sees only wire-out) verifies nothing, and a mismatch count
    of 0 would otherwise read as "checked and clean".
    """

    rows_recorded: int = 0
    rows_checked: int = 0
    rows_unverified: int = 0
    mismatches: int = 0
    # Leaves that carry no comparable row fingerprint: nested tensors (no
    # uniform row shape) and leaves whose leading dim doesn't match the
    # sample count, so a row cannot be attributed to a sample id.
    fields_skipped: int = 0
    # Batches the guard raised on, and so never checked. Same "reads as clean
    # because it checked nothing" hazard as ``fields_skipped``, counted for
    # the same reason. Not ``errors``: ``OpStats.errors`` already means failed
    # transfers, and these are failures of the check, not of the wire.
    guard_failures: int = 0


@dataclass
class DataPlaneStats:
    total_bytes: int = 0
    total_keys: int = 0
    total_ops: int = 0
    # Aggregate wall time across every data-plane call, all statuses. This
    # is the "what did the data plane cost us" number; ``by_op`` splits it.
    total_wall_ms: float = 0.0
    by_op: dict[str, OpStats] = field(default_factory=dict)
    bytes_outstanding: int = 0
    peak_bytes_outstanding: int = 0
    # Anomaly trackers — a wire-format regression that bloats bytes per
    # row (cf. message_log view-aliasing pickle bug) shows up as a
    # sudden spike in ``max_bytes_per_key_seen``.
    max_bytes_per_key_seen: int = 0
    last_put_bytes_per_key: int = 0
    # What measuring cost. Wall time spent inside this wrapper minus the
    # time the inner client was actually working, so a reader can see the
    # observability bill next to the thing it is observing rather than
    # taking a benchmark's word for it.
    self_ms: float = 0.0
    # Jagged pad/unpad CPU cost, drained from the codec timer. Packing runs
    # in the caller before ``put_samples`` and so is invisible to ``by_op``;
    # unpacking runs inside the adapter's ``get_samples`` and is otherwise
    # billed as transport. Kept out of ``total_wall_ms`` so ``frac_of_step``
    # and ``percent_of_dataplane`` keep meaning time spent in the data plane.
    #
    # Same coverage gap as ``comm_volume`` and for the same reason: only a
    # process that drains the codec timer reports its own pad cost, and the
    # rollout actor is not on the policy worker group the fan-out reaches. So
    # ``pack_ms`` omits ``kv_first_write``, the largest single pack in the job.
    # The single-controller path has no fan-out at all, so there it is
    # driver-only on both counters.
    pack_ms: float = 0.0
    unpack_ms: float = 0.0
    hash_verify: HashStats = field(default_factory=HashStats)


class MetricsDataPlaneClient(DataPlaneClient):
    """Wrap a ``DataPlaneClient`` with a per-op callback hook."""

    def __init__(
        self,
        inner: DataPlaneClient,
        on_event: Callable[[DataPlaneEvent], None] | None = None,
        verify_tensor_hash: bool = False,
    ) -> None:
        """Wrap ``inner``, accumulating per-op timing and volume.

        Args:
            inner: The client whose calls are measured.
            on_event: Per-op callback. ``None`` (the default) skips
                building the event dict entirely — with metrics enabled but
                no sink, nothing is paid for a payload nobody reads.
            verify_tensor_hash: Record a per-row ``torch.hash_tensor``
                fingerprint on put and re-check it on get. Debug aid, not a
                metric: it reads every tensor byte again (~2.4 ms for a
                12 MB jagged batch), so it is off unless the config asks.
        """
        self._inner = inner
        self._on_event = on_event
        self._verify_tensor_hash = verify_tensor_hash
        self._stats = DataPlaneStats()
        # Live bytes and live keys per partition. Populated on successful
        # ``put_samples``, released on successful ``clear_samples``. Bounded
        # by the live key population, not by cumulative traffic.
        self._bytes_by_partition: dict[str, int] = {}
        self._keys_by_partition: dict[str, set[str]] = {}
        # partition -> sample_id -> field -> wire-in fingerprint. Same
        # lifetime as ``_bytes_by_partition``: cleared by ``clear_samples``,
        # so it is bounded by the live key population.
        self._hash_by_partition: dict[str, dict[str, dict[str, int]]] = {}
        # partition -> (batch-scoped field names, row count at put). Those
        # digests cover a whole buffer, so they only reconcile against a read
        # of that same batch; the row count is what detects a shard read.
        # partition -> field -> rows the field's digest was reduced over,
        # for batch-scoped fields only. An absent field was reduced per row.
        self._batch_scope: dict[str, dict[str, _WriteScheme]] = {}
        self._hash_mismatches_logged = 0
        # Set by ``_emit`` to the inner client's wall time for the op just
        # run, so the wrapping methods can subtract it and bill the rest to
        # ``self_ms``.
        self._last_inner_ms = 0.0
        # Previous snapshot, for per-step deltas. Owned here rather than by a
        # caller: it is this client's prior reading, and keeping it here lets
        # every trainer use get_step_metrics() without copying the
        # differencing and unit-conversion logic.
        self._prev_snapshot: dict[str, Any] = {}

    def snapshot(self, reset_step_window: bool = False) -> dict[str, Any]:
        """Return cumulative totals plus live byte / key outstanding counts.

        ``total_wall_ms`` is the aggregate data-plane cost; ``by_op`` breaks
        it down per op tag with derived ``mean_ms`` and ``mb_per_s`` so the
        backends can be compared without post-processing. Throughput is
        omitted for ops that move no payload (e.g. ``claim_meta``, whose
        wall time is producer wait, not transfer).

        Args:
            reset_step_window: Zero each op's ``step_max_ms`` after reading
                it, opening a fresh window. A maximum cannot be differenced
                out of a cumulative counter the way ``calls`` and
                ``wall_ms`` can, so the only way to scope one to a step is
                to reset it -- and the reader that consumes it is the one
                that has to. Left off by default so an inspection snapshot
                never disturbs the step series.
        """
        # Gated on reset_step_window for the same reason step_max_ms is: the
        # codec timer is drained destructively, so an inspection snapshot that
        # took it would delete that time from the series the step reader
        # reports. Both callers that consume a step pass True.
        if reset_step_window:
            codec = drain_codec_ms()
            self._stats.pack_ms += codec.get("pack", 0.0)
            self._stats.unpack_ms += codec.get("unpack", 0.0)
        out = asdict(self._stats)
        out["n_keys_outstanding"] = sum(
            len(k) for k in self._keys_by_partition.values()
        )
        _derive_op_metrics(out["by_op"], self._stats.total_wall_ms)
        # Communication volume, derived from by_op so there is one source of
        # truth for bytes. Distinct from ``bytes_outstanding``, which is
        # occupancy (what is held) rather than traffic (what moved).
        by = out["by_op"]
        out["bytes_written"] = sum(by[o]["n_bytes"] for o in _WRITE_OPS if o in by)
        out["bytes_read"] = sum(by[o]["n_bytes"] for o in _READ_OPS if o in by)
        out["comm_volume_bytes"] = out["bytes_written"] + out["bytes_read"]
        if reset_step_window:
            for bucket in self._stats.by_op.values():
                bucket.step_max_ms = 0.0
        return out

    def get_step_metrics(self, step_time_s: float) -> dict[str, float]:
        """Per-step data-plane metrics, as a ready-to-log flat dict.

        Cumulative counters are differenced against the previous call, so this
        reports what the data plane cost *this* step. Mirrors
        ``VllmGeneration.get_step_metrics`` so trainers stay one line.

        ``frac_of_step`` is the metric that decides whether optimising the
        data plane is worth anything: ``percent_of_dataplane`` only says where
        data-plane time went, never whether it mattered against compute.
        """
        # Reading the step maxima is what closes the window: the values
        # just read are this step's, and anything after belongs to the next.
        snap = self.snapshot(reset_step_window=True)
        prev = self._prev_snapshot
        self._prev_snapshot = snap

        wall_ms = snap["total_wall_ms"] - prev.get("total_wall_ms", 0.0)
        # Units are not mixed within one chart: durations charted beside the
        # step clock are seconds, the per-op table is ms throughout, volumes
        # are always MB. GB was the same problem one dimension over -- a
        # realistic step moved 0.00017 GB.
        metrics = _step_deltas(snap, prev)
        metrics["step/frac_of_step"] = (
            (wall_ms / 1e3 / step_time_s) if step_time_s > 0 else 0.0
        )
        metrics.update(
            _hash_deltas(snap.get("hash_verify") or {}, prev.get("hash_verify") or {})
        )
        # The same bill the cluster path reports, under the same name: this
        # process's wrapper time, minus what the inner client was doing.
        # There is no fan-out to add here -- a single process gathers
        # nothing -- so this is the whole of it.
        self_ms = snap["self_ms"] - prev.get("self_ms", 0.0)
        metrics["step/self/overhead_ms"] = self_ms
        metrics["step/self/frac"] = self_ms / wall_ms if wall_ms > 0 else 0.0
        metrics.update(_op_series(snap["by_op"], prev.get("by_op", {})))
        return metrics

    def _record_put(self, partition_id: str, keys: list[str], n_bytes: int) -> None:
        """Attribute put bytes per key so a later ``clear_samples`` can subtract.

        Called after the underlying RPC succeeds so a failed put never
        leaves the accounting inflated.

        ``n_bytes`` is a whole-batch figure, so there was never a per-key
        truth to keep: the old per-key dict stored an even split, and a
        subset clear released the mean either way. Holding one total and one
        key set says the same thing and lets ``set.update`` do the per-key
        work in C — 18.6 us to 3.0 us at 256 keys, which was the single
        largest remaining cost on the put path.

        Args:
            partition_id: Partition the keys were written to.
            keys: Per-sample uids that were written.
            n_bytes: Total bytes written; released pro rata on clear.
        """
        if not keys or n_bytes <= 0:
            return
        self._keys_by_partition.setdefault(partition_id, set()).update(keys)
        self._bytes_by_partition[partition_id] = (
            self._bytes_by_partition.get(partition_id, 0) + n_bytes
        )
        self._stats.bytes_outstanding += n_bytes
        if self._stats.bytes_outstanding > self._stats.peak_bytes_outstanding:
            self._stats.peak_bytes_outstanding = self._stats.bytes_outstanding

    def _record_clear(self, partition_id: str, keys: list[str] | None) -> None:
        """Reverse the put accounting for ``keys``.

        Called after the underlying RPC succeeds so a failed clear keeps
        the accounting consistent with TQ's actual state.

        Bytes are released pro rata: the partition's total times the share
        of its live keys being dropped. Clearing the last key releases the
        remainder exactly, so a partition always reconciles to zero however
        it is chopped up.

        Args:
            partition_id: Partition the keys were dropped from.
            keys: Uids dropped; ``None`` means the whole partition was cleared.
        """
        if self._verify_tensor_hash:
            _pop_partition_keys(self._hash_by_partition, partition_id, keys)
            if keys is None:
                self._batch_scope.pop(partition_id, None)
        live = self._keys_by_partition.get(partition_id)
        if live is None:
            return
        total = self._bytes_by_partition.get(partition_id, 0)
        # Count what was actually live, not what the caller listed. A clear
        # may name uids already dropped or belonging elsewhere, and billing
        # those released bytes this partition never held: clearing 50 live
        # keys alongside 50 unknown ones freed two thirds of a partition
        # that had lost half its keys.
        if keys is None:
            removed = len(live)
        else:
            dropped = live.intersection(keys)
            live -= dropped
            removed = len(dropped)
        if keys is None or not live:
            freed = total
            del self._keys_by_partition[partition_id]
            self._bytes_by_partition.pop(partition_id, None)
        else:
            freed = total * removed // (len(live) + removed) if removed else 0
            self._bytes_by_partition[partition_id] = total - freed
        self._stats.bytes_outstanding -= freed

    def _bill_self(self, entered: float) -> None:
        """Charge this wrapper for the time it spent that was not the RPC.

        One ``monotonic`` per op on top of the two ``_run`` already takes.
        Measuring the measurement is worth that: the alternative is asking a
        reader to trust a benchmark run on some other machine.
        """
        elapsed_ms = (monotonic() - entered) * 1000.0
        self._stats.self_ms += elapsed_ms - self._last_inner_ms

    # ── wire-in / wire-out fingerprinting (opt-in) ─────────────────────

    def _row_fingerprints(
        self,
        td: TensorDict | None,
        sample_ids: list[str],
        batch_scoped_fields: Collection[str] = (),
    ) -> dict[str, _FieldDigest]:
        """``torch.hash_tensor`` fingerprints for each tensor leaf.

        A rectangular leaf reduces per row (``dim=1``), which names the
        sample that diverged. A jagged leaf whose rows happen to be uniform
        is rectangular already — its values buffer reshapes to the rectangle
        as a view — so it takes the same path for free. Only a genuinely
        ragged leaf falls back to one digest over its whole values buffer,
        XORed per row with that row's length, and marked ``batch_scoped``;
        padding it out to a rectangle costs far more than the answer is
        worth. That fallback inherits the blind spot of an XOR reduction: it
        sees any change to the multiset of values, but not a permutation of
        them. ``README.md`` has the detection/attribution table.

        Args:
            td: Leaves to fingerprint; ``None`` yields an empty result.
            sample_ids: Row *i* is attributed to ``sample_ids[i]``, the
                ordering :meth:`DataPlaneClient.get_samples` promises.
            batch_scoped_fields: Fields the *put* side reduced batch-scoped.
                The scheme must follow the field, not the layout in hand: a
                field packed jagged comes back dense whenever its rows are
                uniform (``_from_wire`` densifies those), and choosing per
                layout makes the two sides compute different things. The
                values buffer of a uniform jagged field and the flattened
                dense tensor it densifies into hold the same elements in the
                same order, so replaying the recorded scheme agrees by
                construction.

        Returns:
            Field name -> :class:`_FieldDigest`. Leaves that cannot be
            attributed per row (a leading dim that isn't ``len(sample_ids)``,
            or a non-``jagged`` nested layout) are counted in
            ``fields_skipped`` rather than silently dropped.
        """
        if td is None:
            return {}
        n_rows = len(sample_ids)
        stats = self._stats.hash_verify
        out: dict[str, _FieldDigest] = {}
        for key, v in td.items(include_nested=True, leaves_only=True):
            if not isinstance(v, torch.Tensor) or v.ndim < 1:
                stats.fields_skipped += 1
                continue
            salt = _dtype_salt(v.dtype)
            name = key if isinstance(key, str) else ".".join(key)
            if v.is_nested:
                if v.layout != torch.jagged:
                    stats.fields_skipped += 1
                    continue
                offsets = v.offsets()
                if offsets.numel() - 1 != n_rows:
                    stats.fields_skipped += 1
                    continue
                lengths = (offsets[1:] - offsets[:-1]).tolist()
                # Uniform rows: the values buffer already *is* the rectangle,
                # so reshaping it is a view and the per-row reduction is free.
                uniform = lengths.count(lengths[0]) == n_rows
                rectangle = v.values() if uniform else None
            elif v.shape[0] != n_rows:
                stats.fields_skipped += 1
                continue
            else:
                lengths = [v.shape[1] if v.ndim >= 2 else 1] * n_rows
                rectangle = v
            if rectangle is not None and name not in batch_scoped_fields:
                flat = _as_int_view(rectangle.reshape(n_rows, -1))
                out[name] = _FieldDigest(
                    # Salt on the host, as the batch-scoped path below already
                    # does. Torch has no ``bitwise_xor`` CUDA kernel for
                    # UInt64, so XOR-ing the digest tensor in place raises
                    # ``NotImplementedError`` for any backend whose get returns
                    # device tensors. The digests come back to the host for
                    # comparison either way, so this costs nothing.
                    [d ^ salt for d in torch.hash_tensor(flat, dim=1).tolist()],
                    batch_scoped=False,
                    # One width, not n_rows copies of it: the rows are
                    # uniform by construction on this path.
                    row_lens=(flat.shape[1],),
                )
                continue
            buffer = v.values() if v.is_nested else v
            flat_buffer = _as_int_view(buffer.reshape(1, -1))
            buffer_digest = torch.hash_tensor(flat_buffer, dim=1).tolist()[0] ^ salt
            out[name] = _FieldDigest(
                [buffer_digest ^ length for length in lengths],
                batch_scoped=True,
                row_lens=tuple(lengths),
            )
        return out

    def _hash_guard_failed(self, op: str, exc: Exception) -> None:
        """Absorb a hash-guard failure: count it, log it, never re-raise.

        The guard is a debug aid on a transfer that already succeeded, so a
        bug in it must not take the transfer down. Swallowing is only safe
        because the failure stays visible in ``step/hash/guard_failures`` -- a
        guard that silently stopped checking would report zero mismatches.
        """
        self._stats.hash_verify.guard_failures += 1
        # Logged once, not capped at a handful: whatever makes the guard raise
        # on one batch makes it raise on every batch, so line two would carry
        # nothing line one did not. The count is the series.
        if self._stats.hash_verify.guard_failures == 1:
            logger.warning(
                "data-plane hash guard failed on %s (%s: %s). The transfer "
                "itself is unaffected, but this batch went unchecked -- see "
                "step/hash/guard_failures for how many.",
                op,
                type(exc).__name__,
                exc,
            )

    def _record_hashes(
        self, partition_id: str, sample_ids: list[str], fields: TensorDict | None
    ) -> None:
        """Store wire-in fingerprints for a successful put. Never raises."""
        try:
            self._record_hashes_impl(partition_id, sample_ids, fields)
        except Exception as exc:  # noqa: BLE001 - a debug check must never fail a transfer
            self._hash_guard_failed("put", exc)

    def _record_hashes_impl(
        self, partition_id: str, sample_ids: list[str], fields: TensorDict | None
    ) -> None:
        """Store wire-in fingerprints for a successful put."""
        digests = self._row_fingerprints(fields, sample_ids)
        if not digests:
            return
        partition_hashes = self._hash_by_partition.setdefault(partition_id, {})
        for row, sample_id in enumerate(sample_ids):
            per_field = partition_hashes.setdefault(sample_id, {})
            for name, digest in digests.items():
                per_field[name] = digest.per_row[row]
        # Per field, not per partition: a delta put (write_columns) names only
        # the fields it writes, and must not restate the scheme of the ones it
        # left alone. Recording the batch it was reduced over lets the read
        # side tell a shard of a batch-scoped field from a relayout.
        scheme = self._batch_scope.setdefault(partition_id, {})
        for name, digest in digests.items():
            # Width matters only for the per-row scheme -- uniform rows mean
            # one number describes them all, and lets a later ragged read
            # still check whether any row changed length.
            scheme[name] = _WriteScheme(
                batch_scoped=digest.batch_scoped,
                n_rows=len(sample_ids),
                row_width=digest.row_lens[0] if digest.row_lens else 0,
            )
        self._stats.hash_verify.rows_recorded += len(sample_ids)

    def _check_hashes(self, partition_id: str, sample_ids: list[str], out: Any) -> None:
        """Compare wire-out fingerprints against what was written. Never raises."""
        try:
            self._check_hashes_impl(partition_id, sample_ids, out)
        except Exception as exc:  # noqa: BLE001 - a debug check must never fail a transfer
            self._hash_guard_failed("get", exc)

    def _check_hashes_impl(
        self, partition_id: str, sample_ids: list[str], out: Any
    ) -> None:
        """Compare wire-out fingerprints against what was written."""
        if not isinstance(out, TensorDict):
            return
        scheme = self._batch_scope.get(partition_id, {})
        # Only the batch-scoped names: the scheme also records per-row fields
        # now, and handing the whole mapping over forced every field onto the
        # scoped path.
        scoped_names = {n for n, how in scheme.items() if how.batch_scoped}
        digests = self._row_fingerprints(out, sample_ids, scoped_names)
        if not digests:
            return
        partition_hashes = self._hash_by_partition.get(partition_id, {})
        stats = self._stats.hash_verify
        # A batch-scoped digest covers the whole buffer it was reduced from,
        # so it only means anything against a read of that same batch. Drop
        # those fields on a shard read rather than reporting every row of it
        # as a mismatch — but *count* the drop. Dropping silently is the
        # exact shape of the bug that made this check pass while covering
        # nothing: a field stops being compared and the report still reads
        # clean.
        comparable: dict[str, _FieldDigest] = {}
        length_only: dict[str, tuple[_FieldDigest, int]] = {}
        for name, digest in digests.items():
            recorded = scheme.get(name)
            if not digest.batch_scoped:
                comparable[name] = digest
            elif recorded is None:
                continue  # this process never wrote the field
            elif recorded.batch_scoped:
                if recorded.n_rows == len(sample_ids):
                    comparable[name] = digest
                else:
                    stats.fields_skipped += 1  # a shard of a batch-scoped put
            else:
                # Written with uniform rows, read back ragged. Treating that
                # as a divergence reported 3584 mismatches per step on a
                # healthy run: it is the normal shape of a real pipeline,
                # where a shard is written uniform and read back inside a
                # batch whose other rows differ in length. The row lengths
                # are still comparable, and a row that changed length *is* a
                # divergence, so check that much and count the rest as the
                # abstention it is.
                length_only[name] = (digest, recorded.row_width)
                stats.fields_skipped += 1
        for row, sample_id in enumerate(sample_ids):
            per_field = partition_hashes.get(sample_id)
            if not per_field:
                # Written by another process (rollout actor, policy worker):
                # this client has no wire-in reading to compare against.
                stats.rows_unverified += 1
                continue
            stats.rows_checked += 1
            for name, (digest, width) in length_only.items():
                read_len = _row_len(digest, row)
                if per_field.get(name) is None or read_len == width:
                    continue
                stats.mismatches += 1
                if self._hash_mismatches_logged < _MAX_HASH_MISMATCH_LOGS:
                    self._hash_mismatches_logged += 1
                    logger.error(
                        "data-plane hash mismatch: partition=%s sample=%s "
                        "field=%s row was %d long on the wire in, %d out",
                        partition_id,
                        sample_id,
                        name,
                        width,
                        read_len,
                    )
            for name, digest in comparable.items():
                expected = per_field.get(name)
                if expected is None or expected == digest.per_row[row]:
                    continue
                stats.mismatches += 1
                if self._hash_mismatches_logged < _MAX_HASH_MISMATCH_LOGS:
                    self._hash_mismatches_logged += 1
                    # Scheme and shape on the line: the last two false alarms
                    # were both bookkeeping (a scheme replayed wrong, a
                    # grouping that could not be compared), and neither was
                    # diagnosable from the digests alone.
                    logger.error(
                        "data-plane hash mismatch: partition=%s sample=%s "
                        "field=%s wire_in=%d wire_out=%d "
                        "(%s scheme, row %d of %d, %d long)",
                        partition_id,
                        sample_id,
                        name,
                        expected,
                        digest.per_row[row],
                        "batch-scoped" if digest.batch_scoped else "per-row",
                        row,
                        len(sample_ids),
                        _row_len(digest, row),
                    )

    def _run(
        self,
        op: str,
        partition_id: str,
        fn: Callable[[], Any],
        *,
        n_keys: int = 0,
        n_bytes: int = 0,
    ) -> Any:
        """Run ``fn`` and emit one observability event with wall-time and status.

        Args:
            op: Operation tag (``"put"``, ``"get"``, ``"clear"``, etc.).
            partition_id: Partition the op targets.
            fn: Zero-arg callable that invokes the inner client.
            n_keys: Key count if known up front; otherwise inferred from
                the return value (``KVBatchMeta.sample_ids``).
            n_bytes: Byte estimate; overridden by ``_td_bytes`` when the
                return is a ``TensorDict``.

        Returns:
            Whatever ``fn`` returned.
        """
        t0 = monotonic()
        try:
            out = fn()
        except TimeoutError:
            self._emit(op, partition_id, n_keys, n_bytes, t0, "timeout")
            raise
        except Exception:
            self._emit(op, partition_id, n_keys, n_bytes, t0, "error")
            raise
        # If the call returns a TensorDict, the read-side bytes are more
        # informative than the input estimate.
        if isinstance(out, TensorDict):
            n_bytes = _td_bytes(out)
        elif isinstance(out, KVBatchMeta) and not n_keys:
            n_keys = len(out.sample_ids)
        self._emit(op, partition_id, n_keys, n_bytes, t0, "ok")
        return out

    def _emit(
        self,
        op: str,
        partition_id: str,
        n_keys: int,
        n_bytes: int,
        t0: float,
        status: EventStatus,
    ) -> None:
        wall_ms = (monotonic() - t0) * 1000.0
        self._last_inner_ms = wall_ms
        on_event = self._on_event
        if on_event is not None:
            # Built lazily: with no sink registered nothing reads this dict.
            event: DataPlaneEvent = {
                "op": op,
                "partition_id": partition_id,
                "n_keys": n_keys,
                "n_bytes": n_bytes,
                "wall_ms": wall_ms,
                "status": status,
            }
            on_event(event)
        # Time is charged for every status: a timeout is often the single
        # largest contributor, so dropping it would understate the cost.
        stats = self._stats
        stats.total_wall_ms += wall_ms
        bucket = stats.by_op.get(op)
        if bucket is None:
            # Not setdefault(): its default is evaluated eagerly, building a
            # throwaway OpStats (and its 16-bucket histogram) on every op.
            bucket = stats.by_op[op] = OpStats()
        bucket.calls += 1
        bucket.wall_ms += wall_ms
        if wall_ms > bucket.max_ms:
            bucket.max_ms = wall_ms
        if wall_ms > bucket.step_max_ms:
            bucket.step_max_ms = wall_ms
        bucket.latency_hist[bisect_left(LATENCY_BUCKETS_MS, wall_ms)] += 1
        if status != "ok":
            bucket.errors += 1
            return
        stats.total_bytes += n_bytes
        stats.total_keys += n_keys
        stats.total_ops += 1
        bucket.n_bytes += n_bytes
        bucket.n_keys += n_keys
        bytes_f = float(n_bytes)
        if op == "put" and n_keys:
            per_key = n_bytes // n_keys
            stats.last_put_bytes_per_key = per_key
            if per_key > stats.max_bytes_per_key_seen:
                stats.max_bytes_per_key_seen = per_key

    def register_partition(
        self,
        partition_id,
        fields,
        num_samples,
        consumer_tasks,
        grpo_group_size=None,
        enums=None,
    ):
        self._run(
            "register",
            partition_id,
            lambda: self._inner.register_partition(
                partition_id,
                fields,
                num_samples,
                consumer_tasks,
                grpo_group_size=grpo_group_size,
                enums=enums,
            ),
            n_keys=int(num_samples),
        )

    def claim_meta(
        self,
        partition_id,
        task_name,
        required_fields,
        batch_size,
        dp_rank=None,
        blocking=True,
        timeout_s=60.0,
    ):
        return self._run(
            "claim_meta",
            partition_id,
            lambda: self._inner.claim_meta(
                partition_id,
                task_name,
                required_fields,
                batch_size,
                dp_rank=dp_rank,
                blocking=blocking,
                timeout_s=timeout_s,
            ),
        )

    def get_data(self, meta, select_fields=None):
        entered = monotonic()
        out = self._run(
            "get_data",
            meta.partition_id,
            lambda: self._inner.get_data(meta, select_fields=select_fields),
            n_keys=len(meta.sample_ids),
        )
        if self._verify_tensor_hash:
            self._check_hashes(meta.partition_id, meta.sample_ids, out)
        self._bill_self(entered)
        return out

    def check_consumption_status(self, partition_id, task_names):
        return self._run(
            "check_consumption_status",
            partition_id,
            lambda: self._inner.check_consumption_status(partition_id, task_names),
        )

    def put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        entered = monotonic()
        n_bytes = _td_bytes(fields)
        # Materialize once: ``_run`` consumes its lambda and we also need
        # to attribute bytes per sample after success.
        sample_ids_list = _as_list(sample_ids)
        out = self._run(
            "put",
            partition_id,
            lambda: self._inner.put_samples(
                sample_ids_list,
                partition_id,
                fields=fields,
                tags=tags,
            ),
            n_keys=len(sample_ids_list),
            n_bytes=n_bytes,
        )
        self._record_put(partition_id, sample_ids_list, n_bytes)
        # Fingerprinted after ``_run`` rather than inside it: ``fields`` is
        # the caller's TensorDict and the RPC does not mutate it, so hashing
        # here keeps the check's own cost out of the op's ``wall_ms``.
        if self._verify_tensor_hash:
            self._record_hashes(partition_id, sample_ids_list, fields)
        self._bill_self(entered)
        return out

    def get_samples(self, sample_ids, partition_id, select_fields):
        entered = monotonic()
        sample_ids_list = _as_list(sample_ids)
        out = self._run(
            "get",
            partition_id,
            lambda: self._inner.get_samples(
                sample_ids_list,
                partition_id,
                select_fields=select_fields,
            ),
            n_keys=len(sample_ids_list),
        )
        if self._verify_tensor_hash:
            self._check_hashes(partition_id, sample_ids_list, out)
        self._bill_self(entered)
        return out

    def list_sample_ids(self, partition_id: str) -> list[str]:
        return self._run(
            "list_sample_ids",
            partition_id,
            lambda: self._inner.list_sample_ids(partition_id),
        )

    def clear_samples(self, sample_ids, partition_id):
        entered = monotonic()
        sample_ids_list = _as_list(sample_ids)
        n_keys = len(sample_ids_list) if sample_ids_list is not None else 0
        self._run(
            "clear",
            partition_id,
            lambda: self._inner.clear_samples(sample_ids_list, partition_id),
            n_keys=n_keys,
        )
        self._record_clear(partition_id, sample_ids_list)
        self._bill_self(entered)

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run(
            "save_checkpoint",
            "",
            lambda: self._inner.save_checkpoint(
                checkpoint_dir,
                metadata=metadata,
            ),
        )

    def load_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        return self._run(
            "load_checkpoint",
            "",
            lambda: self._inner.load_checkpoint(checkpoint_dir),
        )

    def close(self) -> None:
        self._run(
            "close",
            "",
            lambda: self._inner.close(),
        )
