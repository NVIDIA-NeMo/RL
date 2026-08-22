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


class _FieldDigest(NamedTuple):
    """One fingerprint per row, plus how far it can be trusted.

    ``batch_scoped`` means the values were derived from the whole batch's
    buffer, so they only reconcile against a read of that same batch. A
    shard of it computes a different buffer digest and must be reported
    unverified rather than as a mismatch.
    """

    per_row: list[int]
    batch_scoped: bool


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


def fit_latency_bandwidth(s: dict[str, Any]) -> dict[str, Any]:
    """Split an op's time into fixed per-request overhead vs transfer.

    Least-squares fit of ``wall_ms ~ fixed_ms + n_bytes / bandwidth`` over the
    op's successful calls, from the accumulated sufficient statistics.

    The fit is only identifiable when request sizes actually vary: if every
    request is the same size, infinitely many (overhead, bandwidth) pairs
    reproduce the data, so ``regime`` reports ``"unidentifiable"`` rather
    than an arbitrary split. That case is common in RL, where a step's
    payloads are often uniform -- vary batch size to break the tie.
    """
    n = s["calls"] - s["errors"]  # successful calls; bytes/time pair only on those
    sx, sy = float(s["n_bytes"]), s["ok_wall_ms"]
    sxx, sxy = s["sum_bytes_sq"], s["sum_bytes_ms"]
    if n < 3 or sx <= 0:
        return {"regime": "insufficient-data"}
    mean_x = sx / n
    var_x = max(sxx / n - mean_x * mean_x, 0.0)
    # Coefficient of variation: how much do request sizes actually differ?
    if (var_x**0.5) / mean_x < 0.05:
        return {
            "regime": "unidentifiable",
            "reason": "request sizes near-uniform; vary payload size to separate",
            "mean_bytes": mean_x,
            "mean_ms": sy / n,
        }
    denom = n * sxx - sx * sx
    if denom <= 0:
        return {"regime": "unidentifiable", "reason": "degenerate fit"}
    slope = (n * sxy - sx * sy) / denom  # ms per byte
    fixed_ms = (sy - slope * sx) / n
    if slope <= 0:
        return {"regime": "noise-dominated", "fixed_ms": fixed_ms}
    transfer_ms_at_mean = slope * mean_x
    # R^2: does an affine model actually fit? Low R^2 means the split below
    # is not trustworthy regardless of how clean the numbers look.
    syy = s["sum_ms_sq"]
    ss_tot = syy - sy * sy / n
    ss_res = syy - fixed_ms * sy - slope * sxy
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "fixed_ms": fixed_ms,
        "bandwidth_mb_s": 1.0 / (slope * 1000.0),
        "transfer_ms_at_mean": transfer_ms_at_mean,
        "mean_bytes": mean_x,
        "r_squared": r_squared,
        # A high R^2 does NOT validate the model: a chunked step function or
        # a quadratic both fit a line at R^2 > 0.93 while producing a
        # meaningless split. A negative intercept is physically impossible
        # (no request costs less than zero to issue) and catches exactly the
        # misspecification R^2 misses, so both must hold.
        "model_trustworthy": r_squared >= 0.8 and fixed_ms >= 0.0,
        "regime": (
            "overhead-dominated"
            if fixed_ms > transfer_ms_at_mean
            else "bandwidth-dominated"
        ),
        "overhead_frac_at_mean": (
            fixed_ms / (fixed_ms + transfer_ms_at_mean)
            if (fixed_ms + transfer_ms_at_mean) > 0
            else 0.0
        ),
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
        stats["pct_of_total_ms"] = (
            100.0 * wall_ms / total_wall_ms if total_wall_ms else 0.0
        )
        stats["fit"] = fit_latency_bandwidth(stats)
        hist = stats["latency_hist"]
        stats["p50_ms"] = percentile_from_hist(hist, 0.50)
        stats["p99_ms"] = percentile_from_hist(hist, 0.99)
        # Tail/mean ratio: a mean hides MR churn and queueing, which show
        # up as p99 pulling away from the mean.
        stats["tail_ratio_p99_mean"] = (
            stats["p99_ms"] / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
        )


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
)
_SNAPSHOT_MAX = ("max_bytes_per_key_seen", "last_put_bytes_per_key")
_OP_SUM = (
    "calls",
    "errors",
    "wall_ms",
    "n_bytes",
    "n_keys",
    "ok_wall_ms",
    "sum_bytes_sq",
    "sum_bytes_ms",
    "sum_ms_sq",
)
_OP_MAX = ("max_ms", "step_max_ms")


def merge_snapshots(snapshots: "list[dict[str, Any]]") -> dict[str, Any]:
    """Combine per-process snapshots into one cluster-wide view.

    This is what the accumulators were shaped for. Latency lives in fixed
    histogram buckets and the latency/bandwidth model lives in sufficient
    statistics precisely so both *add*: summing 256 per-rank histograms
    gives the true cluster distribution, which averaging 256 per-rank
    percentiles cannot. Everything derived — percentiles, throughput, the
    affine fit — is recomputed from the merged totals, never averaged.

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
    merged: dict[str, Any], prev: dict[str, Any], step_time_s: float
) -> dict[str, float]:
    """Per-step cluster metrics from two merged snapshots.

    The single-process equivalent of this lives on the client, which owns
    its own previous reading. A cluster has no such owner, so the caller
    holds ``prev`` and passes it back.

    ``observability_overhead_ms`` is what the measurement itself cost,
    summed over every process -- the wrapper's wall time minus the time its
    inner client was working. It sits beside ``wall_ms`` so the bill is
    visible next to what it bought.
    """
    wall_ms = merged["total_wall_ms"] - prev.get("total_wall_ms", 0.0)
    overhead_ms = merged["self_ms"] - prev.get("self_ms", 0.0)
    metrics: dict[str, float] = {
        "wall_ms": wall_ms,
        "frac_of_step": (wall_ms / 1e3 / step_time_s) if step_time_s > 0 else 0.0,
        "comm_volume_mb": (
            merged["comm_volume_bytes"] - prev.get("comm_volume_bytes", 0)
        )
        / 1e6,
        "bytes_written_mb": (merged["bytes_written"] - prev.get("bytes_written", 0))
        / 1e6,
        "bytes_read_mb": (merged["bytes_read"] - prev.get("bytes_read", 0)) / 1e6,
        "bytes_outstanding_mb": merged["bytes_outstanding"] / 1e6,
        "n_processes": merged.get("n_processes", 0),
        "observability_overhead_ms": overhead_ms,
        "observability_overhead_frac": (overhead_ms / wall_ms if wall_ms > 0 else 0.0),
    }
    prev_ops = prev.get("by_op", {})
    for op, stats in merged["by_op"].items():
        prev_op = prev_ops.get(op, {})
        calls = stats["calls"] - prev_op.get("calls", 0)
        if calls <= 0:
            continue
        metrics[f"{op}/calls"] = calls
        metrics[f"{op}/wall_ms"] = stats["wall_ms"] - prev_op.get("wall_ms", 0.0)
        metrics[f"{op}/max_ms"] = stats["max_ms"]
        # Percentiles are worth reporting here and not per process: this
        # histogram is the sum over every rank, so it is the real cluster
        # distribution rather than one process's handful of calls.
        metrics[f"{op}/p50_ms"] = stats["p50_ms"]
        metrics[f"{op}/p99_ms"] = stats["p99_ms"]
    return metrics


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
    # Sufficient statistics for the least-squares fit wall_ms ~ a + b*n_bytes,
    # which separates fixed per-request overhead (a) from bandwidth (1/b).
    # Successful calls only, so bytes and time refer to the same events.
    # These are additive, so they can be summed across ranks and refit
    # globally -- no need to ship per-event samples off each process.
    ok_wall_ms: float = 0.0
    sum_bytes_sq: float = 0.0
    sum_bytes_ms: float = 0.0
    # Also needed for R^2, which is what tells us whether the affine model
    # describes the data at all -- chunking, retries and queueing all make
    # wall_ms non-linear in n_bytes, and a low R^2 is the signal to stop
    # trusting the overhead/bandwidth split.
    sum_ms_sq: float = 0.0
    # Slowest single call, exact. The histogram below can only place a
    # call in a bucket, so at the handful of calls an op makes in one step
    # a percentile off it is bucket geometry rather than data -- p99 of one
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
        self._batch_scope: dict[str, tuple[frozenset[str], int]] = {}
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

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative totals plus live byte / key outstanding counts.

        ``total_wall_ms`` is the aggregate data-plane cost; ``by_op`` breaks
        it down per op tag with derived ``mean_ms`` and ``mb_per_s`` so the
        backends can be compared without post-processing. Throughput is
        omitted for ops that move no payload (e.g. ``claim_meta``, whose
        wall time is producer wait, not transfer).
        """
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
        return out

    def get_step_metrics(self, step_time_s: float) -> dict[str, float]:
        """Per-step data-plane metrics, as a ready-to-log flat dict.

        Cumulative counters are differenced against the previous call, so this
        reports what the data plane cost *this* step. Mirrors
        ``VllmGeneration.get_step_metrics`` so trainers stay one line.

        ``frac_of_step`` is the metric that decides whether optimising the
        data plane is worth anything: per-op shares only say where data-plane
        time went, never whether it mattered against compute.
        """
        snap = self.snapshot()
        prev = self._prev_snapshot
        self._prev_snapshot = snap
        # Snapshot first, then open a fresh window: the values just read are
        # this step's, and anything after this call belongs to the next one.
        step_maxima = {op: b.step_max_ms for op, b in self._stats.by_op.items()}
        for bucket in self._stats.by_op.values():
            bucket.step_max_ms = 0.0

        wall_ms = snap["total_wall_ms"] - prev.get("total_wall_ms", 0.0)
        vol = snap["comm_volume_bytes"] - prev.get("comm_volume_bytes", 0)
        # Every duration is ms and every volume is MB, with no exceptions:
        # a chart that mixes wall_s against p99_ms puts a 0.008 next to a
        # 24.85 and reads as a bug in the data plane rather than in the
        # axis. GB was the same problem one dimension over -- a realistic
        # step moved 0.00017 GB.
        metrics: dict[str, float] = {
            "wall_ms": wall_ms,
            "frac_of_step": (wall_ms / 1e3 / step_time_s) if step_time_s > 0 else 0.0,
            "comm_volume_mb": vol / 1e6,
            "bytes_written_mb": (snap["bytes_written"] - prev.get("bytes_written", 0))
            / 1e6,
            "bytes_read_mb": (snap["bytes_read"] - prev.get("bytes_read", 0)) / 1e6,
            "bytes_outstanding_mb": snap["bytes_outstanding"] / 1e6,
        }
        if self._verify_tensor_hash:
            hv, prev_hv = snap["hash_verify"], prev.get("hash_verify", {})
            metrics["hash/rows_checked"] = hv["rows_checked"] - prev_hv.get(
                "rows_checked", 0
            )
            metrics["hash/rows_recorded"] = hv["rows_recorded"] - prev_hv.get(
                "rows_recorded", 0
            )
            metrics["hash/rows_unverified"] = hv["rows_unverified"] - prev_hv.get(
                "rows_unverified", 0
            )
            metrics["hash/mismatches"] = hv["mismatches"] - prev_hv.get("mismatches", 0)
            # Logged because a guard that quietly stops covering a field is
            # worse than no guard: it reports 0 mismatches and reads as
            # clean. A step where this climbs is a step where something
            # stopped being checked.
            metrics["hash/fields_skipped"] = hv["fields_skipped"] - prev_hv.get(
                "fields_skipped", 0
            )
        prev_ops = prev.get("by_op", {})
        for op, st in snap["by_op"].items():
            prev_op = prev_ops.get(op, {})
            calls = st["calls"] - prev_op.get("calls", 0)
            if calls <= 0:
                continue
            op_ms = st["wall_ms"] - prev_op.get("wall_ms", 0.0)
            # Four series per op tag. Anything exactly derivable is left
            # out rather than logged: mean is wall_ms/calls, and a dashboard
            # can divide. ``snapshot()`` still carries the full picture,
            # percentiles included, for a one-off inspection.
            metrics[f"{op}/calls"] = calls
            metrics[f"{op}/wall_ms"] = op_ms
            # ``max_ms`` rather than p50/p99. Those come off a histogram
            # that is never reset, so per step they were a lifetime figure
            # that goes flat, quantised to bucket edges. The max is exact
            # and says the same thing at the handful of calls per step.
            metrics[f"{op}/max_ms"] = step_maxima.get(op, 0.0)
            fit = st["fit"]
            if fit.get("model_trustworthy"):
                # The step's time split into the two things that cause it,
                # in ms, rather than a ratio. These stack: together they are
                # the model's estimate of this op's ``wall_ms`` for the
                # step, so charting them against the measured ``wall_ms``
                # shows both the split and how well the model holds.
                #
                # The *coefficients* come from the cumulative fit and should
                # be stable -- that is what a fitted model is for. The
                # *attribution* is per step, because it is applied to this
                # step's calls and bytes. A ratio would have been neither:
                # cumulative and therefore flat, and unitless on an axis of
                # milliseconds.
                op_bytes = st["n_bytes"] - prev_op.get("n_bytes", 0)
                ms_per_byte = 1.0 / (fit["bandwidth_mb_s"] * 1e3)
                metrics[f"{op}/overhead_ms"] = fit["fixed_ms"] * calls
                metrics[f"{op}/transfer_ms"] = ms_per_byte * op_bytes
        return metrics

    def bytes_outstanding_by_partition(self) -> dict[str, int]:
        """Per-partition breakdown of currently-held bytes."""
        return dict(self._bytes_by_partition)

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
        if keys is not None:
            live -= live.intersection(keys)
        if keys is None or not live:
            freed = total
            del self._keys_by_partition[partition_id]
            self._bytes_by_partition.pop(partition_id, None)
        else:
            dropped = len(keys)
            freed = total * dropped // (len(live) + dropped)
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
        sample that diverged. ``hash_tensor`` has no ragged kernel, so a
        jagged leaf instead gets one digest over its whole values buffer,
        XORed per row with that row's length, and is marked
        ``batch_scoped``; padding it out to a rectangle to get per-row
        digests costs far more than the answer is worth. ``README.md`` has
        the resulting detection/attribution table.

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
                buffer, lengths = v.values(), (offsets[1:] - offsets[:-1]).tolist()
            elif v.shape[0] != n_rows:
                stats.fields_skipped += 1
                continue
            elif name in batch_scoped_fields:
                buffer = v
                lengths = [v.shape[1] if v.ndim >= 2 else 1] * n_rows
            else:
                flat = _as_int_view(v.reshape(n_rows, -1))
                out[name] = _FieldDigest(
                    (torch.hash_tensor(flat, dim=1) ^ salt).tolist(),
                    batch_scoped=False,
                )
                continue
            flat_buffer = _as_int_view(buffer.reshape(1, -1))
            buffer_digest = torch.hash_tensor(flat_buffer, dim=1).tolist()[0] ^ salt
            out[name] = _FieldDigest(
                [buffer_digest ^ length for length in lengths], batch_scoped=True
            )
        return out

    def _record_hashes(
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
        scoped = frozenset(n for n, d in digests.items() if d.batch_scoped)
        if scoped:
            self._batch_scope[partition_id] = (scoped, len(sample_ids))
        self._stats.hash_verify.rows_recorded += len(sample_ids)

    def _check_hashes(self, partition_id: str, sample_ids: list[str], out: Any) -> None:
        """Compare wire-out fingerprints against what was written."""
        if not isinstance(out, TensorDict):
            return
        scoped_names, scoped_rows = self._batch_scope.get(
            partition_id, (frozenset(), 0)
        )
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
        # clean. It also fires when a rectangular put comes back jagged
        # (one row truncated makes the batch ragged), which is a real
        # divergence this cannot express as a row-level mismatch.
        comparable = {}
        for name, digest in digests.items():
            if not digest.batch_scoped or scoped_rows == len(sample_ids):
                comparable[name] = digest
            else:
                stats.fields_skipped += 1
        for row, sample_id in enumerate(sample_ids):
            per_field = partition_hashes.get(sample_id)
            if not per_field:
                # Written by another process (rollout actor, policy worker):
                # this client has no wire-in reading to compare against.
                stats.rows_unverified += 1
                continue
            stats.rows_checked += 1
            for name, digest in comparable.items():
                expected = per_field.get(name)
                if expected is None or expected == digest.per_row[row]:
                    continue
                stats.mismatches += 1
                if self._hash_mismatches_logged < _MAX_HASH_MISMATCH_LOGS:
                    self._hash_mismatches_logged += 1
                    logger.error(
                        "data-plane hash mismatch: partition=%s sample=%s "
                        "field=%s wire_in=%d wire_out=%d",
                        partition_id,
                        sample_id,
                        name,
                        expected,
                        digest.per_row[row],
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
        bucket.ok_wall_ms += wall_ms
        bytes_f = float(n_bytes)
        bucket.sum_bytes_sq += bytes_f * bytes_f
        bucket.sum_bytes_ms += bytes_f * wall_ms
        bucket.sum_ms_sq += wall_ms * wall_ms
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

    def clear_samples(self, sample_ids, partition_id):
        sample_ids_list = _as_list(sample_ids)
        n_keys = len(sample_ids_list) if sample_ids_list is not None else 0
        self._run(
            "clear",
            partition_id,
            lambda: self._inner.clear_samples(sample_ids_list, partition_id),
            n_keys=n_keys,
        )
        self._record_clear(partition_id, sample_ids_list)

    def close(self) -> None:
        self._run(
            "close",
            "",
            lambda: self._inner.close(),
        )
