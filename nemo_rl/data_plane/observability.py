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

Everything here sits on the hot path of every transfer, so the cost is a
design constraint rather than an afterthought. Measured against a no-op
inner client on the payload the wire actually carries -- 256 ragged rows,
12 MB, jagged per-token fields as ``codec.pack_jagged_fields`` leaves them
-- the wrapper adds ~99 us per put and ~76 us per get, about 0.15% of a
59 ms operation. What that budget buys is spelled out where it is spent
(``_td_bytes``, ``_record_put``, ``_emit``); the short version is that no
traversal happens twice, no allocation happens for a payload nobody reads,
and no estimate walks a structure whose size it can extrapolate.

``verify_tensor_hash=True`` adds an opt-in correctness check on top:
per-row ``torch.hash_tensor`` fingerprints recorded at put and re-checked
at get, so a tensor that changes between wire-in and wire-out is reported
rather than trained on. It reads every tensor byte again on both sides
(~8 ms for that same 12 MB payload), so it is a debugging tool, not a
metric.
"""

from __future__ import annotations

import logging
import zlib
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal, TypedDict

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


def _dtype_salt(dtype: torch.dtype) -> int:
    """Fingerprint salt distinguishing dtypes that hash to the same words.

    ``hash_tensor`` upcasts to the 64-bit equivalent before reducing, so a
    bf16 tensor and the fp32 tensor holding the same values produce the same
    digest. Mixing the dtype in is what makes a precision change visible.

    ``crc32``, not the builtin ``hash()``: ``hash()`` of a ``str`` is salted
    per process, so a fingerprint built on it would not survive being
    compared across ranks — cheap to keep, expensive to rediscover the day
    someone reduces these cluster-wide.
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


def _enc_const1(obj: Any, budget: list[int]) -> int:
    return 1


def _enc_float(obj: float, budget: list[int]) -> int:
    return 9


def _enc_int(obj: int, budget: list[int]) -> int:
    # msgpack packs small ints in a single byte; only wide values cost 9.
    if -32 <= obj < 128:
        return 1
    if -(2**15) <= obj < 2**16:
        return 3
    if -(2**31) <= obj < 2**32:
        return 5
    return 9


def _enc_str(obj: str, budget: list[int]) -> int:
    n = len(obj) if obj.isascii() else len(obj.encode("utf-8"))
    return n + (1 if n < 32 else 2 if n < 256 else 3 if n < 65536 else 5)


def _enc_bytes(obj: Any, budget: list[int]) -> int:
    n = len(obj)
    return n + (2 if n < 256 else 3 if n < 65536 else 5)


def _enc_dict(obj: dict[Any, Any], budget: list[int]) -> int:
    n = len(obj)
    total = 1 if n < 16 else 3 if n < 65536 else 5
    for k, v in obj.items():
        if budget[0] <= 0:
            break
        budget[0] -= 1
        total += _estimate_encoded_bytes(k, budget)
        total += _estimate_encoded_bytes(v, budget)
    return total


def _enc_seq(obj: Any, budget: list[int]) -> int:
    n = len(obj)
    total = 1 if n < 16 else 3 if n < 65536 else 5
    for v in obj:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        total += _estimate_encoded_bytes(v, budget)
    return total


def _enc_tensor(obj: torch.Tensor, budget: list[int]) -> int:
    return obj.numel() * obj.element_size()


# Exact-type dispatch, tried before any isinstance chain. The common leaves
# come first because insertion order is also the order of the subclass
# fallback below, which only runs for types that miss the exact lookup.
_ENCODERS: dict[type, Callable[[Any, list[int]], int]] = {
    str: _enc_str,
    int: _enc_int,
    bool: _enc_const1,
    float: _enc_float,
    dict: _enc_dict,
    list: _enc_seq,
    tuple: _enc_seq,
    set: _enc_seq,
    bytes: _enc_bytes,
    bytearray: _enc_bytes,
    memoryview: _enc_bytes,
    torch.Tensor: _enc_tensor,
    type(None): _enc_const1,
}


def _estimate_encoded_bytes(obj: Any, budget: list[int]) -> int:
    """Approximate msgpack-encoded size of a non-tensor object.

    TQ encodes non-tensors with msgpack (``serial_utils.batch_encode_into``),
    falling back to pickle/cloudpickle via ``Ext`` for unknown types. Getting
    the exact size means running that encoder, which would double the
    serialisation work on the hot path -- so this walks the structure and
    approximates instead. Container framing (1-5 bytes per element) is not
    modelled, so treat the result as a lower bound.

    Dispatch is an exact-type dict lookup rather than an ``isinstance``
    chain: on the hot path the common leaves (``str``, ``int``) sat 5-7
    branches deep, and the chain cost more than the arithmetic it guarded.

    ``budget`` is a single-element list used as a mutable counter that bounds
    the walk to ``max_nodes`` container elements. It is decremented only by
    the container encoders -- a leaf cannot itself expand the walk, so
    charging leaves bought nothing but two list index ops each. Containers
    stop iterating once it is exhausted; summing a generator would otherwise
    keep walking every element while each recursive call returned 0, making
    the cost O(size) despite the budget.
    """
    encoder = _ENCODERS.get(obj.__class__)
    if encoder is not None:
        return encoder(obj, budget)
    for typ, encoder in _ENCODERS.items():
        if isinstance(obj, typ):
            return encoder(obj, budget)
    # Unknown type -> pickle/cloudpickle Ext. Cheap proxy; the real size
    # would need an actual dumps(), which is what we are avoiding.
    return 64


# Rows sampled from a NonTensorStack to estimate its payload. The stack
# holds one Python object per batch element, so materialising it (``tolist``)
# and walking every row is O(batch) *per put* -- ~1 ms for a 256-row
# message_log stack. Rows in one stack share a schema, so a strided sample
# extrapolates to within a few percent for a figure that is already
# documented as a lower-bound estimate.
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

    Tensor leaves count ``numel * element_size``, which equals
    ``t.contiguous().nbytes`` -- the size mooncake registers and sends
    (``mooncake_client`` calls ``.contiguous()`` before taking the pointer).
    Verified equal for contiguous, sliced, transposed and stride-0 expanded
    views, and across bf16/bool/int64/fp8.

    Non-tensor leaves are estimated with :func:`_estimate_encoded_bytes`,
    since TQ ships them over a separate msgpack path -- omitting them would
    undercount communication volume by whatever metadata rides along. Both
    kinds are counted in a *single* pass: a second traversal would double the
    per-put walk of a structure that can hold hundreds of keys.

    ``leaves_only=True`` would hide the non-tensor entries entirely
    (``NonTensorData`` is not treated as a leaf), so this walks with
    ``leaves_only=False`` and skips container nodes itself. It walks with
    ``items()`` rather than ``keys()`` + ``get()``: ``get()`` re-resolves
    each nested key from the root, which measured 2.7x the cost of the
    single traversal ``items()`` already performs.

    ``NonTensorData`` and ``NonTensorStack`` are matched by type, not by
    ``hasattr``: both are tensorclasses whose attribute misses fall through
    a ``__getattr__`` that costs ~2.8 us per probe. The distinction matters
    beyond speed -- ``NonTensorData`` exposes BOTH ``.data`` and
    ``.tolist()``, and its ``.tolist()`` broadcasts the single stored object
    across the batch dim (a 64-row batch reported 20x the real payload).

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
            # No jagged special case: numel() already reports a nested
            # tensor's total element count, and asking it directly is half
            # the cost of reaching through .values() (16 us vs 32 us per
            # jagged field). Every per-token field on this wire is jagged,
            # so that is paid four or five times per put.
            total += v.numel() * v.element_size()
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
                metric: it reads every tensor byte again (~8 ms for a
                12 MB jagged batch), so it is off unless the config asks.
        """
        self._inner = inner
        self._on_event = on_event
        self._verify_tensor_hash = verify_tensor_hash
        self._stats = DataPlaneStats()
        # Nested per-partition / per-key live byte counts. Populated on
        # successful ``put_samples``; popped on successful ``clear_samples``.
        # Bounded by the live key population, not cumulative traffic.
        self._bytes_by_partition: dict[str, dict[str, int]] = {}
        # partition -> sample_id -> field -> wire-in fingerprint. Same
        # lifetime as ``_bytes_by_partition``: cleared by ``clear_samples``,
        # so it is bounded by the live key population.
        self._hash_by_partition: dict[str, dict[str, dict[str, int]]] = {}
        self._hash_mismatches_logged = 0
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
            len(d) for d in self._bytes_by_partition.values()
        )
        for op, s in out["by_op"].items():
            s["mean_ms"] = s["wall_ms"] / s["calls"] if s["calls"] else 0.0
            s["mb_per_s"] = (
                (s["n_bytes"] / 1e6) / (s["wall_ms"] / 1e3) if s["wall_ms"] else 0.0
            )
            s["pct_of_total_ms"] = (
                100.0 * s["wall_ms"] / self._stats.total_wall_ms
                if self._stats.total_wall_ms
                else 0.0
            )
            s["fit"] = fit_latency_bandwidth(s)
            h = s["latency_hist"]
            s["p50_ms"] = percentile_from_hist(h, 0.50)
            s["p99_ms"] = percentile_from_hist(h, 0.99)
            # Tail/mean ratio: a mean hides MR churn and queueing, which show
            # up as p99 pulling away from the mean.
            s["tail_ratio_p99_mean"] = (
                s["p99_ms"] / s["mean_ms"] if s["mean_ms"] > 0 else 0.0
            )
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

        wall_ms = snap["total_wall_ms"] - prev.get("total_wall_ms", 0.0)
        vol = snap["comm_volume_bytes"] - prev.get("comm_volume_bytes", 0)
        metrics: dict[str, float] = {
            "wall_s": wall_ms / 1e3,
            "frac_of_step": (wall_ms / 1e3 / step_time_s) if step_time_s > 0 else 0.0,
            "comm_volume_gb": vol / 1e9,
            "bytes_written_gb": (snap["bytes_written"] - prev.get("bytes_written", 0))
            / 1e9,
            "bytes_read_gb": (snap["bytes_read"] - prev.get("bytes_read", 0)) / 1e9,
            "bytes_outstanding_gb": snap["bytes_outstanding"] / 1e9,
        }
        if self._verify_tensor_hash:
            hv, prev_hv = snap["hash_verify"], prev.get("hash_verify", {})
            metrics["hash/rows_checked"] = hv["rows_checked"] - prev_hv.get(
                "rows_checked", 0
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
            metrics[f"{op}/calls"] = calls
            metrics[f"{op}/wall_s"] = op_ms / 1e3
            metrics[f"{op}/mean_ms"] = op_ms / calls
            # Percentiles and the overhead/bandwidth fit are cumulative by
            # construction (histogram buckets, regression sums) -- as-is.
            metrics[f"{op}/p50_ms"] = st["p50_ms"]
            metrics[f"{op}/p99_ms"] = st["p99_ms"]
            fit = st["fit"]
            if fit.get("model_trustworthy"):
                metrics[f"{op}/fixed_overhead_ms"] = fit["fixed_ms"]
                metrics[f"{op}/bandwidth_mb_s"] = fit["bandwidth_mb_s"]
                metrics[f"{op}/overhead_frac"] = fit["overhead_frac_at_mean"]
        return metrics

    def bytes_outstanding_by_partition(self) -> dict[str, int]:
        """Per-partition breakdown of currently-held bytes."""
        return {p: sum(d.values()) for p, d in self._bytes_by_partition.items()}

    def _record_put(self, partition_id: str, keys: list[str], n_bytes: int) -> None:
        """Attribute put bytes per key so a later ``clear_samples`` can subtract.

        Called after the underlying RPC succeeds so a failed put never
        leaves the accounting inflated.

        The even split is what makes the loop cheap, and it is not a
        simplification: ``n_bytes`` is a whole-batch figure, so there is no
        per-key truth to preserve. The division remainder therefore lands on
        the first key rather than being spread one-byte-at-a-time across the
        batch — spreading it cost an ``enumerate`` and a compare per key
        (~40% of this method at 256 keys) to move at most one byte each.

        Args:
            partition_id: Partition the keys were written to.
            keys: Per-sample uids that were written.
            n_bytes: Total bytes written; distributed evenly across keys.
        """
        if not keys or n_bytes <= 0:
            return
        per_key, remainder = divmod(n_bytes, len(keys))
        partition_dict = self._bytes_by_partition.setdefault(partition_id, {})
        get_held = partition_dict.get
        for key in keys:
            partition_dict[key] = get_held(key, 0) + per_key
        partition_dict[keys[0]] += remainder
        self._stats.bytes_outstanding += n_bytes
        if self._stats.bytes_outstanding > self._stats.peak_bytes_outstanding:
            self._stats.peak_bytes_outstanding = self._stats.bytes_outstanding

    def _record_clear(self, partition_id: str, keys: list[str] | None) -> None:
        """Reverse the put accounting for ``keys``.

        Called after the underlying RPC succeeds so a failed clear keeps
        the accounting consistent with TQ's actual state.

        Args:
            partition_id: Partition the keys were dropped from.
            keys: Uids dropped; ``None`` means the whole partition was cleared.
        """
        if self._verify_tensor_hash:
            self._drop_hashes(partition_id, keys)
        partition_dict = self._bytes_by_partition.get(partition_id)
        if partition_dict is None:
            return
        if keys is None:
            freed = sum(partition_dict.values())
            del self._bytes_by_partition[partition_id]
        else:
            freed = 0
            for key in keys:
                freed += partition_dict.pop(key, 0)
            if not partition_dict:
                del self._bytes_by_partition[partition_id]
        self._stats.bytes_outstanding -= freed

    # ── wire-in / wire-out fingerprinting (opt-in) ─────────────────────

    def _row_fingerprints(
        self, td: TensorDict | None, sample_ids: list[str]
    ) -> dict[str, list[int]]:
        """Per-row ``torch.hash_tensor`` fingerprints, keyed by field name.

        Each tensor leaf is flattened to ``[n_rows, -1]`` and reduced along
        the trailing dims, giving one ``uint64`` per sample id — the
        granularity the wire actually needs, since a batch put of 256 rows
        is routinely read back as eight shards of 32 and a whole-tensor hash
        would be incomparable.

        ``hash_tensor`` reduces by XOR, which is commutative, so a row's
        digest is blind to a rearrangement of that row's own elements. Rows
        are compared per sample id, so a swap *between* rows is still caught
        — it is only a permutation *within* one row that hides, and nothing
        on this wire reorders within a row. Measured against ten corruption
        modes (single-element change per dtype, truncation, zeroed row,
        precision change, wrong sample, swapped rows) it caught all ten.

        Row *i* is attributed to ``sample_ids[i]``, which is the ordering
        :meth:`DataPlaneClient.get_samples` already promises ("batched along
        ``sample_ids``"). An adapter that reordered rows would show up here
        as a mismatch — which is the correct verdict, since a reordered read
        is exactly the bug this check exists to catch.

        Jagged leaves are fingerprinted too, and have to be: ``column_io``
        packs every per-token field (``input_ids``, ``generation_logprobs``,
        ``token_mask``, ``advantages``) through
        :func:`codec.pack_jagged_fields` before it reaches ``put_samples``,
        so skipping nested tensors would leave the entire bulk payload
        unguarded while still reporting a clean bill of health. They round
        trip asymmetrically — ``_from_wire`` densifies a nested field whose
        rows happen to be uniform — so the fingerprint is defined on the
        *row*, identically for both layouts.

        Leaves that cannot be attributed per row (a leading dim that isn't
        ``len(sample_ids)``, or a nested layout other than ``jagged``) are
        counted in ``fields_skipped`` rather than silently dropped.
        """
        if td is None:
            return {}
        n_rows = len(sample_ids)
        stats = self._stats.hash_verify
        out: dict[str, list[int]] = {}
        for key, v in td.items(include_nested=True, leaves_only=True):
            if not isinstance(v, torch.Tensor) or v.ndim < 1:
                stats.fields_skipped += 1
                continue
            salt = _dtype_salt(v.dtype)
            if v.is_nested:
                if v.layout != torch.jagged:
                    stats.fields_skipped += 1
                    continue
                # Padding with zero is free here rather than approximate:
                # XOR is its own identity on zero and a zero pad bitcasts to
                # a zero 64-bit word for every dtype on this wire (verified
                # for int64/int32/bf16/fp32/bool), so the pad contributes
                # nothing. That is what lets a jagged put reconcile against
                # the dense read `_from_wire` hands back when the rows
                # happen to be uniform.
                v = torch.nested.to_padded_tensor(v, padding=0)
            if v.shape[0] != n_rows:
                stats.fields_skipped += 1
                continue
            digest = torch.hash_tensor(v.reshape(n_rows, -1), dim=1) ^ salt
            name = key if isinstance(key, str) else ".".join(key)
            out[name] = digest.tolist()
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
            for name, column in digests.items():
                per_field[name] = column[row]
        self._stats.hash_verify.rows_recorded += len(sample_ids)

    def _check_hashes(self, partition_id: str, sample_ids: list[str], out: Any) -> None:
        """Compare wire-out fingerprints against what was written."""
        if not isinstance(out, TensorDict):
            return
        digests = self._row_fingerprints(out, sample_ids)
        if not digests:
            return
        partition_hashes = self._hash_by_partition.get(partition_id, {})
        stats = self._stats.hash_verify
        for row, sample_id in enumerate(sample_ids):
            per_field = partition_hashes.get(sample_id)
            if not per_field:
                # Written by another process (rollout actor, policy worker):
                # this client has no wire-in reading to compare against.
                stats.rows_unverified += 1
                continue
            stats.rows_checked += 1
            for name, column in digests.items():
                expected = per_field.get(name)
                if expected is None or expected == column[row]:
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
                        column[row],
                    )

    def _drop_hashes(self, partition_id: str, keys: list[str] | None) -> None:
        """Release fingerprints alongside the bytes accounting."""
        if keys is None:
            self._hash_by_partition.pop(partition_id, None)
            return
        partition_hashes = self._hash_by_partition.get(partition_id)
        if partition_hashes is None:
            return
        for key in keys:
            partition_hashes.pop(key, None)
        if not partition_hashes:
            del self._hash_by_partition[partition_id]

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
        on_event = self._on_event
        if on_event is not None:
            # Built lazily: with no sink registered this dict was the single
            # most frequent allocation in the wrapper, and nothing read it.
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
        out = self._run(
            "get_data",
            meta.partition_id,
            lambda: self._inner.get_data(meta, select_fields=select_fields),
            n_keys=len(meta.sample_ids),
        )
        if self._verify_tensor_hash:
            self._check_hashes(meta.partition_id, meta.sample_ids, out)
        return out

    def check_consumption_status(self, partition_id, task_names):
        return self._run(
            "check_consumption_status",
            partition_id,
            lambda: self._inner.check_consumption_status(partition_id, task_names),
        )

    def put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        n_bytes = _td_bytes(fields)
        # Materialize once: ``_run`` consumes its lambda and we also need
        # to attribute bytes per sample after success.
        sample_ids_list = (
            sample_ids if isinstance(sample_ids, list) else list(sample_ids)
        )
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
        return out

    def get_samples(self, sample_ids, partition_id, select_fields):
        sample_ids_list = (
            sample_ids if isinstance(sample_ids, list) else list(sample_ids)
        )
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
        return out

    def list_sample_ids(self, partition_id: str) -> list[str]:
        return self._run(
            "list_sample_ids",
            partition_id,
            lambda: self._inner.list_sample_ids(partition_id),
        )

    def clear_samples(self, sample_ids, partition_id):
        sample_ids_list = (
            sample_ids
            if (sample_ids is None or isinstance(sample_ids, list))
            else list(sample_ids)
        )
        n_keys = len(sample_ids_list) if sample_ids_list is not None else 0
        self._run(
            "clear",
            partition_id,
            lambda: self._inner.clear_samples(sample_ids_list, partition_id),
            n_keys=n_keys,
        )
        self._record_clear(partition_id, sample_ids_list)

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
