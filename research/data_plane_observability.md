# Data-Plane Observability — Design

**Owner:** zhiyul
**Date:** 2026-05-03
**Status:** Layer 1 (client-side per-op metrics) implemented
**Companions:**
[`data_plane_integration_plan.md`](./data_plane_integration_plan.md),
[`data_plane_test_plan.md`](./data_plane_test_plan.md)

---

## 1. Problem

TransferQueue ops are opaque from the trainer's perspective. We see
GRPO step time, but not:

- bytes / op (does my rollout writeback dominate the step?)
- ops / sec (is the controller a bottleneck?)
- p50 / p99 latency (storage backend swap actually faster?)
- field-level inspection (which field's wire size blew up?)
- error budget (timeouts, transient failures vs hard crashes)
- per-partition lifecycle (register → put → get → clear hygiene)

Without these, the answer to "is the data plane to blame for X" is
guesswork. The integration plan's G1 (backend swap = config flip) is
unenforceable without a measurement that shows Mooncake p99 < SimpleStorage p99.

---

## 2. Goals & non-goals

**Goals**

- G-O1. Every TQ op emits one record with op type, partition_id,
  n_keys, n_bytes, wall_ms, status, fields. Always. Including errors.
- G-O2. The instrumentation does **not** modify
  :class:`DataPlaneClient`'s ABC, the TQ adapter, or any algorithm
  call site. It is a **wrapper**, opt-in via config.
- G-O3. Pluggable output. The same middleware emits to in-memory,
  structured log, wandb, or future Prometheus/OTEL — caller picks.
- G-O4. Composable. Future middleware (integrity check, distributed
  tracing) stack via the same pattern.
- G-O5. Off by default; zero overhead when disabled.

**Non-goals (Phase 1)**

- N-O1. Server-side controller introspection (queue depth, cross-actor
  scheduling stats). Documented as Layer 2; deferred until needed.
- N-O2. Distributed tracing across Ray actors. Ray has its own.
  Cross-actor observability composes with whatever Ray exposes.
- N-O3. Sampling. Every op is recorded. At the rates this layer fires
  (a few hundred ops/step at most), full recording is cheap. If that
  changes, sampling becomes a sink concern, not a middleware concern.
- N-O4. Real-time alerting. The sink interface supports it, but no
  built-in alert sink ships in Phase 1.

---

## 3. Architecture

Three concerns, two layers, one ABC:

```
┌────────────────────────────────────────────────────────────┐
│  Trainer (grpo_train_sync)                                 │
│  └─ dp_client.snapshot()  →  metrics dict  →  wandb.log    │
└────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────┐
│  MetricsDataPlaneClient (this layer — middleware/decorator)│
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │ records each │ →  │  MetricsSink    │ (pluggable)       │
│  │ op event     │    │  - InMemorySink │ default           │
│  └──────────────┘    │  - LogSink      │ structured stdlib │
│                      │  - WandbSink ⌕  │ future            │
│                      └─────────────────┘                   │
└────────────────────────────────────────────────────────────┘
                              │
                  forwards every call unchanged
                              ▼
┌────────────────────────────────────────────────────────────┐
│  TQDataPlaneClient (the production adapter — untouched)    │
└────────────────────────────────────────────────────────────┘
```

**Key invariants:**

1. The middleware **forwards** every method to the inner client. It
   never alters arguments, return values, or semantics.
2. Errors are **recorded then re-raised**. The middleware never
   swallows.
3. The sink is **owned** by the middleware (wired in at construction);
   the middleware doesn't know how the sink publishes.

---

## 4. Wire format — per-op event

Every TQ op produces exactly one event:

```python
{
    "op":           "put" | "get" | "register" | "clear" | "get_meta",
    "partition_id": str,
    "n_keys":       int,        # 0 if not applicable (e.g. register)
    "n_bytes":      int,        # tensor leaf bytes; 0 for control-plane ops
    "wall_ms":      float,      # adapter wall-clock time
    "status":       "ok" | "error" | "timeout",
    "fields":       list[str] | None,  # what crossed the wire
}
```

This is **not** the same as a metrics row — it's a structured event.
The sink decides whether to aggregate (in-memory counters), log
(structured line), publish (wandb), or all three.

---

## 5. Sink interface

```python
class MetricsSink(ABC):
    @abstractmethod
    def record(self, event: dict) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Cumulative flat dict, namespaced under data_plane/<op>/<metric>.
        Trainer merges this into its own log_metrics() payload."""

    def close(self) -> None: ...   # flush; default no-op
```

Sinks are **stateless w.r.t. the middleware** — they receive events,
produce dicts. A sink implementation can be added without changing the
middleware or the ABC.

### Built-in sinks (Phase 1)

| Sink | Use case | Output |
|---|---|---|
| `InMemorySink` (default) | trainer snapshots once per step into wandb metrics | accumulator dict |
| `LogSink` | per-op trace in run log without wandb | DEBUG line per op + accumulator |
| (future) `WandbSink` | direct push, no trainer involvement | wandb.log on flush |
| (future) `OTELSink` | production ops | OpenTelemetry exporter |

### Snapshot semantics

`snapshot()` returns **cumulative** counters — not deltas. The trainer
computes per-step deltas if needed by storing the last snapshot. This
keeps the sink stateless and the integration trivial:

```python
# At end of every grpo step:
metrics.update(dp_client.snapshot())
logger.log_metrics(metrics, total_steps + 1, prefix="train")
```

Deltas are a wandb-side concern (it derives `_runtime` and rates from
cumulatives). Don't push that complexity into the sink.

---

## 6. Configuration

Extend `DataPlaneConfig`:

```python
class DataPlaneConfig(TypedDict):
    enabled:       bool
    impl:          Literal["transfer_queue"]
    backend:       NotRequired[Literal["simple", "mooncake_cpu"]]
    ...
    observability: NotRequired["ObservabilityConfig"]


class ObservabilityConfig(TypedDict):
    enabled: bool
    sink:    NotRequired[Literal["memory", "log"]]
```

YAML example:

```yaml
data_plane:
  enabled: true
  impl: transfer_queue
  backend: simple
  observability:
    enabled: true
    sink: memory     # default
```

The factory wraps automatically when `observability.enabled=true`:

```python
def build_data_plane_client(cfg, *, bootstrap=True):
    inner = TQDataPlaneClient(cfg, bootstrap=bootstrap)
    obs = cfg.get("observability") or {}
    if obs.get("enabled", False):
        from nemo_rl.data_plane.observability import (
            MetricsDataPlaneClient, build_sink,
        )
        return MetricsDataPlaneClient(inner, sink=build_sink(obs.get("sink")))
    return inner
```

---

## 7. Integration with the trainer

In `grpo_sync.py`, the metrics flow into the existing
`logger.log_metrics(...)` payload:

```python
# inside the per-step loop, after policy.train(...) returns:
if hasattr(dp_client, "snapshot"):  # observability enabled
    metrics.update(dp_client.snapshot())
logger.log_metrics(metrics, total_steps + 1, prefix="train")
```

**Note**: this is the only place in the trainer that needs to know
about observability. Trainer code stays clean; one line at the
metrics-merge site.

---

## 8. Composition: future layers stack

The middleware pattern is intentional. Each future concern is a new
class implementing :class:`DataPlaneClient` and wrapping another:

```python
client = TQDataPlaneClient(cfg)
client = MetricsDataPlaneClient(client, sink=...)        # Layer 1 (this doc)
client = IntegrityCheckClient(client)                    # Layer 3 (future)
client = TraceClient(client, exporter=OTLPExporter(...)) # Layer 4 (future)
```

Stacking order is "outermost first" — the trace layer is at the top of
the stack, sees every call before the metrics layer. The factory's
job is to assemble the stack from config; the algorithm layer doesn't
care.

This is the standard middleware idiom (HTTP, gRPC interceptors, AWS
SDK middleware). It works because every layer is a `DataPlaneClient`
implementation — no special "middleware" type, no chain-of-responsibility
boilerplate.

---

## 9. Layer 2 — server-side introspection (deferred)

Things only the controller knows:

- live partitions
- per-partition: num_keys, fields_declared, fields_produced,
  per-task consumption %, oldest_key_age_ms
- queue depth per (partition, task)
- storage utilization (% of `storage_capacity`)

These need **new methods on the ABC** (`list_partitions`,
`partition_stats`, `queue_depth`) and TQ-side support to back them.
Defer until a debug scenario actually needs them — adding to the ABC
is a contract change for every adapter.

When Layer 2 lands:

```python
class DataPlaneClient(ABC):
    @abstractmethod
    def list_partitions(self) -> list[str]: ...

    @abstractmethod
    def partition_stats(self, partition_id: str) -> PartitionStats: ...

    @abstractmethod
    def queue_depth(self, partition_id: str, task_name: str) -> int: ...
```

The metrics middleware then exposes these too (forwards to inner,
records the call shape if useful), and the trainer can call them on
demand for "why is my run stuck" diagnostics.

---

## 10. Layer 3 — integrity check (deferred)

Catches the silent-corruption class of bug (test plan §R-C1, R-C2 —
dtype coercion, scalar unsqueeze, byte-level wire drift).

Same middleware shape:

```python
class IntegrityCheckClient(DataPlaneClient):
    """Hashes payload at put time, attaches hash to tags. On get,
    recomputes hash, asserts equality. Catches silent wire corruption
    (e.g. TQ auto-unsqueezing a 1D tensor to [B,1])."""
```

Cost: ~µs per op for a `xxhash` of the contiguous bytes. Zero
correctness compromise.

---

## 11. Testing

`tests/data_plane/unit/test_observability.py` covers Layer 1 with
:class:`NoOpDataPlaneClient` as the inner client (no TQ, no Ray, no
GPU — runs in the slim Tier 1 venv):

| Test | Asserts |
|---|---|
| `test_put_records_bytes_and_count` | bytes counted from TensorDict, count incremented |
| `test_get_records_after_put` | get is recorded with byte count from returned TD |
| `test_register_and_clear_recorded` | control-plane ops recorded with `n_bytes=0` |
| `test_error_counted_and_reraised` | errors increment `errors`, original exception propagates |
| `test_throughput_metric_emitted` | derived `throughput_MB_s` appears in snapshot |
| `test_build_sink_factory` | sink name → concrete sink resolution + unknown-name rejection |
| `test_close_propagates_to_inner_and_sink` | close cleans up both layers |

Functional / nightly tests would add:

- end-to-end on real TQ adapter, verifying snapshot keys appear in
  wandb after a 10-step GRPO run
- backend parity (simple vs mooncake_cpu) — assert
  `data_plane/get/throughput_MB_s` is greater under Mooncake

---

## 12. Open questions

1. **Wandb auto-flush.** Today the trainer pulls (`snapshot()` →
   `log_metrics`). A `WandbSink` could push directly without trainer
   involvement. Tradeoff: push is more decoupled but couples the sink
   to the trainer's wandb run handle. Defer until WandbSink is
   actually built; the pull pattern works for now.
2. **Per-rank metrics.** The middleware runs in the driver process,
   which sees only the driver's puts/gets. Worker-side puts (Stage 4
   `kv_batch_put` on a worker) wouldn't be visible. Could be addressed
   by also wrapping the worker's `_dp_client` in the same middleware
   class. Defer until someone actually wants per-worker numbers.
3. **Sampling under high op rates.** Phase 1 records every op. At
   rollout scales (hundreds of puts per step) this is fine. If it
   grows to thousands/sec, add a `SamplingSink` decorator over the
   real sink — keeps the middleware unchanged.

---

## 13. References

- `nemo_rl/data_plane/observability/middleware.py` — `MetricsDataPlaneClient`
- `nemo_rl/data_plane/observability/sinks.py` — sink ABC + built-ins
- `nemo_rl/data_plane/factory.py` — auto-wrap based on config
- `tests/data_plane/unit/test_observability.py` — unit coverage
- `data_plane_integration_plan.md` — integration plan (does NOT change)
- `data_plane_test_plan.md` §5.5 — observability tests at functional tier
