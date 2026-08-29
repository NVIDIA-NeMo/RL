# Extending Instrumentation

To add new spans or metrics to NeMo-RL code, use the instrumentation primitives from nemo-lens (`managed_span`, `trace_fn`, `span_cm`). The primitives themselves are documented in [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens); this page covers NeMo-RL conventions.

## The import pattern

Every algorithm instrumentation import should go through
`nemo_rl.telemetry.instrumentation`, which re-exports the lens primitives with
`rl.bucket` tagging applied to leaf spans:

```python
from nemo_rl.telemetry.instrumentation import Bucket, bucket_scope, managed_span, trace_fn
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.span_groups import RLSpanGroup
```

Never import from `nemo.lens.*` directly in algorithm code — that is how a span
ends up with no bucket and invisible to the goodput rollup.

### Goodput tagging

`managed_span` / `trace_fn` from `instrumentation` attach ``rl.bucket`` ∈
``{productive, overhead, idle, wasted}`` for leaf groups (see
`nemo_rl/telemetry/instrumentation.py`). Umbrella groups (`job`, `step`, `rollout`,
…) are **not** tagged. Apps do **not** emit rolled-up ``rl.goodput`` /
``rl.bucket.*`` metrics — the offline monitor SUMs tagged phase / span
durations by bucket.

To override classification for one site, pass the attribute explicitly:

```python
with managed_span(
    RLSpanGroup.GENERATION,
    "rl.vllm.generate",
    **{"rl.bucket": "productive"},
):
    ...
```

To reclassify spans opened *below* you — where the callee cannot tell why it was
called — wrap the region in `bucket_scope` instead. Validation uses this so its
generation counts as `overhead` rather than goodput:

```python
from nemo_rl.telemetry.instrumentation import Bucket, bucket_scope

with bucket_scope(Bucket.OVERHEAD):
    ...  # every leaf span in here is tagged overhead
```

Umbrellas stay unbucketed inside a scope, and an explicit `rl.bucket=` still
wins, so a scope cannot make a parent double-count its children. See
[span groups](span-groups.md).

When adding a **new** span group, update `_DEFAULT_GROUP_BUCKET` /
`UMBRELLA_GROUPS` in `instrumentation.py` and extend `test_instrumentation.py`.

## Instrumenting inside a Ray actor

Two things that are easy to get wrong, because both fail silently as no-ops
rather than as errors.

The actor needs its own providers: call `init_telemetry_worker()` in its
`__init__` (not `post_init`, which some fan-outs run on one rank per group), and
flush before it dies. An actor reaped with `ray.kill` runs no `atexit` handler,
so expose a method that calls `shutdown_telemetry()` and have the driver call it
first — `AsyncTrajectoryCollector.flush_telemetry` is the worked example.

Ray does not propagate OTel context, so the actor's spans form their own trace
unless you carry the parent across. On the driver, inside the span that should
be the root, capture `current_trace_carrier()` and pass it to the actor; in the
actor, wrap the work in `remote_trace_context(carrier)`. Reattach in **every
thread** the actor spawns — OTel context is a `ContextVar` and threads inherit
none — and note the carrier is empty (a harmless no-op) whenever the driver's
enclosing group is disabled. See
[span groups](span-groups.md#getting-the-collector-into-one-waterfall).

## Adding a span

### Decorator — `trace_fn`

For a whole function (this is how `rl.vllm.generate` and the `rl.<algo>.job` spans are done):

```python
@trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate")
def generate(self, ...):
    ...
```

### Group-gated block — `managed_span`

For a hot path where you want minimal cost when the group is disabled:

```python
with managed_span(RLSpanGroup.ROLLOUT, "rl.grpo.generation",
                  **{"rl.iteration": iteration}) as span:
    result = collect()
    if span is not None:
        span.set_attribute("rl.num_generations_per_prompt", n)
```

`managed_span` yields `None` when the group is disabled; the body still runs, so guard attribute-setting with `if span is not None`.

### Always-on block — `span_cm`

`span_cm` always creates a span when telemetry is active (no group gate) — for cold, top-level paths only:

```python
telemetry = get_telemetry_handle()
if telemetry is not None:
    with span_cm("rl.grpo.job", tracer=telemetry.tracer):
        ...
```

## Naming conventions

| Kind | Convention | Example |
|---|---|---|
| Span name | `rl.<algorithm>.<phase>`, matching the block's `Timer` key (the two umbrella spans excepted — see [span groups](span-groups.md#per-algorithm-span-names)) | `rl.grpo.generation` |
| Span tag | `rl.<attr>` categorical | `rl.iteration`, `rl.backend` |
| Resource attribute | `rl.<attr>` / shared `dl.<attr>` | `rl.model`, `dl.tensor_parallel.size` |
| Metric name | `rl.<subsystem>.<metric>` (application scope) | `rl.efficiency.seconds` |

Metric names use the **application scope** (`rl.*`) — never `dl.*`. Attribute names shared across consumers use the constants in `nemo.lens.semconv`; RL-specific short strings are fine hard-coded.

## Choosing a span group

Pick from `RLSpanGroup` before inventing a new one:

- Once per run (setup/whole-job)? → `job`
- Once per training step? → `step`
- Rollout collection? → `rollout`; generation? → `generation`
- Log-probs? → `logprob` (or `reference_policy` for the reference model)
- Reward / advantage / policy update? → `reward` / `advantage` / `policy_update`
- Checkpoint / eval? → `checkpoint` / `evaluate`

## Adding a new span group

If nothing fits, add a group to `RLSpanGroup` in `nemo_rl/telemetry/span_groups.py`:

1. Add the constant, add it to `ALL_GROUPS`, and slot it into the right preset(s) in `_PRESETS`. Decide per preset: `default` is coarse (rarely add here); `per_step` for per-step spans; `all` always includes it.
2. **Leave the fallback stub alone.** The stub `SpanGroup` in that file mirrors only lens's *base* groups and presets, for when nemo-lens is absent; `RLSpanGroup` overrides both `ALL_GROUPS` and `_PRESETS`, so an RL group belongs there and nowhere else. Touch the stub only when lens's own base contract changes.
3. Document the new group in [Span Groups](span-groups.md), and add it to `EMITTED_GROUPS` in `tests/unit/telemetry/test_span_groups.py` so the preset-reachability test covers it.

Keep the base-class contract — shared with lens and its other consumers — consistent when you do this.

## Adding a metric

NeMo-RL records its `rl.*` metrics from the driver rather than scattering record calls through the algorithm code (see [Metrics](metrics.md)). Two cases:

- **You need a brand-new instrument** (a new counter/gauge/histogram, or a value that doesn't go through the Logger). Add it to `nemo.lens.instruments.rl` following the per-Meter `WeakKeyDictionary` caching pattern, then record it from the driver via `telemetry.meter`. The `new-instrument` lens skill covers this.
- **The series is keyed by a NeMo-RL-specific label set** rather than a fixed field — e.g. one value per efficiency category. Lens's `record_rl_metrics` takes fixed keyword fields, so a growing label set doesn't fit it. Define the instrument in `nemo_rl/telemetry/metrics.py` instead (same per-Meter caching pattern) and emit **one dimensioned instrument** with the label as an attribute, not one instrument per label. `rl.efficiency.seconds` is the worked example. This also avoids gating the feature on a lens release.

A training scalar that already flows through `Logger.log_metrics` is a third case with no home yet: mapping NeMo-RL's logger keys onto lens's `record_rl_metrics` fields is still being settled with the lens owners, so the only tee on that hook today is the efficiency one.

Prefer an attribute over a name whenever the label set can grow: `rl.efficiency.seconds{rl.efficiency.category="idle/refit_bubble"}` stays stable as categories come and go, while `rl.efficiency.idle_refit_bubble_seconds` forces an instrument change per category. Keep attribute cardinality bounded — a per-step or per-request value belongs on a span, not a metric label.

Keep `rl.<subsystem>.<metric>` naming and record only non-`None` values. See [lens: metrics](https://github.com/NVIDIA-NeMo/Lens).

## Testing new instrumentation

NeMo-RL telemetry tests live under `tests/` and use lens's in-memory exporter fixtures (global OTel state reset per test). When adding a span:

1. Assert the span is emitted when its group is enabled and absent when disabled.
2. Assert on span name, tags, and parent relationships.

For a pure metrics-tee change, `map_efficiency_seconds` in `nemo_rl/telemetry/metrics.py` is a pure function — unit-test the key mapping directly with no OTel setup.

## When not to add instrumentation

- Inside a tight inner loop (per-token) — even a gated `managed_span`'s frozenset lookup adds up.
- On high-cardinality attributes (raw prompts, tensor shapes) — cardinality explosion at the backend.
- As a replacement for logging — structured logs belong in logs (correlate via the log bridge, `telemetry.logs_enabled: true`).

When in doubt, start with a coarse span at the boundary of a subsystem, not a fine-grained one at every internal call.
