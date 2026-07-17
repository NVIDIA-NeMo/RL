# Extending Instrumentation

To add new spans or metrics to NeMo-RL code, use the instrumentation primitives from nemo-lens (`managed_span`, `trace_fn`, `span_cm`). The primitives themselves are documented in [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens); this page covers NeMo-RL conventions.

```{tip}
If you work in this repo with Claude Code, the `add-span-group` skill (new span group), the `new-instrument` lens skill (new `rl.*` metric), and the `instrumentation-site-helper` agent (new span/metric site) automate the steps below and keep the cross-repo fallback contract in sync. They are optional — everything here can be done by hand.
```

## The import / fallback pattern

Every lens import in NeMo-RL code must go through `nemo_rl.telemetry._fallbacks`, so the code runs unchanged when nemo-lens is not installed:

```python
from nemo_rl.telemetry._fallbacks import managed_span, trace_fn
from nemo_rl.telemetry.setup import get_telemetry
from nemo_rl.telemetry.span_groups import RLSpanGroup
```

`_fallbacks.py` re-exports the real nemo-lens implementations when it is installed, and provides identical no-op stubs when it is not. Never import from `nemo.lens.*` directly in algorithm code. See [lens: optional dependency](https://github.com/NVIDIA-NeMo/Lens).

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
with managed_span(RLSpanGroup.ROLLOUT, "rl.grpo.collect_rollouts",
                  **{"rl.iteration": iteration}) as span:
    result = collect()
    if span is not None:
        span.set_attribute("rl.num_generations_per_prompt", n)
```

`managed_span` yields `None` when the group is disabled; the body still runs, so guard attribute-setting with `if span is not None`.

### Always-on block — `span_cm`

`span_cm` always creates a span when telemetry is active (no group gate) — for cold, top-level paths only:

```python
telemetry = get_telemetry()
if telemetry is not None:
    with span_cm("rl.grpo.job", tracer=telemetry.tracer):
        ...
```

## Naming conventions

| Kind | Convention | Example |
|---|---|---|
| Span name | `rl.<algorithm>.<operation>` | `rl.grpo.collect_rollouts` |
| Span tag | `rl.<attr>` categorical | `rl.iteration`, `rl.backend` |
| Resource attribute | `rl.<attr>` / shared `dl.<attr>` | `rl.model`, `dl.tensor_parallel.size` |
| Metric name | `rl.<subsystem>.<metric>` (application scope) | `rl.reward.mean` |

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
2. **Update the fallback stub** in the same file — the stub `SpanGroup` used when nemo-lens is absent must keep the same constants and presets in lockstep.
3. Document the new group in [Span Groups](span-groups.md).

The `add-span-group` skill walks these steps and keeps the base-class contract (shared with lens and the other consumers) consistent.

## Adding a metric

The `rl.*` gauges are populated by teeing `Logger.log_metrics` (see [Metrics](metrics.md)) — not by scattering `record_rl_metrics()` calls. So there are two cases:

- **The scalar already flows through `Logger.log_metrics`** under a `train` prefix but isn't teed. Add a candidate key (or a new field) to `_RL_OTEL_METRIC_MAP` in `nemo_rl/telemetry/metrics.py`, and add the matching field to `record_rl_metrics` in lens's `nemo.lens.instruments.rl`.
- **You need a brand-new instrument** (a new counter/gauge/histogram, or a value that doesn't go through the Logger). Add it to `nemo.lens.instruments.rl` following the per-Meter `WeakKeyDictionary` caching pattern, then record it from the driver via `telemetry.meter`. The `new-instrument` lens skill covers this.

Keep `rl.<subsystem>.<metric>` naming and record only non-`None` values. See [lens: metrics](https://github.com/NVIDIA-NeMo/Lens).

## Testing new instrumentation

NeMo-RL telemetry tests live under `tests/` and use lens's in-memory exporter fixtures (global OTel state reset per test). When adding a span:

1. Assert the span is emitted when its group is enabled and absent when disabled.
2. Assert on span name, tags, and parent relationships.

For a pure metrics-tee change, `map_rl_metrics` in `nemo_rl/telemetry/metrics.py` is a pure function — unit-test the key mapping directly with no OTel setup.

## When not to add instrumentation

- Inside a tight inner loop (per-token) — even a gated `managed_span`'s frozenset lookup adds up.
- On high-cardinality attributes (raw prompts, tensor shapes) — cardinality explosion at the backend.
- As a replacement for logging — structured logs belong in logs (correlate via the log bridge, `NEMO_RL_OTEL_LOGS_ENABLED=1`).

When in doubt, start with a coarse span at the boundary of a subsystem, not a fine-grained one at every internal call.
