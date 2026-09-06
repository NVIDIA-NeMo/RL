# Extending Instrumentation

To add new spans or metrics to NeMo-RL code, use the instrumentation primitives from nemo-lens (`managed_span`, `trace_fn`, `span_cm`). The primitives themselves are documented in [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens); this page covers NeMo-RL conventions.

## The import pattern

Every algorithm instrumentation import should go through
`nemo_rl.telemetry.instrumentation`, which re-exports the lens primitives with
`rl.bucket` tagging applied to leaf spans:

```python
from nemo_rl.telemetry.instrumentation import (
    Bucket,
    bucket_scope,
    managed_span,
    trace_fn,
    umbrella_span,
    umbrella_trace_fn,
)
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

### Umbrella spans say so at the call site

Whether a span carries a bucket is decided by its group, which used to make the
two cases indistinguishable where it matters most: `GENERATION` and `ROLLOUT`
are both plausible for a generation span, and only one of them enters the
rollup. So umbrella spans are opened through their own helper, with the group
spelled using its `U_` alias:

```python
from nemo_rl.telemetry.instrumentation import umbrella_span, umbrella_trace_fn

with umbrella_span(RLSpanGroup.U_ROLLOUT, "rl.sc.generate_and_push", tracer=tracer):
    ...
```

The `U_` names are aliases — `RLSpanGroup.U_ROLLOUT == RLSpanGroup.ROLLOUT ==
"rollout"` — so presets, the `span_groups` spec and every config are unchanged.
The six are `U_JOB`, `U_STEP`, `U_ROLLOUT`, `U_MODEL_INIT`, `U_EVALUATE` and
`U_SETUP`.

Reach for an umbrella whenever a span can overlap **another instance of
itself**: concurrent spans sum past the wall clock they happened in, so a bucket
on them multiplies rather than measures. The work is still counted by the leaf
spans nested inside.

Neither mistake raises — telemetry does not end a training run. Passing a leaf
group to `umbrella_span` warns once and emits the span as a leaf, so the phase
keeps the bucket it actually has; the reverse, `managed_span` on an umbrella
group, is already correct output. Both are rejected by a drift test, which
catches them at review time rather than partway through a run.

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
unless you carry the parent across. For a per-call actor method, use the
dispatch/receive pair rather than moving the carrier by hand — decorate the
method with `@accepts_trace_context` and dispatch it with one of:

```python
# A direct remote() call.
dispatch_with_trace_context(actor.run_rollouts.options(num_returns="streaming"), batch)

# RayWorkerGroup, which names the method by string and forwards **kwargs.
self.worker_group.run_all_workers_sharded_data(
    "train_presharded", meta=metas, common_kwargs={**common_kwargs, **trace_context_kwargs()}
)
```

Both halves are required and only the decorator is visible at the definition, so
a method decorated but dispatched without a carrier still emits root spans while
looking wired. `test_every_context_accepting_method_is_dispatched_with_a_carrier`
fails the build in that case; when you add a new decorated method, expect that
test to tell you if you forgot the sender.

Reach for `current_trace_carrier()` / `remote_trace_context()` directly only
when the context has to outlive a single call — the async trajectory collector
takes its carrier once at construction and reattaches it in **every thread** it
spawns, because OTel context is a `ContextVar` and threads inherit none. The
carrier is empty (a harmless no-op) whenever the driver's enclosing group is
disabled. See
[span groups](span-groups.md#getting-the-collector-into-one-waterfall).

## Adding a span

### Decorator — `trace_fn` / `umbrella_trace_fn`

For a whole function. `rl.vllm.generate` is a leaf:

```python
@trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate")
def generate(self, ...):
    ...
```

The `rl.<algo>.job` spans are umbrellas, so they take the other one:

```python
@umbrella_trace_fn(RLSpanGroup.U_JOB, "rl.grpo.job")
def grpo_train(...):
    ...
```

### Group-gated block — `managed_span` / `umbrella_span`

For a hot path where you want minimal cost when the group is disabled:

```python
with umbrella_span(RLSpanGroup.U_ROLLOUT, "rl.grpo.generation",
                   **{"rl.iteration": iteration}) as span:
    result = collect()
    if span is not None:
        span.set_attribute("rl.num_generations_per_prompt", n)
```

Both yield `None` when the group is disabled; the body still runs, so guard attribute-setting with `if span is not None`.

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

- Before the first step (Ray init, worker builds, dataloaders)? → `setup`, via `startup_span` / `setup_span`
- Once per run, covering the training loop? → `job`
- Once per training step? → `step`
- Rollout collection? → `rollout`; generation? → `generation`
- Log-probs? → `logprob`, the reference model's included
- Reward / advantage / policy update? → `reward` / `advantage` / `policy_update`
- Checkpoint / eval? → `checkpoint` / `evaluate`
- Transfer-queue op? → `data_plane`

One question cuts across all of those: **how many of these spans will a run
emit?** If the answer scales with the prompt or sample count rather than with
the step count, the span belongs in `per_prompt` regardless of which phase it
measures, so that a user picking `per_step` does not get dataset-sized volume
they did not ask for. `per_prompt` is an umbrella, so such spans carry no
`rl.bucket` — which is usually right anyway, since per-prompt work on the async
paths overlaps itself.

If the span is opened somewhere that cannot see whether its caller is
per-prompt — as `MetricsDataPlaneClient` cannot — wrap the caller's region in
`per_prompt_scope()` and consult `in_per_prompt_scope()` at the span site,
rather than threading a flag through the signatures.

## Adding a new span group

If nothing fits, add a group to `RLSpanGroup` in `nemo_rl/telemetry/span_groups.py`:

1. Add the constant and add it to `ALL_GROUPS`. `ALL_GROUPS` is what `register_span_groups()` declares to lens, so a group missing from it is never selectable — not even under `all`.
2. Slot it into the right preset(s) in `_PRESETS`. Decide per preset: `default` is coarse (rarely add here); `per_step` for per-step spans. There is no `all` to edit — lens resolves `all` as a wildcard over whatever is registered, so a group in `ALL_GROUPS` is reachable that way automatically. A group whose span count scales with dataset size rather than step count belongs in `all` only — see `per_prompt`, and add it to `PER_PROMPT_GROUPS` rather than `EMITTED_GROUPS` in the test below.
3. Document the new group in [Span Groups](span-groups.md), and add it to `EMITTED_GROUPS` in `tests/unit/telemetry/test_span_groups.py` so the preset-reachability test covers it.

Group names live in one flat namespace shared with every other library in the process, so a name Megatron also registers is *shared*: enabling it enables both libraries' spans. That is intended for the phases both genuinely have (`step`, `optimizer`), and worth avoiding otherwise — lens logs a warning naming the other claimant.

## Adding a metric

NeMo-RL records its `rl.*` metrics from the driver rather than scattering record calls through the algorithm code (see [Metrics](metrics.md)). NeMo-RL owns every `rl.*` metric name: they are declared with lens's metric registry, so adding one needs no lens change.

**If the value already flows through `Logger.log_metrics`** — the common case — you only add a row. Append a `_TeedMetric` to `_TRAIN_SCALARS` (or `_VLLM_STEP_METRICS`) in `nemo_rl/telemetry/metrics.py`, giving the logger key, the registry key, the OTel name and the kind:

```python
_TeedMetric(
    "my_logger_key", "my_metric", "rl.my.metric", kind="gauge", description="..."
)
```

The row is both the declaration and the mapping, which is deliberate: an earlier design kept a separate key-to-field map, three of its entries pointed at keys nothing emitted, and those gauges reported a flat line instead of an error. Keep them in one row and `tests/unit/telemetry/test_source_drift.py` fails the build when a logger key stops being emitted.

**If the series is keyed by a growing label set** — e.g. one value per efficiency category — declare **one dimensioned series** and pass the label through `attributes`, rather than one series per label. `rl.efficiency.seconds` is the worked example; `rl.setup.duration` is the same shape for a label set that is not even knowable at declaration time, since `SetupTimingMetrics.extras` is filled in at runtime.

**If the dict arrives under a prefix other than `train`**, add the dispatch in `_tee_rl_metrics_to_otel`. Prefixes are matched rather than merged so neither family is scanned for the other's keys: `timing/setup` arrives once at step 0 and shares no key with the per-step dicts.

**If the value does not go through the Logger at all**, declare it in the same table and call `record_metrics(meter, RL_METRIC_GROUP, {...})` from wherever it is produced.

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
