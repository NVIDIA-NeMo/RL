# Span Groups

Span granularity in NeMo-RL is controlled by `span_groups` in the [`telemetry:` config block](configuration.md). The spec accepts a preset keyword, a comma-separated list of individual group names, or a mix (e.g. `default,generation,reward`).

For the general span-group mechanism — how gating works, why a disabled group costs ~nothing — see [lens: span groups](https://github.com/NVIDIA-NeMo/Lens). This page covers NeMo-RL's groups and the per-algorithm span hierarchy.

## Preset keywords

| Preset | Groups included | Relative cost |
|---|---|---|
| `default` | `job`, `checkpoint`, `evaluate`, `setup`, `model_init` | Lowest — safe for production |
| `per_step` | `step`, `checkpoint`, `evaluate`, `setup`, `model_init`, `rollout`, `generation`, `logprob`, `reward`, `advantage`, `policy_update`, `reference_policy`, `data_processing`, `data_plane`, `efficiency` | Moderate |
| `all` | every group (`job` and `per_prompt` included) | Highest — dev/debug |

### `per_step` deliberately omits `per_prompt`

Every group in `per_step` emits a bounded number of spans per training step, so the preset's cost scales with steps. `per_prompt` is the exception: its spans are emitted once per prompt, so a 10k-prompt rollout produces roughly 20k spans where the phase groups produce a fixed handful. Folding it in would make `per_step` scale with dataset size instead, which is not what someone asking for step detail is asking for.

Ask for it explicitly when debugging an individual rollout:

```yaml
span_groups: per_step,per_prompt
```

or take `all`. See [Per-prompt spans](#per-prompt-spans) for what you get.

### `per_step` deliberately omits `job`

`per_step` **excludes** the `job` group on purpose. `job` is the whole-run root span; if it were enabled alongside `step`, every training step would nest under one giant, ever-growing trace. Omitting `job` makes **each training step its own root trace** — bounded in size and easy to search one step at a time.

`job` lives in `default` (coarse: startup + job + checkpoint + evaluate) and in `all` (one whole-run trace, useful for a short run). Choose `per_step` when you want to inspect individual steps; choose `default`/`all` when you want one trace spanning the run.

## `RLSpanGroup`

Defined in `nemo_rl/telemetry/span_groups.py`, and declared to lens by `register_span_groups()` when that module is imported.

nemo-lens ships **no** span-group names of its own: each consuming library registers what it emits into lens's `SpanRegistry` under its own namespace (NeMo-RL's is `nemo_rl`), and `telemetry.span_groups` resolves against whatever is registered in the process. So the groups marked "base" below — the ones lens used to define — are declared by NeMo-RL too. They keep their original names, which are also the names Megatron registers for the same phases; a group two libraries both register is shared, and enabling it turns on both.

One consequence worth knowing: registration is an *import side effect*, so a group is only selectable once its module has been imported. `nemo_rl/telemetry/setup.py` imports this module before calling `setup_telemetry` for that reason.

| Group | Origin | Controls |
|---|---|---|
| `job` | base | the whole-run root span (`rl.<algo>.job`) |
| `setup` | RL | startup, before the first step: `rl.startup` and the `rl.setup.*` phases under it |
| `checkpoint` | base | `rl.<algo>.checkpointing` |
| `evaluate` | base | `rl.<algo>.evaluate` |
| `model_init` | base | `rl.vllm.load_model` (generation worker), `rl.policy.load_model` / `rl.value.load_model` (training workers) |
| `load_checkpoint` | base | *reserved — bucketed, but no site emits it yet* |
| `step` | base | `rl.<algo>.step` (one per training step) |
| `forward_backward` | base | *reserved — bucketed, but no site emits it yet* |
| `optimizer` | base | *reserved — bucketed, but no site emits it yet* |
| `rollout` | RL | `rl.<algo>.generation` |
| `generation` | RL | the driver-side `rl.vllm.generate` / `rl.vllm.generate_text` spans |
| `logprob` | RL | `rl.<algo>.policy_and_reference_logprobs`, and `rl.distillation.teacher_logprob_inference` on the distillation path |
| `reward` | RL | `rl.<algo>.reward_calculation` |
| `advantage` | RL | `rl.<algo>.advantage_calculation` |
| `policy_update` | RL | `rl.<algo>.policy_training` (and `rl.ppo.value_training` for PPO) |
| `reference_policy` | RL | *reserved — bucketed, but no site emits it yet* |
| `data_processing` | RL | `rl.<algo>.data_processing` |
| `data_plane` | RL | `rl.data_plane.<op>` — one span per transfer-queue operation (`put`, `get`, `claim_meta`, `clear`, …) from a batch-shaped caller |
| `per_prompt` | RL | spans emitted once per prompt: `rl.sc.generate_and_push` and the rollout path's `rl.data_plane.put`. A cardinality axis rather than a phase — see [Per-prompt spans](#per-prompt-spans) |
| `efficiency` | RL | idle phases on async GRPO — driver-side `rl.idle.buffer_starvation`, `rl.idle.refit_bubble`, and collector-side `rl.idle.refit_event_wait`, `rl.idle.generation_limit_pause` |

## Examples

```yaml
telemetry:
  enabled: true

  # Coarse spans only — default
  span_groups: default

  # Per-step traces (rollout / generation / reward / advantage / policy update)
  # span_groups: per_step

  # Coarse job trace + generation spans only
  # span_groups: default,generation

  # Everything
  # span_groups: all
```

## Per-algorithm span names

Span names follow `rl.<algorithm>.<phase>`, where `<phase>` is the `Timer` key the same block records — so a span and the `timing/train/<phase>` metric measuring it carry one name rather than two, and correlating a slow span with its timing series needs no mapping. The `Timer` key is the authority: it is pre-existing and already published as a metric name, so a new span takes its name from the timer rather than the reverse.

Two spans deliberately do not follow it. `rl.<algorithm>.step` wraps `total_step_time` and `rl.<algorithm>.evaluate` wraps `total_validation_time`: a span's duration is intrinsic, so naming one after a `total_*_time` measurement is tautological, and these two are the umbrella spans a reader meets first in a waterfall. They are named after the operation instead. Every span *inside* them matches its timer key.

The controlling group is shown for each; a span is only emitted when its group is enabled *and* the rank is exporting.

| Algorithm | Spans |
|---|---|
| **Startup** (all algorithms) | `rl.startup` and, under it, `rl.setup.ray_init`, `rl.setup.tokenizer`, `rl.setup.data`, `rl.setup.nemo_gym_config`, `rl.setup.workers` — `setup` group; see [Startup](#startup-what-happens-before-the-first-step) |
| **GRPO** (sync + async) | `rl.grpo.job`, `rl.grpo.step`, `rl.grpo.data_processing`, `rl.grpo.generation`, `rl.grpo.reward_calculation`, `rl.grpo.policy_and_reference_logprobs`, `rl.grpo.advantage_calculation`, `rl.grpo.policy_training`, `rl.grpo.checkpointing`, `rl.grpo.evaluate` |
| **GRPO** (async only) | `rl.idle.buffer_starvation`, `rl.idle.refit_bubble` (driver) and `rl.idle.refit_event_wait`, `rl.idle.generation_limit_pause` (collector actor) — `efficiency` group; named after the `Timer` category, not the algorithm |
| **GRPO / PPO** (async only) | `rl.grpo.generation` / `rl.ppo.generation` — `rollout` group, emitted by the collector actor, one span per rollout batch; the name follows the algorithm the collector was built for |
| **PPO** | `rl.ppo.job`, `rl.ppo.step`, `rl.ppo.data_processing`, `rl.ppo.generation`, `rl.ppo.reward_calculation`, `rl.ppo.policy_and_reference_logprobs`, `rl.ppo.advantage_calculation`, `rl.ppo.policy_training`, `rl.ppo.value_training`, `rl.ppo.checkpointing`, `rl.ppo.evaluate` |
| **SFT** | `rl.sft.job`, `rl.sft.step`, `rl.sft.data_processing`, `rl.sft.policy_training`, `rl.sft.checkpointing`, `rl.sft.evaluate` |
| **DPO** | `rl.dpo.job`, `rl.dpo.step`, `rl.dpo.policy_training`, `rl.dpo.checkpointing`, `rl.dpo.evaluate` |
| **RM** | `rl.rm.job`, `rl.rm.step`, `rl.rm.checkpointing`, `rl.rm.evaluate` |
| **Distillation** | `rl.distillation.job`, `rl.distillation.step`, `rl.distillation.data_processing`, `rl.distillation.generation`, `rl.distillation.teacher_logprob_inference`, `rl.distillation.policy_training`, `rl.distillation.checkpointing`, `rl.distillation.evaluate` |
| **SingleController** | `rl.sc.job`, `rl.sc.step`, `rl.sc.logprob_inference_prep`, `rl.sc.policy_and_reference_logprobs`, `rl.sc.value_inference_prep`, `rl.sc.value_inference`, `rl.sc.advantage_calculation`, `rl.sc.training_prep`, `rl.sc.policy_training`, `rl.sc.policy_optimizer_step`, `rl.sc.value_training`, `rl.sc.checkpointing` — opened inside the `SingleControllerActor`, which is where the run actually lives |
| **SingleController** (rollout) | `rl.sc.generate_and_push` — `per_prompt` group (umbrella, so unbucketed), one span per dispatch attempt; and `idle/buffer_starvation` / `idle/refit_bubble` reusing async GRPO's `efficiency` category names |
| **Transfer queue** | `rl.data_plane.<op>` — `data_plane` group, or `per_prompt` when the caller is a rollout; emitted wherever a data-plane client is built (SC actor, `TQPolicy` / `TQValue` workers) |
| **vLLM** (driver-side) | `rl.vllm.generate`, `rl.vllm.generate_text` — `generation` group; nested under the active rollout span |
| **vLLM** (worker-side) | `rl.vllm.load_model` — `model_init` group; a root span in the generation worker's process, since Ray carries no trace context into `__init__` |
| **Policy / value** (worker-side) | `rl.policy.load_model`, `rl.value.load_model` — `model_init` group, `rl.backend` attribute; opened by `traced_worker_init` on the worker's `__init__`, and root spans for the same reason |

`rl.<algo>.job` is a function-level span (via `umbrella_trace_fn`) wrapping the whole run. Under `per_step` it is suppressed, so each `rl.<algo>.step` becomes a root trace.

## Span tags (categorical attributes)

These are set on spans for filtering — they answer "which one?" / "what kind?", not "how much?". Numerical values that change over time are **metrics**, not span tags (see [Metrics](metrics.md)).

| Tag | Meaning |
|---|---|
| `rl.iteration` | training iteration index |
| `rl.epoch` | epoch index (omitted on `rl.sc.step` — see below) |
| `rl.step` | step index |
| `rl.num_generations_per_prompt` | GRPO group size |
| `rl.weight_version` / `rl.target_weight_version` | async rollout batch: the weights it generated from, and the training step it targets |
| `rl.num_prompt_groups` | async rollout batch width, so a gap-filling batch is not read as an unexplained speed-up |
| `rl.rollout.attempt` | SingleController dispatch attempt: `0` is a first try, `> 0` a substitution after a skipped group, whose tokens were discarded |
| `rl.bucket` | goodput bucket: `productive` / `overhead` / `idle` / `wasted` (omit on umbrellas) |

`rl.sc.step` carries `rl.iteration` and `rl.weight_version` but no `rl.epoch`.
In the SingleController the rollout pump advances the epoch on its own clock, so
the epoch counter at the time a train step runs describes how far *generation*
has read into the dataset — not the epoch this step's batch came from. Use the
rollout-side spans for that.

### Span group → `rl.bucket`

Leaf groups are tagged automatically when using
`nemo_rl.telemetry.instrumentation.managed_span` / `trace_fn`. Umbrellas are
timed but **not** tagged so monitors can exclude them from goodput, and they are
opened through `umbrella_span` / `umbrella_trace_fn` with the group's `U_` alias
so the call site shows which of the two it is — see
[Extending](extending.md#umbrella-spans-say-so-at-the-call-site).

| Group | `rl.bucket` |
|---|---|
| `job`, `step`, `rollout`, `model_init`, `evaluate`, `setup`, `per_prompt` (aliased `U_JOB`, `U_STEP`, …) | *(none — umbrella)* |
| `generation`, `reward`, `policy_update`, `forward_backward`, `optimizer` | `productive` |
| `data_processing`, `data_plane`, `checkpoint`, `load_checkpoint`, `logprob`, `advantage`, `reference_policy` | `overhead` |
| `efficiency` | `idle` for the two driver-side phases; *none* for the two collector-side ones (see below) |

Rolled-up `rl.goodput` is **monitor-derived**, not emitted by NeMo-RL.

#### Overriding the bucket for a region: `bucket_scope`

The table above classifies by *what ran*, but a few phases are productive or not
depending on *why* they ran. `bucket_scope(bucket)` reclassifies every leaf span
opened inside it:

```python
with bucket_scope(Bucket.OVERHEAD):
    ...  # generation in here is tagged overhead, not productive
```

The one production use is validation, in `grpo.validate` and `ppo.validate`.
Validation generates through the same `generation` group as a training rollout,
but its tokens are scored and discarded, so `productive` would count a
validation pass as goodput. The span is opened by a decorator on
`VllmGeneration.generate` that cannot see its caller, which is why the scope
travels with the execution context (a `ContextVar`) rather than an argument.

Three properties keep it from creating the double-counting problem it exists to
avoid: umbrellas stay unbucketed inside a scope, an explicit `rl.bucket=` passed
to `managed_span` still wins, and an `efficiency_span` keeps its category's
bucket — that one names the phase it measures, so a caller cannot make
`idle/refit_bubble` productive. It propagates into coroutines started
inside the block — `asyncio.run`, as the rollout entrypoints use — but not into
threads or Ray actors, so a worker-side span is unaffected.

### The `efficiency` group: idle time on async runs

Async GRPO measures its stalls with `Timer` under labels like
`idle/buffer_starvation`, on both sides of the run: the driver waiting on the
collector, and the collector waiting on the driver. `efficiency_span` in
`nemo_rl/telemetry/instrumentation.py` emits those as spans, taking the bucket
from `EFFICIENCY_CATEGORY_BUCKET` so `idle/*` lands in `idle` rather than
defaulting to `overhead`. Each span also carries
`rl.efficiency.category` with the raw label, so idle time can be grouped by
cause without parsing the span name.

Two driver-side phases are wired today, both children of `rl.grpo.step`:

| Span | Category | Bucket | What the driver is waiting on |
|---|---|---|---|
| `rl.idle.buffer_starvation` | `idle/buffer_starvation` | `idle` | replay buffer is empty — the collector is not keeping up |
| `rl.idle.refit_bubble` | `idle/refit_bubble` | `idle` | collector reaching a safe point, then weight sync |

With these enabled, a step's child spans account for much more of the step
duration, so a per-step goodput breakdown leaves a smaller unattributed gap.

#### Why `idle/validation` is not a span

`idle/validation` is driver-side wall-clock like the two above, but it stays
`Timer`-only, because the window it measures is already accounted as
**`overhead`**: `validate()` wraps its generation in `bucket_scope`, so a second
span calling the same interval `idle` would contradict the label and, wherever
those generate spans exist, be counted twice — a rollup sums durations by
`rl.bucket` with no notion of nesting, so the pass would read as nearly double
its wall time.

Whether the children exist depends on the rollout path: sync validation
generates through the traced `rl.vllm.generate`, while async validation goes
through `generate_async`, which carries no span today. The `overhead`
attribution is the same either way, which is why this is `Timer`-only in both.

This is the general rule for `efficiency_span`: **wrap a wait, not a phase that
does instrumented work.** A bucketed span must be a leaf, which is the same
invariant the umbrella groups exist to preserve.

One gap remains: the phase means different things per fleet. The training GPUs
are idle while the generation GPUs do necessary non-training work, and
`overhead` on the generate span describes the latter only. Attributing the
former needs per-fleet accounting, not a per-phase bucket. Note also that the
`val_at_start` pass has no efficiency timer, so it appears in spans (as
`overhead` generation) but not in `efficiency/*`.

#### Trace-only: the collector's two loop waits

`idle/refit_event_wait` and `idle/generation_limit_pause` are emitted as spans
from inside the `AsyncTrajectoryCollector`, but **without** `rl.bucket`:

| Span | Category | Bucket | What it means |
|---|---|---|---|
| `rl.idle.refit_event_wait` | `idle/refit_event_wait` | *none* | collection loop parked while a refit completes |
| `rl.idle.generation_limit_pause` | `idle/generation_limit_pause` | *none* | every target weight already has enough trajectories |

Both are `Event.wait()` calls on the single collection-loop thread, so they are
honest wall-clock durations — but the collector's wall clock runs *concurrently*
with the driver's, so summing them against a driver-side denominator would
overcount. Omitting the attribute keeps them out of a bucket rollup by
construction instead of by convention. The membership list is
`COLLECTOR_LOOP_CATEGORIES` in `nemo_rl/telemetry/instrumentation.py`, which
`UNBUCKETED_SPAN_CATEGORIES` extends with `init/total`.

They still carry `rl.efficiency.category`, so they remain identifiable in a
trace and continue to be reported as `efficiency/*` scalars. As metrics they are
labelled `rl.efficiency.measurement="collector_wall_clock"` — sequential and so
real durations, unlike the batch-worker categories, but on the collector's
timeline rather than the driver's. See
[Metrics — always filter on `rl.efficiency.measurement`](metrics.md#always-filter-on-rlefficiencymeasurement).

#### Still reserved: the rest of the collector-side categories

`idle/buffer_full_backoff` and `wasted/failed_trajectory` stay `Timer`-only.
Both run in the batch-worker threads, so they are genuinely *thread-seconds* —
several workers accumulate at once and the total can exceed the wall time it
happened in. `idle/buffer_full_backoff` also has no clean block to wrap: it is
recorded as a precomputed duration spanning a retry loop. `wasted/failed_trajectory`
covers the same window as the enclosing `rl.grpo.generation` span, so a span
there would duplicate an existing interval.

So goodput on async runs covers driver idle, but not collector-side idle or
wasted work — use the `efficiency/*` metrics for those.

## Startup: what happens before the first step

The `setup` group covers everything between process start and the first training
step. It is in **both** shipped presets: startup is a fixed handful of spans
emitted once per run, so it costs nothing at steady state, and "why was the first
step so late" is a question a coarse preset needs to answer. `model_init` is in
both presets for the same reason and travels with it — without it the worker
build shows as one opaque block with `rl.vllm.load_model`, usually its largest
part, missing from inside.

```
rl.startup                                    (launcher)
├── rl.setup.ray_init                          rl.ray.cluster_source=started_local
├── rl.setup.tokenizer
├── rl.setup.data                              (run_grpo.py)
├── rl.setup.nemo_gym_config                   (single controller, if gym is on)
└── rl.setup.workers
    ├── rl.vllm.load_model                    (generation worker, separate trace)
    ├── rl.policy.load_model                  (training worker,   separate trace)
    └── rl.value.load_model                   (PPO critic,        separate trace)
```

The three worker-side loads carry `rl.backend` — `megatron`, `dtensor` or
`dtensor_v2` for the trainer, so one query compares the same phase across
backends — and each is emitted once per worker process, so a slow rank shows up
as one long span among its peers rather than an average.

`rl.startup` exists because `init_ray()` and the algorithm's `setup()` are
separate top-level calls in the launcher; without a span across them their
phases arrive as unrelated root traces. It closes before training, so the run's
`job` and `step` spans stay separate traces.

`rl.setup.ray_init` is emitted by `init_ray()` itself, so every launcher gets it
without opting in. It carries `rl.ray.cluster_source`:
`attached_external` (a cluster ray.sub or KubeRay already had running),
`reused_local` (one an earlier NeMo-RL run left behind), or `started_local` (paid
to boot a new one). Attaching and booting differ by tens of seconds, which is the
usual reason two otherwise identical runs disagree on time-to-first-step.

### No bucket on any startup span

Unlike the other leaf groups, `setup` spans carry no `rl.bucket` at all. The
phases nest (`rl.startup` over `rl.setup.workers` over `rl.vllm.load_model`) and
the worker builds run *concurrently* under parallel init, so a rollup adding
them by bucket would multiply startup rather than measure it. The flat number
lives in the `rl.setup.duration` metric at `phase=total_setup` — see
[Metrics — startup phases](metrics.md#startup-phases). These spans are for shape.

### The initial buffer fill

On async GRPO the driver blocks before its first step until the replay buffer
holds a full batch. That wait is `rl.init.total` (category `init/total`), a child
of no step — it happens before the loop.

It carries **no** `rl.bucket`, for the same reason as the collector's loop waits:
the generation fleet is busy for that entire window and its
`rl.grpo.generation` spans join the same trace, so bucketing the wait would
charge one stretch of wall clock to two buckets. The `init/total` *metric* keeps
its `overhead` bucket, because it is read as a single per-run number rather than
summed beside sibling spans. The membership list for this rule is
`UNBUCKETED_SPAN_CATEGORIES` in `nemo_rl/telemetry/instrumentation.py`.

Async PPO records the same `init/total` timer but emits no span for it yet.

### Async rollout spans come from the collector actor

In an async run, no rollout is generated on the driver. Every trajectory comes
from inside `AsyncTrajectoryCollector`, a separate Ray actor, which calls
`init_telemetry_worker(rank=0, world_size=1)` in its constructor. Explicit rank
because it is a singleton, not a member of a ranked group, and its `runtime_env`
is a copy of the driver's environment — so without this its spans would carry
whatever `RANK` the driver happened to have. The driver reports itself the same
way, for the same reason.

It flushes through `flush_telemetry()`, which the driver calls before `ray.kill`,
since a kill runs no `atexit` handler. That call stops the collection loop and
waits (bounded) for in-flight batch workers first: the shutdown is terminal, so
a still-running thread would keep opening spans against a dead processor.

Each batch worker opens one `rl.grpo.generation` span — the same name the sync
path uses on the driver, so the two modes read alike — carrying
`rl.weight_version`, `rl.target_weight_version`, `rl.num_generations_per_prompt`
and `rl.num_prompt_groups`. The last one is the batch width: a gap-filling batch
covers a fraction of a full one, so without it a short span looks like an
unexplained speed-up. It is in the `rollout` group, so it is an
umbrella and carries **no** `rl.bucket`: several batch workers run at once, so
their durations sum past wall time and cannot enter a bucket rollup.

`rl.sc.generate_and_push` is an umbrella for the same reason, though it sits in
`per_prompt` rather than `rollout` (see [Per-prompt spans](#per-prompt-spans)).
The SingleController dispatches one asyncio task per prompt group, bounded by
`async_rl.max_inflight_prompts` — `num_prompts_per_step` in most recipes and
`1280` in one — so that many spans can be open at once. Tagged `productive` they
would sum to a large multiple of the wall clock they happened in. On both paths
the productive generation term comes from the worker-side `rl.vllm.generate`
spans, not from these driver-side dispatch spans.

It also carries `rl.rollout.attempt`, because the span is opened inside the
dispatch retry loop: a skipped group is substituted in place and the loop opens
another span. `attempt > 0` is generation whose tokens were discarded — the same
thing async GRPO reports as `wasted/failed_trajectory`.

#### Getting the collector into one waterfall

Ray does not propagate OTel context, so an actor's spans start their own trace
by default. The driver captures its active span as a W3C `traceparent` carrier
with `current_trace_carrier()` — taken inside `rl.grpo.job`, at the point the
collector is constructed — and passes it as the actor's `trace_carrier`
argument. The collector reopens it with `remote_trace_context()` in **both** the
collection-loop thread and every batch-worker thread. Per thread, not once per
process: OTel context is a `ContextVar`, and `threading.Thread` inherits none.

The result is a single trace per run:

```
rl.grpo.job                                   (driver)
├── rl.grpo.step  (iteration 1)               (driver)
│   ├── rl.idle.buffer_starvation
│   └── rl.grpo.policy_training
├── rl.grpo.generation  weight=7              (collector, thread A)
├── rl.idle.generation_limit_pause            (collector, loop thread)
├── rl.grpo.generation  weight=8              (collector, thread B)
└── rl.grpo.step  (iteration 2)               (driver)
```

**This requires the `job` group to be enabled.** `current_trace_carrier()`
returns an empty dict when no span is recording, and `remote_trace_context({})`
is a no-op, so the collector falls back to root spans. The `default` preset has
`job` but not `rollout`/`efficiency`, so the collector emits nothing; `per_step`
has `rollout`/`efficiency` but deliberately omits `job` so each step is its own
bounded trace. For the unified view, ask for both:

```yaml
telemetry:
  span_groups: per_step,job   # or: all
```

Be deliberate about it. A run-long root span means one trace accumulating every
step and every rollout batch for the whole job, which is exactly the trace-size
problem `per_step` exists to avoid. Prefer it for debugging a specific run, not
as a standing default on long jobs.

Two consequences worth internalizing before reading an async trace:

- **One span per batch, not per sample.** `generate_async` is dispatched one
  coroutine per sample, so spanning it would emit thousands of mutually
  overlapping spans per step.
- **There is no `productive` generation span in async mode**, and there cannot
  usefully be one. `rl.vllm.generate` is only reached through the synchronous
  rollout path, and an async run never takes it: `async_grpo_train` requires an
  async generation engine, so even validation goes through `generate_async`,
  which carries no span today. Generation is a
  continuously-batched pipeline overlapping training, so its productive
  contribution is a utilization question — answered by fleet metrics — not a
  span duration. A span-derived goodput ratio on an async run therefore has no
  productive generation term; do not read it as "generation contributed
  nothing."

## Per-prompt spans

Two spans on the SingleController path are emitted once per prompt rather than
once per step or per batch:

| Span | Emitted | Bucket |
|---|---|---|
| `rl.sc.generate_and_push` | one per dispatch attempt, in the SC actor | none (umbrella) |
| `rl.data_plane.put` | one per group commit, nested inside the above | none (umbrella) |

They share the `per_prompt` group, which is a **cardinality axis** rather than a
phase — the one group in `RLSpanGroup` that does not name a stage of work. What
governs whether you want these is not that one is a rollout span and the other a
transfer-queue span; it is that their count scales with the prompt count. A
10k-prompt rollout emits roughly 20k of them, against a fixed handful per step
from every phase group. That is why they are reachable only from `all` or an
explicit `per_step,per_prompt`.

If 20k per rollout is more than you want but you still need the rollout view,
enable `per_prompt` and drop `data_plane`: transfer-queue time inside a rollout
also shows up as a gap between the `rl.sc.*` phase spans.

### Why the group has to come from the caller

`rl.data_plane.put` is emitted by `MetricsDataPlaneClient`, and there is **one
client per process**, shared by callers with wildly different cardinality:

```
SC actor process, one client from build_data_plane_client()
  ├─ RolloutManager → TQReplayBuffer.commit → put_samples()   once per prompt
  └─ _advantage_stage(meta)                → put_samples()    once per batch
```

Same op, same client, counts orders of magnitude apart. So neither the op name
nor a constructor argument can distinguish them, and the client cannot see its
caller. The rollout path therefore marks its region with
`instrumentation.per_prompt_scope()`, a `ContextVar` the client consults — the
same mechanism [`bucket_scope`](#overriding-the-bucket-for-a-region-bucket_scope)
uses for the same reason. Entering the scope also means the put is gated *with*
the rollout span: turn `per_prompt` off and both disappear.

A rollout's put is unbucketed where a batch stage's put is `overhead`. That is
deliberate: rollouts overlap each other and training, so summing their durations
into `overhead` would push that bucket past the wall clock it happened in, by up
to `max_inflight_prompts`. The scope propagates to nested calls and to
coroutines started inside it, but not to raw threads — a data-plane call handed
to a thread pool would read as batch-shaped.

## Coverage gaps

A group being enabled does not guarantee spans: something has to emit them. Known
blanks today, so an empty trace is not read as a broken exporter:

| Area | State |
|---|---|
| SGLang / TRT-LLM / Megatron generation workers | uninstrumented — no `init_telemetry_worker` and no generation spans; only vLLM emits `rl.vllm.*`. (Policy and value workers do initialise telemetry, so their metrics and any future spans are wired.) |
| `VllmGeneration.generate_async` | no span, so async rollouts and async validation have no generate breakdown under `rl.grpo.generation` / `rl.grpo.evaluate` |
| `SyncRolloutActor` | the sync data-plane counterpart of the async collector — uninstrumented, so its rollouts produce no spans |
| Worker flush outside async GRPO | only `async_grpo_train` calls `policy.shutdown()` / `policy_generation.shutdown()`, so on other trainers a worker's last spans depend on the periodic export rather than a flush |
| `load_checkpoint`, `forward_backward`, `optimizer`, `reference_policy` | the groups are defined and bucketed, but no site emits them, so enabling them adds no spans |
| `grpo_sync.py` | no spans |
| Startup phases inside `setup()` | `rl.setup.workers` is one block; its sub-phases run concurrently in worker threads, which OTel context does not reach, so they would detach into their own traces. Read the `rl.setup.duration` metric for the breakdown |
| `rl.startup` outside GRPO | only `run_grpo.py` and `run_grpo_single_controller.py` open the umbrella, so on other launchers `rl.setup.ray_init` is a root span rather than part of a startup waterfall |
| `rl.init.total` on async PPO | the timer is recorded, but `async_ppo_train` is otherwise uninstrumented, so the span is not emitted there |
| SingleController generation / trainer workers | the actor's own phases are instrumented, but `TQPolicy` / `TQValue` / generation workers get no trace context from it, so their spans are separate traces correlated by `run_id` |
| `run_vlm_grpo.py`, `run_grpo_sliding_puzzle.py`, `run_xtoken_off_policy_distillation.py`, `run_eval.py` | these call the instrumented loops but never `init_telemetry_driver`, so a `telemetry:` block in their configs parses, the run succeeds, and nothing is emitted — driver or worker |
| Ranked worker spans | separate traces, correlated by `run_id` — only the async collector's context is propagated |

## Resource attributes (process tags)

Stable-for-the-run values, set once at init and attached to every span/metric: `rl.algorithm`, `rl.model`, `nemo.precision`, `dl.tensor_parallel.size`, `dl.pipeline_parallel.size`, plus `nv.dl.rank` / `nv.dl.world_size`. See [Configuration — Resource attributes](configuration.md#resource-attributes).

## Granularity guidance

| Span groups | Relative cost | Recommendation |
|---|---|---|
| Disabled (`telemetry.enabled: false`) | None | The default |
| `default` | Lowest | Safe for all production runs |
| `per_step` | Moderate | Per-step profiling; each step is its own trace |
| `all` | Highest | Development / deep debugging |

Non-exporting ranks have an empty span-group set — `is_span_group_enabled()` returns `False` everywhere, so no span objects are created at all. The disabled path is a `frozenset` lookup and an immediate return. See [lens: architecture](https://github.com/NVIDIA-NeMo/Lens).
