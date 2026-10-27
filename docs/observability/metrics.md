# Metrics

NeMo-RL emits seven namespaces of metrics: async efficiency metrics (`rl.efficiency.*`), startup phase durations (`rl.setup.duration`), training scalars (`rl.reward.*`, `rl.policy.*`, …), vLLM engine metrics (`rl.vllm.*`), driver-side vLLM generation metrics (`gen_ai.*`, following the OTel GenAI semantic conventions), per-host memory (`rl.host.memory.*`), and rollout checkpoint metrics (`rl.checkpoint.*`).

Metrics are emitted **only when telemetry is exporting**. Most pass through the metrics logger of the process that logs them, via its process-local telemetry handle: the driver in the v1 loops, and the Single-Controller actor in v2, which owns the Logger there and so emits the training scalars, `rl.setup.duration` and `rl.checkpoint.*`. `rl.host.memory.*` is scraped from every node by a driver-side monitor on its own timer. Resource attributes identify the producing process.

Every `rl.*` series is **declared by NeMo-RL**, not by lens. lens provides a consumer-driven metric registry: NeMo-RL calls `register_metric_group("rl", [...])` once per process with a `MetricSpec` per series, then records against it with `record_metrics(meter, "rl", ...)`. The series the tee derives itself are declared in `_metric_specs()` in `nemo_rl/telemetry/metrics.py`; the teed training and vLLM rows are declared next to the code that logs their keys (see below). Either way, adding a series needs no lens release and no agreement on field names. This page names where each family is declared rather than copying the lists, so the code stays the one source of truth. `record_metrics` never raises into the caller and skips `None`.

## The tee

The tee lives outside the algorithm code: `nemo_rl/telemetry/metrics.py` hooks `nemo_rl.utils.logger.Logger.log_metrics`, so after `log_metrics` fans a step out to the file / W&B / MLflow backends it calls `tee_rl_metrics_to_otel(metrics, prefix)`. It is best-effort — the prefixes read include the per-step `train` dicts (`prefix in ("train", "")`) and the one-off `timing/setup` dict at step 0, plus completed `timing/rollout_checkpoint` and `timing/rollout_recovery` dictionaries and the per-attempt `rollout/checkpoint_outcome` dictionary. Other prefixes are skipped, non-scalar values are ignored, and the whole path is a no-op unless telemetry is actively exporting. The numbers you already see in W&B are therefore the same series you get in your OTLP backend, with no double bookkeeping.

Each family has its own exception handler, so a bad value in one cannot silence the others.

## Training scalars

Mirrored as gauges from the keys the algorithms already log. Each series is a `TeedMetric` row pairing a logger key with an `rl.*` name, declared in the module that logs the key:

- `TRAINING_TEED_METRICS` in `nemo_rl/algorithms/metric_utils.py`: the scalars every training loop logs (reward, policy loss, gradient norm, learning rate, response length).
- `LOSS_TEED_METRICS` in `nemo_rl/algorithms/loss/loss_functions.py`: KL divergence and entropy.
- `CRITIC_TEED_METRICS` in `nemo_rl/algorithms/ppo.py`: the PPO value loss.
- `SFT_V2_TEED_METRICS` in `nemo_rl/algorithms/sft_v2.py`: the SFT v2 controller's step timings and throughput. SFT v2 logs from its controller actor, which initializes telemetry before its first metric is logged.

`kl_penalty` is named for the penalty but holds the divergence — the coefficient is divided back out in `loss_functions.py` — which is why the metric is named `rl.kl.divergence`.

Each row uses the same key constant as its logging site, so a rename cannot leave a gauge silently reporting nothing.

## Startup phases

Every algorithm's `setup()` already timed its startup phases and logged them under the `timing/setup` prefix at step 0 (`SetupTimingMetrics` in `nemo_rl/algorithms/metric_utils.py`). Those are teed as one dimensioned gauge, `rl.setup.duration` (`RL_SETUP_DURATION_METRIC` in `nemo_rl/telemetry/metrics.py`), with the phase in `rl.setup.phase`.

Dimensioned rather than one series per phase because the phase set is not knowable at declaration time: alongside its declared fields `SetupTimingMetrics` carries a free-form `extras` dict, and the sparse-refit transports add `vllm_<transport>_sparse_init_time_s` to it at runtime. A row per phase would silently drop whatever it had not been taught about.

The phase name is the logger key with its `_time_s` / `_s` suffix stripped, so `generation_init_time_s` becomes `phase=generation_init`. Requiring one of those suffixes is also the filter that keeps non-durations out — `parallel_init_enabled` is a flag, not a phase.

Representative phases: `generation_init` (and its `generation_init_reserve` / `generation_init_load` parts), `policy_init`, `value_init`, `nemo_gym_init`, `collective_init`, `weight_sync`, `teacher_model_init`, `parallel_wall`, `worker_setup`, `other_setup`, `total_setup`.

### `rl.setup.duration` carries no `rl.bucket`

Unlike `rl.efficiency.seconds`, these phases overlap each other by construction: `total_setup` contains the rest, `parallel_wall` covers the generation and policy builds running concurrently, and `generation_init_reserve` / `generation_init_load` are parts of `generation_init`. Anything summing them by bucket would count startup several times over.

Read one phase at a time. For "what did startup cost", use `phase=total_setup` — that is the one value that cannot double-count. The `rl.setup.*` spans give the same phases their shape in a trace; see [Span groups — Startup](span-groups.md#startup-what-happens-before-the-first-step).

## W&B history step axes

W&B's internal `_step` is a monotonically increasing history-row number, not a
NeMo-RL trainer step. Trainer-correlated metrics are buffered into one history
row and carry the explicit `nemo_rl/step` field. Use `nemo_rl/step` when plotting
or joining training metrics by optimizer step; relying on `_step` also counts
independently committed telemetry rows and can misalign the series.

Independent event streams use their own custom axes and do not carry
`nemo_rl/step`. In particular, Single-Controller rollout benchmark series under
`rollout/throughput/*`, `timing/rollout_checkpoint/*`,
`timing/rollout_recovery/*`, and `rollout/checkpoint_outcome/*` use
`telemetry/wall_time_seconds`, recorded as Unix wall-clock time. This lets those
series continue through a long or paused trainer step and across process restart
without changing the meaning of the trainer-step axis.

## Single-Controller rollout recovery metrics

Wall-clock rollout telemetry sampling is disabled by default. Enable it without
changing the checkpoint cadence by setting, for example:

```yaml
rollout_checkpointing:
  telemetry_interval_s: 30.0
```

This records one throughput and controller-pressure sample every 30 seconds.
Set the value back to `null` to disable the sampler. Detailed per-interval
metric dictionaries are printed to the controller log only when
`async_rl.diagnostics: true`; configured metric backends receive them either
way.

The rollout checkpoint benchmark focuses on the following questions:

1. Did checkpointing reduce raw generation or usable rollout throughput?
2. Which save or restore phase took the time?
3. How much live data-plane work waited behind the checkpoint barrier?
4. Did scheduled checkpoint attempts succeed at the intended cadence?
5. How much completed rollout work was reused after restart?

Each prefix is one dictionary built in `nemo_rl/algorithms/single_controller.py`; the fields are the keys of that dictionary:

- `rollout/throughput`, from `_collect_and_log_rollout_throughput_metrics`: raw backend decoding throughput compared with output committed for training, plus how many live data-plane mutations an exclusive checkpoint delayed and for how long. The raw throughput is absent when the backend does not expose compatible cumulative counters. With token capture, committed tokens are counted exactly from valid staged rows; without it, they are estimated by multiplying the reported per-sample mean by the number of completions, so compare like-for-like runs.
- `timing/rollout_checkpoint`, from `_save_rollout_checkpoint`: end-to-end save latency and its storage, fencing, controller-sidecar and atomic-publication components, and the logical volume the snapshot captured. `controller_sidecar_bytes` excludes the native TQ payload, because the current TQ checkpoint API does not report bytes written; NeMo-RL deliberately does not scan the shared checkpoint directory, because that scan would perturb the benchmark.
- `rollout/checkpoint_outcome`, from `_log_rollout_checkpoint_outcome`: the result, reason and effective cadence of every scheduled snapshot attempt.
- `timing/rollout_recovery`, from `_log_rollout_restore_metrics` and `_redispatch_restored_rollouts` (the load timings start in `single_controller_utils/setup.py`): restore latency, where `total_load_seconds` is the sum of the disjoint restore phases, then the training-ready groups restored without generation, the unfinished groups found for redispatch, and the sibling work preserved or rerun.

Recovery counters describe work reconstructed at restore time, not a promise
that every restored group will eventually train. After recovery, the configured
sampler may still evict a restored group under its normal staleness rules (for
example, when a `windowed` sampler finds that the group's policy version has
fallen outside its valid window).

`barrier_wait_seconds` and mutation wait latency measure opposite sides of the
same fence. The former is how long the checkpoint waits for already-running
mutations; the latter is how long rollout or training mutations wait for the
checkpoint to release its exclusive cut.

The vLLM request, KV-cache, controller-waiter, buffer-occupancy, and per-mutation
kind fields are diagnostic drill-down signals. They are useful when one of the
primary throughput or barrier metrics regresses, but do not need to appear on
the primary rollout checkpoint dashboard.

## Rollout checkpoint metrics (`rl.checkpoint.*`)

The `timing/rollout_checkpoint` and `timing/rollout_recovery` dictionaries
described above also reach Lens. They reuse the controller's existing
measurements; telemetry does not retime the save or restore.

Each series, with its kind and unit, is declared in `_metric_specs()` in
`nemo_rl/telemetry/metrics.py`. The mapping from logger key to
`rl.checkpoint.phase` / `rl.checkpoint.category` is `_CHECKPOINT_DURATION_PHASES`
and `_CHECKPOINT_LEVELS` in the same file. Every point carries
`rl.checkpoint.operation` (`save` or `restore`); durations add
`rl.checkpoint.phase`, and the other series add `rl.checkpoint.category`.

Missing phases are absent, not zero. Save phases nest inside `total` (barrier
wait, exclusive hold, sidecar save and publication all overlap it, and `tq` sits
inside exclusive hold), so **do not sum save phases into checkpoint latency or
goodput**. Restore phases are disjoint and `total` is their sum, excluding
`redispatch_schedule`, which is logged separately after the controller is ready.
The restore total is not allocation-to-ready wall time. These metrics describe
completed operations, not a pending checkpoint's current phase.

All points come from the Single-Controller actor process. The two duration
histograms use the SDK's default bucket bounds (0, 5, 10, 25, ... seconds), so
every phase under 5 s and every skipped attempt land in one bucket. For useful
percentiles, set
`OTEL_EXPORTER_OTLP_METRICS_DEFAULT_HISTOGRAM_AGGREGATION=base2_exponential_bucket_histogram`
in the launch environment (Ray passes it to every actor); the backend must accept
exponential histograms (Prometheus only with native histograms on). No step,
checkpoint, rollout, or model-call IDs are metric labels.

To find *which* save was slow, each completed save also emits one trace-only
`rl.sc.rollout_checkpoint.save` span carrying the same fields plus the snapshot
sequence and trainer step. See
[Span groups — trace-only rollout checkpoint save](span-groups.md#trace-only-the-rollout-checkpoint-save).

### Snapshot attempts

The series above report completed operations only. Every periodic snapshot
attempt, including skipped and failed ones, is also exported from the
`rollout/checkpoint_outcome` dictionary. `_metric_specs()` in
`nemo_rl/telemetry/metrics.py` declares the three series:
`rl.checkpoint.attempts`, counted by `rl.checkpoint.outcome` and
`rl.checkpoint.reason` (their values are `RolloutCheckpointAttemptOutcome` and
`RolloutCheckpointAttemptReason` in `nemo_rl/algorithms/metric_utils.py`);
`rl.checkpoint.attempt_duration`, by outcome (skips return quickly, so read
completed attempts separately); and `rl.checkpoint.seconds_since_last_success`.
That gauge is this controller's time since its last completed snapshot, recorded
only when an attempt finishes: `0` on success, and absent until the first
success after a (re)start. It gets no new point while an attempt is stuck, so do
not alert on it.

Skipping is expected: a snapshot is only valid while the trainer is at a durable
checkpoint step, so attempts after the next optimizer commit skip with
`missing_trainer_anchor` until the following trainer checkpoint. If
`outcome=skipped` climbs while `outcome=completed` stays at zero, snapshots are
not completing; compare the snapshot interval with the trainer checkpoint
cadence and step time. No attempts for several intervals means the pump is off,
telemetry is off, or an attempt is stuck (the save has no timeout). To alert on
stale snapshots, count completed attempts over a window of several snapshot
intervals, e.g. in Prometheus, where the attribute becomes the
`rl_checkpoint_outcome` label:
`increase(rl_checkpoint_attempts_total{rl_checkpoint_outcome="completed"}[30m]) == 0`.

## Async efficiency metrics (`rl.efficiency.*`)

Async GRPO measures where wall time goes with a `Timer` and logs the result as `efficiency/*` scalars (`print_efficiency_summary` in `nemo_rl/algorithms/utils.py`). Those same values are teed to OTel as one **dimensioned** gauge rather than one instrument per category, so adding a category needs no instrument change: `rl.efficiency.seconds`, one point per category in `rl.efficiency.category`, and `rl.efficiency.pct`, the productive share of driver-side wall clock. Both are declared in `_metric_specs()` in `nemo_rl/telemetry/metrics.py` (`RL_EFFICIENCY_SECONDS_METRIC`, `RL_EFFICIENCY_PCT_METRIC`); the attributes they carry are explained below.

The single-controller path reuses two of these category names — `idle/buffer_starvation` for the wait when the training fleet is stalled on the buffer, and `idle/refit_bubble` for `weight_sync` — so a goodput rollup reads the same vocabulary on both paths. Both are spans only there; the single controller logs no `efficiency/*` scalars of its own.

### Always filter on `rl.efficiency.measurement`

Some categories are measured on the driver against wall time; others are summed across concurrent collector threads and **can exceed the wall time they happened in**.

| `rl.efficiency.measurement` | Recorded on | Categories | Safe to sum against elapsed driver time? |
|---|---|---|---|
| `wall_clock` | driver, sequentially | `init/total`, `idle/buffer_starvation`, `idle/refit_bubble`, `idle/validation` | yes |
| `collector_wall_clock` | collector's collection-loop thread, sequentially | `idle/refit_event_wait`, `idle/generation_limit_pause` | no — real durations, but on a timeline that runs concurrently with the driver's |
| `thread_seconds` | collector's batch-worker threads, concurrently | `idle/buffer_full_backoff`, `wasted/failed_trajectory` | no — not durations at all |

Eight rollout threads each backing off for 10s during the same 10s window produce a `thread_seconds` value of 80, not 10. Summing `rl.efficiency.seconds` by `rl.bucket` without filtering therefore overstates idle time — a wrong answer that looks like a real one. Filter to `rl.efficiency.measurement="wall_clock"` before comparing against elapsed time; read the other two per-phase, `thread_seconds` as a saturation signal.

The non-`wall_clock` values also carry `rl.bucket` so all three share one vocabulary with the spans; the `measurement` attribute is what keeps a bucket rollup honest.

Two deliberate metric/span disagreements to know about before comparing a metric rollup against a trace rollup:

- **The two collector-loop categories** carry `rl.bucket="idle"` as metrics and rely on you filtering by `measurement`, but carry no bucket at all as spans, since a trace has no equivalent filter to rely on.
- **`idle/validation`** is `idle` as a metric and `overhead` on every span covering the same seconds. Both are true of different fleets: the training GPUs are idle, which is what the driver's timer measures, while the generation GPUs are doing necessary non-training work, which is what `bucket_scope(Bucket.OVERHEAD)` in `validate()` tags. Attributing this phase properly needs per-fleet accounting; until then, do not expect the two rollups to agree on a validation step. See [span groups — why `idle/validation` is not a span](span-groups.md#why-idlevalidation-is-not-a-span).

### `rl.efficiency.window`: what a value covers in time

`measurement` says whether values may be summed *against each other*; `window` says whether one may be summed *across steps*. They are independent, and the second is the easier one to get silently wrong.

| `rl.efficiency.window` | Categories | Meaning |
|---|---|---|
| `step` | `idle/buffer_starvation`, `idle/refit_bubble`, `idle/validation` | per-step delta — the driver resets its `Timer` every step, so these sum across steps |
| `run` | `init/total`, all four collector-side categories, and `rl.efficiency.pct` on the async PPO path | cumulative since the process started — consecutive points already contain each other, so summing across steps multiplies by the step count |

`init/total` is the driver-side exception: it is measured once, waiting for the first buffer fill before the step loop, then republished unchanged every step so it does not disappear from a dashboard after step 1. Read it as a constant. The collector's `Timer` is never reset, which is why everything from it is `run`.

**`rl.efficiency.pct` carries whichever window its denominator had — read the attribute, do not assume.** `print_efficiency_summary` tags it `window="step"` when the caller passed a per-step wall time and `window="run"` when it fell back to the run's elapsed time. Async GRPO passes the per-step value; async PPO does not, so its points are `run`. The per-step form is the one to trust: its numerator is the three `step`-window idle categories over that step's wall time. `init/total` is deliberately excluded — it is a run constant, so folding it in would charge the whole startup cost to every step — and so are the collector's categories, which are on another clock. Against the run's elapsed time the ratio climbs toward 100% as the run lengthens no matter what the idle time does, which is why a `window="run"` point says little about how the run is actually doing.

## vLLM engine metrics (`rl.vllm.*`)

Read from the vLLM engine's own Prometheus registry, delta'd per step. `snapshot_step_metrics()` takes a baseline before generation and `get_step_metrics()` a second reading after, both fanned out to the DP-leader workers; the same snapshot pair also feeds the spec-decode metrics, so the wider coverage costs no extra RPC.

Each series, its kind and the vLLM family it is derived from are declared in `VLLM_TEED_METRICS` in `nemo_rl/models/generation/vllm/metric_names.py`: prompt and generation token counters, mean prompt and generation lengths, and counters of `ok` and `failed` generations.

The counters carry the step's delta, so summing them across steps reconstructs the run total. The means come from each histogram's `sum`/`count` delta, so they are exact rather than bucket-interpolated; a step in which the engine served no request omits them rather than reporting a zero-length sequence.

**What counts as failed.** vLLM's finish reasons are `stop`, `length`, `abort`, `error` and `repetition`. Only `stop` and `length` leave a usable sample behind, and `length` is a normal RL outcome — a rollout routinely runs to `max_tokens` — so those two are `ok` and *everything else* is `failed`. Counting the remainder rather than an explicit deny-list keeps `ok + failed` equal to the engine's own total even if a future vLLM adds a reason we have not heard of.

These are the family names vLLM registers. The reader returns family names, which `prometheus_client` strips of `_total`, so no suffixed alias is needed on this path. `test_engine_metric_names_match_vllm` checks the copies against vLLM itself. A series that matches nothing is **omitted**, leaving a visible gap in the dashboard rather than a plausible-looking zero, and so is one whose total went backwards, which means an engine restarted mid-step.

## Driver-side generation metrics (`gen_ai.*`)

The driver-side vLLM generation path records token and latency metrics through lens's `record_inference_metrics` with `provider_name="vllm"`, following the [OTel GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

| Metric | Type | Description |
|---|---|---|
| `gen_ai.client.token.usage` | Histogram | Tokens per request, split by `gen_ai.token.type` (`input` / `output`) |

`gen_ai.server.request.duration` is **not** recorded: it is defined per inference request, and one `generate()` call here is a batch across every data-parallel shard. That wall clock goes to `rl.vllm.batch.duration` (histogram, seconds) instead.

These overlap with `rl.vllm.*` on token counts but are not redundant: `gen_ai.*` is derived from the tensors a `generate()` call returns, so it measures what the driver received, while `rl.vllm.*` is the engine's own accounting. When the two disagree, the gap is work the engine did that never reached the driver — the aborted requests, which appear in no returned tensor at all.

These ride the normal `http/protobuf` OTLP path and reach the same backend as everything else. They are distinct from vLLM's **native** engine tracing (opt-in, and exported over vLLM's own gRPC-by-default exporter) — see [vLLM Tracing](vllm-tracing.md).

## Host memory (`rl.host.memory.*`)

Every raylet publishes its node's memory on its own Prometheus endpoint. `HostMemoryMonitor` in `nemo_rl/telemetry/host_memory.py` scrapes all of them from the driver on a timer and records one point per node — the same source `RayGpuMonitorLogger` already reads for the W&B `ray/node.*.mem_gb` rows, but keeping the node's identity instead of flattening it into the metric name.

The two series, `rl.host.memory.used` and `rl.host.memory.total`, are observable gauges declared in `HOST_MEMORY_RECORDED_METRICS` in the same module, read from the raylet's `RAY_NODE_MEM_USED` / `RAY_NODE_MEM_TOTAL` series and labelled with `host.name` and `rl.worker_groups`.

Bytes, not the gigabytes the W&B path reports. The interval follows `logger.gpu_monitoring.collection_interval` so the host-memory and GPU series line up, but collection is **not** gated on `logger.monitor_gpus`: that flag governs what reaches W&B, and an OTel series should not vanish when someone turns off an unrelated W&B feature. It is gated on telemetry exporting, and on nothing else.

The monitor is started by `init_telemetry_driver` and stopped by `shutdown_telemetry`, which puts it in the **driver** process on every entry point. That placement is load-bearing rather than incidental — see `rl.worker_groups` below — which is why it does not live in `Logger` beside the GPU monitor it otherwise resembles.

### Why these are observable rather than pushed

The monitor samples on its own timer, once per collection interval, which does not line up with the exporter's cycle — and a synchronous gauge is reported only in the cycles that follow a `set`. On the locked SDK (1.44.0) the cycles in between would carry no point at all; from 1.45 the SDK holds a synchronous gauge's last value *per attribute set* instead, which trades the gaps for a staler problem, since nothing in the OTel API can retract an attribute set. A host's `rl.worker_groups` label does change during a run — it is `none` until the worker groups are built, which is minutes in, and changes again when they regroup or shut down — so on 1.45 every host's startup-time `none` point would sit next to its correctly labelled one for the rest of the run, and `sum by (host.name)` would count each host twice.

An observable gauge is read on every export instead, so it always reports the latest snapshot and a label the callback stops reporting simply ends. The monitor therefore scrapes on its own timer into a snapshot that it *replaces* wholesale, and the callbacks return whatever is in it. Three consequences worth knowing:

- Scraping stays on the monitor's thread rather than inside the callback, so the exporter never blocks on HTTP to an unreachable node — a wait that would otherwise hold up every other metric in the process and grow with the cluster.
- The scrape and export intervals are independent, so a point can be up to one collection interval old, and scrapes taken between two export cycles are never seen. Both are normal for a gauge.
- A node that fails a scrape leaves a gap rather than a flat line at its last known value, because the snapshot is replaced and not merged. For "how much memory does this host have in use right now" a gap is the honest answer.

The instruments are created by `create_rl_metric_instruments()` when the monitor starts. This is not optional bookkeeping: nothing ever records these series, so without that call no instrument exists and the callbacks are never polled.

`host.name` is a metric attribute here, not the resource attribute of the same name that lens already sets. The resource one names the host the *emitting* process runs on, which for every one of these points is the head node; the driver is reporting about other hosts. It prefers Ray's `NodeManagerHostname` and falls back to `NodeManagerAddress` (the IP) when Ray resolved no hostname.

### `rl.worker_groups` is a descriptor, not a partition key

It lists the worker groups resident on the host, sorted and comma-joined (`lm_policy,vllm_policy`), or `none` for a host running no group — the head node, or any node whose groups had not been built when the sample was taken. `RayWorkerGroup` registers its placement as it is created, resolving bundles to nodes through `placement_group_table`, so no worker is asked anything.

There is exactly one point per `host.name` per cycle. That matters where several raylets share a machine, as in a local multi-node cluster: each publishes the same whole-host memory on its own endpoint, so the monitor scrapes a host once and merges the worker groups of every node on it, rather than emitting a point per node that `sum by (host.name)` would add together.

That registry is an ordinary module global, so it is visible only to the process that wrote it. This is why the monitor has to run on the driver: the single-controller entry points (`run_grpo_single_controller.py`, `run_sft_v2.py`) build their worker groups on the driver but their `Logger` inside the controller actor, so a monitor owned by the `Logger` would read an empty map and label every host `none` — right numbers, wrong label, on every point.

One known gap follows from the same property. `RewardModelEnvironment` is a Ray actor and builds its `Policy` inside itself, so that worker group registers in the actor's copy of the map and the driver never sees it: hosts running only the reward model report `none`. The memory figure is unaffected; only the label is incomplete. Closing it would mean sharing the map through a named actor and paying a round trip per sample, which is more than a descriptive label is worth.

**Do not aggregate on it.** The value it labels is the *whole host's* memory:

- Under colocation `lm_policy` and `vllm_policy` share GPUs and therefore a host, so `sum by (rl.worker_groups)` would count that host's memory once per group.
- No worker group owns the raylet, the Ray **object store** (often tens of GB, and frequently the thing that actually ran a node out of memory) or — on the head node — the driver, yet all of them are in the number.

Group on it to decide *which* hosts to look at; read `host.name` for the number itself. There is no breakdown inside `ray_node_mem_used`: getting one means per-process sampling (`psutil` RSS from inside each worker, where `rl.worker_group` and `nv.dl.rank` are already on the resource and nothing has to be inferred), and attributing a single process's RSS to tensors versus pinned buffers versus allocator arenas is a profiler's job, not a metric's.

## Metric vs span tag vs resource attribute

The one rule that trips people up. Classify each value before you emit it:

| Kind | Use | Example |
|---|---|---|
| **Metric** | numerical value that changes over time | per-category efficiency seconds → `rl.efficiency.seconds` |
| **Span tag** | categorical per-span context for filtering | `rl.iteration`, `rl.bucket`, `rl.num_generations_per_prompt`, `rl.weight_version` |
| **Resource attribute** | stable for the whole run | `rl.algorithm`, `rl.model`, `nv.dl.topology.size.tp` |

Do **not** put a time-series number (loss, reward) on a span attribute — it produces no useful series in your backend and wastes storage. Do **not** put a per-step categorical (iteration number) on a metric label — that is unbounded cardinality. See [lens: metrics — metric vs span attribute vs resource attribute](https://github.com/NVIDIA-NeMo/Lens).

### Goodput (monitor-derived)

NeMo-RL does **not** emit `rl.goodput` or `rl.bucket.*` rollup metrics.
Leaf spans carry `rl.bucket` ∈ `{productive, overhead, idle, wasted}`;
umbrella spans (`job` / `step` / `rollout`) omit it. `rl.efficiency.seconds`
carries the same `rl.bucket` tokens, but it is a per-category duration, not a
rollup — and it needs the `rl.efficiency.measurement` filter described above.
Offline monitors (e.g. wandb-monitor) SUM span / phase GPU-time by `rl.bucket`
and compute:

```text
rl_goodput = productive_gpu_s / (productive + overhead + idle + wasted)_gpu_s
```

See [Span groups — goodput buckets](span-groups.md) and `nemo_rl/telemetry/instrumentation.py`.

Metric names use the **application scope** (`rl.*`); attribute names in the `rl.*` namespace are NeMo-RL's own, defined in `nemo_rl/telemetry/`, not imported from lens's `semconv.py`.

## Filtering across runs

Every `rl.*` data point carries the `nemo.run.id` resource attribute. Use it to isolate or compare runs in your backend (Grafana/Prometheus, or any OTLP-compatible backend). See [Configuration — Run identification](configuration.md#run-identification).
