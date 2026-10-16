# Prefix rollout checkpoint telemetry

RL observes the outer checkpoint lifecycle. Gym owns participant preparation,
commit, restore, and token-capture details. This instrumentation complements Gym
PRs [#3903](https://github.com/NVIDIA-NeMo/Gym/pull/3903) and
[#4218](https://github.com/NVIDIA-NeMo/Gym/pull/4218); it does not copy their timers
or change the Gym pin. Enable the checkpoint span group in Gym as well as RL.

```yaml
telemetry:
  enabled: true
  span_groups: default,checkpoint
  metrics_enabled: true
```

## Save trace

`rl.rollout_checkpoint.save` starts when `_save_rollout_checkpoint` is invoked.
There is one parent per attempt, including skipped, failed, and cancelled
attempts. It ends when that call returns or raises. Attributes identify the
attempt, operation, outcome, and result reason. The same attempt identity is
passed to Gym. The parent includes waiting for a previous save's lock and for a
durable trainer anchor; the existing `total_save_seconds` retains its original,
shorter timing boundary.

Child spans use `rl.rollout_checkpoint.<phase>`:

| Boundary | Phases |
|---|---|
| Serialization and eligibility | `save_lock`, `trainer_anchor`, `snapshot_paths` |
| Admission and generation | `dispatch_pause`, `generation_fence`, `generation_release`, `retire_dropped`, `drain_replies` |
| RL fanout over Gym deployments | `gym_prepare`, `gym_renew`, `gym_commit`, `gym_resume`, `gym_retire` |
| Consistent data-plane cut | `dp_barrier_wait`, `dp_barrier_hold`, `validate_cut`, `validate_staging`, `capture_state` |
| Durable publication | `tq_save`, `sidecars`, `publication` |
| Generation worker | `prefix_control`, `prefix_cut`, `prefix_batch_put` |

`dispatch_pause` covers the dispatch gate's closed body, not an engine-wide
pause: existing rollouts can still run and vLLM can continue decoding into active
buffers. The terminal fence controls staging. Publication happens **after** Gym,
dispatch, and generation-fence release in this implementation. Consequently,
parent save time is not an all-worker pause measurement. Read the pause and
publication spans separately.

`prefix_batch_put` includes the sink's tensor encoding/construction and TQ
round trip. It is not an isolated measurement of storage or network latency.

The barrier's acquisition and exclusive hold are separate spans. All new spans
are trace-only: nested spans and concurrent workers carry no `rl.bucket`, and
their durations must not be summed for goodput. Batch puts are observed once per
batch, with a row count, rather than once per rollout. Gym's existing record
build/digest and participant phase measurements remain the source for those
costs. Prefix control uses its existing executor and propagates the caller's
context into it; no pool capacity or checkpoint deadline is changed.

## Cross-process parentage

Checkpoint prepare, renew, commit, resume, retire, and restore calls pass the
existing W3C carrier through Ray. The receiving `NemoGym` methods attach it before
calling Gym's coordination library. Gym's traced HTTP client can then continue
that trace into participants. The generation fence fanout also passes the
carrier, and worker thread executors preserve the context.

This requires Gym's telemetry code and HTTP instrumentation to be present and
enabled in the deployed checkout. No Gym APIs or source changes are bundled in
this RL PR. A shared checkpoint ID alone is not trace propagation.

## Restore boundaries

`rl.rollout_checkpoint.restore` covers replay-index and controller/Gym recovery
before the rollout/train pumps start. Children include `replay_restore`,
`recovery_prepare`, `gym_restore`, `gym_resume`, and `restore_prompts`.

Native `tq_load` executes earlier, in setup or actor initialization. Its adapter
span belongs to that caller's trace context; it is **not included** in the
controller-restore span. `redispatch_schedule` executes later in the rollout
pump. Neither the controller span nor these diagnostics claim an
allocation-to-ready restart duration or GPU KV-cache prefill duration.

Individual prefix fetch and reconstruction spans can be very numerous. They
require both `checkpoint` and the existing `per_prompt` group:

```yaml
telemetry:
  span_groups: default,checkpoint,per_prompt
```

These detailed spans are `prefix_restore_fetch` (a coalesced fetch callback) and
`prefix_restore_reconstruct` (one call). A coalescing thread without a caller
context may emit a root span, identified by its process resource attributes.
They are disabled in the normal coarse configuration. Gym's low-overhead capture
metrics remain available without per-prefix tracing.

## Live progress and timeout diagnostics

When telemetry exports, an attempt owns a scoped observation task. Every 30
seconds it records the active phases, elapsed time, time since the last observed
progress, and unanswered Gym deployment count. Structured
`rollout_checkpoint_progress` log records include the attempt identity and up to
16 pending deployment names. Normal replies update pending state; simply polling
it is not counted as progress. Concurrent lease renewal has its own phase and
does not overwrite the phase waiting on storage.

On exception or cancellation, `rollout_checkpoint_phase_interrupted` records the
active phase and unanswered deployments before unwinding. Attempts also emit
`rollout_checkpoint_interrupted`. Progress snapshots contain no rollout identities, staging keys, capture
payloads, or credentials. The observation task is cancelled and
awaited on exit; telemetry-disabled runs create no observer.

The observer uses the controller event loop. If that loop is blocked, the next
sample is delayed too. It is **not an independent hang watchdog**. Executor/loop
diagnostics from RL [#4732](https://github.com/NVIDIA-NeMo/RL/pull/4732) help
separate queueing, function execution, and delayed coroutine resumption.

These additional metrics have bounded operation/phase labels and no checkpoint,
rollout, or deployment identifiers:

| Metric | Kind/unit | Meaning |
|---|---|---|
| `rl.rollout_checkpoint.phase_elapsed` | Gauge, seconds | Current phase age; reset to zero on phase exit |
| `rl.rollout_checkpoint.pending_deployments` | Gauge, deployments | Replies still awaited by RL; reset on exit |
| `rl.rollout_checkpoint.progress_age` | Gauge, seconds | Age of last phase entry or deployment reply; reset on exit |
| `rl.rollout_checkpoint.live_phase.duration` | Histogram, milliseconds | Additional RL fanout, dispatch-pause, fence/release, reply-drain, and worker prefix phase durations |

The phase-duration histogram deliberately excludes native TQ persistence,
sidecars, publication, and barrier timings already exported from completed
logger measurements by RL [#4730](https://github.com/NVIDIA-NeMo/RL/pull/4730).
RL fanout includes scheduling/transport and waiting for **all** Gym deployments;
Gym's coordination and participant metrics measure work inside each deployment.
These are different boundaries, and should be compared, not added together.
Worker histogram samples use operation `worker`; they are not controller
wall-clock totals. Live gauges sample in-progress work, while histograms record
completed phase durations. Metrics remain available when the checkpoint span
group is off.

## Validation

Unit coverage uses real in-memory Lens/OTel exporters and actual asyncio/thread
executors. It checks trace parentage through the real coordinator and Gym actor
entrypoint, timeout pending-instance diagnostics, cancellation and lock/barrier
cleanup, disabled paths, metric-label bounds, and optional per-prefix tracing.
The actual controller save/context-manager methods run against lightweight
resources to verify release-before-publication and release-on-failure behavior.
No CUDA engine or distributed Gym deployment is mocked as a performance result.

Before a production rollout, combine the Gym telemetry dependencies in an
integration checkout, verify one prepare/commit trace across HTTP processes, and
run the existing 12-node GPU smoke. Keep that integration branch separate from
the focused PR. This PR does not add terminal batching or background cleanup.
