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
| RL fanout over Gym deployments | `gym_prepare`, `gym_renew`, `gym_commit`, `gym_resume`, `gym_retire`, `gym_forget` |
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

With RL telemetry metric export enabled, generation cut requests also emit one
`prefix_cut_probe` JSON log summary per inventory. These summaries separate
cut-control queue wait, execution and result delivery; engine route-snapshot
wait; route-history concatenation and prefix assembly; cumulative/delta route
encoding and record construction/digests; sink route decoding and batch
construction; and the TQ `put_samples` call/acknowledgement. They include phase
wall time, calling-thread CPU time, invocation/error counts and maximum latency.
Counters report cumulative/delta route bytes, chunk/token counts and batch
tensor payload sizes. Tensor bytes exclude serialization overhead, and repeated
cumulative observations are work-volume counters, not unique live memory.
Phases are inclusive and overlap; do not sum them. Thread CPU excludes Torch's
auxiliary threads. Cancelled HTTP requests can leave executor work running, so
summaries labelled `interrupted` may contain only the phases completed so far.
No token contents, route payloads, model-call IDs or staging keys are logged.

Router-replay cuts cache joined engine routes per physical request. Only new
immutable pieces are validated and concatenated with NumPy; a capacity-growing
buffer amortizes history copies. Its capacity can approach twice the observed
route rows and is additional to vLLM's own chunk storage. Assembly runs outside
the token-observation lock. `route_history_chunks` and `route_history_bytes`
count newly absorbed pieces, not the entire history on each cut. The cache is
observed state, so failed staging does not rewind it or advance durable tokens.

Live route lookups first freeze bounded groups (at most 64 calls, additionally
limited by the configured prefix-batch row/token limits), then copy that group's
route-piece references directly on the verified GIL path or in one engine-owner
callback on the fallback path. Token observation stays
unlocked; each call's lifecycle lock remains owned until staging or rollback.
Missing/invalid requests fail individually, and an owner-loop timeout rolls back
the affected group without starting a fresh timeout per call. Existing TQ
batching and route digests are unchanged. Completed calls with captured terminal
routes and calls needing no live routes skip the owner lookup.

`engine_snapshot_groups`, `engine_snapshot_requests`, and
`max_engine_snapshot_group_calls` distinguish logical group count from request count.
The `engine_snapshot_wait` measures one lookup per group, not per call; the
fallback dispatch/callback/wakeup phases measure one handoff per group.
`transaction_prepare_total` covers freezing up
to the deferred lookup; `transaction_route_resume` covers resumed record work.
On this path `route_fetch_total` covers route materialization after the shared
lookup, not its scheduling wait. Summing per-call waits across a shared handoff
would overcount, so the owner wait is recorded only once per group.

Fresh router-replay cuts use digest layout 2, identified by
`nrl_generation_cut_digest_layout: 2` in the first staged row's call extras.
Each staged chunk still commits to its route payload. The cumulative prefix
digest commits to tokens, call extras, and `nrl_generation_cut_chunks_sha256`,
a domain-separated SHA-256 over the ordered chunk digests, without encoding
cumulative routes again. Restore verifies each fragment and reconstructs this
commitment. Unknown layouts fail closed. Unmarked legacy and canonical terminal
rows retain the old cumulative-route digest; resumed legacy chains continue
using that layout. Old readers cannot restore layout-2 prefixes and fall back
to regeneration; use the updated reader to preserve that checkpointed work.

On the verified vLLM 0.29.0 layout, CPython with the GIL enabled can copy the
published route-piece references directly from the cut thread. Setup logs
`generation_route_snapshot mode=direct_refs` when this path is eligible;
unverified layouts still fail setup and unsupported runtimes/container types
retain the owner-loop path. The reader copies built-in lists only, never tensor
bytes, and does not change token freezing, coverage checks, receipts or TQ
batching. An instance-local completion hook retains the route list before vLLM
removes a finished request. Only requests being consumed by this adapter are
retained, until the HTTP consumer has observed the terminal output (or aborts).
Both read modes use this fallback, counted as `engine_snapshot_retained_reads`.
The hook copies no route bytes; exhaustion, cancellation and shutdown release
its references. Unknown requests still return no live state. This fast path relies
on the pinned vLLM's append-only, immutable published pieces and must be audited
again on upgrades. It bypasses owner-loop scheduling, not process-wide GIL/GC
pauses. Prefix summaries count `engine_snapshot_direct_reads` and time
`engine_snapshot_direct_read`; the existing group/request counts still describe
logical route-fetch groups, not necessarily actual owner-loop handoffs.
For matched A/B runs, set
`policy.generation.disable_direct_route_snapshot=true` to force the existing
owner-loop path. False or absent retains the guarded direct-reference path;
the setup mode log confirms which path is actually active.

For an opt-in control-queue experiment, set
`++policy.generation.generation_cut_control.max_workers=2`. The absent/default
configuration uses one thread; only 1 and 2 are accepted. Setup logs
`generation_cut_control max_workers=... guarded_admission=enabled`.
Two threads overlap independent inventories from the same checkpoint, not GPU
decode or refit work. Admission reserves all model-call IDs before executor
entry, so overlapping inventories cannot acquire lifecycle locks in opposite
orders. Identical in-flight inventories share one operation and receipt.
Different checkpoint IDs never execute concurrently, and a queued checkpoint
barrier cannot be overtaken. Once the next checkpoint starts, uncached requests
for an older checkpoint are rejected rather than cutting stale state. A
disconnected caller does not release a running cut's reservations: they remain
until its blocking write actually finishes.
Queued work with no callers is cancelled; expired queued work never starts.
Acknowledgements still follow successful TQ staging and transaction sealing.
Shutdown refuses queued/new work without prematurely releasing running cuts.

`control_queue_wait` still includes both guarded admission and executor
scheduling. `control_admission_wait` is its nested admission component, not an
additional duration. Counters include `control_duplicate_join`,
`control_cancelled_before_start`, and `control_admission_deadline_expired`.
`control_stale_checkpoint_rejected` counts rejected older-epoch work.
Compare end-to-end snapshot time, terminal/fence latency, throughput and memory,
not just queue time: two threads can increase GIL contention, concurrent route
buffer memory and TQ traffic. Validate save/restore on a small allocation first.
Use a fresh experiment/checkpoint directory for the initial A/B comparison;
bootstrap compatibility fingerprints can include this configuration setting.

On the fallback path, the route lookup round trip is split into
`engine_snapshot_dispatch_wait`, `engine_snapshot_callback` (owner-thread wall
and CPU time), and `engine_snapshot_return_wakeup`. A lookup that times out
before completing records `engine_snapshot_unresolved_wait` instead; its
dispatch/callback phases may be incomplete. The outer `engine_snapshot_wait`
remains the end-to-end boundary and must not be added to its children. Unix
timestamps on inventory summaries allow correlation with owner-loop reports.
Cut-work timers also cover capture freezing/copying, UTF-8 boundary checks,
registry lookup, full route fetch/materialization, acknowledgement building,
and transaction preparation, sealing and release. These boundaries are nested.

When metrics export is enabled, the vLLM worker attaches reversible observers
to its existing output processor and completion-output construction. The
`vllm_engine_probe_start` log records effective output chunk size, Torch/OMP
thread settings and CPU affinity. A one-second probe runs on the engine-owner
loop, independently of the existing HTTP-loop probe. `vllm_engine_probe` JSON
summaries report 30-second windows of owner-loop lag, synchronous output-chunk
wall/thread-CPU time, output/finished-request counts and finished route bytes
and chunk counts. `terminal_output_build` includes route concatenation, text
selection and output-object construction; it is **not** a pure NumPy concat
timer. Slow operations (at least one second) emit `vllm_engine_slow`, limited
to one record per phase per ten seconds. No per-token logs are emitted. Slow
records overlap the periodic summaries; neither nested timings nor records
from different windows should be added to infer exclusive checkpoint time.
Observers restore their hooks and cancel their probe at worker shutdown.

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

## Integration-only Gym actor diagnostics

With RL telemetry enabled, the mounted integration Gym checkout emits bounded
`gym_actor_probe` JSON logs. These are diagnostics, not checkpoint behavior changes:

- `prepare_enter` records the caller's submission timestamp and actor-entry delay;
  `prepare_return` records handler execution time. Compare these with the existing
  outer `gym_prepare` span to identify delayed reply delivery. Cross-host entry
  delays depend on synchronized clocks; handler durations use a monotonic clock.
- Thirty-second cumulative summaries include synchronous startup phase wall/thread
  CPU times, active rollout-call counts, process CPU, and GC collections/pauses.
  Nested startup phases are inclusive and must not be summed. Separate phases
  cover config HTTP fetch, JSON decode, OmegaConf reconstruction, client creation,
  routing validation, and routing-index construction.
- A one-second watchdog checks a half-second actor-loop heartbeat. Stalls over
  two seconds log at most one 12-frame stack sample every ten seconds. Samples
  contain only filenames, function names, and line numbers, never locals or
  source. The watchdog itself can be delayed by the GIL.
- `gym_row_ownership` logs correlate failed physical episode IDs with subsequent
  retry cleanup without changing ownership, retirement, or retry policy.

The extra heartbeat, watchdog, GC callback, and startup scopes are disabled when
RL telemetry is disabled. Set `RAY_DEDUP_LOGS_ALLOW_REGEX` to
`gym_actor_probe|gym_row_ownership` for a diagnostic submission so Ray does not
collapse successive numeric summaries. No request bodies or credentials are logged.

The integration Gym actor opts into `RolloutCollectionHelper(reuse_server_client=True)`
for its fixed deployment. It resolves the server client/config once per helper,
instead of fetching and reconstructing the same config for every prompt group.
Changing the supplied head config invalidates the one-entry cache; callers can
explicitly reload an unchanged deployment with `setup_server_client(refresh=True)`.
Standalone helpers keep per-call reload behavior by default. Startup phase counters
remain enabled, so compare config reconstruction calls/time per actor against the
uncached baseline, along with loop lag and prepare-entry delay.

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
