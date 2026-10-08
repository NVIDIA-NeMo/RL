# Profile GRPO GPU workers

Four-phase capture records weight refit, generation, logprobs, and policy
training in broad worker windows. The shared run identity and phase boundaries
let ntrace produce a single HTML report with native GPU breakdowns for each
phase. CUPTI runs inside the policy and vLLM GPU workers. Profiling is opt-in
and adds no hard dependency on ntrace. The same hooks also support standalone
policy-update and rollout profiling, described below.

## Enable capture

Install the current ntrace package and its native CUPTI backend in every selected
policy and internal vLLM worker environment. Set these variables in the driver
before constructing workers:

```bash
export NRL_NTRACE_FOUR_PHASE=1
export NRL_POLICY_PROFILER_CLASS=ntrace.NemoRLTraceController
export NRL_ROLLOUT_PROFILER_CLASS=ntrace.NemoRLRolloutTraceController
export NTRACE_INCLUDE_MEMOPS=1
export NTRACE_RUN_ID=my-unique-run-id
```

An explicit `NTRACE_RUN_ID` takes precedence. Otherwise the driver reuses
`NEMO_RL_OTEL_RUN_ID` or `NEMO_LENS_RUN_ID` when nonblank, then falls back to a
new UUID. This shares run identity with Lens without requiring telemetry export.
Configure the plugin's rank selection, output directories, and iteration ranges
separately for policy and rollout. Use fresh output directories; keep both roles'
artifacts together with their effective configuration and worker completion logs.
Configure capture before benchmark-specific policy warmup and pass the same
controller into GRPO, as described below. Warmup does not own capture steps.

The supported combination is the legacy GRPO driver with Megatron policy
workers and vLLM generation. Synchronous GRPO supports colocated or separate
devices. Asynchronous GRPO uses separate policy and generation devices. The
TransferQueue driver is not part of the four-phase contract: setup rejects
`NRL_NTRACE_FOUR_PHASE=1` with `data_plane.enabled=true` before allocating workers.
Standalone policy and rollout profiling remain available without
`NRL_NTRACE_FOUR_PHASE=1`.

Rollout topology support is TP >= 1, PP = 1, and EP = 1 or TP. Each workload and
topology needs GPU qualification. Choosing an asynchronous vLLM engine does not
by itself select asynchronous GRPO.

## Window ownership

Synchronous GRPO opens one broad window on both roles before refit or wake, then
closes it after logprobs, training (including optimizer work), and any periodic
or final validation in the step. Dynamic-sampling attempts have distinct attempt
IDs. Both sides of weight transfer and vLLM wake belong to refit; reference
logprobs include reference-weight swaps. Validation generation and its cleanup
have a separate `validation_generation` annotation with `purpose=validation`
and the post-update weight version. Initial validation before the first broad
step remains outside the capture.

Asynchronous GRPO has one rollout capture owner per process-wide weight epoch.
After the existing native generation pause or pending-generation drain, the
collector closes the generation phase and epoch, then opens the next epoch
before refit. Generation resumes after refit. This replaces per-batch profiler
ownership while preserving concurrent request scheduling. Requests can cross
weight epochs; possible target weight versions are not observed request-to-update
assignments. Reports join role windows by recorded timestamps.

Final shutdown quiesces generation and closes the current epoch without creating
another. Native quiescence uses a bounded control RPC. Artifact serialization
uses a separate awaited RPC so large trace saves do not consume that budget.
Failed quiescence, profiler close, or serialization invalidates the capture.
Generic engine shutdown still follows the engine's own cleanup policy.

For three asynchronous policy updates starting at step zero, capture policy
ordinals 0–2 and rollout ordinals 0–3:

```bash
export NTRACE_CAPTURE_ITER=0 NTRACE_NUM_ITERS=3
export NTRACE_ROLLOUT_CAPTURE_ITER=0 NTRACE_ROLLOUT_NUM_ITERS=4
```

This includes the initial and final weight transfers. The last rollout epoch
may contain only refit if generation has not resumed before shutdown.

## Review the artifacts

Generate the combined report in an environment with ntrace installed:

```bash
uv run python -m ntrace grpo-step \
  --policy-dir /results/policy --rollout-dir /results/rollout \
  --policy-rank 0 --rollout-rank 0 --output-dir /results/report
```

Check completion, exact run/rank/window counts, device identities, graph
provenance, and memory-operation capture in both source artifacts. Successful
worker or scheduler exit alone does not qualify a trace. The report covers the
selected ranks and windows, not all ranks or the entire job. Phase boundaries
synchronize the GPU; measured elapsed time includes instrumentation overhead.

CPU-only contract tests execute the actual routing and lifecycle methods without
importing Ray, PyTorch, or vLLM:

```bash
uv run --no-sync python -m pytest -q -o addopts='' \
  tests/standalone/test_four_phase_profiling.py
```

These tests cover synchronous validation success/failure and asynchronous epoch
ownership, quiescence, and finalization. They complement fresh GPU mechanism and
workload captures; they do not qualify GPU execution by themselves.

## Synthetic warmup and deferred evaluation

A launcher that warms up workers before entering GRPO must construct
`GrpoCapture(policy, policy_generation, schedule="async", colocated=False)`
after setup and before warmup, then pass that same object as `capture=capture`
to `async_grpo_train` (or use `schedule="sync"` and `grpo_train`). Configuration
runs once. Refit, logprobs, and monolithic or split training outside an open
GRPO owner window do not consume capture ordinals; synthetic warmup can retain
its normal begin/microbatch/abort behavior. Close the capture before destroying
workers if warmup fails, retaining the original workload exception.

For training-only capture with a separate evaluator, explicitly set
`NRL_NTRACE_FOUR_PHASE=0` and both profiler selectors to empty strings in the
evaluator driver and its policy/generation worker environments. The actor
configurations are `policy.megatron_cfg.env_vars` and
`policy.generation.vllm_cfg.env_vars`; explicit entries override inherited
environment values. Do not reuse training output identity for new evaluator
actors.

## Standalone profilers and plugin contracts

Leave `NRL_NTRACE_FOUR_PHASE` unset and set either profiler selector to an
installed, fully qualified class name to capture only that role:

```bash
export NRL_POLICY_PROFILER_CLASS=my_profiler.package.MyPolicyProfiler
# Or: export NRL_ROLLOUT_PROFILER_CLASS=my_profiler.package.MyRolloutProfiler
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_math_8B_megatron.yaml grpo.max_num_steps=3
```

Install the plugin and its dependencies in the policy/vLLM actor environments,
including internal vLLM GPU processes for TP>1 or async engines; installation
in the driver alone is insufficient.

Both classes receive `__init__(*, rank: int)` and must implement `close()`.
The role-specific contracts are defined in
[`PolicyProfiler`](../nemo_rl/models/policy/profiling.py) and
[`RolloutProfiler`](../nemo_rl/models/generation/profiling.py):

| Role | Required lifecycle methods | Captured work |
|---|---|---|
| Policy | `begin_train_step()`, `finish_train_step()`, `abort_train_step(*, reason)` | One complete non-evaluation update: forward/backward, gradient reduction and optimizer work |
| Rollout | `begin_engine_initialization()`, `end_engine_initialization(token)`, `begin_rollout(*, step_id)`, `finish_rollout()`, `abort_rollout(*, reason)` | Engine startup in a separate window, then complete generation attempts or async batches |

Four-phase plugins additionally implement
[`CaptureController`](../nemo_rl/models/four_phase_profiling.py): configuration,
step begin/finish/abort and phase begin/end. Tokens remain in their owning GPU
process; they are never serialized through Ray. Capture ranges, rank selection,
output paths and native runtime constraints remain the plugin's responsibility.

The existing Ray worker environment propagation carries profiler settings.
Explicit actor settings override inherited values, including an empty selector
to disable profiling. Missing modules, invalid classes, missing required role
methods and constructor failures raise during worker initialization. With empty
selectors, no profiler package is imported. Output paths must be writable at
the same absolute location from every selected worker.

Policy profilers are created after distributed setup with the worker's rank;
ModelOpt Megatron workers inherit the integration. Split training opens once in
`begin_train_step`, spans all microbatches and closes in `finish_train_step`.
Errors abort without masking the original exception; explicit caller aborts
do not repeat an already completed finish/abort callback. Policy evaluation,
logprobs and generation are outside standalone policy windows.

For synchronous TP1 vLLM, the model-owning NeMo actor hosts the profiler. TP>1
or an asynchronous engine requires a profiler in each internal GPU worker,
using the owning NeMo rank plus internal worker rank as a dense rollout rank.
The built-in NIXL worker is supported; arbitrary custom `worker_cls` values
and ModelOpt modes replacing the internal GPU worker are not. Synchronous TP1
ModelOpt inherits the outer-actor integration. The initialization window spans
worker/engine startup through warmup and graph creation; its exact begin token
is passed back at end.

Standalone synchronous rollout capture works with legacy and TransferQueue
GRPO. Each `stepN/attemptM` window includes all generation turns and
`finish_generation()`, excluding validation, policy scoring and training.
Continuous async GRPO assigns one batch nonblocking ownership of the
process-wide profiler. Other batches continue concurrently, so their GPU work,
refit, KV-cache reset and NCCL can appear in the owner's window. Its
`generationN/targetM/attemptK` window closes before CPU transfer, teacher
inference and replay-buffer enqueue. Shutdown prevents new owners and bounds
the drain; a timeout fails the run and leaves its artifacts untrusted. Async
PPO profiling is rejected because it lacks this drain contract.

## Relationship to NeMo-Lens

Lens supplies optional host-side spans and context. GPU capture remains
independent of telemetry enablement, sampling and exporter availability.
The policy rank and Ray environment come from existing NeMo worker setup;
vLLM capture still initializes in the GPU process before graph construction.

Do not infer GPU phases from host span names: for example, NeMo's combined
logprob span can exist when both actual logprob passes are skipped. Capture
hooks follow the actual worker operations and do not force skipped work to run.
Likewise, best-effort telemetry flush cannot replace awaited capture saving.

Custom launchers that also want Lens telemetry must call
`init_telemetry_driver` before `init_ray` and `shutdown_telemetry` during final
cleanup, as in [`examples/run_grpo.py`](../examples/run_grpo.py). A telemetry
YAML block alone does not initialize a custom launcher.
