# Profile GRPO GPU workers

Four-phase capture records weight refit, generation, logprobs, and policy
training in broad worker windows. A shared run identity and phase boundaries let
ntrace produce one report with native GPU breakdowns for each phase. CUPTI runs
inside the policy and vLLM GPU workers. Profiling is opt-in and adds no hard
dependency on ntrace.

## Enable capture

Install ntrace and its native CUPTI backend in the policy and vLLM worker
environments, including the internal vLLM GPU workers. Set these variables in
the driver before constructing workers:

```bash
export NRL_NTRACE_FOUR_PHASE=1
export NRL_POLICY_PROFILER_CLASS=ntrace.NemoRLTraceController
export NRL_ROLLOUT_PROFILER_CLASS=ntrace.NemoRLRolloutTraceController
export NTRACE_RUN_ID=my-unique-run-id
```

An explicit `NTRACE_RUN_ID` takes precedence. Otherwise the driver reuses a
nonblank `NEMO_RL_OTEL_RUN_ID` or `NEMO_LENS_RUN_ID`, then falls back to a new
UUID. Rank selection, capture ranges, and output directories are plugin
settings; use fresh output directories for each run.

Supported runs use the legacy GRPO driver with Megatron policy workers and vLLM
generation: synchronous GRPO on colocated or separate devices, and asynchronous
GRPO on separate devices. Rollout profiling supports TP >= 1, PP = 1, and EP = 1
or TP. These combinations fail before workers start or when the worker starts:

- `data_plane.enabled=true` (the TransferQueue driver);
- a profiler selector in a Megatron or vLLM worker without
  `NRL_NTRACE_FOUR_PHASE=1`;
- other policy or generation backends;
- a custom `vllm_kwargs.worker_cls` other than the built-in NIXL worker, and
  ModelOpt fake quantization.

## Window ownership

Synchronous GRPO opens one window on both roles before refit or wake and closes
it after logprobs, training (including optimizer work), and any validation in
the step. Dynamic-sampling attempts have distinct rollout IDs. Both sides of
weight transfer and vLLM wake belong to refit; reference logprobs include
reference-weight swaps. Validation generation is a separate phase labeled with
`purpose=validation`. Skipped logprob passes record no logprob phase.

Asynchronous GRPO opens a policy window for each update, starting with the
bootstrap refit. The rollout workers have one capture epoch per weight version:
after the native generation pause (or pending-generation drain), the collector
closes the generation phase and epoch, then opens the next epoch before refit.
Requests can cross weight epochs, so their target weight versions are labels,
not observed assignments. Reports join role windows by recorded timestamps.

At shutdown, generation quiesces within a bounded control RPC, then the driver
separately awaits the save of the final epoch before closing capture and
tearing down workers. Failed quiescence, saves, or close invalidate the capture
without masking the original workload error.

## Warmup and deferred evaluation

A launcher that warms up workers before training constructs
`GrpoCapture(policy, policy_generation, schedule="async", colocated=False)` after
setup and before warmup, then passes it as `capture=` to `async_grpo_train` (or
uses `schedule="sync"` with `grpo_train`). Work outside an open GRPO window does
not consume capture windows.

To run a separate evaluator without capture, set `NRL_NTRACE_FOUR_PHASE=0` and
empty profiler selectors in its driver environment and in
`policy.megatron_cfg.env_vars` and `policy.generation.vllm_cfg.env_vars`. Actor
settings override the inherited driver environment.

## Plugin contract

Both profiler classes receive `__init__(*, rank: int)` and implement
[`CaptureController`](../nemo_rl/models/profiling.py): `configure_capture`,
`begin_step`, `finish_step`, `abort_step`, `begin_phase`, `end_phase`, and
`close`. Rollout profilers also implement `begin_engine_initialization` and
`end_engine_initialization(token)`. vLLM workers construct the rollout profiler
before model loading and CUDA graph creation, using the owning NeMo rank plus
the internal worker rank. Phase tokens stay in their GPU process. Missing
modules, invalid classes, missing methods, and constructor failures raise
during worker initialization.

## Review the artifacts

Generate the combined report where ntrace is installed:

```bash
uv run python -m ntrace grpo-step \
  --policy-dir /results/policy --rollout-dir /results/rollout \
  --policy-rank 0 --rollout-rank 0 --output-dir /results/report
```

The report covers the selected ranks and windows, not the whole job. Phase
boundaries synchronize the GPU, so measured time includes instrumentation
overhead. NeMo-Lens telemetry is independent: GPU capture does not require span
export, and host span names do not imply that a GPU phase ran.
