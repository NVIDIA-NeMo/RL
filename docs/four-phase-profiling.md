# Capture all four GRPO phases

Four-phase capture records weight refit, generation, logprobs, and policy
training in broad worker windows. The shared run identity and phase boundaries
let ntrace produce a single HTML report with native GPU breakdowns for each
phase. CUPTI runs inside the policy and vLLM GPU workers.

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

The driver creates and distributes a UUID when `NTRACE_RUN_ID` is absent.
Configure the plugin's rank selection, output directories, and iteration ranges
separately for policy and rollout. Use fresh output directories; keep both roles'
artifacts together with their effective configuration and worker completion logs.
Disable benchmark-specific policy warmup callbacks before entering GRPO; the
capture identity must be configured before the first worker step.

The supported combination is the legacy GRPO driver with Megatron policy
workers and vLLM generation. Synchronous GRPO supports colocated or separate
devices. Asynchronous GRPO uses separate policy and generation devices. The
TransferQueue driver is not part of the four-phase contract: setup rejects
`NRL_NTRACE_FOUR_PHASE=1` with `data_plane.enabled=true` before allocating workers.
Existing standalone
[policy](policy-profiler.md) and [rollout](rollout-profiler.md) profiling remain
available without `NRL_NTRACE_FOUR_PHASE=1`.

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
