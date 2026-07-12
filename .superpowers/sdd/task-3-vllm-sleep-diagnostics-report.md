# vLLM sleep diagnostics implementation report

Status: `DONE_WITH_CONCERNS`

Base commit: `32f5addc7ed2b474b243b254c67aeee20d55a7ad`

## Implementation

- `VllmGeneration.finish_generation()` logs one bounded, structured INFO record mapping `discard_weights` to the requested sleep level before dispatching a colocated worker sleep.
- A typed host-memory snapshot helper reports current-process RSS and system available memory in GiB through `psutil`.
- Sync and async model-owner sleep paths resolve the requested level once, then emit exactly two structured INFO records: one immediately before the vLLM sleep call and one after vLLM sleep, `gc.collect()`, and `torch.cuda.empty_cache()`.
- After-sleep records include signed process-RSS and system-available-memory deltas relative to the before-sleep snapshot.
- No recipe, runtime configuration, refit behavior, cluster script, remote job, or environment logging was added.

## Files in scope

- `nemo_rl/models/generation/vllm/vllm_generation.py`
- `nemo_rl/models/generation/vllm/vllm_worker.py`
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `tests/unit/models/generation/test_vllm_generation.py`
- `.superpowers/sdd/task-3-vllm-sleep-diagnostics-report.md`

The shared worktree also contains independently owned edits to the CuTeDSL lifecycle wrapper and its test. Those files are excluded from this task and will not be staged or committed here.

## TDD evidence

The focused tests were changed before production code. They retain the lifecycle harness's hard-coded names:

- `test_sync_sleep_uses_requested_sleep_level`
- `test_async_sleep_uses_requested_sleep_level`

The tests now cover the resolved level, deterministic before/after RSS and available-memory values, signed deltas, and both sync and async logger names. The existing driver parameterization covers both `discard_weights=False -> level 1` and `discard_weights=True -> level 2`, now including the structured request log.

The exact project test command was attempted first:

```bash
uv run pytest tests/unit/models/generation/test_vllm_generation.py \
  -k 'sync_sleep_uses_requested_sleep_level or async_sleep_uses_requested_sleep_level or finish_generation_maps_weight_discard_to_sleep_level' -vv
```

It stopped before collection because the lock supports Linux only while this host is macOS arm64:

```text
error: The current Python platform is not compatible with the lockfile's supported environments:
`platform_machine == 'x86_64' and sys_platform == 'linux'`,
`platform_machine == 'aarch64' and sys_platform == 'linux'`
```

Before implementation, a source contract check confirmed the expected RED state:

```text
RED: required telemetry event strings are absent from all three production files
```

An attempt to construct a local macOS-only environment was stopped because it would not reproduce the locked Linux runtime. The initially resolved incompatible `transformers==5.13.1` was replaced with lock-compatible `transformers==5.8.1`; no tracked dependency or lockfile changed. The authoritative focused GREEN result must come from the locked nightly Linux container.

## Static verification

```bash
ruff check \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  tests/unit/models/generation/test_vllm_generation.py
```

Result: exit 0, `All checks passed!`.

```bash
ruff format --check \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  tests/unit/models/generation/test_vllm_generation.py
```

Result: exit 0, `4 files already formatted`.

`git diff --check` also exited 0.

```bash
python3 -m py_compile \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  tests/unit/models/generation/test_vllm_generation.py
```

Result: exit 0.

## Independent review

A fresh read-only reviewer inspected only the three production files, focused test file, and this report. The reviewer found no high-confidence Critical or Important issues and assessed the source diff as ready for authoritative locked-container testing. The reviewer independently reproduced passing Ruff, format, and whitespace checks and did not edit any file.

The reviewer's supplemental local pytest attempt stopped during collection on missing `prometheus_client`. No further local environment repair was attempted because the authoritative environment is the locked Linux nightly container.

## Commit

Planned signed commit subject: `chore: add vLLM sleep memory diagnostics`.

## Concerns

- Focused pytest has not yet produced authoritative GREEN evidence. It must run in the repository's locked nightly Linux container before this work is classified complete.
- The telemetry measures the model-owner actor process RSS, not aggregate RSS for all engine subprocesses. This is intentional and matches the requested current-process diagnostic boundary.
