# Task 8 Report

## Scope

Implemented Task 8 from baseline `2bb62c50`: exact native MXFP8 configuration
validation, focused acceptance/rejection coverage, and task-owned OCI-HSG
launch/configuration artifacts. No GPU jobs were submitted.

## Validation Change

`check_nccl_reshard_refit_support()` now accepts only the new native branch:
`fp8_param=true`, `fp8_recipe=mxfp8`, generation `precision=fp8`, and
`is_mx is True`. BF16-to-MXFP8 and matching blockwise-FP8 behavior is unchanged.
Native MXFP8 to BF16 remains rejected, and native MXFP8 to a missing, false, or
truthy-nonboolean `is_mx` value reports an explicit format mismatch. Existing
pre-communicator guards continue to reject ETP greater than one, colocated
generation, non-vLLM generation, DTensor training, invalid vLLM EP/PP, and
other unsupported parallelism.

Source extraction already rejects invalid TE metadata, including GEMM-swizzled
scales, before any collective. Task 8 does not change that implementation.

## Experiment Artifacts

The new launcher provides `render`, `test-only`, and `submit` actions for
Qwen3-30B-A3B and Nemotron-3 Nano, each with explicit false/true `fp8_param`
overlays. It uses `sbatch --test-only` for scheduler validation, does not poll
the scheduler, and documents filtered monitoring at intervals of at least 60
seconds.

All source remains under `/home`; build/cache/temporary state is under
`/raid/scratch`; only durable containers, source model cache, logs, and run
artifacts are under `/lustre`. Node-local environments are keyed by source SHA,
compiled caches are additionally scoped by the FP8 parameter arm, and model
cache staging copies only known model directories once per node.

## Evidence And Limitations

The requested `uv run pytest` command cannot collect locally: the lockfile is
Linux-only on this macOS host, and `uv run --no-sync` lacks `torch`, `ray`, and
`pytest`. A dependency-free AST execution of the actual validator captured the
RED blockwise-only rejection, then passed the native acceptance and format/ETP
rejection matrix after the minimal change.

Local static verification must still include YAML parsing, shell syntax and
ShellCheck, `ACTION=render` for every model/arm, Ruff on the modified Python
files, and a Linux environment's focused pytest command. GPU smoke evidence is
intentionally deferred to Task 10. The existing runtime does not log a complete
per-module native/BF16 storage inventory; the experiment README makes that
required inspection explicit, but worker-side instrumentation is outside Task
8 ownership.
