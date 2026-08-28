# Task 8 Report

## Scope

Implemented Task 8 from baseline `2bb62c50`: exact native MXFP8 configuration
validation, focused acceptance/rejection coverage, task-owned OCI-HSG
launch/configuration artifacts, and the review-fix prerequisite plumbing. No
GPU jobs were submitted.

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

`_apply_precision_config()` now uses Megatron-Core's
`load_quantization_recipe()` API to attach the parsed recipe to
`model_cfg.quant_recipe` before provider finalization. It also maps
`first_last_layers_bf16`, `num_layers_at_start_in_bf16`, and
`num_layers_at_end_in_bf16` directly onto the provider. The typed schema and
baseline configuration contain the same fields, so the task overlays no longer
depend on inert YAML keys. The setup-level test creates a real recipe and
asserts routed-FC1 MXFP8 selection, BF16 catch-all selection, and the Nano
zero/last-eight count values.

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

Native true-arm overlays select a bounded metadata assertion in the policy
worker. Before building the NCCL refit plan, it checks actual native E4M3 plus
uint8-scale entries for routed experts, BF16 misc entries for shared experts,
router, QKVO, and LM head, and all Nano routed experts in the final eight
layers. It logs a machine-readable `[native-mxfp8-inventory]` JSON record and
raises on mismatch, making the smoke command fail nonzero. `RAY_TMPDIR_ROOT` is
exported by the launcher before `ray.sub`; `ray.sub` resolves its job-scoped
directory before either Ray daemon starts.

## Evidence And Limitations

The requested focused `uv run pytest` commands cannot collect locally because
the lockfile is Linux-only on this macOS host. The RED attempt stopped before
collection with the exact unsupported-platform error. A dependency-free AST
harness executed the changed precision function and confirmed recipe loading
plus all first/last assignments; a direct inventory harness passed the expected
Nano JSON inventory and rejection cases. The existing validator AST evidence
continues to cover native acceptance and format/ETP rejection.

Local checks passed: YAML parsing, `bash -n`, scoped ShellCheck (with existing
`ray.sub` warnings suppressed), all four `ACTION=render` combinations, Ruff,
and direct Pyrefly for the new inventory module. A broad standalone Pyrefly
run remains host-limited by missing runtime packages and existing diagnostics
in the large Megatron worker. GPU smoke and Linux-native pytest evidence remain
intentionally deferred to Task 10/Task 9.
