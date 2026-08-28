# Task 5 Report: Native MXFP8 Megatron Source Mapping

## Changed Files

- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - Selects the native path only when `fp8_param=true`, the recipe is
    `mxfp8`, generation precision is `fp8`, and `is_mx=true`.
  - Builds ordered `weight` and `weight_scale` shape metadata and tuple-keyed
    `(logical_name, role)` source specs while retaining the legacy producer
    path for non-native refits.
  - Handles dense, per-expert, and pre-grouped FFN FC1/FC2 mappings, including
    direct row-parallel dense FC2, PP placeholders, and pre-collective source
    validation.
  - Refreshes both components on every refit. Grouped storage is reopened with
    `create_if_missing=False`, then each member is extracted and split/stacked;
    the aggregate grouped tensor is never extracted.
- `tests/unit/models/policy/test_megatron_worker.py`
  - Covers the exact native gate, metadata component transfer order, and
    missing-role failure before a collective.
- `tests/unit/models/megatron/test_group_experts.py`
  - Covers dense and expert FC1/FC2 mappings, both component roles, direct and
    grouped refresh, numeric expert order, PP placeholders, unsupported bulk
    mappings, ordered metadata, and EP-global expert metadata.
- `.superpowers/sdd/2026-08-27-native-mxfp8-nccl-reshard-refit/task-5-report.md`
  - Records Task 5 scope and verification evidence.

The native path does not call the Bridge blockwise exporter, perform receiver
quantization, or call quantize, dequantize, or contiguous. Attention/QKV,
embeddings, shared experts, router, and `lm_head` remain outside this first
native bulk mapping. No vLLM destination adapter was added.

## RED Evidence

The native focused pytest command was attempted before implementation:

```text
uv run pytest -q tests/unit/models/policy/test_megatron_worker.py -k native_mxfp8_export_selection
```

It was blocked before collection on macOS because the repository lockfile only
supports Linux:

```text
error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine == 'x86_64' and sys_platform == 'linux'`, `platform_machine == 'aarch64' and sys_platform == 'linux'`
```

An isolated AST worker harness then failed with the expected missing production
surface:

```text
AssertionError: missing Task 5 worker methods: ['_build_native_mxfp8_shape_metadata', '_is_native_mxfp8_export', '_iter_local_native_mxfp8_param_components', '_materialize_native_grouped_component']
```

After adding the EP-global metadata regression, the isolated focused test
failed before its production fix:

```text
FAILED test_native_mxfp8_per_expert_metadata_expands_global_expert_axis
AssertionError: [2, 4, 64] != [4, 4, 64]
```

## GREEN Evidence

Final focused isolated source-mapping rerun:

```text
uv run --no-project --python 3.12 --with pytest --with torch python /tmp/task5_group_harness.py
```

Result: exit 0, `11 passed in 0.19s`.

Final isolated worker rerun:

```text
uv run --no-project --python 3.12 --with torch python /tmp/task5_worker_harness.py
```

Result: exit 0, `Task 5 worker harness passed`.

The native pytest command was rerun after implementation and produced the same
Linux-only lockfile error before collection. This report does not claim native
pytest success on macOS.

Final formatting and diff checks:

```text
uvx ruff check nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/megatron/test_group_experts.py
uvx ruff format --check nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/megatron/test_group_experts.py
git diff --check
```

All exited 0; Ruff reported all checks passed and all three files formatted.

## Commits

Implementation range: `afe33a01..aa591b1e`.

- `058be4c9 test(refit): define native MXFP8 source mapping`
- `aa591b1e feat(refit): expose native MXFP8 source components`

Both implementation commits are signed off with `git commit -s`.

## Remaining Concerns

- Native repository pytest remains unavailable on this macOS host because the
  checked-in lockfile supports Linux only. The focused behavior was validated
  with isolated CPU harnesses, but Linux native pytest and TE/MCore GPU runtime
  validation remain external verification gates.
- The intentionally excluded model surfaces and the vLLM destination adapter
  remain for their planned follow-on tasks.

## Review Fix

Review-fix implementation range: `8a5ed16e..788dba42`.

- `2640c98d test(refit): cover native MXFP8 review failures`
- `788dba42 fix(refit): repair native MXFP8 grouped sources`

Both commits are signed off with `git commit -s`.

The grouped native setup now builds singular grouped FC1/FC2 tasks through
their suffixed registry entries before Bridge validates the remaining ordinary
tasks. This makes the path reachable for owner and non-owner PP ranks while
preserving placeholders and global transfer order. Simple routed-expert FC1
`AutoMapping` sources now use a direct, unsplit `up_proj.weight` descriptor for
both local-expert and numeric-suffix Megatron names. Fused expert names accept
an optional input `.weight` suffix and emit one canonical suffix.

Before any transfer loop, every owned grouped member is reopened with
`create_if_missing=False`, extracted, and view-validated for both roles. The
per-parameter `pre` hook still performs the current refit's split and stack.
Direct components are extracted once per task refresh and exposed through the
shared source iterator.

### Review RED Evidence

The new production-path tests initially failed with:

```text
ValueError: No mapping found for decoder.layers.0.mlp.experts.linear_fc1.weight
```

This occurred for both owner and non-owner PP cases because strict Bridge task
construction ran before the custom grouped builder. Additional focused RED
failures were:

```text
ValueError: Unsupported native MXFP8 source 'model.layers.0.mlp.experts.3.up_proj.weight' role 'weight' with mapping AutoMapping
AssertionError: emitted model.layers.0.mlp.experts.gate_u.gate_proj.weight
AssertionError: expected one component extraction, observed four
AttributeError: 'NoneType' object has no attribute 'base'
AssertionError: expected no transfer before grouped validation, observed one
```

The numeric-suffix `linear_fc1.weight3` form also failed with the same
unsupported `AutoMapping` error before its fix.

### Review GREEN Evidence

Final isolated verification results:

```text
Task 5 grouped/source suite: 20 passed in 0.18s
Task 5 worker harness: Task 5 worker harness passed
Task 2 regressions: 80 passed in 2.37s
Task 4 regressions: 27 passed in 0.72s
Focused test Pyrefly: INFO 0 errors
Ruff check: All checks passed!
Ruff format: 3 files already formatted
git diff --check: exit 0
```

Native repository pytest remains blocked before collection by the Linux-only
lockfile on this macOS host, so no native pytest success is claimed. Focused
Pyrefly for the changed test file is clean. Whole-file Pyrefly on the worker
still reports 43 pre-existing diagnostics outside the Task 5 implementation
region; this review fix did not broaden scope to repair that baseline.
