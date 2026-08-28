# Task 8 Fix Round 3 Report

## Scope

Resolved the remaining routed-expert inventory false pass from
`task-8-rereview-round-2.md`. The validator now checks canonical exported
projection names independently for every expected global layer:

- Qwen3-30B-A3B: `gate_proj`, `up_proj`, and `down_proj` across 48 layers.
- Nemotron Nano: `up_proj` and `down_proj` across 52 layers.

The native refit metadata builder emits those canonical names before invoking
this inventory hook, so `linear_fc1` and `linear_fc2` source-side names are not
accepted as aliases. In particular, they cannot substitute for Qwen
`gate_proj`.

## RED Evidence

Added `test_qwen_inventory_rejects_missing_all_gate_proj_entries`. Before the
implementation change, a dependency-free harness loaded the existing module
directly and accepted a Qwen inventory containing only all 48 `up_proj` and
`down_proj` entries (96 routed native entries), with every `gate_proj` absent.

## GREEN Evidence

A dependency-free post-fix harness accepted valid Qwen and Nano inventories,
then rejected each counterexample:

- Qwen native shared expert.
- Qwen with every `gate_proj` absent.
- Qwen with `linear_fc1` entries in place of every `gate_proj`.
- Nano early BF16 `up_proj`.
- Nano missing final-layer BF16 `down_proj`.

## Validation

Completed successfully:

- `ruff check nemo_rl/models/policy/workers/native_mxfp8_inventory.py tests/unit/models/policy/test_native_mxfp8_inventory.py`
- `uvx pyrefly==0.24.2 check nemo_rl/models/policy/workers/native_mxfp8_inventory.py`
- `python3 -m py_compile nemo_rl/models/policy/workers/native_mxfp8_inventory.py tests/unit/models/policy/test_native_mxfp8_inventory.py`
- `git diff --check`
- Dependency-free inventory harness described above.

The normal focused pytest command could not run on this macOS worktree:
`uv run pytest` rejects the Linux-only lockfile, and `python3 -m pytest` stops
in `tests/unit/conftest.py` because `ray` is unavailable. The focused tests are
present and their behavior is covered by the direct dependency-free harness.

## Files

- `nemo_rl/models/policy/workers/native_mxfp8_inventory.py`
- `tests/unit/models/policy/test_native_mxfp8_inventory.py`
- `.superpowers/sdd/2026-08-27-native-mxfp8-nccl-reshard-refit/task-8-fix-round-3-report.md`

## Commit

- `604f72ce fix(refit): validate MXFP8 routed projections` (signed and signed-off)

No GPU or scheduler jobs were submitted.
