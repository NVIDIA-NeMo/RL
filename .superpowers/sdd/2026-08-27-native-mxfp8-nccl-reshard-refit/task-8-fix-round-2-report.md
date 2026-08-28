# Task 8 Fix Round 2 Report

## Commit

- `0ff06f415d1d19f1642e9885f7e38850fe01db5e`
  `fix(refit): enforce native MXFP8 inventory scope`
- Signed and signed off.

## RED Evidence

Added regressions for a native Qwen shared expert, a Nano early BF16 routed
FC1, and an omitted Nano final-tail routed FC2. Before the fix, the
dependency-free harness accepted all three invalid inventories.

`uv run pytest -q tests/unit/models/policy/test_native_mxfp8_inventory.py`
could not collect on this macOS host because the repository lockfile supports
Linux x86_64/aarch64 only.

## GREEN Evidence

- Dependency-free inventory harness accepted valid Qwen/Nano inventory and
  rejected all three regressions.
- `uvx pyrefly==0.24.2 check nemo_rl/models/policy/workers/native_mxfp8_inventory.py`: passed.
- `ruff check nemo_rl/models/policy/workers/native_mxfp8_inventory.py tests/unit/models/policy/test_native_mxfp8_inventory.py`: passed.
- `python3 -m py_compile` for both focused Python files: passed.
- `git diff --check`: passed.

## Files

- `nemo_rl/models/policy/workers/native_mxfp8_inventory.py`
- `tests/unit/models/policy/test_native_mxfp8_inventory.py`
- `experiments/native_mxfp8_source_refit/README.md`

The inventory now uses fixed task-owned expected FC1/FC2 layer/module sets:
Qwen layers 0-47 and Nano layers 0-51, with Nano BF16 tail layers 44-51.
Every BF16-only scope rejects native entries regardless of whether that model
is expected to contain the scope. No GPU or SLURM job was submitted.
