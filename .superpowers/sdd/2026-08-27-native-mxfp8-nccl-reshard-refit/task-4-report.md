# Task 4 Report: Canonical Native MXFP8 Source Storage

## Changed Files

- `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
  - Added the TE-free `NativeMXFP8Components` dataclass and
    `extract_native_mxfp8_components()`.
  - Reads only `get_metadata()` rowwise storage, validates the logical shape,
    E4M3 metadata format when supplied, compact non-swizzled scales, uint8
    storage, exact value bytes, contiguity, and scale capacity.
  - Returns E4M3 and E8M0 views without calling `contiguous`, transpose,
    quantize, or dequantize, and never selects columnwise storage.
- `pyrefly.toml`
  - Registered the new extractor module in `project-includes`.
- `tests/unit/models/policy/test_mxfp8_refit_source.py`
  - Added TE-free structural fakes and coverage for dense and grouped shapes,
    padded scale cropping, shared-storage pointers, columnwise fallback
    rejection, metadata/storage validation, swizzled scales, and format
    metadata mismatch.

## RED Evidence

Native command:

```text
uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py
```

Result: blocked before collection on macOS because the repository lockfile
supports Linux x86_64 and aarch64 only:

```text
error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine == 'x86_64' and sys_platform == 'linux'`, `platform_machine == 'aarch64' and sys_platform == 'linux'`
```

The isolated CPU harness stubs only the parent package import boundaries and
lets pytest import the requested source module. Before the module existed it
failed as intended:

```text
ModuleNotFoundError: No module named 'nemo_rl.models.policy.workers.mxfp8_refit_source'
```

## GREEN Evidence

Focused isolated CPU harness:

```text
uvx --with pytest --with torch --with numpy python -c '<namespace-package setup>; pytest.main(["-q", "--confcutdir=tests/unit/models/policy", "tests/unit/models/policy/test_mxfp8_refit_source.py"])'
```

Result: exit 0, `22 passed in 0.55s`.

The same native `uv run pytest` command was rerun after implementation and
remained blocked by the documented Linux-only lockfile. This report does not
claim native pytest success.

Focused Ruff checks:

```text
uvx ruff check nemo_rl/models/policy/workers/mxfp8_refit_source.py tests/unit/models/policy/test_mxfp8_refit_source.py
uvx ruff format --check nemo_rl/models/policy/workers/mxfp8_refit_source.py tests/unit/models/policy/test_mxfp8_refit_source.py
```

Both exited 0. `git diff --check` also exited 0 before each code commit.

## Commits

- `ff9de2c5 test(refit): define native MXFP8 source extraction`
- `496ff646 feat(refit): extract native MXFP8 components`

Both commits are signed off with `git commit -s`.

## Remaining Concerns

- Native Transformer Engine MXFP8 runtime validation remains a Linux GPU gate.
  The unit suite is intentionally TE-free and validates the documented
  `get_metadata()` storage contract with structural fakes.
- Direct Pyrefly verification is unavailable locally: the repository config
  resolves site packages from the absent `.venv`, so `uvx --with pyrefly
  --with torch --with numpy pyrefly check ...` reports `Cannot find module
  torch`. The new module is registered in `pyrefly.toml`; locked Linux
  environment validation remains for Task 9.
- This task intentionally does not identify MXFP8 tensors in worker code or
  select the native refit path. Task 5 owns those runtime gates.
