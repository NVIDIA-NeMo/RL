# Task 2 Report: Version-Neutral Refit Component Contract

## Changed Files

- `nemo_rl/weight_sync/refit_components.py`
- `nemo_rl/weight_sync/nccl_reshard_utils.py`
- `pyrefly.toml`
- `tests/unit/weight_sync/test_refit_components.py`
- `tests/unit/weight_sync/test_nccl_reshard_utils.py`

## RED Evidence

Native command:

```bash
uv run pytest -q tests/unit/weight_sync/test_refit_components.py
```

Output (exit 2):

```text
error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine == 'x86_64' and sys_platform == 'linux'`, `platform_machine == 'aarch64' and sys_platform == 'linux'`
```

Isolated macOS import harness:

```bash
uvx --with torch python -c 'import importlib; import sys; import types; package = types.ModuleType("nemo_rl.weight_sync"); package.__path__ = ["nemo_rl/weight_sync"]; sys.modules[package.__name__] = package; importlib.import_module("nemo_rl.weight_sync.nccl_reshard_utils"); importlib.import_module("nemo_rl.weight_sync.refit_components")'
```

Output (exit 1):

```text
ModuleNotFoundError: No module named 'nemo_rl.weight_sync.refit_components'
```

## GREEN Evidence

Focused isolated macOS harness:

```bash
uvx --with pytest --with torch --with numpy python -c 'import sys; import types; import pytest; package = types.ModuleType("nemo_rl.weight_sync"); package.__path__ = ["nemo_rl/weight_sync"]; sys.modules[package.__name__] = package; raise SystemExit(pytest.main(["-q", "--confcutdir=tests/unit/weight_sync", "tests/unit/weight_sync/test_refit_components.py", "tests/unit/weight_sync/test_nccl_reshard_utils.py"]))'
```

Output (exit 0):

```text
78 passed in 1.34s
```

Lint command:

```bash
uvx ruff check nemo_rl/weight_sync/refit_components.py nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_refit_components.py tests/unit/weight_sync/test_nccl_reshard_utils.py
```

Output (exit 0):

```text
All checks passed!
```

`git diff --check` and `git diff --cached --check` both exited 0 before their
respective commits.

## Commit Range

Task 2 implementation range: `4022b42b..667d88d3`

- `5722aa69 test(refit): define ordered component contract`
- `667d88d3 refactor(refit): add ordered weight components`

## Remaining Concerns

- Native `uv run pytest` did not execute on macOS because `uv.lock` supports
  Linux x86_64/aarch64 only. The isolated harness is non-native evidence and
  must not be reported as native pytest success. Task 9 needs Linux-container
  or cluster validation with the locked environment.
- `uvx pyrefly check` cannot resolve `torch` from the absent local `.venv` and
  also reports existing unbound-flow diagnostics in `nccl_reshard_utils.py`.
  The new module is registered in `pyrefly.toml`; Task 9 should run Pyrefly in
  the locked Linux environment.
- This task defines metadata and role-aware lookup only. Ordered component
  transfer and destination binding remain the later Task 5 and Task 7 work.

## Native Component Ordering Fix

Commit: `5550aa90 fix(refit): enforce native component ordering`

Regression RED, using the isolated macOS harness:

```bash
uvx --with pytest --with torch --with numpy python -c 'import sys; import types; import pytest; package = types.ModuleType("nemo_rl.weight_sync"); package.__path__ = ["nemo_rl/weight_sync"]; sys.modules[package.__name__] = package; raise SystemExit(pytest.main(["-q", "--confcutdir=tests/unit/weight_sync", "tests/unit/weight_sync/test_refit_components.py", "-k", "rejects_reversed"]))'
```

Output before the fix (exit 1):

```text
FAILED tests/unit/weight_sync/test_refit_components.py::test_native_mxfp8_rejects_reversed_value_and_scale
Failed: DID NOT RAISE ValueError
```

The fix accepts only role sequences `("weight",)` and
`("weight", "weight_scale")`; it rejects a reversed native pair without
reordering it.

GREEN evidence:

- The same isolated regression command exited 0: `1 passed, 11 deselected`.
- The isolated focused component and NCCL utility suite exited 0:
  `79 passed in 1.16s`.
- `uvx ruff check nemo_rl/weight_sync/refit_components.py tests/unit/weight_sync/test_refit_components.py`
  exited 0 with `All checks passed!`.
