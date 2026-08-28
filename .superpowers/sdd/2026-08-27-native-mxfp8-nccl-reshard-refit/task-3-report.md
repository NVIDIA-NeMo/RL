# Task 3 Report: Validate Refit Plans Before NCCL Setup

## Commit Range

Task 3 implementation range: `a3f98564..63689a2d`

- `65acaa33 test(refit): cover NCCL plan validation`
- `63689a2d fix(refit): validate plans before NCCL setup`

## Changed Files

- `nemo_rl/weight_sync/nccl_reshard_utils.py`
- `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- `tests/unit/weight_sync/test_weight_synchronizer.py`
- `tests/unit/weight_sync/test_nccl_reshard_utils.py`

## RED Evidence

Native command:

```bash
uv run pytest -q tests/unit/weight_sync/test_weight_synchronizer.py -k validates_refit_plan_before_collectives
```

Output (exit 1 before collection):

```text
error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine == 'x86_64' and sys_platform == 'linux'`, `platform_machine == 'aarch64' and sys_platform == 'linux'`
```

The focused isolated utility test failed before production changes (exit 1):

```text
FAILED tests/unit/weight_sync/test_nccl_reshard_utils.py::test_restore_refit_info_rejects_plan_digest_mismatch
Failed: DID NOT RAISE ValueError
```

An isolated production-module harness exercised the synchronizer with the same
event-recording fakes as the unit test. Before production changes, its order
assertion failed with:

```text
['policy.init_collective', 'generation.init_collective', 'policy.init_reshard', 'generation.init_reshard', 'policy.prepare_refit_info', 'generation.prepare_refit_info']
AssertionError
```

## GREEN Evidence

The isolated CPU utility suite passed after the implementation:

```bash
uvx --with pytest --with torch --with numpy python -c '... pytest.main(["-q", "--confcutdir=tests/unit/weight_sync", "tests/unit/weight_sync/test_nccl_reshard_utils.py"])'
```

Output (exit 0): `68 passed in 1.32s`.

The isolated production-module synchronizer harness passed its digest and
ordering assertions after the implementation, recording:

```text
['policy.prepare_refit_info', 'generation.prepare_refit_info', 'policy.init_collective', 'generation.init_collective', 'policy.init_reshard', 'generation.init_reshard']
```

Focused Ruff passed:

```bash
uvx ruff check nemo_rl/weight_sync/nccl_reshard_utils.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_nccl_reshard_utils.py
```

Output (exit 0): `All checks passed!`

## Behavior

- Policy metadata is prepared, given a canonical `plan_digest`, converted to
  the wire representation, and validated by generation before either NCCL
  communicator family is initialized.
- Placement restoration retains a received digest, restores nested component
  placements, recomputes the canonical digest, and raises an actionable
  `ValueError` with both digest values on mismatch.
- Refit metadata without `plan_digest` retains the existing restoration
  behavior for legacy direct callers.

## Concerns

- Native `uv run pytest` was not executed because this macOS host is excluded
  by the Linux-only lockfile; the isolated checks are not native pytest
  evidence. Linux locked-environment validation remains required by Task 9.
- This task intentionally adds no communicator rollback. Validation completes
  before the first initializer, so validation failures create no communicator.
