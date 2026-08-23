# Task 6 producer report

## Scope

Implemented the bounded producer layer on top of `156f9905c`, preserving the
reviewed receiver/MTP/NCCL work. The change threads `WeightSyncSelection`
through the policy IPC and collective send paths and makes target-only sync
skip Megatron draft preflight and draft payload generation. It also propagates
the draft schedule mode into factory creation and rejects unsupported direct
and remote-sparse modes before worker setup.

No controller decisions, apply receipts, cadence-science capability, or
fixed/adaptive controller execution was added.

## RED

The permanent behavioral tests were added first in:

- `tests/unit/models/policy/test_lm_policy_collective.py`
- `tests/unit/single_controller/test_single_controller_setup.py`

The normal pytest environment cannot be created on this workstation:

```sh
uv run --group test pytest -q tests/unit/models/policy/test_lm_policy_collective.py
```

Result: exit 1 before pytest because the lockfile supports only Linux x86_64
and Linux aarch64, while this worktree is macOS arm64. Retrying without sync
reaches collection but exits 1 because `tests/unit/conftest.py` imports Ray and
the local environment has no `ray` package.

To obtain an observable dependency-free RED against the real producer source,
the following command executes the Task 6 harness. It extracts and invokes the
producer method from the exact base commit with a minimal worker-group stub:

```sh
uv run --no-project python .superpowers/sdd/2026-08-22-draft-update-cadence/task-6-producer-harness.py
```

Result: exit 1 with `TypeError: broadcast_weights_for_collective() got an
unexpected keyword argument 'selection'`, wrapped as `AssertionError: RED:
Policy collective producer does not accept component selection`.

## GREEN

The same dependency-free harness then invokes the current producer and prints:

```text
GREEN: selection accepted and default call shape preserved
```

It also invokes the real Megatron IPC and collective sender methods and prints:

```text
GREEN: Megatron IPC and collective target-only transfers skip draft preflight, names, and bytes; full recovery verified
```

It exercises full, target-only, full on one reusable worker for each transport,
asserting that target-only has no draft names, zero draft bytes, and no draft
preflight/PP-collective invocation.

The following completed with exit 0:

```sh
ruff check nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/models/policy/test_lm_policy_collective.py tests/unit/single_controller/test_single_controller_setup.py
ruff format --check nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/models/policy/test_lm_policy_collective.py tests/unit/single_controller/test_single_controller_setup.py
uv run --no-project python -m compileall -q nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py
git diff --check
```

## Files

- `nemo_rl/models/policy/interfaces.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `nemo_rl/algorithms/grpo.py`
- `nemo_rl/algorithms/single_controller_utils/setup.py`
- `tests/unit/models/policy/test_lm_policy_collective.py`
- `tests/unit/single_controller/test_single_controller_setup.py`
- `.superpowers/sdd/2026-08-22-draft-update-cadence/task-6-producer-harness.py`

## Concern

The permanent pytest regressions were not executable in this macOS arm64
worktree. They require a Linux-supported `uv` environment with Ray and the
Megatron test dependencies installed. The dependency-free harnesses cover the
new producer behavior locally; run the listed pytest tests in the Linux CI or
container environment before integration.
