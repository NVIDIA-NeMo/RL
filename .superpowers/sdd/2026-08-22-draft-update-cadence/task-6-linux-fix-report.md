# Task 6 Linux gate compatibility fix

## Scope

Fixed the sole product-test blocker reported by Linux gate job `6460700` on
top of reviewed product commit `bbea05d3e3160444d160e2bfa8eecfb7d0e742cc`
and the existing gate-harness commit `d490c22cde6e23a42fcf8fe49dc83a1d54e61c97`.

The change is test-only. Product behavior and the reviewed target-only transfer
implementation are unchanged.

## Root cause

Two target-only worker tests attempted to recover a local class through
`MegatronPolicyWorkerImpl.__ray_metadata__.modified_class`. Ray 2.56.1 does not
provide that private attribute on this class.

The production module already follows the repository's stable actor-extension
pattern: `MegatronPolicyWorkerImpl` is the undecorated implementation, while
only its empty `MegatronPolicyWorker` subclass is decorated with `@ray.remote`.
The tests now instantiate `MegatronPolicyWorkerImpl` directly and continue to
invoke its real IPC and collective methods.

## RED

The Linux gate failure was:

```text
tests/unit/models/policy/test_lm_policy_collective.py:197
MegatronPolicyWorkerImpl.__ray_metadata__.modified_class
AttributeError under Ray 2.56.1
```

The permanent dependency-free compatibility contract was added first and run:

```sh
uv run --no-project python \
  .superpowers/sdd/2026-08-22-draft-update-cadence/task6_linux_test_compat_contract.py
```

It exited 1 with:

```text
AssertionError: RED: test_megatron_worker_target_only_skips_draft_preflight_and_payload depends on private Ray actor metadata
```

## GREEN

After replacing both private metadata lookups with the explicit implementation
class, the same contract exited 0:

```text
TASK6_LINUX_TEST_COMPAT_GREEN
```

The contract also verifies that each test calls the corresponding production
method and does not replace that method on the worker instance. The existing
dependency-free producer harness then exercised the real extracted production
methods and exited 0:

```sh
uv run --no-project python \
  .superpowers/sdd/2026-08-22-draft-update-cadence/task-6-producer-harness.py
```

```text
RED: base producer rejects component selection
GREEN: selection accepted and default call shape preserved
GREEN: Megatron IPC and collective target-only transfers skip draft preflight, names, and bytes; full recovery verified
GREEN: colocated checkpoint-engine transports reject before worker setup
```

The official focused pytest command remains unavailable on this macOS arm64
workstation because the locked project supports Linux x86_64 and Linux aarch64
only:

```sh
uv run --frozen --group test pytest -q \
  --confcutdir=tests/unit/models/policy \
  tests/unit/models/policy/test_lm_policy_collective.py
```

It exits 2 before test collection with the lockfile platform error. The same
focused test is the MCore phase entry in `task6_linux_gate.sbatch` and should be
rerun in the Linux gate after this commit is pushed by the integrator.

## Files

- `tests/unit/models/policy/test_lm_policy_collective.py`
- `.superpowers/sdd/2026-08-22-draft-update-cadence/task6_linux_test_compat_contract.py`
- `.superpowers/sdd/2026-08-22-draft-update-cadence/task-6-linux-fix-report.md`

## Concern

The complete Ray/MCore pytest was not executable locally due to the declared
platform lock. No product code changed, and the dependency-free contracts cover
both actor-class compatibility and the original target-only behavior without
mocking the production methods under test.
