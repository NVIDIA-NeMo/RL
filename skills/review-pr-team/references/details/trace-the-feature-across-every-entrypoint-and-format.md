# trace-the-feature-across-every-entrypoint-and-format

> For a new opt-in feature or config guard, build the full coverage matrix across BOTH axes — every algorithm entrypoint that reads the flag AND every format/mode/family member — before proposing a guard; verify the guard's call site executes for every case it must catch, and put it at the shared choke point

When a PR adds an opt-in feature with a setup-time handshake or guard, "I found a path that skips it"
is only the start. Finish the matrix on **both** axes before writing the comment:

**Axis 1 — every entrypoint.** Don't stop at the algorithm files the PR touched. Enumerate every
`setup()`/`init_communicator()` and resolve which trainer loops share one. In NeMo-RL, `grpo.setup()`
serves `grpo_train`, `async_grpo_train` AND `grpo_sync.grpo_train_sync` (they import it), so one call
site covers three loops — saying "grpo" without that is ambiguous. `ppo.py` calls the helper
unconditionally because it has no transport branches at all; `distillation.py` uses a 2-way branch;
`grpo.py` a 3-way branch; `single_controller_utils/setup.py` is a separate setup that skips it. Report
the table, not a sample. **N call sites with M different conditional shapes is the finding** — and the
root cause is usually a signature mismatch (the helper took a `PolicyConfig` the synchronizers don't
have, which is *why* one was hand-inlined instead of reused).

**Axis 2 — every format/mode the flag could plausibly apply to.** A flag that reads as generic
("prequantize") but is implemented under one format's branch is silently ignored for the others. Grep
for sibling formats (nvfp4/mxfp4/fp8) *and* check whether each even reaches the code holding your
proposed guard.

Axis 2 covers any family the new path dispatches over, not just quantization formats: every
advantage estimator, every backend, every transport. Walk each member through the new path. On PR
#4326 the new advantage stage was checked for grpo and opd only. `gdpo` passes the startup
`unsupported` check but always raises on the first advantage call, after rollouts and logprobs,
because the SC payload never fills the multi-reward fields it needs. The user found it by asking "does
the PR deal with GDPO properly?" It was already true on main, so the ask was one line: add gdpo to
the startup check.

**Then verify the guard's call site executes for every case it must catch.** This is the trap: I
proposed a `raise` in `init_fp8`'s `else:` arm, but `init_fp8` is only called when
`precision == "fp8"` (`vllm_worker.py:453`), so it could never see the NVFP4 case the guard was
supposed to cover. A guard placed where the failing config never arrives is worse than no guard — it
reads as fixed. The right home is the shared config-validation choke point.

**Walk the guard's enclosing conditions.** A guard nested inside
`if not colocated and remote_transport is None and checkpoint_engine_config is None:` protects only
the happy path; each excluded branch is a silent drop or a deep crash. Also check the guard's
*content* covers the combinations that reach it: hoisting a guard that never checks
`refit_transport` still won't catch the `refit_transport` conflict.

**Put it at the choke point.** NeMo-RL has many sibling drivers (`grpo.py`, `grpo_sync.py`, `ppo.py`,
`distillation.py`, `opd.py`, `single_controller.py`, `xtoken_off_policy_distillation.py`) that share
helpers by import. A guard in one algorithm's `setup()` leaves the others unchecked, and the next
person to wire the feature must remember to copy it. Recommend the shared choke point instead: the
factory that builds the component, the component's `__init__`, or the existing `normalize_*_config`
every algorithm already calls.

**A defaulted-`False` bool that turns off a feature the user switched on is its own bug.** Grep every
caller of a helper that takes it. On PR #2971 the guard was called from one site in `grpo.setup()`,
three combinations bypassed it, and `nccl_reshard_refit` was read only in `grpo.py`. Worse,
`grpo_sync.py` (picked by `data_plane.enabled: true` through the same `setup()` in
`examples/run_grpo.py`) imports `refit_policy_generation` and calls it at three sites without
forwarding `nccl_reshard_refit=` (default `False`). Setup built the communicators and every refit
then quietly used the old broadcast path. The team's Critical finding stopped at "hoist the guard";
The user's push ("it might be added to opd/grpo_sync/single_controller, so it needs to be in the
right place") is what found the live silent no-op.

**Why:** On PR #3294 the review was staged with both defects until the user asked two questions —
"is it covered for grpo sync AND async AND single_controller AND PPO? is this centralized?" and "does
refit_prequantize only apply to mxfp8? what about mxfp4/nvfp4 — is there an assert?" The first surfaced
a second uncovered path the whole 7-agent team had missed (`grpo.py`'s `remote_transport` branch, not
just the SC path); the second showed the proposed fix was unreachable for the case it targeted, and
that MXFP4 doesn't exist in the repo at all. Six agents plus an adversarial pass had all anchored on
"three synchronizers discard the return" without completing either axis.

**How to apply:** for any new opt-in feature, before drafting: (1) `grep -rn "<helper>|<flag>"
nemo_rl/algorithms/ nemo_rl/weight_sync/` and build the entrypoint table incl. which loops share a
`setup()`; (2) grep sibling formats and check each one's dispatch; (3) for the proposed guard, trace
its enclosing condition back to the entrypoint and confirm the failing config reaches it; (4) list the
callers of any helper that carries the flag as a defaulted parameter. Related:
[new-option-check-existing-selector](new-option-check-existing-selector.md) (a proper selector makes most of this structural),
[partial-fix-check-the-sibling-branch](partial-fix-check-the-sibling-branch.md) (same instinct on if/elif arms),
the comment skeleton in /review-pr-team.

**Seen:** #3294 entrypoints + formats (user) · #4326 gdpo through the SC advantage stage (user) · #2971 guard in one setup(), grpo_sync silently off (user)
