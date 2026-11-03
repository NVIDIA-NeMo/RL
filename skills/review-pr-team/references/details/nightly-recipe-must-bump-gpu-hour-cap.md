# nightly-recipe-must-bump-gpu-hour-cap

> Any NeMo-RL PR that adds a line to tests/test_suites/nightly.txt must also bump test_nightly_compute_stays_below_NNNN_hours (function name + assert + message) in the same commit, or that test goes red in the CPU config checks job, which cancels the rest of a PR run

`tests/unit/test_recipes_and_test_suites.py` has a hard budget assert:
`test_nightly_compute_stays_below_NNNN_hours` runs `DRYRUN=1 ./tools/launch <every nightly.txt
entry>` and asserts the printed total is `<= NNNN`. The headroom is deliberately tiny — the cap is
reset to "current total + a small margin" each time — so almost ANY new nightly entry breaks it.

**Check this on every PR that touches `nightly.txt`.** Compute the new entry's cost from its
wrapper's CONFIG block. `tools/launch` computes
`gpu_hours = NUM_RUNS * NUM_NODES * GPUS_PER_NODE * NUM_MINUTES / 60` (L147), with
`NUM_RUNS = ceil(MAX_STEPS / STEPS_PER_RUN)`. The part that trips parsers: **`GPUS_PER_NODE`
defaults to 8** at L130 (`export GPUS_PER_NODE=${GPUS_PER_NODE:-8}`), so most driver scripts
legitimately omit it. A parser that skips scripts without all four vars wrongly calls most of them
unparseable; with the default applied, replaying the formula over `nightly.txt` reproduces the suite
total to within ~2 scripts (#2884). A routine 1n8g / 240-minute recipe costs **32 GPU-hours**, and
the margin is often under 20.

**When this applies:** only to a PR that adds nightly compute, and only by that entry's own cost, in
the same commit. Never ask to raise the cap ahead of need or by more than the new entry costs — the
cap is a deliberate ratchet (see `wait-for-devil-advocate-before-staging` (see [lane-adjudication](../lane-adjudication.md)), #3262).

**The fix is the repo's own convention, not a judgment call.** `git log -L` on the test function
shows every prior compute-adding PR bumped the cap in the same commit: 4024 -> 4048 -> 4139 -> 4149
-> 4157 -> 4181 -> 4197 -> 4266 -> 4280. Ask for all THREE literals to move together — the function
name (the number is in it), the `assert`, and the f-string message — or the test name lies about
what it checks. Do not propose "shrink the job" as the primary fix without checking the delegation:
a TQ wrapper's CONFIG block only sizes the Slurm job, while the delegated base script re-reads its
own `MAX_STEPS` and hardcodes metric assertions at that step count.

**Why it is worse than one red test:** the module runs in the `CPU config checks` job
(`cpu-config-tests` in `cicd-main.yml`). The repo's pytest `addopts` includes `-x`, so the
failure stops that job on its first failing test, and on a PR run the job then cancels the rest
of the run, GPU jobs included.

**Why:** on PR #4024 the new nightly entry took the total 4248 -> 4280 against a 4266 cap. Nobody
had noticed because the PR carried no `CI:L*` label, so no test job had ever run on it — the entry
had survived several force-pushes and a since-dismissed approval. CI being quiet is not evidence a
gate passes; see [run-gh-pr-checks-before-any-lint-or-test-claim](run-gh-pr-checks-before-any-lint-or-test-claim.md).

**How to apply:** `tools/launch` needs GNU `sed` and will not run on macOS, so replay its arithmetic
in a script — reproduce the quirk that `NUM_RUNS`/`GPUS_PER_NODE` are never reset between loop
iterations (a few scripts omit them and inherit the previous entry's values), then compare the
merge-base total against the head total and confirm the delta equals the new entry alone. Ask the
author to re-run the real tool on Linux and paste the number, since you could not.
