---
name: nemo-rl-mcore-watchdog
description: Run and act on the standing NeMo-RL-against-megatron-core-main watch. Rebuilds the integration branch of unmerged fixes, runs the functional suite against mcore main, owns any breakage that is not attributable to a labeled PR, and maintains one tracking issue. Use when asked to run the watchdog, refresh the baseline, check whether NeMo-RL is healthy against mcore main, or find out who owns a pre-existing failure.
---

# NeMo-RL vs megatron-core `main` watchdog

## Why this is a separate job

Before it existed, a bug on `main` was paid for by whichever labeled PR happened
to arrive first. That author's comment filled with failures they did not cause,
and the agent burned its entire fix budget on someone else's bug — then did it
again on the next PR, because nothing remembered.

This job takes ownership of that class of breakage. It runs on a schedule rather
than on a PR, it fixes what it finds, and it parks those fixes where every later
PR run picks them up. The per-PR agent is then only responsible for what is
genuinely new.

## One pass

```bash
.agents/nemo-rl-testing-agent/scripts/watchdog.sh --suite l1
```

That does five things, in order:

1. `sync_integration.sh` rebuilds `nrlta/integration` on the fork: NeMo-RL `main`
   plus every open PR whose branch matches `mcore-*-fix`.
2. `known_issues.py refresh` asks GitHub about each registry entry's fix PR and
   retires the merged ones.
3. `ensure_baseline.sh --force` runs the suite: megatron-core `main` × the
   integration branch × the Bridge sha NeMo-RL pins.
4. `known_issues.py annotate` labels each failure with what we already know.
5. `post_tracking_issue.py` renders the tracking issue.

Add `--publish` to actually create or edit the GitHub issue; without it the body
is only written to `$STATE_DIR/watchdog-<suite>-issue.md`. Use `--skip-run` to
re-render from the cached baseline without occupying the cluster.

## Reading the result

The output of step 4 is the whole point:

- `known` — already diagnosed. Nothing to do unless its fix has stalled in
  review, in which case chase the reviewer rather than the bug.
- `STALE` — the registry says a fix exists **and that fix is applied to the
  branch that just failed**. Something is wrong with the entry, or two bugs share
  a signature. Investigate as new, and correct the entry when you know which.
- `new` — unclaimed breakage. This is yours.

## What to do with new breakage

1. Confirm it is real and not infrastructure. `NRLTA_PREP_FAIL` means the harness
   broke; `NRLTA_PREP_FAIL_INTEGRATION` means the pinned Bridge and mcore `main`
   are incompatible, which is a genuine finding about the stack.
2. Diagnose it, using `.claude/skills/megatron-pr-failure-triage/SKILL.md`.
3. Fix it if you can, following
   `.claude/skills/megatron-pr-fix-delivery/SKILL.md`. The branch convention here
   is `mcore-main-<slug>-fix` rather than `mcore-<PR>-fix`, since no labeled PR
   is responsible.
4. Record it either way:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/known_issues.py record \
  --id <stable-slug> \
  --test <test name> \
  --signature "<the error_signature from the results JSON>" \
  --diagnosis "One or two sentences a PR author can act on." \
  --repo NVIDIA-NeMo/RL --fix-pr <n>
```

   Recording it **without** a fix PR is still worth doing: it stops every later
   PR re-deriving the diagnosis, and the tracking issue will show it as awaiting
   a fix.

5. Re-run `sync_integration.sh` so the new fix reaches subsequent PR runs
   immediately rather than at the next scheduled pass.

## Getting the signature right

The registry matches on test name plus a normalized error signature, so the
signature has to identify the *bug*, not the *run*. Copy it verbatim from the
results JSON — `normalize()` already strips pids, paths, line numbers and
literal values. Do not hand-write a signature that includes a measured value:
`median(...) < 1.1 (measured 3.37)` will not match the same failure measuring
2.01 next week, which is exactly the case this is meant to catch.

## When a fix will not apply

`sync_integration.sh` reports `CONFLICT applying #N` and leaves that fix out
rather than shipping it half-applied. The branch stays usable, but that fix is no
longer protecting anyone, and the tracking issue says so under **Needs a human**.
Rebase the PR — do not work around it by excluding it permanently.

## Learnings from a watchdog pass

This job runs unattended and against a moving `main`, so it is usually the first
to notice that a skill has gone out of date with the stack. Record those with
[nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md) and
promote them at the end of the pass; a watchdog run has no PR sweep to batch
them into, so its own draft PR is the end of the job.
