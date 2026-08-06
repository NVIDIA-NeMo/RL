---
name: megatron-pr-fix-delivery
description: Publishes a validated fix for a NeMo-RL functional test failure as a branch plus a draft pull request in the correct repository (NeMo-RL, Megatron-LM, or Megatron-Bridge), and returns the review link for the results table. Use once a fix attempt has actually made the failing test pass.
when_to_use: A fix attempt worked and needs to be raised for review; choosing which repo a Megatron-LM-related fix belongs in; writing the branch name, sign-off, and draft PR body for an agent-authored fix.
---

# Delivering a validated fix

Stage 4 of the nemo-rl-testing-agent pipeline. Only runs when a fix has been
**verified by a passing targeted re-run** — never publish an untested patch.

Every fix is delivered as a **draft** PR. The agent proposes; a human reviews.

## Which repo, which base

| Fix lives in | Repo | Base branch | Local path |
|---|---|---|---|
| NeMo-RL call site, config, or test wiring | `NVIDIA-NeMo/RL` | `main` | repo root |
| Genuine Megatron-LM bug the PR introduced | `NVIDIA/Megatron-LM` | **the PR's own branch** | `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM` |
| Bridge model mapping / provider / checkpoint glue | `NVIDIA-NeMo/Megatron-Bridge` | `main` | `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` |

Targeting the PR's own branch for a Megatron-LM fix matters: it lets the author
merge the correction into their PR with one click instead of untangling it from
`main`. If the PR is from a fork you cannot push to, push the branch to the
upstream repo anyway (or your fork) and open the draft PR against the fork's
branch, then say so in the PR body.

## Steps

### 1. Branch and commit

```bash
git checkout -b mcore-5700-fix
git add <only the files that constitute the fix>
git commit -s -m "fix: follow megatron-core InferenceConfig rename in the mcore worker"
```

- The branch is named **`mcore-<PR>-fix`**, where `<PR>` is the labeled
  Megatron-LM PR number that surfaced the failure — `mcore-5700-fix`. The same
  name is used in whichever repo the fix lands in, so the fix is traceable back
  to the PR that exposed it from any of the three repos.
- If one PR needs independent fixes in the *same* repo, suffix them
  (`mcore-5700-fix-refit`) and keep one draft PR per logical fix.
- Branch from a clean base, not from whatever is checked out. Local worktrees
  routinely carry unrelated work in progress; commit only the fix files, and
  restore the original working state afterwards.
- `-s` (DCO sign-off) is mandatory in all three repos.
- Push over ssh — `git push git@github.com:NVIDIA-NeMo/<repo>.git <branch>`. The
  https remote a plain clone hands you is refused with a 403 and a SAML SSO
  authorization prompt, and it only surfaces at push time, after the fix is
  committed. That 403 is the org's SSO enforcement, not missing write access;
  `gh` API calls (comments, issues, `pr create`) go through either way.
- The subject must be [Conventional Commits](../contributing/SKILL.md); the
  `semantic-pull-request` check enforces it on the PR title.
- Commit only the fix. Ledger files, results JSON, and scratch logs live outside
  the repo under `$STATE_DIR` and must never be committed.

### 2. Open the draft PR

```bash
gh pr create --repo NVIDIA-NeMo/RL --draft \
  --base main --head mcore-5700-fix \
  --title "fix: follow megatron-core InferenceConfig rename in the mcore worker" \
  --body "$(cat <<'EOF'
Found by the nemo-rl-testing-agent while validating NVIDIA/Megatron-LM#5700
against the NeMo-RL Megatron functional suites.

## Failure
`grpo_megatron_generation_async` (L1) failed with:
```
<error signature>
```

## Root cause
<one or two sentences>

## Fix
<what changed and why it is the right layer>

## Validation
Re-ran `grpo_megatron_generation_async` against megatron-lm `<sha>` on
oci-hsg (GB200, 4 GPUs): passing. Other suite tests unaffected.

Draft: raised by an agent, needs a human review before merge.
EOF
)"
```

For a NeMo-RL PR, follow up with `/ok to test <full-sha>` per
[contributing](../contributing/SKILL.md) so CI runs.

### 3. Record the link

Put the PR URL in the results JSON entry's `comment` (that is what the author
sees) and in `ledger.md`. Set the entry's `status` to `fixed` so the table
renders it distinctly from a plain pass — the test only passes *with* that PR
merged, which is exactly what the author needs to know.

When the failure was pre-existing, the comment must say both things: that this
PR did not cause it, **and** that a fix is up for review with the link. The
author needs to know they are not blocked on it; whoever owns the broken area
needs to know it is being handled.

### 4. Make the fix reach the next PR

A raised fix is not yet a merged fix, and until it merges the same test fails on
every labeled PR that arrives. Two commands close that gap, and skipping them is
the difference between fixing a bug once and fixing it every week:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/known_issues.py record \
  --id <stable-slug> --test <test name> \
  --signature "<error_signature verbatim>" \
  --diagnosis "..." --repo NVIDIA-NeMo/RL --fix-pr <n>

.agents/nemo-rl-testing-agent/scripts/sync_integration.sh
```

The first makes later runs recognise the failure instead of re-debugging it. The
second puts the fix on the integration branch, so later runs do not hit the
failure at all. Only NeMo-RL fixes ride the integration branch. A Megatron-Bridge
or Megatron-LM fix is checked out inside the container instead, and the registry
entry is the only thing that knows which branch that is — so record it with
`--fix-branch`, and `run_suite.sh` will carry it until the fix merges. Leave it
off and the branch you just pushed changes nothing about the next run.

`sync_integration.sh` only picks up branches named `mcore-*-fix` whose PR is
open, which is why the naming convention matters. If it reports
`CONFLICT applying #N`, your fix no longer applies on current `main`: rebase it,
or it protects nobody.

## Guardrails

- One PR per logical fix. Do not bundle unrelated repairs from different tests.
- Only a code fix gets an `mcore-*-fix` branch. Corrections to the agent's own
  skills, scripts, or config go out separately per
  [nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md),
  since anything matching that pattern is cherry-picked onto the integration
  branch that every later run is tested against.
- Do not push to the labeled Megatron-LM PR's branch directly, even with write
  access. Propose; let the author take it.
- Do not open a PR for a change that only silences a test (see the "never do
  these" list in
  [megatron-pr-failure-triage](../megatron-pr-failure-triage/SKILL.md)).
- If the fix touches an upstream API, verify the signature against the actual
  source at the revision under test before proposing it.
