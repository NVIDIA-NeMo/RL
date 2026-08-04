---
name: megatron-pr-discovery
description: Finds open Megatron-LM pull requests carrying the NemoRLTest label, brings each branch up to date with main, and maintains the per-PR ledger that the nemo-rl-testing-agent works through. Use when starting or resuming a NemoRLTest sweep, or when asked which Megatron-LM PRs still need NeMo-RL validation.
when_to_use: Starting a NemoRLTest sweep; "which megatron PRs need testing"; "update the PR branch before testing"; resuming an interrupted sweep; deciding whether a PR already has fresh results.
---

# Discovering and preparing Megatron-LM PRs

Stage 1 of the nemo-rl-testing-agent pipeline. Produces, for each candidate PR,
a head SHA that is known to contain the latest `main`, plus a ledger entry that
every later stage appends to.

All tunables (repo, label, base branch, state dir) live in
`.agents/nemo-rl-testing-agent/config.env`. Never hard-code them.

## Ledger layout

One directory per PR under `$STATE_DIR` (default `~/.nemo-rl-testing-agent`):

```
pr-5700/
├── ledger.md          # append-only narrative: what ran, what failed, what was tried
├── pr.env             # KEY=value from update_branch.sh (HEAD_SHA, BASE_SHA, ...)
├── l1.results.json    # parse_results.py output, comments enriched by the agent
├── l2.results.json
└── baseline/          # results of re-running failed tests on plain main
```

The ledger is the recovery point. Write to it after every meaningful step so an
interrupted sweep can resume without re-running GPU jobs.

## Steps

### 1. List the candidates

```bash
.agents/nemo-rl-testing-agent/scripts/list_prs.sh
```

Returns a JSON array with `number`, `headRefOid`, `isDraft`,
`isCrossRepository`, `mergeable`, `mergeStateStatus`, `updatedAt`.

Process PRs **one at a time**, oldest `updatedAt` first, so the longest-waiting
author gets feedback first. Draft PRs are still tested (the label is the opt-in
signal, not the draft state).

### 2. Skip PRs that already have fresh results

If `pr-<N>/pr.env` records the same `HEAD_SHA` that GitHub reports now, and the
results JSON for both suites exists, the sticky comment is already current.
Skip unless the user explicitly asked for a re-run.

### 3. Update the branch

```bash
.agents/nemo-rl-testing-agent/scripts/update_branch.sh 5700 | tee "$STATE_DIR/pr-5700/pr.env"
```

This uses GitHub's "Update branch" button semantics (merge `main` into the PR
branch, no history rewrite). Exit code 2 means the update was refused; the
reason is in `UPDATE_ERROR`:

| Refusal | What to do |
|---|---|
| Merge conflicts with `main` | Do not test. Report every test as `not run` with the conflict as the comment, then move to the next PR. |
| Fork without maintainer edits (`isCrossRepository: true`) | Test the PR head **as-is** and say so in the report comment: results are against the un-merged branch. |
| Any other GitHub error | Retry once, then report `not run` with the error text. |

After a successful update, re-read `HEAD_SHA` — it is the new merge commit and
is what every test run must pin.

### 4. Pre-flight: does this PR need more than a source checkout?

Test runs swap the Megatron-LM revision by checking it out inside the container
over the **editable** `megatron-core` install. That covers pure-Python changes
only. Check what the PR touches:

```bash
gh pr view 5700 --repo NVIDIA/Megatron-LM --json files --jq '.files[].path'
```

Flag the PR if it changes any of:

- `setup.py`, `pyproject.toml`, `requirements*.txt` — new or bumped dependencies
  will **not** be installed by the checkout.
- `*.cu`, `*.cpp`, `*.h`, `*.pyx` — compiled extensions are **not** rebuilt; the
  image's prebuilt `.so` files stay in place.

These runs still proceed, but note the caveat in the report comment for any
failure that could plausibly come from a stale build artifact, and tell the user
an image build would be needed for a fully faithful result.

### 5. Record and hand off

Append to `ledger.md`: PR number, title, author, URL, head SHA, base SHA, update
outcome, and any pre-flight flags. Then continue with
[megatron-pr-test-run](../megatron-pr-test-run/SKILL.md).

## Ref to test

Every later stage pins the revision with a fetchable ref plus the SHA:

- PR under test: `--mcore-ref refs/pull/<N>/head --mcore-sha <HEAD_SHA>`
- Baseline: `--mcore-ref refs/heads/main` (no SHA pin; record what it resolved to)

`refs/pull/<N>/head` works for fork PRs too, which is why it is preferred over a
branch name.

## If this stage surprised you

A refusal reason not in the table above, or a pre-flight signal this skill does
not check for, is a gap in these instructions rather than a quirk of one PR.
Record it with
[nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md) and
keep going.
