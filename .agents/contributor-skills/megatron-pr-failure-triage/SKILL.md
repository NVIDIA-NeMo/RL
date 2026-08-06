---
name: megatron-pr-failure-triage
description: Diagnoses a failing NeMo-RL functional test on a Megatron-LM PR, decides whether the PR caused it, attributes the break to NeMo-RL, Megatron-LM, or Megatron-Bridge, and runs a bounded two-attempt fix loop. Use after a functional suite reports a failure for a NemoRLTest-labeled PR.
when_to_use: A functional test failed on a megatron PR; deciding if a failure is pre-existing; "try to fix this test failure"; attributing a Megatron-LM API break to the right repo; deciding when to stop trying.
---

# Triaging a functional-test failure

Stage 3 of the nemo-rl-testing-agent pipeline. Input: a `fail` entry in a
results JSON. Output: either a validated fix, or a written explanation of what
broke and what was tried.

**Look it up before you investigate it.** Most failures on a labeled PR are not
that PR's, and many have already been diagnosed on an earlier one. Re-deriving a
known diagnosis wastes the budget; raising a second fix PR for a bug that already
has one is actively harmful. Step 0 below is not optional.

**Every unclaimed failing test gets a real fix attempt — including a pre-existing
one.** A red test blocks the same people whether or not this PR turned it red,
and the agent is already holding all the context needed to chase it. Never close
out a failure with "not caused by this PR" and nothing else.

Two budgets bound the work:

- `MAX_FIX_ATTEMPTS` (default 4) hypothesis/edit/re-run cycles **per failing
  test**.
- **Two hours of wall-clock per PR** across all of its failures. When it runs
  out, stop mid-investigation and report what was learned.

Neither budget is a target. Stopping early with "here is exactly what broke, and
the call belongs to a human" beats a speculative patch that greens a test.

## Order of work

### 0. Ask what is already known

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/known_issues.py annotate \
  --results ~/.nemo-rl-testing-agent/pr-<N>/l1.results.json \
  --integration ~/.nemo-rl-testing-agent/integration.json
```

Each failure comes back in one of three states.

**`known`** — already diagnosed, with its fix linked. The entry's diagnosis is
written into the test's comment for you, and the status drops to
`fail (pre-existing)`, because an entry only ever exists for a break the PR did
not cause. Do not investigate; do not open another fix branch. If the fix has
been sitting in review, chasing the reviewer is the useful action, not
re-debugging the bug.

That downgrade is also why you must not record a break the PR *did* cause. The
registry is cross-PR memory: an entry excuses this failure on every future run,
including the one where it is somebody's fault. Report those against the author
instead.

Do not lean on `apply_baseline.py` to do that downgrade. It only fires when the
cached baseline failed the same test, and the baseline runs on a stack that
carries pending fixes — so the moment a fix exists, the baseline goes green and
stops covering for anybody. That combination once had three innocent PRs
reported as `fail` with a link to the fix printed directly underneath.

**`STALE`** — the registry claims a fix **and that fix is already applied to the
branch this run tested**, yet the test failed anyway. This is a genuine finding.
Either the entry is wrong, or a second bug produces the same signature.
Investigate it as new, and correct the registry entry once you know which.

**`new`** — nothing on record. This one is yours; continue to step 1.

A registry entry retires itself when its fix PR merges, so a `known` label always
means "there is an open fix for this". After a merge, the same failure comes back
as `new`, which is correct: it is a regression now, not a known issue.

### 1. Rule out infrastructure

Already covered by [megatron-pr-test-run](../megatron-pr-test-run/SKILL.md)
step 3. Do not start diagnosing a CUDA error that was really a preempted node.

One marker is worth calling out because it looks like infrastructure and is not:
`NRLTA_PREP_FAIL_INTEGRATION` means the megatron-core revision under test could
not be imported together with the Megatron-Bridge that NeMo-RL pins. That is a
real finding about the revision — usually an API the PR removed and Bridge still
uses — so it goes through baselining and attribution like any other failure. Plain
`NRLTA_PREP_FAIL` is the harness's own problem and is never the author's.

### 2. Establish whether the PR caused it

This is the single most valuable step in the whole pipeline: without it, a broken
`main` gets reported as five different authors' faults.

The baseline is a cached daily run of the suite against megatron-core `main`, so
this is normally a lookup rather than a cluster job:

```bash
baseline="$(.agents/nemo-rl-testing-agent/scripts/ensure_baseline.sh --suite l1)"
uv run --script .agents/nemo-rl-testing-agent/scripts/apply_baseline.py \
  --results ~/.nemo-rl-testing-agent/pr-<N>/l1.results.json \
  --baseline "${baseline}" \
  --baseline-meta "${baseline%.json}.env"
```

`ensure_baseline.sh` reuses a baseline younger than `--max-age-hours` (24 by
default) and otherwise runs one and caches it. `apply_baseline.py` rewrites any
failure that also fails on `main` to `fail (pre-existing)` with the main sha in
the comment, and reports which failing tests were absent from the baseline so you
can decide about them explicitly.

The baseline decides **attribution and wording, never whether to investigate**:

- Reclassified to `fail (pre-existing)` → say plainly that the PR did not cause
  it, then fix it anyway. The fix lands in NeMo-RL (or Bridge) on its own branch
  and is reviewed independently of this PR.
- Still `fail` after applying the baseline → the PR caused it. The fix probably
  belongs in NeMo-RL's call site or in the PR itself.
- Absent from the baseline (a test the PR adds or renames) → re-run just that
  test against `main` with `--tests`, and say so in the comment either way.

Run the baseline **before** blaming a PR, not after. Establishing it up front also
catches NeMo-RL/Bridge/mcore integration drift on its own, instead of letting it
surface as a mystery failure on somebody's unrelated PR.

Note what the baseline is taken against: megatron-core `main` × the **integration
branch** × the pinned Bridge. So `fail (pre-existing)` now means "fails on `main`
even with every fix we have already raised applied" — a stronger and more useful
statement than it used to be. Pre-existing breakage found here belongs to
[nemo-rl-mcore-watchdog](../nemo-rl-mcore-watchdog/SKILL.md); if the watchdog has
not claimed it yet, fix it and record it so it is claimed from now on.

### 3. Attribute the break to a repo

Read the error, then the PR diff, then the NeMo-RL call site. Typical patterns:

| Symptom | Usually belongs in |
|---|---|
| `TypeError: ... unexpected keyword argument`, `AttributeError` on a mcore class, changed default | **NeMo-RL** — the PR intentionally changed a Megatron API and NeMo-RL's call site must follow. Check `nemo_rl/models/generation/megatron/`, `nemo_rl/models/megatron/`, `nemo_rl/models/policy/megatron_policy_worker.py`. |
| Config key no longer accepted / renamed | **NeMo-RL** config plumbing, or **Megatron-Bridge** if the key is mapped there. |
| Wrong numerics: `token_mult_prob_error` above threshold, NaN loss, prefix-cache misses | **Megatron-LM** — a real behaviour regression in the PR. Do not paper over it in NeMo-RL. |
| Crash inside `megatron/core/**` with no NeMo-RL frame in the traceback | **Megatron-LM**. |
| Model provider / weight mapping / dist-ckpt load errors | **Megatron-Bridge** (`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`). |

Verify the API against the actual source in
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM` at the
PR's revision before concluding — do not guess a signature.

### 3b. Do not trust a green test

`rollouts.py` wraps per-sample generation in `except Exception: print(...); break`,
so a backend that raises on **every** sample still produces a rollout, a metric,
and `rc=0`. This is not hypothetical: when megatron-core stopped echoing
`prompt_tokens`, two async L1 tests exited 0 while every single generation raised
`AttributeError`.

`parse_results.py` marks these `pass (suspect)` and records the swallowed error.
Treat a suspect pass as a failure for triage purposes — same investigation, same
budget — and say so in the comment. A test that reports success while doing no
work is worse than a red one, because nobody goes looking.

### 4. Fix loop (at most `MAX_FIX_ATTEMPTS` per test)

Each attempt is: one hypothesis, the smallest edit that tests it, one targeted
re-run of that test alone.

```
Attempt N:
- [ ] State the hypothesis in one sentence in ledger.md
- [ ] Make the minimal edit in the repo identified in step 3
- [ ] Re-run just that test (fresh --run-name, -a<N>)
- [ ] Record the outcome in ledger.md whether it passed or not
```

For a NeMo-RL fix, edit the local worktree and re-run with
`--nemo-rl-ref worktree --allow-dirty`; prep overlays the tree into the container
and the diff is saved next to the run so the result stays interpretable. Runs
without that flag test the integration branch instead, which will not contain
your uncommitted edit. For a Megatron-LM or Megatron-Bridge fix, push the branch
first and re-run with `--mcore-ref refs/heads/<branch>`; see
[megatron-pr-fix-delivery](../megatron-pr-fix-delivery/SKILL.md).

**Never do these to make a test pass:**

- Loosen a `check_metrics.py` threshold or delete an assertion.
- Comment out, skip, or shorten a test.
- Reduce step counts, sequence lengths, or batch sizes below what the test
  specifies.
- Catch and swallow the exception at the call site.

If the only way to green a test is one of the above, the correct outcome is a
reported failure with that finding written down. A threshold that legitimately
needs to move is a decision for the PR author and the reviewer, not the agent —
say so in the comment.

### 5. Stop and write it down

After a successful fix, or once either budget is spent, write the comment
material into the results JSON entry. **A failure never gets a bare "not caused
by this PR"** — one of these two outcomes is mandatory:

- **Fixed**: the root cause in one sentence plus the branch/PR link from
  [megatron-pr-fix-delivery](../megatron-pr-fix-delivery/SKILL.md).
- **Still failing**: the error signature, the most likely root cause, and each
  attempt with what it changed and why it did not work. Someone picking this up
  cold should not have to re-derive anything. Naming the one thing you would try
  next is worth more than a summary of what you already tried.

Then record it in the registry — **both outcomes, not just the fixed one**. An
unfixed failure with a written diagnosis still saves the next PR the whole
investigation:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/known_issues.py record \
  --id <stable-slug> --test <test name> \
  --signature "<error_signature verbatim from the results JSON>" \
  --diagnosis "One or two sentences a PR author can act on." \
  --repo NVIDIA-NeMo/RL --fix-pr <n> \        # omit --fix-pr if there is no fix
  --fix-branch <branch> \                     # required for a Bridge or mcore fix
  --first-seen-megatron-pr <N>
```

Copy the signature verbatim; `normalize()` strips the run-specific parts. Never
hand-write one containing a measured value — `median(...) < 1.1 (measured 3.37)`
will not match the same bug measuring 2.01 next week.

`--first-seen-megatron-pr` records which run surfaced the break, for tracing it
back later. It is not a culprit: the break is usually older than that PR and
often in another repository, so nothing reads it as blame.

`--fix-branch` is load-bearing. It is what carries a **Megatron-Bridge or
Megatron-LM** fix into subsequent runs at all: a
NeMo-RL fix rides the integration branch, but a fix in another repository is
checked out inside the container, and `run_suite.sh` looks the branch up here to
decide. Omit it and the fix exists in review while every run keeps reproducing
the bug it fixes — for as long as review takes.

The registry entry is about the bug. If the investigation also taught you
something about *how to investigate* — a place this skill sent you that was
wrong, a symptom that means something other than what the attribution table
says, an attempt shape that reliably wastes one of the two — that belongs in
[nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md),
which is where this skill gets corrected. The fix budget is small enough that
one misleading sentence here costs a real attempt.

Then hand off to [megatron-pr-reporting](../megatron-pr-reporting/SKILL.md).
