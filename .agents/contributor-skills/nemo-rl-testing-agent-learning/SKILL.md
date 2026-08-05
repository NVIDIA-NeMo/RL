---
name: nemo-rl-testing-agent-learning
description: Capture what a NemoRLTest sweep taught the agent and write it back into the skills, scripts, config, and agent definition that drive the next sweep. Use whenever a run contradicts the instructions, a step needed knowledge the skills did not carry, or at the end of a sweep to promote the queued learnings into a reviewable PR.
when_to_use: The skill told you to do something that turned out to be wrong; you needed a flag or a step nobody wrote down; a failure mode has now cost two sweeps; closing out a sweep; "update your own instructions"; deciding whether something belongs in a skill, a script, config, or the ledger.
---

# Writing back what the sweep taught you

Cross-cutting stage of the nemo-rl-testing-agent pipeline. Every other stage
consumes instructions; this one repairs them.

The pipeline's whole value is not re-deriving things. `known_issues.py` gives
that to failures and `ensure_baseline.sh` gives it to `main`'s breakage, but the
agent's own operating knowledge — which flag prep needs, which symptom means the
image is stale, which order avoids a wasted allocation — has no such memory. It
lives in one sweep's context window and dies with it, so the next sweep pays the
cluster hours again.

**The bar is not "was this interesting" but "would a future run act differently
if it knew".** If the answer is no, it belongs in `ledger.md`.

## 1. Capture it the moment it happens

Do not stop the sweep to edit a skill. Recording takes a second and the queue
survives an interrupted run:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/learnings.py record \
  --id bridge-pin-needed-with-mcore-swap \
  --trigger "Every test failed at import; prep had swapped mcore but left the image's Bridge." \
  --lesson "Swapping megatron-core alone tests an incompatible pair. Always pin Bridge in the same prep step and record all three shas." \
  --target .agents/contributor-skills/megatron-pr-test-run/SKILL.md \
  --severity routine --context-pr 5700
```

Write `--lesson` as an instruction to a future run, not as a description of what
happened — that is what `--trigger` is for. A lesson naming a run name, a
`/lustre` path, or a home directory is rejected: those are specific to one run,
and skill files are read by contributors outside NVIDIA.

`--target` is the file the edit belongs in (see the routing table below). Pass a
path under `.claude/skills/` and the script records the real file behind the
symlink, since that is what a commit has to touch.

### `--severity blocking` changes what you do next

`routine` (default) means the queue can wait for the end of the sweep.
`blocking` means **the results this sweep is producing are wrong or unreliable
without the fix** — a prep step that silently tests the wrong revision, a parse
rule that mislabels failures as passes. For those, stop, apply the edit to the
working tree immediately so the rest of the sweep benefits, and say so in the
run's report. The queue entry is still promoted with the others at the end.

## 2. Where the learning goes

Putting it in the wrong place is how a correction gets ignored. Route by what
kind of knowledge it is, not by which stage happened to surface it:

| What you learned | Where it goes |
|---|---|
| A specific failure's diagnosis and fix | `known_issues.py record` — not here |
| A fact about one PR, node, or run | `ledger.md` — not here |
| A tunable: timeout, partition, image, path, budget | `.agents/nemo-rl-testing-agent/config.env` |
| A step, ordering, or symptom that changes how a stage is run | that stage's `SKILL.md` |
| A gotcha the agent has now hit twice despite it being written down | a guard in the relevant script, plus a test |
| A harness bug or a brittle assumption in a script | the script under `.agents/nemo-rl-testing-agent/scripts/`, plus a test |
| An orchestration rule: what to do first, what never to do | `.claude/agents/nemo-rl-testing-agent.md` |

Two rules of thumb keep this honest. **A value is never a lesson**: if the
correction is a number or a path, it goes in `config.env` and the skill keeps
pointing at the config key. **The second occurrence changes the remedy**: prose
that has already failed once will not work better in bold, so promote it to
something mechanical — a check in a script, a required flag, a line in the
agent's loop checklist.

`learnings.py record` tells you when that has happened. Re-recording an id that
was already promoted prints `REGRESSED` with the file it was written into. Treat
that the same way triage treats a `STALE` known-issue match: it is a real
finding about the instructions, not a duplicate to wave through.

## 3. Promote at the end of the sweep

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/learnings.py list
```

**No script does this part.** `learnings.py` only holds the worklist; you open
each target file and rewrite it yourself, editing it exactly as you would any
other file in the repo. A learning left sitting in the queue has changed
nothing, because no stage reads the queue — the next sweep reads the skills.

### What one promotion looks like

Take `bridge-pin-needed-with-mcore-swap`, targeting
`megatron-pr-test-run/SKILL.md`. Find the sentence that let the mistake happen.
Here it was the description of the mcore swap, which said what prep did to
megatron-core and nothing about Bridge:

> Swap the megatron-core revision with `--mcore-ref`; prep checks it out over
> the editable install.

The edit replaces that sentence — it does not add a warning underneath it:

> Swap the megatron-core revision with `--mcore-ref`; prep pins Megatron-Bridge
> in the same step, defaulting to the sha NeMo-RL itself pins. A current mcore
> against the image's older Bridge fails every test at import and reads as the
> author's bug.

That shape is not hypothetical. The Bridge pin the skill documents today exists
because of exactly that failure — the one recorded next to
`CONTAINER_BRIDGE_DIR` in `config.env` — and it is the kind of correction this
queue exists to make routine instead of accidental.

Then check the old wording is gone rather than merely outnumbered:

```bash
rg -n 'checks it out over the editable install' .agents/ .claude/agents/
```

Two copies of an instruction that disagree is the worst outcome available: the
next sweep may follow either. If the search still hits, the promotion is not
done.

Once every entry is edited:

```bash
git checkout -b nrlta-learnings-<yyyy-mm-dd>
git add .agents/ .claude/agents/nemo-rl-testing-agent.md
git commit -s -m "docs: fold nemo-rl-testing-agent sweep learnings back into the skills"
```

Read the diff before committing. It should be small, and every hunk should
correspond to a queue entry — a promotion that rewrites a section nobody
recorded a learning about is scope creep in the one place it is hardest to
review.

One branch and one **draft** PR against `NVIDIA-NeMo/RL` `main` per sweep, not
per learning — these are small correlated edits and a reviewer wants them
together.

The branch is deliberately **not** named `mcore-<N>-fix`: `sync_integration.sh`
cherry-picks every open branch matching that pattern onto the integration
branch, and instruction edits have no business riding along with the code fixes
that runs are tested against. Keep the two kinds of PR separable.

The PR body is the queue itself:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/learnings.py list --format markdown
```

Same delivery rules as any agent-authored change, per
[megatron-pr-fix-delivery](../megatron-pr-fix-delivery/SKILL.md): DCO sign-off,
Conventional Commits subject, draft status, and nothing from `$STATE_DIR`
committed. Then close each entry out so the next sweep does not re-promote it:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/learnings.py resolve \
  --id bridge-pin-needed-with-mcore-swap --as promoted --pr <n>
```

Decided against acting on one? `--as rejected --note "<why>"`. The note is
mandatory: an entry that disappears with no reason gets re-recorded next month
and re-argued from scratch.

If you touched anything under `.agents/nemo-rl-testing-agent/scripts/`, run the
tests before opening the PR:

```bash
for t in .agents/nemo-rl-testing-agent/tests/*.sh; do bash "$t"; done
```

## 4. Editing a skill without wrecking it

These files are the agent's working memory, and they degrade in a predictable
way: by accretion, until the instruction that matters is buried in the ones that
do not.

- **Edit in place; never append a "Learnings" section.** A correction means an
  existing sentence is wrong. Find it and replace it. A growing log at the bottom
  of a skill is read last and followed never.
- **Delete what the learning invalidates.** Leaving both the old and new
  instruction in the file is worse than either alone.
- **Keep the rationale, drop the anecdote.** Say why the rule exists in one
  clause. The war story goes in `ledger.md` or the PR body.
- **Respect the file's job.** A skill says what to do and why; a script does it;
  `config.env` holds the values. Do not inline a tunable into prose, and do not
  put a procedure into `config.env` as a comment.
- **Update the front matter when the scope changes.** `description` and
  `when_to_use` are what causes the skill to be loaded at all; a skill that has
  grown a new responsibility nobody can discover is not doing that work.
- **Keep a skill readable in one sitting.** The existing ones run 100–250 lines.
  A promotion that pushes one well past that is a signal to split or cut, not to
  keep appending.

## 5. Things not to write down

- A fix for a specific test failure. That is `known_issues.py`.
- Anything you have not actually verified. An unconfirmed theory recorded as an
  instruction is worse than no instruction, because the next sweep will follow
  it. Leave the theory in `ledger.md` until a run confirms it.
- A restatement of something the skill already says. If the skill said it and the
  run got it wrong anyway, the remedy is structural — see step 2.
- Secrets, tokens, internal hostnames, or cluster paths. These files are public.
