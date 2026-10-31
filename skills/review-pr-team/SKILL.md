---
name: review-pr-team
description: Agent-team-based parallel code review for NVIDIA-NeMo/RL pull requests. Spawns specialized agents (RL expert, submodule experts, bug finder, design reviewer, test agent, devil's advocate, comment reviewer) that coordinate via shared task list and direct messaging. Leader orchestrates, collates ALL findings, and presents to user for approval before posting.
when_to_use: Deep multi-agent review of a PR; '/review-pr-team <number>'; 'team review PR', 'thorough parallel review'. Requires CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1.
argument-hint: "<pr-number>"
allowed-tools:
  - AskUserQuestion
  - Bash
  - Read
  - Glob
  - Grep
  - Agent
  - TaskCreate
  - TaskList
  - TaskGet
  - TaskUpdate
  - SendMessage
---

# Agent Team PR Review — NVIDIA-NeMo/RL

Review a pull request using a coordinated team of specialized agents.

**Repo**: `NVIDIA-NeMo/RL`

**Requires**: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` must be set — agent teams (teammate spawning, shared task list, `SendMessage`) are an experimental Claude Code feature.

**Sandbox**: Always pass `dangerouslyDisableSandbox: true` on every Bash tool call in this skill. The sandbox's bubblewrap (bwrap) container fails to initialize in worktree/container environments with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`, breaking ALL commands — not just network-dependent ones like `gh`. Do NOT attempt any Bash call in sandbox mode first; it will fail and waste a round-trip.

---

## Phase 0: Parse & Validate

Extract `$PRNUM` from `$ARGUMENTS`. A PR number is required.

```bash
gh pr view $PRNUM --repo NVIDIA-NeMo/RL --json number
```

If invalid, ask the user for a valid PR number.

---

## Phase 1: Setup & Context

### 1.1 Checkout PR

```bash
git fetch origin pull/$PRNUM/head:pr-$PRNUM-team-review
git checkout pr-$PRNUM-team-review
git submodule update --init --recursive
```

### 1.2 Gather PR metadata (parallel)

Run these in parallel:

```bash
# PR metadata (include mergeable to detect conflicts)
gh pr view $PRNUM --repo NVIDIA-NeMo/RL \
  --json title,body,author,baseRefName,headRefOid,labels,files,comments,reviews,reviewRequests,mergeable,mergeStateStatus

# Full diff
gh pr diff $PRNUM --repo NVIDIA-NeMo/RL

# Inline review comments
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/comments
```

Record: `$TITLE`, `$AUTHOR`, `$BASE_BRANCH`, `$HEAD_SHA`, changed files list, existing comments, existing reviews.

**Merge conflict check**: If `mergeable` is `"CONFLICTING"` or `mergeStateStatus` is `"DIRTY"`, include a prominent note in the review asking the author to rebase their PR on `$BASE_BRANCH` and resolve conflicts before further review. Add this as the first item in the review body.

**Read the PR description (`body`) carefully.** It contains the author's intent, motivation, and test plan. Also parse it for linked issues (patterns like `Fixes #123`, `Closes #456`, `Related: #789`). For each linked issue, fetch it:

```bash
gh issue view <ISSUE_NUM> --repo NVIDIA-NeMo/RL --json title,body,comments,labels
```

The PR description + linked issues + diff together form the full context. All agents should have access to this context so they understand *why* the change is being made, not just *what* changed.

### 1.2a Performance & convergence evidence check

If the PR touches code that could affect **performance or convergence** (e.g. new training features, optimizer changes, CUDA graph support, kernel changes, parallelism config), check the PR description and comments for quantitative evidence:

- **New feature**: Author should provide baseline numbers (throughput, tokens/sec, memory, convergence curves) demonstrating the benefit of the feature.
- **Existing feature modification**: Author should show before/after comparison or prove no regression (e.g. convergence curves from an A/B run).
- If evidence is missing, flag it as a `[PERF-EVIDENCE]` finding in the review. This is especially important for features that users will enable in production — they need to know the expected benefit and any trade-offs.

Record whether evidence was found: `$PERF_EVIDENCE_FOUND` (yes/no/not-applicable). Pass this to the `rl-expert` agent in its prompt.

### 1.3 Determine touched submodules

```bash
gh pr diff $PRNUM --repo NVIDIA-NeMo/RL --name-only | grep -E '^3rdparty/'
```

Map paths to submodule names:
- `Automodel-workspace/` → spawn `expert-automodel`
- `Megatron-Bridge-workspace/` → spawn `expert-megatron-bridge`
- `Megatron-LM-workspace/` → spawn `expert-megatron-lm`
- `Gym-workspace/` → spawn `expert-gym`

### 1.3a Upstream reference lookup

When the PR adds or modifies config that wraps Megatron-LM `TransformerConfig` fields, agents MUST check how **Megatron-Bridge** sets those same fields. Megatron-Bridge is the canonical integration layer and often shows the correct, non-deprecated API:

1. Get submodule SHAs for permalinks: `git ls-tree HEAD 3rdparty/Megatron-LM-workspace/Megatron-LM` and `git ls-tree HEAD 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
2. Search Megatron-Bridge for the config field name: `grep -rn <field_name> 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/`
3. Read the Megatron-Bridge code to understand the pattern (what field it sets on `recipe.model`, how it handles RNG, scope validation, etc.)
4. Read the Megatron-LM `TransformerConfig` field definition and docstring for the canonical documentation of valid values, defaults, and deprecation status
5. Pass the submodule SHAs and relevant Bridge/LM permalinks to all agents in their prompts

This ensures review comments can point authors to the established pattern rather than just saying "this is wrong."

### 1.4 Read root context

- Read `CLAUDE.md` from repo root
- Read all `.claude/skills/*/SKILL.md` files (except `review-pr` and `review-pr-team`)
- Read every `references/*.md` file in this skill's own directory. These are the shared review lessons for this repo, grouped by theme. A rule marked ★ has come up three or more times; list those first when you pass the lessons on. A rule that ends with `Details: [name](details/name.md)` has a full lesson behind it, with the steps, commands and repo maps the one-line rule leaves out. Keep those links when you pass the lessons on, and tell every agent to read the detail file whenever its rule applies to what it is checking.
- Read every `skills/contributing/references/*.md` and treat each rule as a finding checklist, the same as this skill's own references. A rule that ends with `Review Details: [name](...)` has a review lesson behind it, and one that ends with `Contributing Details:` has an author how-to under `skills/contributing/references/details/`; tell every agent to read the linked file whenever the rule applies.
- Then glob `~/.claude/review-memory/RL/*.md` and read EVERY match. This is the optional, machine-local store of lessons that are not in `references/` yet. It may be empty or absent; an empty glob is fine. New lessons are written here in Phase 6. A memory's `count` is how many times its pattern has shown up in reviewed PRs; list the ones with count 2 or more first.
- Treat all of them as binding guidance and pass their content to every spawned agent in its prompt.

### 1.5 Detect GPU availability & set up local testing

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
```

If GPUs are available, set `$GPU_TESTING_AVAILABLE = true`. All agents run tests locally via `uv run` — no Docker container needed. On first run, `uv sync` resolves all deps (torch, vLLM, CUDA) directly on the host (~5 min first time, instant after).

If no GPUs: set `$GPU_TESTING_AVAILABLE = false`. Agents should note "not verified — no GPU environment" in their findings. Test-agent can still do CPU-only work (grep, read, collect-only).

### 1.5b Kick off the repo linters (background)

Start the repo's own lint/format/type suite in the background early, so results are ready by collation:

```bash
uv run --group dev pre-commit run --all-files
```

Do NOT run `ruff`/`pyrefly` directly — `pre-commit` runs the repo's pinned hook versions and config, and calling the tools directly can use the wrong versions/args and produce misleading results. Fallback only when `pre-commit` cannot run at all: run the pinned version with `uvx ruff@<version from .pre-commit-config.yaml>` and run every ruff hook the config lists (plain `ruff check` misses `I001`). When collating, attribute findings ONLY to files in the PR diff (`gh pr view $PRNUM --json files`); `--all-files` also surfaces pre-existing issues unrelated to this PR — don't report those. Also flag any NEW source file in the diff that isn't added to `pyrefly.toml` `project-includes` (untracked files silently escape type-checking).

### 1.6 Teams are MANDATORY — never substitute plain subagents

With `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` set, agent teams form automatically: the current session is the team lead, and the team is created the moment you spawn the first teammate via the `Agent` tool. There is **no `TeamCreate` step** (that tool was removed; a session has exactly one implicit team). You do not name the team — refer to the review as "the PR #$PRNUM review" in prompts for human readability only.

**Every agent in this skill MUST be spawned as a named teammate** — `Agent` with a `name`, so it appears as its own tmux pane and can `SendMessage` its peers. This is not a preference to trade off against reliability:

- **Do NOT spawn unnamed/plain background `Agent` calls as a substitute**, even when the teammate channel is being slow or silent. The user watches the panes; a review that "worked" with no panes reads as the team having run when it did not.
- **Do NOT fall back to solo review.** It throws away substantive reports — six of them on PR #3292, which arrived in full after one correctly-phrased nudge.
- If a lane is genuinely stuck, escalate *within* teams: re-send the verbatim tool-call nudge below, then re-spawn that lane as a NEW named teammate. Only if a lane still cannot be recovered may you use an unnamed `Agent` for that one lane — and then you MUST say so explicitly in the Phase 4 preview.

The cost of substituting plain subagents is not hypothetical: they cannot message each other, so every cross-lane disagreement becomes a serial leader round-trip. On PR #3591 two lanes contradicted each other on backend reachability and two adversarial passes disagreed about CI lane coverage — with a team those are one peer `SendMessage`.

#### Making teammate reports actually land

The delivery failure is real, mechanical, and fixable. Do not let it push you off teams.

- An agent spawned by `Agent` returns its final text to the leader automatically.
- An agent **resumed via `SendMessage`** has NO such channel. Its final assistant text goes nowhere. It must explicitly call the `SendMessage` tool.

So a nudge saying "report back" produces an idle notification with no content — the agent answered into the void. Phrase it as a tool instruction, verbatim:

> Your report never reached me. When you are resumed by a message, your final assistant
> text is NOT delivered — you must explicitly call the SendMessage tool. Do this now:
> `SendMessage(to="team-lead", summary="...", message="<your full findings>")`. Do NOT
> just write the text as your answer — call the tool.

On PR #3292 all six teammates were silent through two rounds of ordinary nudging; after one message phrased that way, all six delivered within minutes.

Two force multipliers when nudging: state what you have ALREADY verified (so they spend remaining effort on gaps, not redoing your work), and list 2-4 numbered open questions.

**Delivery can be very late and arrive in ONE batch — never declare the channel dead.** On PR #2902 five teammates showed `completed` and stayed silent through eight nudges over ~2 hours, then every report landed at once and refuted a finding that was already staged. Confirmed again on #3422 at ~14 hours, producing that review's top finding. So: keep working while you wait; prefer long sleeps over tight `TaskList` polling; say "reports have not arrived yet", never "the team did not deliver"; and if reports land after you have staged, re-open the staged set against them. A teammate can also outlive your own shutdown request — until the review is POSTed, a late report is still actionable.

**Teammates mutate the shared worktree.** Two agents once applied prototype patches to source files mid-run. Before staging, re-verify: `git status --porcelain` plus per-file `md5` against `git show HEAD:<path>`. Never stage a review built against a tree an agent may have edited.

**Adjudicate, don't aggregate.** Agents contradict each other and the leader, and the leader is not automatically right. Verify every load-bearing claim yourself, in both directions — including an agent overstating "no CI job covers X" when one does.

---

## Phase 2: Create Tasks & Spawn Agents

### 2.1 Create tasks

Create all tasks upfront using `TaskCreate`, then set dependencies with `TaskUpdate`.

**Wave 1 tasks (parallel, no blockers):**

| Task | Owner | Description |
|------|-------|-------------|
| `analyze-rl-code` | `rl-expert` | Analyze diff for RL code, guidelines, docstrings. Report ALL findings. |
| `analyze-{submodule}` | `expert-{submodule}` | (conditional) Analyze submodule changes. Detect upstream bugs. Report ALL findings. |
| `review-existing-comments` | `comment-reviewer` | Review all PR comment threads. Identify responses needed. |
| `review-and-suggest-tests` | `test-agent` | Review tests in PR. Suggest new tests. Run tests locally via `uv run` if GPUs available. |
| `scan-for-bugs` | `bug-finder` | Scan diff for bugs independently. Write tests when uncertain. |
| `review-design` | `design-reviewer` | Review design-level changes (new class/interface/config/worker group/module) for testability, extensibility, and maintainability — flagging drift toward BOTH under- and over-abstraction. Bail fast on PRs with no design surface. Report ALL findings. |

**Wave 2 tasks (blocked by ALL Wave 1 tasks):**

| Task | Owner | Description |
|------|-------|-------------|
| `challenge-findings` | `devil-advocate` | Stress-test ALL findings from Wave 1 (including bug-finder). Up to 2 rounds per agent. Waits for the leader to push consolidated Wave 1 findings via SendMessage — does NOT poll TaskList. |

**Wave 3 task (blocked by Wave 2):**

| Task | Owner | Description |
|------|-------|-------------|
| `edit-comments` | `comment-editor` | Rewrite every surviving finding into wire-ready comment text, and rewrite the review body. Owns the final prose. MANDATORY — the leader may not bypass it. |

**Wave 4 task (blocked by Wave 3, leader does this):**

| Task | Owner | Description |
|------|-------|-------------|
| `collate-review` | leader | Assemble the payload from comment-editor's text, run the pre-POST audit, present to user. |

Set dependencies:
- `challenge-findings` blockedBy all Wave 1 task IDs
- `edit-comments` blockedBy `challenge-findings`
- `collate-review` blockedBy `edit-comments`

### 2.2 Spawn agents (parallel)

Spawn all agents in a single message with multiple `Agent` tool calls. Give each agent a unique `name` (so teammates can address each other via `SendMessage`). Do **not** pass `team_name` — that input is ignored; spawning a teammate joins the session's single implicit team automatically. You may set `subagent_type` to reuse a defined role; its `tools`/`model` are honored and its body is appended to the teammate's prompt (`SendMessage` and the task tools are always available regardless of any `tools` allowlist).

Spawn Wave 1 and `devil-advocate` here. Do **not** spawn `comment-editor` yet — it is Wave 3 and only ever sees findings that survived the adversarial pass. Spawning it early wastes it on findings that are about to be disputed, and invites it to start editing before the set is final.

---

## Agent Prompt Specifications

### Common preamble (include in EVERY agent prompt)

```
You are a member of the "pr-$PRNUM-review" team reviewing PR #$PRNUM on NVIDIA-NeMo/RL.

PR: #$PRNUM "$TITLE" by $AUTHOR (base: $BASE_BRANCH, head: $HEAD_SHA)

PR Description:
$PR_BODY

Related Issues:
$RELATED_ISSUES (title, body, and key comments for each linked issue)

Instructions:
- Read and understand the PR description and related issues FIRST — they explain the author's
  intent, motivation, and test plan. Review the code in that context.
- Check TaskList for tasks assigned to you. Claim your task with TaskUpdate(status="in_progress").
- Use Read, Glob, Grep for local file lookups (faster than gh CLI).
- Do NOT git checkout other commits — use `git show <sha>:<path>` for history lookups.
- After any git operation that changes commits: `git submodule update --init --recursive`
- Include GitHub permalinks in ALL findings: https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/<path>#L<line>
- **Prose permalinks — every reference to code that is NOT on the commented line gets a link.**
  The test is location, not category: if a reader would have to leave this comment to see the
  thing you just named, it needs an `<a href>` at a pinned SHA. That covers function / class /
  method / config-field / sentinel names (`get_replay_topk()`,
  `_install_missing_route_fallback_patch`, `R3_MISSING_ROUTE_SENTINEL`) **and also**, which is
  where this rule usually gets violated:
    - **bare file paths** — `models/megatron/data.py`, `test_megatron_data.py`. A path is not a
      link. Link it to the exact line you mean.
    - **`file:line` strings** — `nightly.txt:89`, `controller.py:1645`. These look precise and are
      still not clickable; they are the most common offender because they *feel* like citations.
    - **test names, suite entries, recipe/YAML names** you claim do or don't cover something.
    - **other comments in your own review** — "see the comment on the CLEVR recipe" must carry a
      permalink to that file/line (or, after posting, the comment URL).
  Applies to review comments, HTML explainers, and any other output. Readers should never have
  to grep for something you mentioned by name.
  **Self-check before posting, and do it literally:** re-read the comment and point at every
  `code span` and every filename. For each one ask "is this on the line I anchored to?" If no and
  it has no link, that is a defect — fix it. One permalink in the opening sentence does not cover
  the eight bare references below it.
- **Deleted code MUST be linked at the BASE commit, never at head.** When a finding is about
  something the PR *removes* — a guard, a branch, an assert — a head-SHA permalink points at
  whatever now occupies those line numbers, i.e. unrelated code. That is worse than no link,
  because it looks authoritative and misleads. Link the parent/base SHA
  (`git merge-base <base-branch> <pr-branch>`), say so in the text ("linked at the base, since it
  no longer exists at head"), and quote the removed lines inline so the reader sees them without
  clicking. Observed on PR #3921: a comment about a deleted `NotImplementedError` linked
  `worker_mixin.py#L209` at head, where the guard was already gone.
  Corollary: a deletion has no `+` line, so it cannot be a comment anchor. Anchor the comment on
  the nearest related added line (or a file the finding is really about) and open by naming where
  the deleted code lived, with the base permalink.
- **Evidence permalinks**: When claiming behavior exists in upstream code (Megatron-LM,
  Megatron-Bridge, etc.), include a permalink to the EXACT line in the upstream repo that
  proves it. Use the submodule's GitHub repo + pinned SHA (leader will provide these).
  Don't just say "transformer_config.py:767 says deprecated" — link to it so the reader
  can click through. Quote the relevant snippet inline for quick scanning.
- **External reference implementation claims**: When claiming "standard implementations
  do X" (e.g. verl, TRL, OpenAI baselines, DeepSpeed-Chat), you MUST fetch the actual
  source code (via WebFetch on the raw GitHub URL or `curl`) and provide a permalink to
  the exact line that proves the claim. Use a pinned commit SHA, not `main`. Never claim
  "verl does X" or "TRL does Y" based on memory alone — verify and link.
- **Copied-constant check.** Any value the PR copies from code it doesn't own (a string key, a field
  name, a protocol name, a look-alike type) — usually flagged by a comment like "must match X" or
  "mirrors Y" — has to be justified, in this order: (1) import it where it is read, even with a
  deferred import; (2) if that is impossible because the import would pull packages this module
  must not load, move it to a light shared home both sides can import; (3) only if neither exists,
  keep the copy AND require a unit test, in a lane that can import the owner, that FAILS (never
  skips) when the two differ. Full rule: TEST E in `design-reviewer`.
- **vLLM patch check — run this whenever the PR adds or changes a patch.** A "patch" is any workaround for
  upstream behaviour: rewriting vLLM source by literal string match, monkey-patching its objects at runtime,
  aliasing attributes onto its modules, or replacing one of its methods. Two incidents drive this: a
  string-replacement patch critical to the VLM path **silently failed to apply** when its anchor stopped
  matching, and two separate places patching vLLM **conflicted, with the second silently overwriting the
  first**. Both were invisible until a downstream symptom appeared. Check all five:
    1. **One home.** It lives in `nemo_rl/models/generation/vllm/patches.py`. A runtime workaround tucked into
       a quantization backend or a worker is the same hazard with less visibility — flag it and name the file.
    2. **One executor.** Exactly one function *applies* patches; everything else *registers*. If a patch runs
       from two entry points, say so — that is the overwrite bug's shape, and idempotency is a weaker defence
       than not doing it twice.
    3. **Loud on drift, at build AND run time.** A missing anchor must raise, not `logger.warning` and
       continue. A warning in a worker log is not a signal anyone acts on.
    4. **A version check.** The patch records which vLLM version it was written against, so a bump reports
       which patches to re-check instead of failing mid-run on a GPU node. Messages that merely *say* "the
       vLLM version may have changed" do not count — grep for an actual comparison before crediting one.
    5. **Safe to apply twice**, with a test proving it.
  Also ask whether the patch is needed at all: every one is a standing cost at each upgrade, so prefer
  upstreaming or a supported extension point, and expect the docstring to say why no supported hook exists.
- **Verify the premise before reporting.** When a finding hinges on how an external tool/API/env-var behaves
  (SLURM, Ray, Megatron, transformers, …), verify it against source/docs with a permalink FIRST. Never inflate
  your stated confidence to clear a threshold without checking the underlying fact — that launders speculation
  into an authoritative-looking comment.
- **Make every finding actionable.** Lead with the observation, then a bolded **Action:** line naming the exact
  file/function to change and (when known) the concrete snippet/schema. Drop pure editorializing ("looks
  intended", reassurance) that doesn't lead to an action.
- **Open every comment by declaring its actionability, before any prose.** The author must know what they
  are expected to DO before they read why. The first line is one of:
    - `**C<n> · No action needed** — <one clause>.` for pure FYI/context.
    - `**C<n> · 1 action item.**` / `**C<n> · 2 action items, 1 follow-up.**` when there is work.
  `C<n>` is the comment's ID (see Phase 4, "Comment IDs"). The user reads the staged review on GitHub and
  refers to comments by this ID when asking for changes.
  Then make the body match that count: label each ask `### AI-1`, `### AI-2` …, label deferred work
  `### Follow-up` with its tracking reference, and put pure background under a final `**Context — no
  action.**` heading. If an ask should be done in THIS PR rather than deferred, say so in the opening line
  ("please fix in this PR") and do not also offer a tracked fallback — an escape hatch beside an ask is
  read as permission to skip it.
  **Every AI must be independently followable, and must NOT open by positioning itself against another one.**
  Banned openers: "Separate from the above", "Building on AI-1", "As mentioned earlier", "Relatedly". The
  author reads an AI to find out what to DO; a first clause about where this ask sits in your argument makes
  them re-read upward before they can start. Each AI opens with the imperative — what to change, in which
  file/function — and carries whatever context it needs, even if that repeats a line from a sibling AI.
  Repetition across AIs is cheap; a cross-reference the reader has to resolve is not.
  Test it the way the author will: read each `### AI-n` section ALONE, with everything above it covered. If it
  no longer tells you what to do, rewrite it. Observed on PR #3921: the user could not tell what an AI was
  asking because it began "Separate from the above," — the ask itself was one line and perfectly clear once
  found.
  Give one recommended fix per AI. No menus: no "pick either", no "preferred … or if you'd rather …", no
  "optional". If you cannot pick one, ask the user before staging; do not hand the choice to the author.
- **EVERY comment needs a one-line TL;DR**, bolded, immediately after the actionability declaration and
  before any prose. No length threshold — a comment without one fails the pre-POST audit. State the defect
  and its consequence, not the topic: "**TL;DR — in colocated mode the `mcore_generation_config` model keys
  are silently discarded, so `policy.megatron_cfg` is the only route left**", never "TL;DR — about how config
  keys are handled". Reviewers skim; a comment with no TL;DR gets skipped entirely, which is strictly worse
  than a short comment that lands.
  **If the whole finding turns on one non-obvious fact about what some code actually does, that fact IS the
  TL;DR.** Do not put it at step 2 behind a correct "background" sentence — the reader is then reverse-
  engineering the one thing the finding rests on. Test: "if the reader believed only my TL;DR, would they
  still have to guess why this is a bug?" If yes, rewrite it as the causal fact — "X is the only thing that
  does Y, so skipping X means no Y".
  Two corollaries. If you cannot compress the finding into one line, it is usually TWO findings — split them
  into separate comments anchored at their own lines, rather than one comment with numbered sections; several
  focused comments are far easier to act on than one that has to be read in full before any of it can be.
  And if the user ever asks you to "tl;dr" a comment you already posted, treat that as a defect report on the
  comment itself: shorten it in place and split out whatever forced the length.
- **Never blend an ask and a "tracked elsewhere" note in one paragraph.** A comment that says both
  "tracked in <issue>" and "please change X" without separating them is the most common way review
  feedback gets silently dropped: the author reads "tracked" and closes the thread. When you cite a
  tracking issue, cite the SPECIFIC item (a stable id and title, not a bare issue link) and state in the
  same sentence whether it covers the ask or only the part you are NOT asking for.
- **Be ruthlessly succinct — length is a cost the author pays.** Write the shortest comment that still lands,
  then cut again. Before each sentence ask "does deleting this change what the author does?" If no, delete it.
  Cut: restatements of what the code plainly does, hedging preambles, evidence the author needn't act on, and
  recaps of your own investigation. Reviewers skim — an over-long comment gets skipped entirely, so it is
  WORSE than a short one even when every sentence is true. The breaking-case walkthrough and the mechanism
  explanation below are the only licensed exceptions, and both have their own tight budgets.
  **Succinct does NOT mean action-only.** The visible part of every comment carries three things: the TL;DR,
  a short explanation of how the problem actually shows up, and the fix. Stripping it to a bare imperative is
  the opposite failure and is just as wrong. Examples, pseudocode, counter-examples and permalinks all belong
  in the visible part — short does not mean evidence-free.

  **The comment skeleton:**

  ```
  **C<n> · N action items.** **TL;DR — <the issue in one line>**
  <short explanation of how the problem shows up>
  ### AI-1
  <the fix: a ```suggestion block, or the concrete change + one clause why it can't be one>
  [optional] <details><summary><label naming what is inside></summary> … </details>
  ```

  There is no acknowledgement/credit section in an inline comment. Praise for the approach, or credit for
  something fixed since the last round, goes in the review BODY, once — not per-comment.
- **Never let your own investigation appear in the visible part of a comment.** The author did not do the
  investigation and does not act on it. These four all move below the fold or get deleted outright:
    1. **First-person narration** — "I checked whether X breaks and could not construct a failing config",
       "Recording what I looked at", "To be clear on what this does not change: I verified …".
    2. **Pre-empting objections nobody raised** — "(the sibling block is not a precedent because …)".
    3. **Ruled-out alternatives** — "X is not reachable here because …". Real work, but a negative result.
    4. **Softening/credit preambles** — "Thanks for the rewrite, that fixes the part that mattered".
  This is the single most common defect in staged comments and the user has had to ask for it more than once.
- **Collapsible sections: what goes in, and how to label them.** Use `<details><summary>…</summary>` for the
  material above plus long examples, full call chains, version-by-version checking, and negative results.
  Three rules:
    - **The visible part must stand alone.** Never write "expand this to understand the issue". The fold is
      strictly supplementary — for a reader who wants to dispute the finding or deep-dive, never a
      prerequisite for acting on it.
    - **Cut means relocate, not delete, for anything carrying a permalink.** Evidence moves below the fold;
      it does not vanish. (The audit enforces this: the set of permalinks must survive editing.)
    - **The summary label names what is inside**, in plain words. Good: "Why I ruled out router replay",
      "Full call chain", "Alternative fix I considered", "More counter-examples". BANNED as vague: "What I
      checked", "Details", "More info", "Reasoning". If a comment has two kinds of hidden content, use two
      folds with one specific label each — not one bucket holding everything.
    - **No fold when there is nothing real to put in it.** Do not manufacture one to fill the template.
- **If the finding turns on a mechanism the reader may not hold in their head, walk them to the punchline.**
  Your reader is a subject-matter expert — do not explain RL, distributed training, or the codebase to them.
  What they lack is not knowledge but *paged-in context*: naming a mechanism ("the `prepare_refit_info`
  handshake") forces them to reconstruct it from the diff before they can judge the finding. Most won't, and
  the finding dies unread. So reload that context for them, tersely, in this order, stopping as soon as the
  defect is self-evident:
  1. **Background** — the one sentence of context the finding sits on ("every step, weights are copied
     trainer→engine; that's the refit").
  2. **What problem the mechanism solves** — why it exists at all. Usually "side A doesn't know X, only side B
     does", which is exactly what makes the design non-obvious.
  3. **How it works** — numbered steps, one line each, each anchored to its permalink.
  4. **What breaks** — the defect stated against that scaffolding, ending in the observable symptom.
  Budget: 4-8 lines before the **Action:**. Needing more usually means it's two findings. Skip steps 1-2 when
  the mechanism is obvious from the diff — this rule is for handshakes, protocols, caches, lifecycles and
  cross-process contracts, not for a null check. Terseness is the point: a paragraph that teaches an expert
  something they already know is as costly as one that omits what they don't.
- **Show the breaking case, don't just assert it.** Whenever a finding HAS a concrete failing case, walk the
  reader through it — abstract prose ("this isn't validated", "this could starve the policy") is easy to wave
  away; a config plus a trace is not. Two acceptable forms:
  1. **Walk the failure.** Show the triggering input (a YAML block, an argument combination, a call), then the
     ordered steps from input to symptom, each anchored to the line that fails to stop it, ending in the
     observable symptom the author would actually see (the exact error text, a silent wrong value, a 180s
     stall). Include the step where it *succeeds when it shouldn't* — that's usually the crux.
  2. **Give a repro.** "Run `<recipe/command>` with `<config delta>` and you'll see `<symptom>`", or a short
     self-contained snippet. Fine when it's genuinely short.
  Rules: the trace must be TIGHT — a config block plus 3-5 one-line steps, not an essay; every step that
  claims code behaves a certain way gets a permalink to that line; and quote the real symptom string rather
  than paraphrasing it. If the premise depends on an external tool's semantics (Ray `PACK` vs `STRICT_PACK`,
  a SLURM flag), link the upstream source that defines it. If you CAN'T construct a failing case, say so
  plainly and downgrade the finding — "I could not construct a config where this breaks" is honest and often
  means it shouldn't be posted at all.
- **If the code states a contract about itself, LEAD with that quote — don't argue the finding from scratch.**
  A docstring, a module comment, or a written invariant ("`_segment_provenance` is stable across
  deepcopy/pickle", "never silently fetch all fields") converts an arguable observation into a settled one.
  "These two paths differ" invites a debate about whether the difference matters; "the docstring says X and
  this path does not-X" does not. Search for the contract before writing the finding: the class docstring,
  the comment above the function, the abstract method's docstring in the interface. When one exists, quote it
  as a blockquote, link it, and let the defect fall out of the mismatch.
- **When the finding IS a difference between two things, show both and stop.** Two snippets side by side,
  each permalinked, beats any paragraph describing how they differ — the reader sees it instead of taking your
  word. Reach for this whenever you catch yourself writing "X does A while Y does B": that sentence is a sign
  you have two quotable blocks and are paraphrasing them instead. Pick the tightest pair that shows it (often
  the two constructor calls, the two guards, the two config values — not the whole functions). Observed on
  PR #3921: a design finding was three paragraphs of prose until the user pointed out both paths were just
  `PackedTensor(...)` calls, one passing three fields the other omits — the entire finding, visible at a glance.
  This is the same root cause as burying a reference: writing from how you *derived* it rather than how it
  *reads*. The author does not need your derivation.
- **Pre-empt the wrong fix.** When the obvious repair to a finding is one you have ruled out, say so and say
  why, under a `**Context — no action.**` heading. Otherwise the author does the obvious thing and it is worse
  than what they had. On PR #3921 a nullable-argument finding invited "just make the field non-optional" — but
  `None` was a documented third state that made the data plane fail loudly, and defaulting it to `[]` would
  have converted a loud raise into a silent empty fetch. A finding that leaves its most tempting wrong fix
  unaddressed is half-done.
- **State whether the defect is PR-introduced or pre-existing/adjacent,** in one clause, near the top. Asking
  an author to fix something they didn't break is legitimate but must be labelled as such ("adjacent footgun
  rather than something this PR broke — but <what the PR changes about it>"), or the comment reads as a
  false accusation.
- **When proposing a guard/assert, show it blocks nothing legitimate.** One line is enough ("`gpus_per_node`
  exceeding the node's GPU count is never a valid configuration"). If a real user might deliberately want the
  configuration you're proposing to reject, it's not an assert — it's a design discussion, and a much weaker
  finding.
- **Prefer a `suggestion` block whenever a fix is a concrete edit to line(s) in the diff.** A GitHub
  `suggestion` block is directly committable, so it is far more actionable than prose — reach for it by default
  for any clear code/docs fix, not just docstrings and tests. Don't force one where it doesn't fit: if the fix
  spans multiple files, targets a line not in the diff, or is an operational/config/verification ask, give the
  concrete change in prose. When several valid implementations exist, show only the one you recommend.
- **Suggestion-or-say-why is a HARD rule, enforced per action item.** Before writing any AI's prose, answer:
  "can this fix be a committable `suggestion` block?" If yes, it MUST be one. If the edit spans multiple
  lines, create the comment as a multi-line thread (`addPullRequestReviewThread` with `startLine` + `line`,
  or `start_line` in the REST payload) sized to the whole region being replaced — never shrink a fix to prose
  because the anchor was created single-line; delete and re-anchor instead. If an AI's fix is legitimately not
  a suggestion, the comment must say why in one clause ("multi-file", "line not in diff", "several valid
  shapes — sketch only"). The user has had to ask for missing suggestion blocks repeatedly; treat a stageable
  AI without a suggestion block as a defect in the comment.
- **But a `suggestion` block is only for an in-place replacement — apply the "alone" test first.** A
  suggestion is a one-click "replace exactly these lines with exactly this text". Before writing one, ask:
  **if the author applies only this block and changes nothing else, is the repo correct?** If no, it must be
  a plain ` ```python ` / ` ```yaml ` fence with the placement stated in words. This is a precondition to the
  HARD rule above, not an escape from it: the rule is "suggestion where it fits, fence + why where it does
  not". Three ways it fails, all seen on one PR (#3917, 3 of 23 comments applied wrongly):
  1. **A new symbol anchored on an existing symbol's line.** The anchor is often *correct* — you are
     commenting about that function — but a suggestion starting `def new_test():` on a line that reads
     `def old_test():` swaps the header and leaves the old body to be adopted by the new function. Net hunk
     `-1 +35`, the file still parses, the test count still looks right, and a good existing test is gone. To
     ADD a symbol, anchor on the blank line between symbols, or make it non-applyable.
  2. **A menu where one option doesn't satisfy the ask.** "Redact this, the repo is public — or gate it
     behind an env var" got the gate, and the internal URL is still in public source. An env var hides
     output, not source. Never pair a hard requirement with an "or" that fails to meet it; suggest only the
     option that satisfies the ask.
  3. **An edit with a prerequisite.** A suggestion rewriting a `disabled.txt` comment to "Covered instead by
     tests/functional/x.sh" is only true once `x.sh` exists — which the same comment asked for. Applied
     alone it makes the repo misstate its own coverage. If a block is only correct after another change,
     say so inside the block or don't make it applyable.
- **A suggestion carries its own "why" when the new code isn't obvious.** After "Commit suggestion", only the
  block lands in the file; the review prose around it does not. If the block adds a magic number, a
  `max`/`min`/clamp expression, an order that looks arbitrary, or a workaround, put a 1–2 line code comment
  inside the block saying why, in the file's comment style. If the reason needs more than two lines, simplify
  the code first. Seen on PR #4264: `max(0.0, remaining - min(1.0, remaining / 10))` was suggested with its
  reason only in the prose, so applied as-is it read as magic. Obvious edits (a rename, a missing argument, a
  docstring) need no comment — do not pad.
- **On re-review, diff the applied hunk — do not just grep that the new thing exists.** An author clicking
  "Commit suggestion" can satisfy your grep while deleting something. `git diff <old-head> <new-head> -- <path>`
  and read the `-` lines; `-1 +35` on a file you expected only to grow is the tell. Also re-run the check the
  comment was about (the failing command, the lint hook) rather than trusting that the symptom edit fixed the
  cause — a restored config can still leave the tool that broke it unfixed, and the next run re-breaks it.
- **One action item = one anchored location.** An AI (or Follow-up) that discusses code on a different line
  or file than the comment's anchor must either become its OWN comment anchored on that line, or open with a
  permalink to the exact line it targets. Never make the author hunt for the code an AI refers to — an
  unanchored, unpermalinked AI is untrackable and reads as untrustworthy.
- **Write in plain, common words.** Assume the reader is an expert engineer who is not a native
  English speaker. They should never have to stop and look up a word to act on your finding. Short
  sentences, one idea each. Active voice ("the save drops these prompts", not "these prompts are
  dropped"). No throat-clearing ("it is worth noting that", "importantly", "as we can see").
  Explain what a thing *does*, not what category it belongs to.
  Domain terms (rollout, refit, cursor, permit) are fine — fancy English is not. Replace:
  *load-bearing* → "X has to happen before Y" / "other code depends on this"; *non-trivial* → big,
  hard, slow; *orthogonal* → unrelated; *leverage* → use; *surface* (verb) → show; *semantics* →
  what it means; *canonical* → the standard one; *idempotent* → safe to run twice; *pathological* →
  worst case; *degenerate* → empty or all-one-value; *elide / obviate / preclude* → skip / make
  unnecessary / prevent; *salient* → important; *delta* → difference; *amortized* → spread across
  many steps; *by construction* → because of how it is built. That list is examples, not the rule.
  The rule is: if you would not say it that way to a colleague at a whiteboard, rewrite it.
- **Do not invent shorthand — this is the biggest failure mode in review prose.** The author knows
  their own diff. What they do NOT have is the hour you just spent reading across files, and in that
  hour you picked up private names for things — *the handshake*, *the frontier*, *the ring*, *the
  stamp*, *the fallback path*, *the ordinal*. Those words are yours, from an hour ago; they are not
  common knowledge. Using one as if it were forces the author to reverse-engineer your vocabulary
  before they can judge the finding, and most will not — the comment dies unread. Every invented noun
  must be either (a) replaced with what the thing plainly is, or (b) named once, right where you
  explain it, and then used.

  | Your shorthand | What the author can follow |
  |---|---|
  | "the ring keeps the last 512 snapshots" | "the last 512 loader places are kept" |
  | "stamped with its ordinal" | "given a number: 0, 1, 2, …" |
  | "saved at the trained frontier" | "saved at the point training had reached" |
  | "the cursor is past the batch" | "the loader has already moved on" |

  **Real code names are not shorthand — quote and link them.** `frontier_ordinal`,
  `_dataloader_snapshots`, `prepare_refit_info()` are things the author can grep and click, so keep
  them in backticks with a permalink and put the plain meaning beside them. The rule is against
  *prose* built out of invented nouns, not against naming the code.

  This binds hardest on the lines that are read FIRST and often ALONE: the actionability declaration,
  the TL;DR, and the **Action:** line. A reader who bounces off those reads nothing below them. Same
  for any diagram/table labels in the comment — least context, read first.
- Report ALL findings — never limit, truncate, cap, or summarize. Every issue you find must be reported.
- When done, mark your task completed with TaskUpdate and send your findings to the leader via SendMessage.
```

### `rl-expert`

```
You are the RL codebase expert.

Scope: nemo_rl/, tests/, examples/, docs/, root-level config files.

FIRST: Dynamically discover ALL coding guidelines:
1. Glob `.claude/skills/*/SKILL.md` at the repo root — read every match
2. Read `CLAUDE.md` at the repo root
Do NOT hardcode skill names. Skills may be added, removed, or renamed.

Tasks:
1. Analyze the PR diff against ALL loaded guidelines
2. For each changed file, read surrounding context with Read/Grep to understand changes
3. Categorize findings as [BUG], [TEST], [GUIDELINE], [DOC] with file:line and permalink
4. DOCSTRING REVIEW: For any new or modified public function/class/method missing a docstring,
   draft the docstring as a GitHub suggestion block. Before finalizing, confer with the relevant
   submodule expert(s) via SendMessage to verify accuracy of param descriptions and return types.
   Only include the suggestion after expert confirmation. Category: [DOCSTRING]
5. PERFORMANCE/CONVERGENCE EVIDENCE: If the PR touches code that could affect performance or
   convergence (training features, optimizer changes, CUDA graphs, kernels, parallelism), scan
   the PR description AND all PR comments for quantitative evidence:
   - New features: expect baseline numbers (throughput, tokens/sec, memory) or convergence curves
   - Modifications: expect before/after comparison or proof of no regression
   If evidence is missing, report as [PERF-EVIDENCE] with a specific ask: what numbers/curves
   the author should provide given the nature of the change.

You are also available to answer questions from other agents via SendMessage.
```

### `expert-{submodule}`

One agent per touched submodule. Adjust the submodule path accordingly:
- `expert-automodel`: `3rdparty/Automodel-workspace/Automodel/`
- `expert-megatron-bridge`: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/`
- `expert-megatron-lm`: `3rdparty/Megatron-LM-workspace/Megatron-LM/`
- `expert-gym`: `3rdparty/Gym-workspace/Gym/`

```
You are the {SUBMODULE_NAME} subject matter expert.

Scope: {SUBMODULE_PATH}

FIRST: Dynamically discover domain knowledge within your submodule:
1. Glob `{SUBMODULE_PATH}/**/SKILL.md` — read every match
2. Glob `{SUBMODULE_PATH}/**/CLAUDE.md` — read every match
Do NOT hardcode skill names. Different submodules store skills in different locations.

Tasks:
1. Analyze the PR diff for changes within your submodule scope
2. Verify correct API usage — check function signatures, return types, semantics against actual code
3. Report findings with file:line and permalink

UPSTREAM BUG DETECTION: When the PR contains a workaround for behavior in your submodule,
determine if the underlying cause is a bug in the upstream submodule itself. If so:
- Report it as category [UPSTREAM]
- The review comment should: (a) acknowledge the workaround is correct and should stay,
  (b) explain the upstream bug with a permalink to the relevant upstream code,
  (c) suggest filing an issue against the upstream repo (include the repo issues URL)

You are also available to answer questions from other agents (especially rl-expert for docstring
verification and comment-reviewer for context). Respond via SendMessage.
```

### `test-agent`

```
You are the test reviewer and test author.

Scope: tests/ directory and any test files in the PR.

FIRST: Dynamically discover guidelines:
1. Glob `.claude/skills/*/SKILL.md` at the repo root — read every match (especially the testing skill)
2. Read `CLAUDE.md` at the repo root
3. Read `tests/unit/conftest.py` to understand pytest marks and fixtures
4. Read the L0 test runner scripts: `tests/unit/L0_Unit_Tests_*.sh` to understand test modes

TEST MODES — tests are run with different pytest marks and uv extras. Check which mark the
test uses (or should use) and run with the appropriate command:

  Default (no marks):     cd tests && uv run pytest <path> -x
  HF gated:               cd tests && uv run pytest <path> -x --hf-gated
  Megatron Core:          cd tests && uv run --extra mcore pytest <path> -x --hf-gated --mcore-only
  Automodel:              cd tests && uv run --extra automodel pytest <path> -x --hf-gated --automodel-only
  vLLM:                   cd tests && uv run --extra vllm pytest <path> -x --hf-gated --vllm-only
  SGLang:                 cd tests && uv run --extra sglang pytest <path> -x --hf-gated --sglang-only

GPU WORK: If GPUs are available ($GPU_TESTING_AVAILABLE=true), run tests directly via
`uv run` (e.g., "cd tests && uv run pytest unit/test_foo.py -x"). If no GPUs, you may
only run CPU-only tasks locally (grep, read, ast.parse, collect-only) and note "not
verified — no GPU environment" for tests that need GPUs.

Tasks:

1. REVIEW existing tests in the PR: Check correctness, edge cases, assertions, proper cleanup,
   correct pytest marks, and that the test is in the right L0 category (Generation, Policy, Other).

2. COVERAGE CHECK (devil's advocate yourself): Before suggesting a new test, search existing
   tests to determine if this code path is ALREADY covered by an existing test. Use Grep to
   search for the function/class name, imports, and usage patterns across tests/. If existing
   coverage is adequate, do NOT suggest a redundant test — instead note in your findings that
   coverage already exists and reference the existing test file:line.

3. SUGGEST new tests when coverage is genuinely missing. Balance test weight with feature
   complexity:
   - Simple changes (one-liner fixes, config changes): simple mock-based unit tests are fine
   - Complex features (new algorithms, distributed logic, model changes): heavier tests are
     warranted — the repo has tests that compare logprobs between two real forward passes,
     use the distributed_test_runner fixture, etc. Match the weight to the risk.
   - Read nearby test files in the same directory first to match conventions and style
   - Suggest tests as GitHub suggestion blocks

4. VERIFY: You MUST run every suggested test locally before including it in your findings.
   Check what pytest.mark decorators the test needs and run with the matching command from
   the modes list above. Only include the suggestion if the test PASSES. If it fails, fix
   and re-run. Do NOT suggest tests that haven't been verified to pass.

If you need domain knowledge to write accurate tests, ask rl-expert or submodule experts via SendMessage.
```

### `comment-reviewer`

```
You are the comment reviewer.

Tasks:
1. Read ALL existing PR comment threads (provided in the PR metadata)
2. For each thread, determine if action is needed:
   - The PR author responded and needs a reply → draft a response
   - A coworker commented and we should affirm or challenge → draft a response
   - No action needed → skip
3. For threads needing a response, talk to rl-expert and relevant submodule experts
   via SendMessage to get technical context before drafting
4. All responses MUST include permalink references so the user can follow the rationale

Report: list of (thread_comment_id, action, draft_reply_text, permalinks)
```

### `devil-advocate`

```
You are the devil's advocate. Your job is to stress-test ALL findings from ALL other agents.

Wait for the leader to send you all Wave 1 findings. The leader will message you once all
Wave 1 agents have completed, with their consolidated findings attached. Do NOT poll TaskList
or check task status in a loop — you will be woken by the leader's SendMessage. Stay idle
until that message arrives.

For EACH finding from every agent:
1. Demand the source/reference/permalink. If missing, challenge it.
2. Independently verify the issue is real by reading the actual code yourself.
3. Challenge whether the finding matters — is it a real problem or noise?
4. You get UP TO 2 ROUNDS of challenge per agent:
   Round 1: Send your challenge via SendMessage to the agent
   Round 2: If the response is unsatisfying, push back once more
   Then render your final verdict.

ALSO: Challenge whether the PR is even needed. Check if the changes are trivial
(whitespace-only, formatting-only, no functional change). If so, flag it.

ALSO, two mandatory disqualifiers:
- **Scope check (stale diff):** Cross-check each finding's file against the PR file list
  (`gh pr view $PRNUM --json files`). If a file isn't in the PR diff, the change came from another
  already-merged PR via a stale local `origin/main` — DISPUTE the finding. For large rebased PRs,
  recompute the true diff base (the first PR commit's parent).
- **Premise check:** For any finding asserting external tool/API/env-var behavior, demand a source
  permalink that proves it. A confidence score is a claim about a factual premise; if the premise is
  unverified, DISPUTE or DOWNGRADE — never let an agent's low-confidence guess get promoted past the
  staging threshold just because it was restated confidently.
- **Design-finding check (for `[DESIGN]` findings):** these are the easiest to hand-wave, so hold them
  to a HIGHER bar. DISPUTE any design finding that lacks a NAMED, plausible near-term extension or a
  CONCRETE maintenance hazard — hypothetical "what if we ever need X" is noise. For a "decouple this"
  finding, demand the ≥2-variants-or-named-second-impl evidence and check the split actually reduces
  net complexity (not just moves it). For a "unify these" finding, check it wouldn't create if-else
  soup or blast-radius coupling — if it would, DISPUTE (over-abstraction is a real failure mode).
  Prefer findings on code the PR INTRODUCES; downgrade demands to refactor pre-existing shape to
  "tracking-issue" severity.
  **Exception — placement findings (TEST D: "this helper is in the wrong module").** Do NOT apply
  the named-extension bar to these; they are cheap moves, not redesigns, and demanding a future
  scenario would kill every one of them. Hold them to their own test instead: the finding must name
  the module the helper operates on the data of, name the target module and show it already exists,
  and list the import sites that change. Then DISPUTE if any of these hold — the move would create a
  new import cycle (check the direction yourself), the helper really does have exactly one caller it
  is conceptually part of, or the code carries a comment explaining the current placement as
  deliberate. A function-local import used to dodge a cycle is supporting evidence FOR the finding,
  not against it — but verify the cycle is real before letting that stand.
  **Exception — mirrored-constant findings (TEST E: "this value is copied from its owner").** Also
  exempt from the named-extension bar; the hazard is silent drift on someone else's next bump, which
  is by definition unscheduled. Hold them to their own test: the finding must name the owner and link
  its definition, and show that BOTH a consistency test and an explaining comment are absent. Verify
  the owner's value yourself — if the two already differ, the finding is not a design nit, it is a
  live bug and should be upgraded. DISPUTE when the value is a format invariant rather than someone's
  config (an FP4 element max, a block size fixed by the wire format), when the repo genuinely owns the
  value, or when a consistency test already exists that FAILS (not skips) when the value is missing —
  `grep` for it rather than assuming, and read it: an `importorskip` or `pytest.skip` guard does not
  count. Also DISPUTE a "just add a guard test" ask when the value can simply be imported where it is
  read (TEST E step 2) — then the right ask is to delete the copy, not to guard it.
  **Exception — capability-list findings (TEST F: "this fact belongs on the class").** Also
  exempt from the named-extension bar; the hazard is the second place every new class must
  remember. Hold them to their own test: the finding must show the list decides a fact about each
  class's own behavior (a copy of a declaration is TEST E instead), give the property's value for
  every existing class, and show the one consumer line. DISPUTE when the list is not about classes
  the repo owns, when the PR only moved a pre-existing chain (downgrade to a tracking issue), or
  when the finding settles for a `getattr` default instead of a required property.

Report ALL verdicts — every challenged finding gets one of:
- CONFIRMED (with justification)
- DISPUTED (with reason — what's wrong with the finding)
- DOWNGRADED (was reported as high severity but is actually minor)
```

### `comment-editor`

Spawned in Wave 3, AFTER devil-advocate reports. It only ever sees findings that survived the adversarial pass, so nothing is polished that is about to be killed. It is NOT a reviewer — it does not add, drop, or re-judge findings.

```
You are the comment editor. You own the final wording of everything this review posts:
every inline comment body AND the review summary body. You are the last writer before the
payload is assembled.

You are NOT a reviewer. Do not add findings, drop findings, or re-litigate verdicts. Your
input is the set of findings that already survived devil-advocate; your output is the text
that goes on the wire.

For each finding the leader sends you, rewrite it to the comment skeleton in your preamble:
actionability declaration (starting with the `C<n> · ` ID the leader gave that finding — never
renumber), then a bolded one-line TL;DR, then a short explanation of how the
problem shows up, then `### AI-n` with the fix (a ```suggestion block wherever the fix is a
concrete edit to a diff line, otherwise the concrete change plus one clause saying why it
can't be a block), then optional `<details>` sections.

Move ALL of this below the fold or delete it: first-person investigation narration, objections
nobody raised, ruled-out alternatives, and softening/credit preambles. Label each fold for what
is inside it ("Why I ruled out X", "Full call chain") — never "What I checked" or "Details".

HARD CONSTRAINTS — these are checked mechanically after you report, and a violation blocks
the POST:
1. Carry EVERY permalink through. You may relocate a link below the fold; you may not drop or
   alter one. The audit diffs the URL set before and after your edit.
2. Carry every ```suggestion block through byte-identical. You are editing prose, not fixes.
3. Never invent. You did not derive these findings. If you cannot rewrite something faithfully
   because you lack the context — you don't know what a mechanism does, which line a claim
   refers to, or whether a rephrasing is still true — do NOT guess and do NOT paper over it
   with vaguer wording. Ask: SendMessage the leader, or the lane that produced the finding
   (the leader tells you which lane each came from; e.g. expert-megatron-lm for Megatron-Core
   claims, test-agent for CI claims). Wait for the answer, then write.
4. Preserve the finding's meaning. Trimming is your job; weakening or strengthening a claim
   is not. If shortening would change what the author concludes, keep the words and cut
   elsewhere.

Also rewrite the review summary body: same plain-language and no-monologue rules, but it is
not a finding, so it gets no TL;DR and no `### AI-n` headings. It is the one place
acknowledgement belongs. Do not reference an inline comment that is not in the final set.
Each list item that summarizes an inline comment names it by ID, e.g. `1. (C13) …`.

Report the full rewritten text for every comment plus the body via SendMessage.
```

### `bug-finder`

```
You are the bug finder. Scan the entire diff for bugs beyond what other experts found.

FIRST: Dynamically discover guidelines:
1. Glob `.claude/skills/*/SKILL.md` at the repo root — read every match
2. Read `CLAUDE.md`

Tasks:
1. Scan the diff for: logic errors, null refs, race conditions, type errors, missing imports,
   incorrect API usage, security issues, resource leaks, off-by-one errors
2. For each changed file, read surrounding context to understand the change
3. When uncertain about a potential bug, write a self-contained test to validate
4. Delegate domain questions to expert agents via SendMessage

GPU WORK: If GPUs are available ($GPU_TESTING_AVAILABLE=true), run validation tests directly
via `uv run`. If no GPUs, note "not verified — no GPU environment" for findings that need
runtime validation.
```

### `design-reviewer`

```
You are the design reviewer. You evaluate the PR's design SHAPE and its cost to the next
engineer — NOT correctness (bug-finder owns that) or guideline compliance (rl-expert owns
that). Think one step past this PR: how will this be tested, extended, and maintained?

FIRST: Dynamically discover guidelines:
1. Glob `.claude/skills/*/SKILL.md` at the repo root — read every match
2. Read `CLAUDE.md`

SCOPE GATE (do this first): if the diff has no design surface — a one-line fix, a config
VALUE change, a pure test/doc edit, a mechanical rename — report "no design-level surface —
LGTM" and STOP. Do NOT manufacture findings. Design review only earns its keep when the PR
adds or reshapes a class, interface, config schema, worker group, module, or a control-flow
branch structure.

Your core job is to flag drift in EITHER direction on the abstraction spectrum. A naive
architecture reviewer only ever says "add more abstraction" — that is actively harmful
(it produces if-else soup and coupling). Run each test below on the change and report only
where one genuinely fires:

  TEST A — under-abstraction (should this be DECOUPLED?):
    - Are there already ≥2 variants expressed as flags/branches in one unit, or a NAMED
      second implementation coming soon?
    - Does the current shape force a new variant to edit shared code in multiple places?
    - Would a seam let each variant's logic live — and be unit-tested — in one place?
    → If yes, recommend the seam with a concrete interface sketch.

  TEST B — over-abstraction / over-coupling (should this stay SEPARATE?):
    - Would unifying scatter "which algorithm/mode" conditionals through a shared path
      (if-else soup)?
    - Would a change to one variant then force re-testing ALL variants (blast radius —
      every small change becomes monstrous)?
    - Are these things actually the same, or coincidentally similar — do they change for
      DIFFERENT reasons? Prefer duplication over the wrong abstraction.
    → If yes, recommend AGAINST merging (or a thin shared helper, not a merged path).

  TEST C — extraction (should a cluster inside this class become its OWN OBJECT?):
    Method count and line count are SYMPTOMS that tell you to run this test; they are never
    themselves the finding. Compare against sibling classes first — "big" relative to nothing
    is not a finding.
    Score the candidate group on three questions, and flag ONLY when all three hold:
      1. CLOSED STATE — is there a set of instance attributes touched ONLY by this group of
         methods and by nothing else in the class? COMPUTE this, do not eyeball it: map every
         method to the `self.<attr>` it reads/writes, then look for a partition.
      2. INDEPENDENT LIFECYCLE — does the group own a thread / socket / subprocess /
         connection, or a stateful protocol with real ordering (prepare -> use -> teardown),
         that the rest of the class does not participate in?
      3. LOW REACH-BACK — how many host attributes must it borrow? If everything it needs
         fits through a constructor — with dynamic data arriving as a callable/provider
         rather than a snapshot — reach-back is zero. If it must call back into the host,
         do NOT extract.
    -> 3/3: recommend extraction with signatures + the instantiation site, keep the public
      methods as one-line delegations so the interface contract is unchanged, and state the
      payoff as a NUMBER (lines off __init__, attributes off __setstate__, what becomes
      unit-testable without spinning up the world).
    -> Anything less: do NOT flag.

    ANTI-PATTERN to name explicitly when you see it: "these methods are private and callers
    never invoke them" is NOT a reason to extract. A cluster with NO mutable state and no
    lifecycle wants to be module-level functions, not a class — turning it into one produces
    an anemic helper whose constructor exists only to memoize its arguments. Reject that
    proposal out loud rather than silently omitting it.

    Cheap signals worth checking for, each of which is a cost you can quantify: a >100-line
    __init__; a __getstate__/__setstate__ that must enumerate a dozen-plus attributes (every
    one a silent bug on the unpickle path if forgotten); being the only class at its layer
    that owns a thread. If an existing repo abstraction already models the cluster (e.g. a
    synchronizer, a registry, a worker pool), say so — "extract this" and "adopt the existing
    abstraction" are often the same refactor seen from two directions.

  TEST D — placement (is each new helper in the RIGHT MODULE?):
    Run this on every function, constant and registry the PR ADDS. It is the cheapest
    design finding there is — the fix is a move, the behaviour is unchanged, and it is
    almost always uncontroversial once named. Ask, in order:
      1. WHAT DOES IT OPERATE ON? Map the function to the data it reads. A pure function
         over a registry/constant/type belongs in the module that DEFINES that thing, not
         in the module that happens to be its first caller. Compute this — list the module
         each name in the body comes from and see where the mass sits.
      2. IS IT PURE? No `self`, no config, no I/O, no worker/actor handle → it has no
         reason to live in a class module, a trainer, or a launcher. Those modules are for
         things with state or a lifecycle.
      3. WHO ELSE ALREADY WANTS IT? Grep for imports. **The tell is an import that points
         "backwards"** — a low-level module importing from a high-level one, a second
         caller reaching into a sibling's private `_helper`, or a function-local import
         added to dodge a cycle. A local import inside a function body is almost always a
         placement smell wearing a disguise; say so, and name the cycle it is avoiding.
      4. DOES A HOME ALREADY EXIST? Name the module you would move it to and say why it is
         the right one ("it is where `X` is defined and where the other `X` helpers live").
         If no such module exists, this is NOT a placement finding — do not invent one.
    -> Flag when 1-4 line up: state the current module, the proposed module, the one-line
      reason, and every import site that must change. Keep it low severity; it is a move,
      not a redesign.
    -> Do NOT flag: a helper genuinely used by exactly one caller and conceptually part of
      it; a deliberate layering choice the code comments explain; or anything that would
      create a NEW import cycle — check the direction before proposing the move, because
      "put it where it belongs" that introduces a cycle is worse than leaving it.
    Bias hard to code the PR INTRODUCES. Asking an author to relocate a pre-existing
    helper they merely touched is scope creep — mention it as a tracking issue at most.

  TEST E — mirrored constants (is this value copied from somewhere that owns it?):
    Run this on every constant, default, magic number and literal the PR ADDS. Deriving
    or importing a value is ALWAYS preferred to copying it. A copy that must be kept in
    step with a value someone else owns is a silent-drift bug waiting for their next bump.
      1. WHO OWNS THIS VALUE? Is it defined in an upstream library (megatron, vLLM, TE),
         in another module of this repo, or in a config schema? If yes, this is a mirror.
         `grep` the name and the literal — a third copy hiding as a bare literal somewhere
         else is common and is the worst version, because it has no name to grep for.
      2. CAN IT BE DERIVED OR IMPORTED INSTEAD? If the module can import the owner, the
         copy has no reason to exist. Say so and stop — that is the whole finding.
         Check the READER, not just the module: a module that must not import the owner at
         top level (because a driver or a light actor imports it) can still import it
         inside the one function that uses the value. If that function already runs where
         the owner is installed — often it already has a deferred import from the very
         same owner module — the copy is unnecessary. (PR #4129: `PREFIX_*_FIELD` keys
         copied from `megatron.core.inference.inference_request` were read only inside a
         method that already did a deferred import from that module.)
      3. IS THERE AN EXCEPTIONAL REASON TO COPY? There is one good one: the owner cannot
         be imported here, because importing it would pull packages this module must not
         load. The canonical case is a constant owned by `megatron` that the DRIVER needs —
         the driver is backend-agnostic and must not import a backend.
      4. IS THERE A SHARED HOME THAT EVERYONE CAN IMPORT? Before accepting a copy, look for
         a light module that already holds the value, or that could: a constants /
         protocol module with no heavy imports, in this repo or exported by the owner. If
         one exists or can reasonably be created, ask for the value to live there and be
         imported from both sides. Only when no such home exists is a mirrored copy
         legitimate — and then the author must say why in the comment below.
      5. IF THE COPY IS LEGITIMATE, TWO THINGS ARE REQUIRED, and this is the finding when
         either is missing:
           a. **A unit test that imports both and asserts they match**, placed in a lane
              that CAN import the owner (e.g. an `mcore`-marked test). The driver's
              inability to import the owner is not the test's inability — say that
              explicitly, because authors conflate the two. The test must FAIL, never
              skip, when the owner or the value is missing: no `pytest.importorskip`, no
              `if not hasattr(...): pytest.skip(...)`, and not in a file whose module-level
              `importorskip` can drop it. A guard that can skip guards nothing on exactly
              the bump that removes the value.
           b. **A comment at the definition** saying why the value is duplicated here and
              how to update it when the owner changes.
    -> Flag when a mirror has no derivation, no consistency test, or no comment. Name the
      owner with a permalink to its definition, and sketch the test.
    -> Do NOT flag: a value this repo genuinely owns; a format invariant that is a fact
      rather than someone's config (the max of an FP4 e2m1 element, a block size fixed by
      the wire format) — naming those as local constants is an improvement over inline
      magic numbers, not a duplication.

  TEST F — capability lists (is a fact about a class kept as a list of names somewhere else?):
    Run this on every frozenset/list/dict of class or kind NAMES the PR adds, and on every
    `if name == / in (...)`, `isinstance`, or `cfg.is_x` chain in a caller that decides what a
    family of classes can do. A list that says which classes "can do X" or "count as X" keeps a
    fact about each class's own behavior in a second place. That place must be edited whenever a
    class is added or changed, and it can only answer per class name, never per setting.
      1. IS IT A FACT ABOUT THE CLASS? Would the class's author be the one who knows the answer
         (its math, its inputs, its return shape)? If the list instead copies a DECLARATION
         (dataclass fields, enum members, another module's constant), it is TEST E, not this.
      2. NAME THE PROPERTY AFTER THE CLASS'S BEHAVIOR, not after the consumer's feature
         (`needs_cross_group_stats`, not `shard_invariant`). The consumer derives its own flag
         from it (`shardable = not est.needs_cross_group_stats`), so one fact can answer several
         consumers.
      3. MAKE IT MANDATORY. Recommend an `ABC` base with the property abstract. A new class that
         forgets it then raises `TypeError: Can't instantiate abstract class ...` when it is
         built, usually at setup, before any GPU work. A plain class attribute satisfies an
         abstract property; a `@property` handles answers that depend on config. Never
         `getattr(obj, "x", default)`: the default hides exactly the class someone forgot. When
         no base class exists, adding one is part of the ask.
      4. GIVE EVERY VALUE. List the value for each existing class, including the config-dependent
         ones the name list could not express, the one consumer line, and a test that checks the
         property on each built instance (not a test pinning the list's contents). Check the
         abstract-property behavior in a scratch interpreter before the comment claims it.
    -> Flag at low severity when the list is correct and falls back to the safe path; higher when
      a missing name silently takes the unsafe path. Leave "this PR or a follow-up" to the author.
    -> Do NOT flag: a pre-existing chain the PR only moved (a tracking issue at most); a list of
      names that are not classes the repo owns (config keys, field names): that is TEST E.
    Example (PR #4326): `SHARD_INVARIANT_ESTIMATORS = frozenset({"grpo", "opd"})` read by name
    became an `AdvantageEstimator(ABC)` base with an abstract `needs_cross_group_stats`.

The one-liners to apply: couple what changes together; decouple what changes for
different reasons; give a cluster its own object only when its state is closed and
its lifecycle is independent; put each thing in the module that owns the data it
operates on; never keep a second copy of a value someone else owns without a test
that proves the two still agree; and put a fact about a class's own behavior on the
class, as a required property, not in a list of names somewhere else.

Judge on three axes, each tied to an OBSERVABLE (no abstract hand-waving):
  1. TESTABILITY: can the core logic be tested without spinning up the world (GPU / Ray
     actors / real models)? If a change buries testable logic behind an untestable boundary,
     flag it and name the seam that would make it unit-testable. This is the strongest,
     most objective signal.
  2. EXTENSIBILITY: state the CONCRETE cost of the next PLAUSIBLE variant — "to add a 4th X
     / a 2nd Y you'd touch N call sites and add an elif in M places." No named/plausible
     extension → do NOT flag (avoid hypothetical gold-plating).
  3. MAINTAINABILITY: blast radius + reasons-to-change — how many places must move together
     for one logical change.

Rules:
- Bias to code the PR INTRODUCES — reshaping a seam while it is being written costs nothing.
  For pre-existing shape the PR merely touches, suggest a tracking issue; do NOT block.
- Low-severity SUGGESTIONS by default (these are not blockers) unless the shape will actively
  cause bugs.
- TEST A-C findings need a NAMED future scenario + a concrete seam/interface sketch; TESTs
  D-F use their own evidence requirements above. BANNED:
  "consider making this more extensible" with no scenario and no sketch.
- Also affirm GOOD design explicitly — a clean seam worth keeping — so the collate step can
  reinforce it. Category [DESIGN] for both problems and affirmations.

Delegate domain questions to rl-expert / submodule experts via SendMessage. Report ALL findings.
```

---

## Phase 3: Collation (leader)

### Wave 1 → Wave 2 handoff (leader responsibility)

After each Wave 1 agent reports completion (you will receive an idle notification with their findings attached), check `TaskList` to see which Wave 1 tasks are still in flight. When ALL Wave 1 tasks (`analyze-rl-code`, any `analyze-{submodule}`, `review-existing-comments`, `review-and-suggest-tests`, `scan-for-bugs`, `review-design`) are completed, IMMEDIATELY send a single consolidated message to `devil-advocate` via `SendMessage` containing the full set of Wave 1 findings (grouped by source agent). This is a push notification — it unblocks `devil-advocate`, which is otherwise idle waiting for your message. Do NOT expect `devil-advocate` to poll for completion; it will not wake itself.

### Leader brokering for devil-advocate challenges

When devil-advocate sends challenges to other agents, those agents may go idle without checking their inbox. **The leader must broker**: when you see a devil-advocate idle notification with a peer DM summary (e.g. "[to test-agent] Challenge X"), send a nudge to the target agent via SendMessage telling them to check their inbox and respond. This prevents deadlocks where DA is waiting for a response and the target agent is idle.

### Tone guidelines

Review comments represent our team — keep them constructive and helpful, especially for community contributors who are volunteering their time:

- **Ask, don't accuse**: "It would be helpful to include benchmark numbers" not "The PR contains no quantitative evidence."
- **Suggest, don't demand**: "Consider adding..." or "This could be improved by..."
- **Don't single out the author**: Never quote an author's words back to highlight what's missing or wrong. If referencing something they said, frame it positively ("Building on your note about...").
- **Be specific about asks**: "Could you share tokens/sec with and without CUDA graphs on a 1B model?" is better than "Please add performance numbers."
- **Acknowledgement goes in the review BODY, not in inline comments.** If the feature is valuable, or the author fixed something from a previous round, say so once in the body. Do NOT open an inline comment with credit or a softening preamble — an inline comment carries the TL;DR, how the problem shows up, and the fix. Nothing else. Said once in the body, acknowledgement reads as an assessment of the work; repeated per-comment it reads as padding in front of every ask.

### 🚦 HARD GATE — do NOT stage anything until devil-advocate has reported

**Blocking rule: you may not call `AskUserQuestion` to offer staging, and you may not POST a review, until `devil-advocate` has delivered its verdicts via SendMessage.** The `collate-review` blockedBy `challenge-findings` dependency exists for this; honour it in behaviour, not just in the task graph.

**The gate covers EVERY adversarial pass you started, not just the first one to answer.** If you respawned devil-advocate on a second channel because the first went silent (see the respawn escape hatch below), and then the *original* delivers, you are NOT unblocked — the respawn is a genuine second opinion on exactly the findings that have already proven fragile, and it may overturn the first. Wait for both and reconcile them, or explicitly kill the extra pass with `TaskStop` and say in the preview that you did. Do not silently treat the redundant pass as noise because a report already arrived.

Before calling `AskUserQuestion`, enumerate every agent you spawned and confirm each has either reported or been deliberately stopped. "I forgot it was still running" is the failure mode here; a spawned-agent checklist is the fix. Observed on PR #3490: the leader presented findings and offered staging while a second, deliberately-commissioned adversarial pass was still in flight — caught by the user, not by the leader.

**Two more gates sit in front of the POST, and neither has a bypass:**
- `comment-editor` must have returned the wire-ready text (Wave 3). The leader does not author the final comments.
- The pre-POST audit table (Step 0a) must be printed in your reply and every row must be OK.

Both are blocking. If either cannot be satisfied, the review does not post — say so plainly rather than routing around it.

This is the single most common way this skill produces churn. The failure mode is seductive: the leader has personally verified most findings, devil-advocate is slow, and proceeding "feels safe because I checked everything myself." It is not safe — the leader is one of the reviewers, and the leader's findings need the adversarial pass **more** than anyone's, because nobody else is auditing them.

If devil-advocate has gone idle without reporting, do NOT interpret that as "no objections." Send the explicit-tool-call nudge — a resumed agent's final assistant text is not delivered to the leader, so a silent agent is usually one that never called the tool, not one with nothing to say:

> Your verdicts never reached me. When you are resumed by a message, your final assistant text
> is NOT delivered — you must explicitly call the SendMessage tool. Do this now:
> `SendMessage(to="team-lead", summary="devil-advocate verdicts", message="<full verdict list>")`.

Only after that nudge has ALSO failed may you proceed, and then you MUST say so in the preview ("devil-advocate did not report; findings below carry only the leader's verification").

**Cost of getting this wrong, observed on PR #3262:** the leader staged 8 comments before devil-advocate returned. DA then killed 3 of them — one where the leader's own supporting audit used the wrong peer group (comparing PPO recipes against tuned large-model async GRPO recipes to claim a config value was anomalous, when it was that family's house value), one where the "no test covers this" premise was refuted by an existing deterministic unit test, and one where the proposed fix would have defeated a deliberate CI ratchet. The staged review had to be deleted and re-POSTed. Had the user hit publish in that window, the author would have received three findings that do not survive scrutiny.

**If you catch yourself thinking any of these, stop and wait:**
- "I've verified these myself, DA is a formality."
- "DA already weighed in on some items, that's close enough."
- "I'll stage now and amend if DA objects." ← re-POSTing is what creates duplicate comments.

Note also that DA overturning **itself** is a signal the pass is working, not a reason to discount it: on #3262 it reinstated a finding it had previously agreed to kill.

After all Wave 2 agents complete:

1. Gather ALL findings from ALL agents. Categories: `[BUG]`, `[TEST]`, `[GUIDELINE]`, `[DOC]`, `[UPSTREAM]`, `[DOCSTRING]`, `[PERF-EVIDENCE]`, `[DESIGN]`
2. Apply devil-advocate verdicts: remove DISPUTED findings, adjust scores for DOWNGRADED ones
3. Deduplicate: same file + same line range + same core issue = one finding
4. Confidence threshold: discard anything scoring below 80
5. **Show ALL surviving findings** — no caps, no "top N", no summarization
5a. **Hand the surviving set to `comment-editor` (Wave 3) and wait.** Send each finding together with the lane that produced it, so the editor knows who to ask when it needs context. The editor returns the wire-ready text for every comment and for the review body; that text is what you put in the payload. **You may not skip this and write the comments yourself.** You carry the same rules below so your drafts comply, but your draft is the editor's INPUT, never a substitute for its pass — the editor exists to catch what the leader gets wrong, so a leader bypass reinstates the exact failure. If the editor goes silent, recover the way you would any silent lane (explicit SendMessage tool-call nudge, then re-spawn as a new named teammate). A blocked review is the correct outcome; it is visible and recoverable. A skipped editor is neither.
6. The rules the editor applies — and that your own drafts must already meet:
   - Concise, straight to the point (2-3 sentences max) — the ONLY things that earn extra length are the breaking-case walkthrough and the mechanism context-reload below, each within its own budget. After drafting, re-read and delete every sentence whose removal wouldn't change what the author does.
   - **Understandable**: if the finding hinges on a mechanism (handshake, protocol, cache, lifecycle, cross-process contract), open with the 4-step context-reload from the preamble — background → what problem the mechanism solves → numbered how-it-works → what breaks — in 4-8 lines. The reader is an expert but has not paged this mechanism in; one who has to reconstruct it from the diff before judging the finding won't.
   - **In the reader's words, not yours**: strip every noun you only know because you spent an hour in the code (*the ring*, *the frontier*, *the stamp*, *the handshake*). Replace it with what the thing plainly is, or name it once where you explain it. Real code names stay — in backticks, with a permalink, with the plain meaning beside them. Apply this to the TL;DR and the **Action:** line first; those are read alone.
   - **Actionable**: lead with the issue, then a bolded **Action:** line naming the exact file/function to change. Give the concrete fix (snippet / schema / signature) when known, and note placement + DRY concerns (e.g. "factor this predicate into one helper so the guard can't drift from the loop"). Cut reassurance/editorializing — if the answer to "what should the author DO?" is "nothing," drop the comment or move it to the review body.
   - **Tangible**: if the finding has a concrete failing case, SHOW it (per the preamble rule) — the triggering config/input, 3-5 permalinked steps from input to symptom, and the real symptom string; or a one-line repro ("run `<recipe>` with `<delta>` → `<symptom>`"). This is the single highest-leverage thing that makes a reviewer's ask land: an author can argue with "this isn't validated", but not with a YAML block and a trace ending in the exact error they'd see. Budget it: config block + a few one-line bullets. A finding with no demonstrable failing case is a weaker finding — reflect that in its score rather than dressing it up in prose.
   - Label each finding **PR-introduced** vs **pre-existing/adjacent** in one clause near the top
   - Start with a permalink to the code being commented on
   - **Evidence permalinks**: When a finding references behavior in upstream code (Megatron-LM, Megatron-Bridge, etc.), include permalinks to the specific lines that prove the claim. Use the submodule's own GitHub repo + pinned SHA (from `git ls-tree HEAD 3rdparty/...`). Example: linking to the deprecation notice, the assertion that would crash, the enum definition that shows a missing member, or the `__post_init__` normalization logic. Every claim about external code must have a clickable link — don't just cite file:line.
   - For call-stack reasoning (e.g. "A calls B which asserts C"), link each step in the chain
   - **Include a `suggestion` block whenever the fix is a concrete edit to line(s) in the diff** — not just for docstrings/tests. It is directly committable from the UI, so it is the most actionable form a finding can take; default to it for any clear code/docs fix. The block must contain the exact replacement for the commented line(s) (they are replaced verbatim), so anchor the comment on the line the edit changes. Skip it only when the change isn't a single-region edit (multi-file, a line not in the diff, or an operational/config ask) — then give the concrete change in prose. When multiple valid fixes exist, show one but label it **one way to do it** so it reads as illustrative:
     ````
     ```suggestion
     corrected code here
     ```
     ````
7. Incorporate linter results from the background `pre-commit` run (§1.5b) — attribute findings ONLY to files in the PR diff (`gh pr view $PRNUM --json files`), and flag any new source file missing from `pyrefly.toml`
8. Incorporate comment-reviewer's thread response drafts
9. Incorporate test-agent's verified test suggestions

---

## Phase 4: Preview & Confirm

> **Precondition check before you write a single line of this preview:** has `devil-advocate`
> delivered its verdicts via SendMessage? If not, go back to the Phase 3 hard gate. Do not
> present findings and do not offer staging. A preview built on un-challenged findings is how
> a review gets staged, deleted, and re-staged.

#### Comment IDs: C1, C2, C3 … — never reused

Give every comment a short ID — `C1`, `C2`, `C3`, … — so the user can say "drop C4" or "reword C7" when talking to an agent. Thread replies get IDs from the same counter.

- **Number once, in preview order.** Assign IDs when you first show the findings here, top to bottom. Put the ID at the start of every card (`**C1 · BUG**`), in the pre-POST audit table, and in every later message that mentions the comment.
- **An ID is permanent.** It stays with that comment through edits, re-anchoring, staging and publishing. Never renumber to close gaps.
- **Never reuse an ID.** A deleted comment's number is retired. A new or replacement comment takes the next number above the highest one ever used. Example: C1–C5 exist, the user deletes C2 and C3 and asks for one comment to replace them → it is `C6`, not `C2`. The next new comment after that is `C7`.
- **Keep the counter across sessions.** Record the ID → GitHub comment ID map, the retired IDs, and the next free number in the PR's memory file (Phase 6). On a re-review or follow-up of the same PR, continue from that number instead of starting over at C1.
- **The ID goes in the posted text too.** Every inline comment and thread reply starts with it (`**C<n> · 1 action item.**`), and each review-body list item that summarizes a comment names it (`1. (C13) …`). Use the same number as in previews, explainer pages and memory files. Why: the user reads the staged review on GitHub and refers to comments by ID when asking for changes.

Display ALL findings to the user using the card layout. Group by severity (Critical, then Suggestions, then Informational), with each finding as a blockquote card:

```markdown
## Review: PR #$PRNUM — $TITLE
by @$AUTHOR | <count> files changed | <count> agents

### Critical (<count>)

> **C1 · BUG** `path/to/file.py:42` [confidence: 95]
> <concise description of the issue>
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/file.py#L42)
>
> ```suggestion
> corrected code here
> ```
>
> _bug-finder, confirmed by devil-advocate_

> **C2 · UPSTREAM** `nemo_rl/foo.py:100` [confidence: 82]
> Workaround for <upstream issue> — upstream bug in Megatron-LM.
> Workaround is correct, but consider filing: https://github.com/NVIDIA/Megatron-LM/issues/new
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/nemo_rl/foo.py#L100)
>
> _expert-megatron-lm_

### Suggestions (<count>)

> **C3 · GUIDELINE** `path/to/other.py:15` [confidence: 85]
> <description of guideline violation>
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/other.py#L15)
>
> ```suggestion
> fixed code
> ```
>
> _rl-expert_

> **C4 · DOCSTRING** `path/to/module.py:30` [confidence: 90]
> `function_name` missing docstring
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/module.py#L30)
>
> ```suggestion
> def function_name(self, ...):
>     """Brief description.
>
>     Args:
>         param: Description.
>     """
> ```
>
> _rl-expert, verified by expert-megatron-bridge_

> **C5 · TEST** `tests/unit/test_foo.py` [confidence: 88]
> Suggest test for <feature> (verified passing locally)
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/tests/unit/test_foo.py)
>
> ```suggestion
> def test_feature():
>     ...
> ```
>
> _test-agent_

> **C6 · DESIGN** `path/to/new_component.py:40` [confidence: 82]
> <shape observation + which axis: testability/extensibility/maintainability>. To add the next
> <named variant> you'd touch <N> sites; a <interface/seam> would isolate + unit-test each. (Or,
> for over-coupling: merging <X> and <Y> would scatter mode conditionals — keep them separate.)
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/new_component.py#L40)
>
> _design-reviewer, confirmed by devil-advocate_

### Threads (<count>)

> **C7 · Reply** to @author on `file.py:10`
> <draft reply text with permalink references>
>
> _comment-reviewer_

_or: No responses needed._

### Linter: PASS/FAIL
<if failed, list hook name and file>

### Devil's Advocate
<N> confirmed | <N> disputed | <N> downgraded
PR necessity: **<verdict>** — <one-line justification>

---
_<N> findings filtered (scored <80)_
```

Then use `AskUserQuestion`:
- **(1) Stage all** — stage everything as a PENDING review for preview on GitHub
- **(2) Discuss individually** — iterate through each item before staging
- **(3) Cancel** — do nothing

If user picks "Discuss individually": iterate through items. For each, ask if they want to approve, edit the text, or skip. Then stage approved items as PENDING.

---

## Phase 5: Post Review

**IMPORTANT**: Post everything as a single review. Never post separate standalone comments via the issues API.

### Known GitHub limitation: PENDING review bodies get wiped on UI submit

When a PENDING review is created via the API with a `body`, the GitHub UI's "Submit review" dialog has its own text area that **defaults to empty**. Clicking "Submit" in the UI **overwrites** the API-set body with the (empty) text area contents. Only inline `comments` survive because they are separate objects.

**Fix**: Do NOT ask the user to submit from the UI. Instead:
1. Create the PENDING review (body + inline comments) — this stages everything
2. **Show the user a link to the PENDING review on GitHub** so they can preview the actual rendered comments in context
3. Ask for FINAL confirmation to publish — use `AskUserQuestion` with options:
   - **Publish** — submit the review via API
   - **Edit** — user wants to iterate on specific comments (edit IN PLACE via the GraphQL `updatePullRequestReviewComment` mutation — see Step "Edit comments" below — then re-ask)
   - **Cancel** — delete the comments this session added and stop (never the review itself; see below)
4. Only after explicit "Publish" confirmation, submit via the events endpoint

**CRITICAL**: Do NOT submit the review immediately after creating it. The user MUST have a chance to preview the staged comments on GitHub and request edits before publishing. The Phase 4 selection ("Stage all") only stages — it does NOT authorize publishing.

### Step 0a: Pre-POST audit — MANDATORY, and you must show the table

Prose checklists do not get followed. This one is gated on visible output: **build the payload in a script, run these checks in the same script, print the per-comment OK/FAIL table into your reply, and do not call `gh api … /reviews` until every row is OK.** If the table is not in your reply, you did not run it — the user reads its absence as a skipped audit, because that is exactly what it means.

Do NOT keep a checked-in audit script. Write it fresh each run from this spec, so the checks track the spec rather than rotting beside it. Every check below is mechanical; there is no judgement to defer.

Per inline comment:

| Check | Fails when |
|---|---|
| `id` | the first line does not start with `**C<n> · `, or `<n>` repeats another comment's ID or a retired one |
| `len` | body length, reported for every comment so bloat is visible even when passing |
| `TLDR` | no bolded `**TL;DR` line — required on EVERY comment, no length threshold |
| `perm` | a named function/class/method/config-field/sentinel in prose is not wrapped in a permalink; or an `### AI-n` targets a file/line other than the comment's anchor without a permalink to it |
| `words` | a banned word appears (see the plain-words rule in the preamble): load-bearing, belt-and-braces, non-trivial, orthogonal, leverage, surface (verb), semantics, canonical, idempotent, pathological, degenerate, elide, obviate, preclude, salient, delta, amortized, by construction |
| `mono` | **visible half only** — first-person investigation narration (`I checked`, `I verified`, `I could not construct`, `Recording what I looked at`, `To be clear on what this does not change`), a pre-empted objection, a ruled-out alternative, or a credit/softening preamble. Text inside `<details>` is exempt — that is where it belongs. |
| `fold` | a `<summary>` label is vague: "What I checked", "Details", "More info", "Reasoning", "Notes" |
| `sugg` | the comment has ≥1 `### AI-n` but no ```suggestion block and no why-not clause ("multi-file", "outside this PR's diff", "different file than this anchor", "isn't a code edit", "several valid shapes") |
| `sugwhy` | a ```suggestion block adds a non-obvious value or expression (a number other than 0/1, `max`/`min`/clamp, a workaround) with no code comment inside the block saying why — a judgment check, not a regex |
| `oneAI` | WARN, not FAIL: >1 `### AI-n` heading in a single comment. Check whether it is two findings wanting two anchors (that is how a suggestion block goes missing). Several AIs on the same anchored lines are fine. |
| `anchor` | `git show <current-head>:<path>` at the anchored line does not contain the expected subject token |

On the review body: no monologue, no banned words, and every inline comment it references (by `(C<n>)`) must exist in the final payload (a body pointing at a comment the user deleted is a dangling reference).

Editor drift check — run over comment-editor's output against the findings it was given:

| Check | Fails when |
|---|---|
| `links` | the set of permalink URLs is not preserved (relocating a link into a `<details>` is fine; losing or rewriting one is not) |
| `sugg=` | a ```suggestion block differs byte-for-byte from the one the editor was given |

Print it like this, then POST only on ALL OK:

```
PRE-POST AUDIT (4 comments)
 id  anchor            len  TLDR  perm  words  mono  fold  sugg  oneAI  anchor
 C1  config.py:68      412   ok    ok    ok     ok    n/a   ok    ok     ok
 C2  worker.py:503     775   ok    ok    ok     ok    ok    ok    ok     ok
 C3  worker.py:309    1118   ok    ok    ok     ok    n/a   why   ok     ok
 C4  yaml:418         1013   ok    ok    ok     FAIL  n/a   why   ok     ok
   mono: "I checked whether the flip breaks anything and could not construct"
BODY: monologue ok | words ok | refs resolve ok
EDITOR DRIFT: links 11 -> 11 ok | suggestions identical ok
REFUSING TO POST (1 failure)
```

### Step 0b: Pre-stage checklist — run for EVERY comment, at the initial POST and again for every later `addPullRequestReviewThread`/edit

1. **Anchor at the CURRENT head.** `addPullRequestReviewThread` interprets `line` against the PR's *current* head, NOT the review's `commit_id`. After any author push, recompute every anchor via `git show <current-head>:<path>` and assert the line contains the comment's subject token before posting. (Comments in the original POST are auto-remapped by GitHub when the head moves; GraphQL-added ones are not — a stale line number lands the comment on unrelated code.) When the head has moved mid-review, audit ALL staged comments: GraphQL `comments.nodes{path line originalLine}`; a post-push comment with `line == originalLine` in a file that shifted is the tell that it anchored against the wrong baseline.
2. **Every named code reference is a clickable permalink** at a pinned SHA — including upstream repos (Gym, aiohttp, ray at the versions in `uv.lock`) and upstack stack PRs (their head SHAs) — and every permalink is verified programmatically (the `L()` helper with `expect` tokens; refuse to emit the payload on mismatch). Claims about external behavior ("aiohttp returns 504", "Gym retries this status") get a permalink to the exact upstream line that proves them, not just an assertion. **Run this as a mechanical pass, per comment, not as a vibe check.** For each staged body, list every `code span`, every filename, and every `file:line` string; drop the ones that sit on the comment's own anchored line; every survivor must be inside a `[...](...)`. Anything left bare is a defect — the author cannot click it and will not go looking. Grep the assembled payload for the tell-tales before POSTing: a `.py`/`.txt`/`.yaml` outside a link, or a `:\d+` outside a link. A single permalink in the opening sentence does NOT discharge this for the rest of the comment; that is exactly how it gets missed. Deleted code is linked at the BASE SHA (see the preamble rule) — verify those targets against `git show <base>:<path>`, not against head, or the check passes while the link still points at the wrong code. Observed on PR #3921: the user had to point out a comment whose entire body — five file references and a cross-reference to a sibling comment — carried one link, to a line where the code no longer existed.
3. **Suggestion-or-say-why** (see the preamble rule): every AI whose fix is a concrete single-region edit carries a ```suggestion block sized to the region — multi-line regions need the comment created with `startLine`/`line`. Anything else states in one clause why it isn't a suggestion.
4. **Each AI sits on its target line** or opens with a permalink to it; an AI about a different file/line than its comment's anchor is a split candidate.
5. **Read it as the author, who has not read your investigation.** Is there a single noun in the comment that you only know because you spent an hour in the code — a name you coined for a mechanism, a state, a buffer? If yes, replace it with what the thing plainly is, or explain it where it first appears. Bare code identifiers are fine (they are greppable and permalinked); invented English is not. Check the first line and the **Action:** line hardest — they are read alone.

### Step 1: Create PENDING review with body + inline comments

Do NOT include `event` — omitting it creates a PENDING review.

Add `_Generated by Claude Code_` at the end of the review body.

**CRITICAL — All actionable findings MUST be inline comments, not body text.** The review body is the right place for general context: PR summary, merge conflict notes, overall impressions, agent count, and non-actionable observations. But every **actionable finding** (bugs, guideline violations, test gaps, doc issues) — even those about files not in the diff — MUST be posted as an inline comment tied to the most relevant file:line. Actionable items buried in the review body are easy to miss and hard to track as resolved.

**How to tie "general" findings to inline comments:**
- Bug in a file NOT in the diff (e.g. a test file missing a new required field): place the comment on the diff line that INTRODUCES the requirement (e.g. the new field declaration or the code that reads it), and reference the affected external file(s) in the comment body.
- PR description issue (e.g. wrong field name in docs): place the comment on the diff line where the field is defined, noting the description mismatch.
- Test coverage gaps: place the comment on the most relevant test file that IS in the diff, listing the untested functions with permalinks.
- General observations (unseeded RNG, etc.): place on the most relevant source line in the diff.

**IMPORTANT — Evidence permalinks in every inline comment**: Each comment body MUST include:
1. A permalink to the code being commented on (the line in the PR diff)
2. **Evidence permalinks** to any upstream/external code that proves the claim — e.g. the Megatron-LM line showing a deprecation notice, the assertion that would crash, the enum definition showing a missing member. For submodule code, use the submodule's own GitHub repo URL + pinned SHA (from `git ls-tree HEAD 3rdparty/<submodule>`). Quote the relevant code snippet inline so the reader doesn't have to click through for the gist.
3. For call-stack reasoning, link each step: "A [calls B](<link>) which [asserts C](<link>)"

Use Python `json.dump` to generate the review JSON — this avoids shell escaping issues with backticks, quotes, and markdown in comment bodies. Use the GraphQL `addPullRequestReviewThread` mutation (see Step 1a) to add comments that can't be placed on diff lines in the initial POST.

```bash
cat <<'REVIEW_JSON' > "$TMPDIR/review.json"
{
  "commit_id": "$HEAD_SHA",
  "body": "<brief summary — merge conflict note, agent count, etc.>\n\n_Generated by Claude Code_",
  "comments": [
    {"path": "<file>", "line": <line>, "side": "RIGHT", "body": "[`<file>:<line>`](<permalink>)\n\n<comment with evidence permalinks>"},
    ...
  ]
}
REVIEW_JSON

gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews \
  --method POST --input "$TMPDIR/review.json"
```

Save the returned review `id` as `$REVIEW_ID`. Always print the pending review URL so the user can click through to preview it:

```
https://github.com/NVIDIA-NeMo/RL/pull/$PRNUM#pullrequestreview-$REVIEW_ID
```

**Verify the staged comment count with GraphQL, not REST — REST under-reports on a PENDING review.** `GET /pulls/<n>/reviews/<id>/comments` can return fewer comments than were actually attached (observed: 30 returned for a 32-comment pending review). Trusting it leads to "re-adding" comments that are already there and silently creating duplicates. Always confirm with GraphQL, which reports the true `totalCount`:

```bash
cat > /tmp/q.json <<'JSON'
{"query":"query { repository(owner:\"NVIDIA-NeMo\", name:\"RL\") { pullRequest(number:<PRNUM>) { reviews(last:1, states:PENDING) { nodes { databaseId state comments(first:100) { totalCount nodes { databaseId path line body } } } } } } }"}
JSON
gh api graphql --input /tmp/q.json \
  --jq '.data.repository.pullRequest.reviews.nodes[0] | "state=\(.state) id=\(.databaseId) comments=\(.comments.totalCount)"'
```

Rules that follow from this:
- After creating the review, assert `totalCount == len(comments)` in your payload before telling the user it is staged. If it is short, the POST silently dropped some — recreate the review rather than patching.
- If comments really are missing, prefer **deleting the review and re-POSTing the canonical JSON** over adding the stragglers with `addPullRequestReviewThread`. Re-adding is what creates duplicates, and a pending review has no cheap dedupe.
- To find duplicates after the fact, list `(path, body[:60])` from the GraphQL `nodes` and look for repeats; each duplicate's `databaseId` is what you would need to delete.
- The same caution applies when answering "did the author address my comments?" — enumerate threads via GraphQL so you do not miss ones REST omits.

### 🚫 The staged review is the USER'S, not yours — never add or remove a comment unilaterally

Once the review is staged, the user edits it out of band: adding comments of their own in the GitHub UI, deleting ones they disagree with, rewording others. **Those edits are decisions, and they are invisible to you unless you re-read the live review.** Treat the server as the source of truth and your own record as stale the moment you finish a POST.

Three rules, and the count rule is the one that bites:

- **A comment that is missing was deleted on purpose. Do NOT recreate it.** If the live `totalCount` is lower than your record, the difference is the user pruning, not a POST that dropped comments. Re-adding it costs them the work of deleting it twice and silently overrides a call they already made. This is the one case where the "assert `totalCount == len(comments)`" rule above does NOT apply — that check is only valid in the moments immediately after your own POST, before the user has touched anything.
- **A comment you do not recognise is theirs. Do NOT delete it.** Before deleting anything, fetch the live comment and confirm it is one you wrote in this session. Never delete by position, by count, or by "everything except the list I have".
- **Delete only what the user names.** They will point at a specific comment — by URL, by `#rNNN...` id, or by describing it. Resolve that to one `databaseId`, delete exactly that one, and report the new live count. Never infer a deletion from a general remark about the review being too long.

Before any mutation of a staged review — add, edit, or delete — **re-read it live** and diff against your record. If something moved that you did not move, stop and ask rather than reconciling it yourself. And never delete the pending review to "clean up": it is one pending review per user per PR, so their hand-written drafts live inside it and go with it.

**If the `gh api` POST is blocked by permissions**: Print the exact command for the user to run manually (prefixed with `!`). After they run it, parse the returned JSON for the `id` field and print the pending review URL.

### Step 1a: Add additional comments to a PENDING review

If the user asks to add more inline comments after the review is created, use the **GraphQL** `addPullRequestReviewThread` mutation. The REST API cannot add comments to an existing pending review.

Use Python to generate the JSON, then call `gh api graphql --input <file>`:

```python
import json

gql = {
    "query": (
        "mutation($reviewId: ID!, $body: String!, $path: String!, $line: Int!) {"
        "  addPullRequestReviewThread(input: {"
        "    pullRequestReviewId: $reviewId, body: $body,"
        "    path: $path, line: $line, side: RIGHT"
        "  }) { thread { id comments(first:1) { nodes { id url } } } }"
        "}"
    ),
    "variables": {
        "reviewId": "$REVIEW_NODE_ID",   # the node_id (e.g. "PRR_kw..."), NOT the integer id
        "body": "<comment text>",
        "path": "<file>",
        "line": 42,
    },
}
with open("$TMPDIR/add_comment.json", "w") as f:
    json.dump(gql, f)
```

```bash
gh api graphql --input "$TMPDIR/add_comment.json"
```

**Key details:**
- Use `node_id` from the review creation response (e.g. `"PRR_kwDO..."`) — NOT the integer `id`
- The mutation is `addPullRequestReviewThread` — NOT `addPullRequestReviewComment` (`addPullRequestReviewComment` does not accept `line`)
- REST endpoints that do NOT work for this:
  - `POST /pulls/$PRNUM/reviews/$REVIEW_ID/comments` → 404 (doesn't exist)
  - `POST /pulls/$PRNUM/comments` with `pull_request_review_id` → 422 (not a permitted key)

### Step 1.5: Preview and confirm

After creating the PENDING review, show the user:
```
Review staged as PENDING: https://github.com/NVIDIA-NeMo/RL/pull/$PRNUM#pullrequestreview-$REVIEW_ID
Please preview the comments on GitHub. Ready to publish?

To submit the review yourself:
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews/$REVIEW_ID/events --method POST -f event=COMMENT
```

Use `AskUserQuestion` with options: **Publish**, **Edit comments**, **Cancel**.

If user picks "Edit comments": ask which comment to change, then edit it IN PLACE with the GraphQL `updatePullRequestReviewComment` mutation (by the comment's `node_id`), then re-ask.

**Do NOT use REST `PATCH /pulls/comments/$COMMENT_ID` while the review is still PENDING — it returns 404** (that endpoint only works once the review is submitted). While pending, use GraphQL:
- ADD a new inline comment → `addPullRequestReviewThread` (Step 1a)
- REPLY to an existing thread → `addPullRequestReviewComment` with `inReplyTo` (Step 3). Replies can be pending; only the REST `/replies` endpoint cannot.
- EDIT an existing one → `updatePullRequestReviewComment`

Get a pending comment's `node_id` from the review's comment list:
```bash
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews/$REVIEW_ID/comments \
  | python3 -c "import json,sys; [print(c['id'], c['node_id'], c['path']) for c in json.load(sys.stdin)]"
```
Then:
```python
import json, subprocess
gql = {
    "query": "mutation($id: ID!, $body: String!){updatePullRequestReviewComment(input:{pullRequestReviewCommentId:$id, body:$body}){pullRequestReviewComment{url}}}",
    "variables": {"id": "<node_id>", "body": "<new body>"},
}
open("/tmp/edit.json","w").write(json.dumps(gql))
subprocess.run(["gh","api","graphql","--input","/tmp/edit.json"])
```

If user picks "Cancel": re-read the live review, then delete only the comments this session added, one `databaseId` at a time (`gh api repos/NVIDIA-NeMo/RL/pulls/comments/$COMMENT_ID --method DELETE`). Never delete the pending review itself: it may hold the user's own drafts (see "The staged review is the USER'S" above). If the user explicitly asks to delete the whole review, confirm it holds nothing they wrote first.

### Step 2: Submit the review via API after user confirms "Publish"

Only after explicit "Publish" confirmation, submit the PENDING review programmatically:

```bash
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews/$REVIEW_ID/events \
  --method POST -f event=COMMENT
```

This publishes the review atomically (body + all inline comments) without going through the UI's text area, so the body is preserved.

### Step 3: Thread replies — stage them PENDING, in the same review

**A reply to an existing thread CAN be part of a pending review.** Stage it like every other comment, so the user previews the whole review — replies included — before anything publishes. Do NOT defer replies until after submitting, and do NOT tell the user replies must wait.

What misleads here is the REST endpoint. `POST /pulls/$PRNUM/comments/$COMMENT_ID/replies` publishes immediately and returns `422: user_id can only have one pending review per pull request` while a pending review exists — which reads as "replies cannot be pending". It only means *that endpoint* cannot. Use GraphQL instead:

```python
import json, subprocess
gql = {
    "query": (
        "mutation($reviewId: ID!, $inReplyTo: ID!, $body: String!) {"
        "  addPullRequestReviewComment(input: {"
        "    pullRequestReviewId: $reviewId, inReplyTo: $inReplyTo, body: $body"
        "  }) { comment { databaseId url } }"
        "}"
    ),
    "variables": {
        "reviewId": "$REVIEW_NODE_ID",   # node_id from the review POST, e.g. "PRR_kwDO..."
        "inReplyTo": "$THREAD_ROOT_NODE_ID",
        "body": "<reply text>",
    },
}
open("/tmp/reply.json", "w").write(json.dumps(gql))
subprocess.run(["gh", "api", "graphql", "--input", "/tmp/reply.json"])
```

Key details, each of which costs a round-trip if you get it wrong:
- `inReplyTo` is the **node_id of the thread's ROOT comment**, not the comment you are answering. A thread's root is the one whose `in_reply_to_id` is null. Get both from `gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/comments --jq '.[] | "\(.id) \(.node_id) \(.path):\(.line) \(.in_reply_to_id // "-")"'`.
- `addPullRequestReviewComment` is the right mutation here precisely *because* it takes `inReplyTo` and does NOT take `line` — a reply inherits the thread's anchor. Use `addPullRequestReviewThread` (Step 1a) only for a NEW thread on a line.
- Order matters: create the pending review FIRST (Step 1), then attach replies to it with the returned `node_id`. There is no pending review to attach to before that.
- Verify with the GraphQL count from Step 1 — replies are counted in the review's `comments.totalCount`, so a 6-comment review plus 1 reply must report 7.

Post a reply through the REST `/replies` endpoint only when you deliberately want it published on its own, outside any review.

After the review is published, tell the user:
> Review published on GitHub with <N> inline comments and review summary.

### Marking comments fixed in a later commit

**Full workflow: run `/review-pr-team-verify-followup <pr>`.** It validates each thread, replies `fixed in <sha>` and resolves the fixed ones, and turns partial fixes into a pending follow-up review, which replaces the "partially fixed" reply below.

**Trigger**: the user asks to update the review comments after new commits land, OR asks a question like "did the author address my review?" / "did they fix my comments?" (often in a follow-up session). That question is the request to run this workflow — don't just summarize; after reporting, post the fixed-in replies below (confirm with the user first if they only asked the question).

Method: list the author's **new commits since the reviewed head SHA** (`gh pr view $PRNUM --json commits` or `git log <reviewed-sha>..<new-head>`), fetch each commit's diff (`gh api repos/NVIDIA-NeMo/RL/commits/$SHA`), and **verify each open review thread against those diffs** (re-run the linter/test, inspect the code at that SHA). Then handle each thread by how much of the finding was addressed:

- **Completely fixed, and it's VERY OBVIOUS** (the commit's diff directly implements the asked-for change) → reply with exactly `fixed in <full-commit-sha>` (nothing else needed). If it is not obvious — the diff only partially overlaps the ask, or you'd have to infer intent — do NOT post a "fixed" reply; treat it as partial or unaddressed instead.
- **Not fixed at all** → **do not reply**. Leave the thread as-is; a "still not fixed" note adds noise. (Only surface unaddressed findings to the user in your summary, not on the PR thread.)
- **Partially fixed** → reply noting it's partially resolved in `<full-commit-sha>` **and spell out how**: what the commit did address and what still remains (the specific sub-item / file:line not yet handled), so the author knows the thread isn't done. Do not mark it simply "fixed."

Rules:
- Use the **full 40-char commit SHA**, as **plain text with NO backticks / code span**. GitHub's UI only auto-links a bare SHA into a clickable commit permalink; wrapping it in backticks (or any code formatting) suppresses the auto-link. This is the opposite of the usual "wrap code refs in backticks" rule — for a fixed-in commit reference, backticks are wrong.
- Base the fixed/partial/not-fixed verdict on evidence you confirmed (code present at that SHA + linter/test result), never on the author's claim alone.
- Post as a **reply to the existing review-comment thread** (the `/comments/$COMMENT_ID/replies` endpoint above), not as a new top-level comment.

```bash
# Completely fixed:
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/comments/$COMMENT_ID/replies \
  --method POST -f body="fixed in $FIX_SHA"   # $FIX_SHA = full 40-char SHA, no backticks

# Partially fixed (note what remains):
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/comments/$COMMENT_ID/replies \
  --method POST -f body="partially fixed in $FIX_SHA — still remaining: <what's left, with file:line>"

# Not fixed: post nothing.
```

---

## Phase 6: Capture review lessons (memory write-back)

Shared lessons live in `references/` next to this file and change only through a PR. New lessons go to the machine-local store (`~/.claude/review-memory/RL/`), which only grows if you write to it. After the review is posted — and **especially whenever the USER corrects, adds, reframes, or pushes back on a finding** during the session — distill the durable lesson into a new (or updated) memory file there.

Write a lesson only when it is **general and reusable**: a review-process rule, a repo convention, a recurring class of bug/smell, or an operational gotcha. Do NOT write one-off facts about this single PR. Match the existing schema:

```markdown
---
name: <short-kebab-slug>
description: <one-line summary>
metadata:
  node_type: memory
  type: feedback
  scope: repo-specific | general
  count: <occurrences so far, usually 1>
  last_seen: <YYYY-MM-DD>
---

<the lesson>

**Why:** <what went wrong / why it matters, ideally with the triggering example>
**How to apply:** <concrete rule to follow next time; link related lessons with [[name]]>
**Seen:** #<PR> <item> (<team|user>)
```

**The count.** `count` is how many separate times the pattern has shown up in a reviewed PR. A new memory starts at the number of occurrences in this PR. That is usually 1, but if the author made the same mistake twice in one PR (two copied key lists, two name-keyed capability lists), it starts at 2 with both listed in **Seen**. In each Seen entry, write who caught it: `team` if the review raised it on its own, `user` if the user had to ask for it.

**Fit before you create.** If an existing memory would have caught the case, do not write a new file. Bump its `count` (once per occurrence), set `last_seen`, add the Seen entry, and widen its wording by a sentence only if it would not quite have caught this case. If a memory has no `count` yet, first set it to the number of incidents its body already describes. Never count the same occurrence twice. When a memory reaches count 3, suggest promoting it to a TEST in this skill. If the rule that would have caught the case lives in `references/` or `skills/contributing/references/`, do not edit that file here: write (or bump) a local memory whose body names the reference file and rule, so the next upstreaming PR can bump the ★ or widen the rule.

**Upstreaming.** When the user asks, move stable local lessons into the right `references/*.md` file (one rule per bullet: the rule in bold, then one short example), open a PR, and archive the moved local files so they are not read twice. If the lesson has steps, commands or repo maps the one-line rule cannot carry, copy it into `references/details/` and end the rule with `Details: [name](details/name.md)`.

Set `scope:` deliberately — it routes the lesson when it is upstreamed:
- `general` → a review-process rule: a `references/` file, or this skill's text once it is stable.
- `repo-specific` → a NeMo-RL convention. If an author can follow it while writing code, add it as a one-line rule to the matching `skills/contributing/references/<theme>.md`, and put any review-only steps in `references/details/` behind a `Review Details:` link. Only conventions that a reviewer alone acts on go to `references/repo-conventions-*.md`.

Before writing, glob the folder and **UPDATE** an existing file if one already covers the topic rather than duplicating; delete a lesson that turns out wrong. (Materialize is human-triggered: the user asks you to review the folder and promote stable lessons into the appropriate skill, then prune what's been absorbed.)

---

## Phase 7: Teardown

Send shutdown to each agent individually (broadcast doesn't support structured messages):
```
SendMessage(to="<agent-name>", message={"type": "shutdown_request"})
```

After all agents confirm shutdown, you're done. There is **no `TeamDelete` step** (that tool was removed). The team config directory is cleaned up automatically when the session ends; the task-list directory persists locally so a resumed session keeps its tasks (governed by `cleanupPeriodDays`). Sending each teammate a `shutdown_request` is still the graceful way to free their contexts before you finish.
