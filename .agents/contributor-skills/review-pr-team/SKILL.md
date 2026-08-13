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

**Requires**: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` must be set — agent
teams (teammate spawning, shared task list, `SendMessage`) are an experimental
Claude Code feature.

**Sandbox**: Always pass `dangerouslyDisableSandbox: true` on every Bash
tool call in this skill. The sandbox's bubblewrap (bwrap) container fails to
initialize in worktree/container environments with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`, breaking ALL
commands — not just network-dependent ones like `gh`. Do NOT attempt any Bash
call in sandbox mode first; it will fail and waste a round-trip.

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

**Merge conflict check**: If `mergeable` is `"CONFLICTING"` or `mergeStateStatus` is `"DIRTY"`,
include a prominent note in the review asking the author to rebase their PR on `$BASE_BRANCH`
and resolve conflicts before further review. Add this as the first item in the review body.

**Read the PR description (`body`) carefully.** It contains the author's intent, motivation, and test plan. Also parse it for linked issues (patterns like `Fixes #123`, `Closes #456`, `Related: #789`). For each linked issue, fetch it:

```bash
gh issue view <ISSUE_NUM> --repo NVIDIA-NeMo/RL --json title,body,comments,labels
```

The PR description + linked issues + diff together form the full context. All agents should have access to this context so they understand *why* the change is being made, not just *what* changed.

### 1.2a Performance & convergence evidence check

If the PR touches code that could affect **performance or convergence** (e.g. new training
features, optimizer changes, CUDA graph support, kernel changes, parallelism config), check
the PR description and comments for quantitative evidence:

- **New feature**: Author should provide baseline numbers (throughput, tokens/sec, memory,
  convergence curves) demonstrating the benefit of the feature.
- **Existing feature modification**: Author should show before/after comparison or prove no
  regression (e.g. convergence curves from an A/B run).
- If evidence is missing, flag it as a `[PERF-EVIDENCE]` finding in the review. This is
  especially important for features that users will enable in production — they need to know
  the expected benefit and any trade-offs.

Record whether evidence was found: `$PERF_EVIDENCE_FOUND` (yes/no/not-applicable). Pass
this to the `rl-expert` agent in its prompt.

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

When the PR adds or modifies config that wraps Megatron-LM `TransformerConfig` fields, agents
MUST check how **Megatron-Bridge** sets those same fields. Megatron-Bridge is the canonical
integration layer and often shows the correct, non-deprecated API:

1. Get submodule SHAs for permalinks: `git ls-tree HEAD 3rdparty/Megatron-LM-workspace/Megatron-LM`
   and `git ls-tree HEAD 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
2. Search Megatron-Bridge for the config field name:
   `grep -rn <field_name> 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/`
3. Read the Megatron-Bridge code to understand the pattern (what field it sets on
   `recipe.model`, how it handles RNG, scope validation, etc.)
4. Read the Megatron-LM `TransformerConfig` field definition and docstring for the
   canonical documentation of valid values, defaults, and deprecation status
5. Pass the submodule SHAs and relevant Bridge/LM permalinks to all agents in their prompts

This ensures review comments can point authors to the established pattern rather than
just saying "this is wrong."

### 1.4 Read root context

- Read `CLAUDE.md` from repo root
- Read all `.claude/skills/*/SKILL.md` files (except `review-pr` and `review-pr-team`)
- Glob `~/.claude/review-memory/RL/*.md` — read EVERY match. These are durable, cross-session
  PR-review lessons for this repo (NVIDIA-NeMo/RL). Treat them as binding guidance and pass their
  content to every spawned agent in its prompt. (Scoped by repo name so lessons never bleed across
  repos.) This store is personal and machine-local — it may be empty or absent on first use; an
  empty glob is fine. New lessons are written back here in Phase 6.

### 1.5 Detect GPU availability & set up local testing

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
```

If GPUs are available, set `$GPU_TESTING_AVAILABLE = true`. All agents run tests locally
via `uv run` — no Docker container needed. On first run, `uv sync` resolves all deps
(torch, vLLM, CUDA) directly on the host (~5 min first time, instant after).

If no GPUs: set `$GPU_TESTING_AVAILABLE = false`. Agents should note "not verified — no
GPU environment" in their findings. Test-agent can still do CPU-only work (grep, read,
collect-only).

### 1.5b Kick off the repo linters (background)

Start the repo's own lint/format/type suite in the background early, so results are ready by collation:

```bash
uv run --group dev pre-commit run --all-files
```

Do NOT run `ruff`/`pyrefly` directly — `pre-commit` runs the repo's pinned hook versions and config, and
calling the tools directly can use the wrong versions/args and produce misleading results. When collating,
attribute findings ONLY to files in the PR diff (`gh pr view $PRNUM --json files`); `--all-files` also surfaces
pre-existing issues unrelated to this PR — don't report those. Also flag any NEW source file in the diff that
isn't added to `pyrefly.toml` `project-includes` (untracked files silently escape type-checking).

### 1.6 The team is implicit — no setup step

With `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` set, agent teams form automatically:
the current session is the team lead, and the team is created the moment you spawn the
first teammate via the `Agent` tool. There is **no `TeamCreate` step** (that tool was
removed; a session has exactly one implicit team). You do not name the team — refer to
the review as "the PR #$PRNUM review" in prompts for human readability only.

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

**Wave 3 task (blocked by Wave 2, leader does this):**

| Task | Owner | Description |
|------|-------|-------------|
| `collate-review` | leader | Merge all findings, apply verdicts, deduplicate, present to user. |

Set dependencies:
- `challenge-findings` blockedBy all Wave 1 task IDs
- `collate-review` blockedBy `challenge-findings`

### 2.2 Spawn agents (parallel)

Spawn all agents in a single message with multiple `Agent` tool calls. Give each agent a
unique `name` (so teammates can address each other via `SendMessage`). Do **not** pass
`team_name` — that input is ignored; spawning a teammate joins the session's single implicit
team automatically. You may set `subagent_type` to reuse a defined role; its `tools`/`model`
are honored and its body is appended to the teammate's prompt (`SendMessage` and the task
tools are always available regardless of any `tools` allowlist).

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
- **Prose permalinks**: When prose text mentions a specific function, class, method, config
  field, or sentinel value by name (e.g. `get_replay_topk()`, `_install_missing_route_fallback_patch`,
  `R3_MISSING_ROUTE_SENTINEL`), wrap the `<code>` span in an `<a href>` permalink to where it's
  defined. This applies to review comments, HTML explainers, and any other output. Readers
  should never have to grep for something mentioned by name — every named code reference
  should be one click away.
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
- **Compare new components to their nearest existing analog.** When the PR adds a component that
  parallels an existing one (a new worker group ↔ `lm_policy.py`, a new advantage estimator ↔ the
  existing estimators, a new config block ↔ `MasterConfig`), diff the new one against the established
  pattern and flag missing affordances: backend dispatch, override hooks (e.g. `resolve_policy_worker_cls`),
  guards/validation, type annotations, and return-shape consistency. "It works for the shipped recipe" is
  not enough if the new component silently diverges from its sibling's contract.
- **Fail-loud check.** For each new config option / branch, ask: what does the worst plausible misconfiguration
  do? If it silently produces wrong results (zeroed logprobs, an unrouted alias, an ignored backend, a config
  combination no warning covers), flag it and suggest a setup-time assert/raise.
- **Verify the premise before reporting.** When a finding hinges on how an external tool/API/env-var behaves
  (SLURM, Ray, Megatron, transformers, …), verify it against source/docs with a permalink FIRST. Never inflate
  your stated confidence to clear a threshold without checking the underlying fact — that launders speculation
  into an authoritative-looking comment.
- **Make every finding actionable.** Lead with the observation, then a bolded **Action:** line naming the exact
  file/function to change and (when known) the concrete snippet/schema. Drop pure editorializing ("looks
  intended", reassurance) that doesn't lead to an action.
- **Open every comment by declaring its actionability, before any prose.** The author must know what they
  are expected to DO before they read why. The first line is one of:
    - `**No action needed** — <one clause>.` for pure FYI/context.
    - `**1 action item.**` / `**2 action items, 1 follow-up.**` when there is work.
  Then make the body match that count: label each ask `### AI-1`, `### AI-2` …, label deferred work
  `### Follow-up` with its tracking reference, and put pure background under a final `**Context — no
  action.**` heading. If an ask should be done in THIS PR rather than deferred, say so in the opening line
  ("please fix in this PR") and do not also offer a tracked fallback — an escape hatch beside an ask is
  read as permission to skip it.
- **Any comment over ~1500 characters needs a one-line TL;DR**, bolded, immediately after the actionability
  declaration and before any prose. State the defect and its consequence, not the topic: "**TL;DR — in
  colocated mode the `mcore_generation_config` model keys are silently discarded, so `policy.megatron_cfg` is
  the only route left**", never "TL;DR — about how config keys are handled". Reviewers skim; a long comment
  with no TL;DR gets skipped entirely, which is strictly worse than a short comment that lands.
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
  concrete change in prose. When several valid implementations exist, still show one but label it explicitly as
  **one way to do it** (e.g. "one option — of several: …") so the author reads it as illustrative, not
  prescriptive.
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

Report ALL verdicts — every challenged finding gets one of:
- CONFIRMED (with justification)
- DISPUTED (with reason — what's wrong with the finding)
- DOWNGRADED (was reported as high severity but is actually minor)
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

The one-liners to apply: couple what changes together; decouple what changes for
different reasons; and give a cluster its own object only when its state is closed and
its lifecycle is independent.

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
- Compare the new component to its nearest existing analog (a new worker group ↔ lm_policy,
  a new estimator ↔ the existing estimators): does it match that pattern's extension points
  (backend dispatch, override hooks, guards), or diverge without reason?
- Bias to code the PR INTRODUCES — reshaping a seam while it is being written costs nothing.
  For pre-existing shape the PR merely touches, suggest a tracking issue; do NOT block.
- Low-severity SUGGESTIONS by default (these are not blockers) unless the shape will actively
  cause bugs.
- Every finding needs a NAMED future scenario + a concrete seam/interface sketch. BANNED:
  "consider making this more extensible" with no scenario and no sketch.
- Also affirm GOOD design explicitly — a clean seam worth keeping — so the collate step can
  reinforce it. Category [DESIGN] for both problems and affirmations.

Delegate domain questions to rl-expert / submodule experts via SendMessage. Report ALL findings.
```

---

## Phase 3: Collation (leader)

### Wave 1 → Wave 2 handoff (leader responsibility)

After each Wave 1 agent reports completion (you will receive an idle notification with their
findings attached), check `TaskList` to see which Wave 1 tasks are still in flight. When ALL
Wave 1 tasks (`analyze-rl-code`, any `analyze-{submodule}`, `review-existing-comments`,
`review-and-suggest-tests`, `scan-for-bugs`, `review-design`) are completed, IMMEDIATELY send a single
consolidated message to `devil-advocate` via `SendMessage` containing the full set of Wave 1
findings (grouped by source agent). This is a push notification — it unblocks
`devil-advocate`, which is otherwise idle waiting for your message. Do NOT expect
`devil-advocate` to poll for completion; it will not wake itself.

### Leader brokering for devil-advocate challenges

When devil-advocate sends challenges to other agents, those agents may go idle without
checking their inbox. **The leader must broker**: when you see a devil-advocate idle
notification with a peer DM summary (e.g. "[to test-agent] Challenge X"), send a nudge
to the target agent via SendMessage telling them to check their inbox and respond. This
prevents deadlocks where DA is waiting for a response and the target agent is idle.

### Tone guidelines

Review comments represent our team — keep them constructive and helpful, especially for
community contributors who are volunteering their time:

- **Ask, don't accuse**: "It would be helpful to include benchmark numbers" not "The PR
  contains no quantitative evidence."
- **Suggest, don't demand**: "Consider adding..." or "This could be improved by..."
- **Don't single out the author**: Never quote an author's words back to highlight what's
  missing or wrong. If referencing something they said, frame it positively ("Building on
  your note about...").
- **Acknowledge the work**: If the feature is valuable, say so before listing issues.
- **Be specific about asks**: "Could you share tokens/sec with and without CUDA graphs on
  a 1B model?" is better than "Please add performance numbers."

### 🚦 HARD GATE — do NOT stage anything until devil-advocate has reported

**Blocking rule: you may not call `AskUserQuestion` to offer staging, and you may not POST a
review, until `devil-advocate` has delivered its verdicts via SendMessage.** The
`collate-review` blockedBy `challenge-findings` dependency exists for this; honour it in
behaviour, not just in the task graph.

This is the single most common way this skill produces churn. The failure mode is seductive:
the leader has personally verified most findings, devil-advocate is slow, and proceeding
"feels safe because I checked everything myself." It is not safe — the leader is one of the
reviewers, and the leader's findings need the adversarial pass **more** than anyone's, because
nobody else is auditing them.

If devil-advocate has gone idle without reporting, do NOT interpret that as "no objections."
Send the explicit-tool-call nudge — a resumed agent's final assistant text is not delivered to
the leader, so a silent agent is usually one that never called the tool, not one with nothing to say:

> Your verdicts never reached me. When you are resumed by a message, your final assistant text
> is NOT delivered — you must explicitly call the SendMessage tool. Do this now:
> `SendMessage(to="team-lead", summary="devil-advocate verdicts", message="<full verdict list>")`.

Only after that nudge has ALSO failed may you proceed, and then you MUST say so in the preview
("devil-advocate did not report; findings below carry only the leader's verification").

**Cost of getting this wrong, observed on PR #3262:** the leader staged 8 comments before
devil-advocate returned. DA then killed 3 of them — one where the leader's own supporting audit
used the wrong peer group (comparing PPO recipes against tuned large-model async GRPO recipes to
claim a config value was anomalous, when it was that family's house value), one where the "no
test covers this" premise was refuted by an existing deterministic unit test, and one where the
proposed fix would have defeated a deliberate CI ratchet. The staged review had to be deleted and
re-POSTed. Had the user hit publish in that window, the author would have received three findings
that do not survive scrutiny.

**If you catch yourself thinking any of these, stop and wait:**
- "I've verified these myself, DA is a formality."
- "DA already weighed in on some items, that's close enough."
- "I'll stage now and amend if DA objects." ← re-POSTing is what creates duplicate comments.

Note also that DA overturning **itself** is a signal the pass is working, not a reason to
discount it: on #3262 it reinstated a finding it had previously agreed to kill.

After all Wave 2 agents complete:

1. Gather ALL findings from ALL agents. Categories: `[BUG]`, `[TEST]`, `[GUIDELINE]`, `[DOC]`, `[UPSTREAM]`, `[DOCSTRING]`, `[PERF-EVIDENCE]`, `[DESIGN]`
2. Apply devil-advocate verdicts: remove DISPUTED findings, adjust scores for DOWNGRADED ones
3. Deduplicate: same file + same line range + same core issue = one finding
4. Confidence threshold: discard anything scoring below 80
5. **Show ALL surviving findings** — no caps, no "top N", no summarization
6. For each finding, construct the review comment:
   - Concise, straight to the point (2-3 sentences max) — the ONLY things that earn extra length are the
     breaking-case walkthrough and the mechanism context-reload below, each within its own budget. After
     drafting, re-read and delete every sentence whose removal wouldn't change what the author does.
   - **Understandable**: if the finding hinges on a mechanism (handshake, protocol, cache, lifecycle,
     cross-process contract), open with the 4-step context-reload from the preamble — background → what problem
     the mechanism solves → numbered how-it-works → what breaks — in 4-8 lines. The reader is an expert but has
     not paged this mechanism in; one who has to reconstruct it from the diff before judging the finding won't.
   - **Actionable**: lead with the issue, then a bolded **Action:** line naming the exact file/function to
     change. Give the concrete fix (snippet / schema / signature) when known, and note placement + DRY
     concerns (e.g. "factor this predicate into one helper so the guard can't drift from the loop"). Cut
     reassurance/editorializing — if the answer to "what should the author DO?" is "nothing," drop the comment
     or move it to the review body.
   - **Tangible**: if the finding has a concrete failing case, SHOW it (per the preamble rule) — the triggering
     config/input, 3-5 permalinked steps from input to symptom, and the real symptom string; or a one-line
     repro ("run `<recipe>` with `<delta>` → `<symptom>`"). This is the single highest-leverage thing that
     makes a reviewer's ask land: an author can argue with "this isn't validated", but not with a YAML block
     and a trace ending in the exact error they'd see. Budget it: config block + a few one-line bullets. A
     finding with no demonstrable failing case is a weaker finding — reflect that in its score rather than
     dressing it up in prose.
   - Label each finding **PR-introduced** vs **pre-existing/adjacent** in one clause near the top
   - Start with a permalink to the code being commented on
   - **Evidence permalinks**: When a finding references behavior in upstream code (Megatron-LM,
     Megatron-Bridge, etc.), include permalinks to the specific lines that prove the claim.
     Use the submodule's own GitHub repo + pinned SHA (from `git ls-tree HEAD 3rdparty/...`).
     Example: linking to the deprecation notice, the assertion that would crash, the enum
     definition that shows a missing member, or the `__post_init__` normalization logic.
     Every claim about external code must have a clickable link — don't just cite file:line.
   - For call-stack reasoning (e.g. "A calls B which asserts C"), link each step in the chain
   - **Include a `suggestion` block whenever the fix is a concrete edit to line(s) in the diff** — not just for
     docstrings/tests. It is directly committable from the UI, so it is the most actionable form a finding can
     take; default to it for any clear code/docs fix. The block must contain the exact replacement for the
     commented line(s) (they are replaced verbatim), so anchor the comment on the line the edit changes. Skip it
     only when the change isn't a single-region edit (multi-file, a line not in the diff, or an operational/config
     ask) — then give the concrete change in prose. When multiple valid fixes exist, show one but label it **one
     way to do it** so it reads as illustrative:
     ````
     ```suggestion
     corrected code here
     ```
     ````
7. Incorporate linter results from the background `pre-commit` run (§1.5b) — attribute findings ONLY to files
   in the PR diff (`gh pr view $PRNUM --json files`), and flag any new source file missing from `pyrefly.toml`
8. Incorporate comment-reviewer's thread response drafts
9. Incorporate test-agent's verified test suggestions

---

## Phase 4: Preview & Confirm

> **Precondition check before you write a single line of this preview:** has `devil-advocate`
> delivered its verdicts via SendMessage? If not, go back to the Phase 3 hard gate. Do not
> present findings and do not offer staging. A preview built on un-challenged findings is how
> a review gets staged, deleted, and re-staged.

Display ALL findings to the user using the card layout. Group by severity (Critical, then
Suggestions, then Informational), with each finding as a blockquote card:

```markdown
## Review: PR #$PRNUM — $TITLE
by @$AUTHOR | <count> files changed | <count> agents

### Critical (<count>)

> **BUG** `path/to/file.py:42` [confidence: 95]
> <concise description of the issue>
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/file.py#L42)
>
> ```suggestion
> corrected code here
> ```
>
> _bug-finder, confirmed by devil-advocate_

> **UPSTREAM** `nemo_rl/foo.py:100` [confidence: 82]
> Workaround for <upstream issue> — upstream bug in Megatron-LM.
> Workaround is correct, but consider filing: https://github.com/NVIDIA/Megatron-LM/issues/new
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/nemo_rl/foo.py#L100)
>
> _expert-megatron-lm_

### Suggestions (<count>)

> **GUIDELINE** `path/to/other.py:15` [confidence: 85]
> <description of guideline violation>
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/other.py#L15)
>
> ```suggestion
> fixed code
> ```
>
> _rl-expert_

> **DOCSTRING** `path/to/module.py:30` [confidence: 90]
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

> **TEST** `tests/unit/test_foo.py` [confidence: 88]
> Suggest test for <feature> (verified passing locally)
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/tests/unit/test_foo.py)
>
> ```suggestion
> def test_feature():
>     ...
> ```
>
> _test-agent_

> **DESIGN** `path/to/new_component.py:40` [confidence: 82]
> <shape observation + which axis: testability/extensibility/maintainability>. To add the next
> <named variant> you'd touch <N> sites; a <interface/seam> would isolate + unit-test each. (Or,
> for over-coupling: merging <X> and <Y> would scatter mode conditionals — keep them separate.)
> [view on GitHub](https://github.com/NVIDIA-NeMo/RL/blob/$HEAD_SHA/path/to/new_component.py#L40)
>
> _design-reviewer, confirmed by devil-advocate_

### Threads (<count>)

> **Reply** to @author on `file.py:10`
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

**IMPORTANT**: Post everything as a single review. Never post separate standalone comments
via the issues API.

### Known GitHub limitation: PENDING review bodies get wiped on UI submit

When a PENDING review is created via the API with a `body`, the GitHub UI's "Submit review"
dialog has its own text area that **defaults to empty**. Clicking "Submit" in the UI
**overwrites** the API-set body with the (empty) text area contents. Only inline `comments`
survive because they are separate objects.

**Fix**: Do NOT ask the user to submit from the UI. Instead:
1. Create the PENDING review (body + inline comments) — this stages everything
2. **Show the user a link to the PENDING review on GitHub** so they can preview the
   actual rendered comments in context
3. Ask for FINAL confirmation to publish — use `AskUserQuestion` with options:
   - **Publish** — submit the review via API
   - **Edit** — user wants to iterate on specific comments (edit IN PLACE via the GraphQL
     `updatePullRequestReviewComment` mutation — see Step "Edit comments" below — then re-ask)
   - **Cancel** — delete the PENDING review and stop
4. Only after explicit "Publish" confirmation, submit via the events endpoint

**CRITICAL**: Do NOT submit the review immediately after creating it. The user MUST have
a chance to preview the staged comments on GitHub and request edits before publishing.
The Phase 4 selection ("Stage all") only stages — it does NOT authorize publishing.

### Step 1: Create PENDING review with body + inline comments

Do NOT include `event` — omitting it creates a PENDING review.

Add `_Generated by Claude Code_` at the end of the review body.

**CRITICAL — All actionable findings MUST be inline comments, not body text.** The review
body is the right place for general context: PR summary, merge conflict notes, overall
impressions, agent count, and non-actionable observations. But every **actionable finding**
(bugs, guideline violations, test gaps, doc issues) — even those about files not in the
diff — MUST be posted as an inline comment tied to the most relevant file:line. Actionable
items buried in the review body are easy to miss and hard to track as resolved.

**How to tie "general" findings to inline comments:**
- Bug in a file NOT in the diff (e.g. a test file missing a new required field): place the
  comment on the diff line that INTRODUCES the requirement (e.g. the new field declaration
  or the code that reads it), and reference the affected external file(s) in the comment body.
- PR description issue (e.g. wrong field name in docs): place the comment on the diff line
  where the field is defined, noting the description mismatch.
- Test coverage gaps: place the comment on the most relevant test file that IS in the diff,
  listing the untested functions with permalinks.
- General observations (unseeded RNG, etc.): place on the most relevant source line in the diff.

**IMPORTANT — Evidence permalinks in every inline comment**: Each comment body MUST include:
1. A permalink to the code being commented on (the line in the PR diff)
2. **Evidence permalinks** to any upstream/external code that proves the claim — e.g. the
   Megatron-LM line showing a deprecation notice, the assertion that would crash, the enum
   definition showing a missing member. For submodule code, use the submodule's own GitHub
   repo URL + pinned SHA (from `git ls-tree HEAD 3rdparty/<submodule>`). Quote the relevant
   code snippet inline so the reader doesn't have to click through for the gist.
3. For call-stack reasoning, link each step: "A [calls B](<link>) which [asserts C](<link>)"

Use Python `json.dump` to generate the review JSON — this avoids shell escaping issues with
backticks, quotes, and markdown in comment bodies. Use the GraphQL `addPullRequestReviewThread`
mutation (see Step 1a) to add comments that can't be placed on diff lines in the initial POST.

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

Save the returned review `id` as `$REVIEW_ID`. Always print the pending review
URL so the user can click through to preview it:

```
https://github.com/NVIDIA-NeMo/RL/pull/$PRNUM#pullrequestreview-$REVIEW_ID
```

**Verify the staged comment count with GraphQL, not REST — REST under-reports on a PENDING review.**
`GET /pulls/<n>/reviews/<id>/comments` can return fewer comments than were actually attached (observed:
30 returned for a 32-comment pending review). Trusting it leads to "re-adding" comments that are already
there and silently creating duplicates. Always confirm with GraphQL, which reports the true `totalCount`:

```bash
cat > /tmp/q.json <<'JSON'
{"query":"query { repository(owner:\"NVIDIA-NeMo\", name:\"RL\") { pullRequest(number:<PRNUM>) { reviews(last:1, states:PENDING) { nodes { databaseId state comments(first:100) { totalCount nodes { databaseId path line body } } } } } } }"}
JSON
gh api graphql --input /tmp/q.json \
  --jq '.data.repository.pullRequest.reviews.nodes[0] | "state=\(.state) id=\(.databaseId) comments=\(.comments.totalCount)"'
```

Rules that follow from this:
- After creating the review, assert `totalCount == len(comments)` in your payload before telling the user
  it is staged. If it is short, the POST silently dropped some — recreate the review rather than patching.
- If comments really are missing, prefer **deleting the review and re-POSTing the canonical JSON** over
  adding the stragglers with `addPullRequestReviewThread`. Re-adding is what creates duplicates, and a
  pending review has no cheap dedupe.
- To find duplicates after the fact, list `(path, body[:60])` from the GraphQL `nodes` and look for repeats;
  each duplicate's `databaseId` is what you would need to delete.
- The same caution applies when answering "did the author address my comments?" — enumerate threads via
  GraphQL so you do not miss ones REST omits.

**If the `gh api` POST is blocked by permissions**: Print the exact command for the
user to run manually (prefixed with `!`). After they run it, parse the returned JSON
for the `id` field and print the pending review URL.

### Step 1a: Add additional comments to a PENDING review

If the user asks to add more inline comments after the review is created, use the
**GraphQL** `addPullRequestReviewThread` mutation. The REST API cannot add comments
to an existing pending review.

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
- The mutation is `addPullRequestReviewThread` — NOT `addPullRequestReviewComment`
  (`addPullRequestReviewComment` does not accept `line`)
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

If user picks "Edit comments": ask which comment to change, then edit it IN PLACE with the GraphQL
`updatePullRequestReviewComment` mutation (by the comment's `node_id`), then re-ask.

**Do NOT use REST `PATCH /pulls/comments/$COMMENT_ID` while the review is still PENDING — it returns 404**
(that endpoint only works once the review is submitted). While pending, use GraphQL:
- ADD a new inline comment → `addPullRequestReviewThread` (Step 1a)
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

If user picks "Cancel": delete the PENDING review (`gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews/$REVIEW_ID --method DELETE`) and stop.

### Step 2: Submit the review via API after user confirms "Publish"

Only after explicit "Publish" confirmation, submit the PENDING review programmatically:

```bash
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/reviews/$REVIEW_ID/events \
  --method POST -f event=COMMENT
```

This publishes the review atomically (body + all inline comments) without going through
the UI's text area, so the body is preserved.

### Step 3: Post thread replies (after review is submitted)

Thread replies **cannot** be posted while a PENDING review exists (GitHub returns
`422: user_id can only have one pending review per pull request`). Post them after
the review is submitted in Step 2:

```bash
gh api repos/NVIDIA-NeMo/RL/pulls/$PRNUM/comments/$COMMENT_ID/replies \
  --method POST -f body="<reply text>"
```

After posting, tell the user:
> Review published on GitHub with <N> inline comments and review summary.

### Marking comments fixed in a later commit

**Trigger**: the user asks to update the review comments after new commits land, OR asks a
question like "did the author address my review?" / "did they fix my comments?" (often in a
follow-up session). That question is the request to run this workflow — don't just summarize;
after reporting, post the fixed-in replies below (confirm with the user first if they only asked
the question).

Method: list the author's **new commits since the reviewed head SHA**
(`gh pr view $PRNUM --json commits` or `git log <reviewed-sha>..<new-head>`), fetch each commit's
diff (`gh api repos/NVIDIA-NeMo/RL/commits/$SHA`), and **verify each open review thread against
those diffs** (re-run the linter/test, inspect the code at that SHA). Then handle each thread by
how much of the finding was addressed:

- **Completely fixed, and it's VERY OBVIOUS** (the commit's diff directly implements the asked-for
  change) → reply with exactly `fixed in <full-commit-sha>` (nothing else needed). If it is not
  obvious — the diff only partially overlaps the ask, or you'd have to infer intent — do NOT post
  a "fixed" reply; treat it as partial or unaddressed instead.
- **Not fixed at all** → **do not reply**. Leave the thread as-is; a "still not fixed" note adds
  noise. (Only surface unaddressed findings to the user in your summary, not on the PR thread.)
- **Partially fixed** → reply noting it's partially resolved in `<full-commit-sha>` **and spell out
  how**: what the commit did address and what still remains (the specific sub-item / file:line not
  yet handled), so the author knows the thread isn't done. Do not mark it simply "fixed."

Rules:
- Use the **full 40-char commit SHA**, as **plain text with NO backticks / code span**. GitHub's UI
  only auto-links a bare SHA into a clickable commit permalink; wrapping it in backticks (or any
  code formatting) suppresses the auto-link. This is the opposite of the usual "wrap code refs in
  backticks" rule — for a fixed-in commit reference, backticks are wrong.
- Base the fixed/partial/not-fixed verdict on evidence you confirmed (code present at that SHA +
  linter/test result), never on the author's claim alone.
- Post as a **reply to the existing review-comment thread** (the `/comments/$COMMENT_ID/replies`
  endpoint above), not as a new top-level comment.

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

The review-memory store (`~/.claude/review-memory/RL/`) only grows if you write to it. After the review is
posted — and **especially whenever the USER corrects, adds, reframes, or pushes back on a finding** during the
session — distill the durable lesson into a new (or updated) memory file there.

Write a lesson only when it is **general and reusable**: a review-process rule, a repo convention, a recurring
class of bug/smell, or an operational gotcha. Do NOT write one-off facts about this single PR. Match the
existing schema:

```markdown
---
name: <short-kebab-slug>
description: <one-line summary>
metadata:
  node_type: memory
  type: feedback
  scope: repo-specific | general
---

<the lesson>

**Why:** <what went wrong / why it matters, ideally with the triggering example>
**How to apply:** <concrete rule to follow next time; link related lessons with [[name]]>
```

Set `scope:` deliberately — it routes the lesson during the periodic "materialize" pass:
- `general` → a review-process rule that should graduate into THIS skill's text.
- `repo-specific` → a NeMo-RL convention that should graduate into the repo's contributor-skills
  (`linting-and-formatting`, `config-conventions`, `error-handling`, `review-pr`, …).

Before writing, glob the folder and **UPDATE** an existing file if one already covers the topic rather than
duplicating; delete a lesson that turns out wrong. (Materialize is human-triggered: the user asks you to
review the folder and promote stable lessons into the appropriate skill, then prune what's been absorbed.)

---

## Phase 7: Teardown

Send shutdown to each agent individually (broadcast doesn't support structured messages):
```
SendMessage(to="<agent-name>", message={"type": "shutdown_request"})
```

After all agents confirm shutdown, you're done. There is **no `TeamDelete` step** (that tool
was removed). The team config directory is cleaned up automatically when the session ends; the
task-list directory persists locally so a resumed session keeps its tasks (governed by
`cleanupPeriodDays`). Sending each teammate a `shutdown_request` is still the graceful way to
free their contexts before you finish.
