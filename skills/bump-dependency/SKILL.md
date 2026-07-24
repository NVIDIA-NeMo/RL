---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, vLLM, SGLang, Megatron-LM / Megatron-Bridge / Automodel / Gym submodules, etc.), regenerate the lockfile, open a PR, and drive it to green by attaching a watchdog to the `CICD NeMo RL` workflow and quarantining failing functional tests in `tests/test_suites/disabled.txt` until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml` or one of the `3rdparty/*` submodules and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump vLLM', 'bump SGLang', 'bump submodule', 'bump Megatron-LM', 'bump Megatron-Bridge', 'bump Automodel', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'run all tests for this bump'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in NeMo-RL. Optimised
for the case where TE, vLLM, SGLang, or one of the `3rdparty/*`
submodules moves forward — which often surfaces flakes in functional or
convergence tests that have to be quarantined before the PR can land.

The pipeline is always: **edit → relock → push → /ok to test → watchdog
→ quarantine on red → re-trigger → repeat until green**.

## When to reach for this skill

- Bumping a git-source pin in `pyproject.toml` (TE, vLLM, SGLang, ...).
- Bumping any of the `3rdparty/*` submodules — `Megatron-LM`,
  `Megatron-Bridge`, `Automodel`, `Gym`.
- Any change that touches `uv.lock` and needs the full L1 (functional) +
  L2 (convergence) matrix to prove out before merge.

For pure dep additions/removals without a CI loop, the
`build-and-dependency` skill is enough.

## Required context

Read first, then follow the steps below:

- @skills/contributing/SKILL.md — Conventional Commits PR title (bare,
  no `[area]` prefix), DCO sign-off.
- @skills/build-and-dependency/SKILL.md — `uv lock` mechanics, container
  choice (`nemo-rl:latest`).
- @skills/cicd/SKILL.md — `copy-pr-bot`, `/ok to test`, the `CI:L0|L1|L2`
  label (no label → `test_level=none` → quality-check stays red).
- @skills/testing/SKILL.md — recipe naming, the `tests/test_suites/`
  layout. The `disabled.txt` quarantine surface is documented here in
  Step 7 (it's not yet in the testing skill).

## Step 1 — Worktree and edit

Create a worktree off `main` per @skills/contributing/SKILL.md. **Before
any `uv lock`** for a submodule bump:

```bash
git submodule update --init --recursive
```

Edit the pin. Two common shapes:

**Git-source override** (TE, vLLM, ...) — edit the matching line in
`pyproject.toml`. There are usually **two places** to keep in sync (the
runtime `override-dependencies` block and the build-time helper list
near the bottom of the file). Search for the package name and update
*every* occurrence so they don't drift:

```bash
grep -n "TransformerEngine.git@" pyproject.toml
```

```toml
"transformer-engine[pytorch,core_cu12] @ git+https://github.com/NVIDIA/TransformerEngine.git@<new-ref>"
```

Use a **branch name** (`release_v2.15`) only when you want to track a
moving tip; use a full SHA for reproducibility. TE branches use
`release_vX.Y` (underscore), not `release/vX.Y`. Verify with
`git ls-remote https://github.com/NVIDIA/TransformerEngine.git`.

**Submodule bump** (e.g. `3rdparty/Megatron-LM-workspace/Megatron-LM`):

```bash
cd 3rdparty/<name>-workspace/<name>
git fetch origin && git checkout <new-ref>
cd -
git add 3rdparty/<name>-workspace/<name>
```

## Step 2 — Regenerate the lockfile

Run `uv lock` inside the project container per
@skills/build-and-dependency/SKILL.md. Then confirm only the intended
packages moved:

```bash
git diff --stat pyproject.toml uv.lock
git diff uv.lock | grep -E '^\+\+\+|^---|name = ' | head -50
```

If the diff carries changes you didn't ask for (transitive movements
you can't explain), stop and investigate before pushing.

## Step 3 — Commit and push

Sign-off + signed-commit + commit-title format per
@skills/contributing/SKILL.md and @skills/cicd/SKILL.md. **Bare
Conventional Commits — no `[area]` prefix** (NeMo-RL's
`semantic-pull-request` job rejects bracket-prefixed titles):

```bash
git add pyproject.toml uv.lock 3rdparty/   # 3rdparty/ only if a submodule moved
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin <branch-name>
```

## Step 4 — Open the PR

Title and labels per @skills/contributing/SKILL.md +
@skills/cicd/SKILL.md. The bump-specific requirement: **`CI:L2` is
mandatory** — it's the only way to expand the matrix to L1 (functional)
+ L2 (convergence). Without any `CI:*` label, `pre-flight` resolves
`test_level=none` and the quality-check job stays red.

The PR body template — durable record of the bump:

```markdown
<details><summary>Claude summary</summary>

## What
- Bump <package> to <ref>.
- Regenerate `uv.lock`.

## Lockfile delta
```
Updated <package> <old> -> <new>
```

## Test plan
- [ ] L1 CI green (functional)
- [ ] L2 CI green (convergence; required for any bump that can shift numerics)

## Quarantined tests (this bump)
_None yet — will be appended as flakes are identified during CI iteration._

</details>
```

To update the PR title or body later, use `gh api -X PATCH
"repos/NVIDIA-NeMo/RL/pulls/<N>" -F "body=@/tmp/pr-body.md"` — never
`gh pr edit`.

## Step 5 — Trigger CI on the exact SHA

Trigger mechanics + `pull-request/<N>` mirror branch live in
@skills/cicd/SKILL.md "How CI Is Triggered". For this loop the rule is
simple: **on every new SHA you push, post `/ok to test
$(git rev-parse HEAD)`**. Use the **full** SHA — the short form
silently fails to match.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **`CICD NeMo RL`** workflow only. Other workflows
(copyright check, semantic-pull-request, secrets scan, automodel
submodule check, wheel build) are noise here.

**Always attach a watchdog with the Monitor tool. Never schedule wakeups
or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to reason
  about.
- Natural early termination via `TaskStop` once the run is green.

### Watchdog script

Save to `/tmp/watchdog-<PR>.sh` and chmod +x:

```bash
#!/usr/bin/env bash
# Watchdog: monitor "CICD NeMo RL" runs on pull-request/<PR> and emit
# per-job state changes. Stays alive across re-runs (new commits).
set -u
PR=<PR>
REPO=NVIDIA-NeMo/RL
BRANCH="pull-request/$PR"

prev_run_id=""
declare -A prev_state

emit() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

while true; do
  run_json=$(gh run list --repo "$REPO" --workflow "CICD NeMo RL" \
    --branch "$BRANCH" --limit 1 \
    --json databaseId,status,conclusion,headSha 2>/dev/null || echo "[]")
  run_id=$(echo "$run_json" | jq -r '.[0].databaseId // empty')
  run_status=$(echo "$run_json" | jq -r '.[0].status // empty')
  run_conclusion=$(echo "$run_json" | jq -r '.[0].conclusion // empty')
  run_sha=$(echo "$run_json" | jq -r '.[0].headSha // empty')

  if [[ -z "$run_id" ]]; then
    sleep 30; continue
  fi

  if [[ "$run_id" != "$prev_run_id" ]]; then
    emit "RUN ${run_id} STARTED sha=${run_sha:0:8} status=${run_status}"
    prev_run_id="$run_id"
    unset prev_state
    declare -A prev_state
  fi

  jobs_json=$(gh run view "$run_id" --repo "$REPO" --json jobs 2>/dev/null || echo "{}")
  while IFS=$'\t' read -r name status conclusion; do
    [[ -z "$name" ]] && continue
    cur="${status}/${conclusion}"
    if [[ "${prev_state[$name]:-}" != "$cur" ]]; then
      case "$status" in
        completed)
          emit "JOB ${name} -> ${conclusion}" ;;
        in_progress)
          if [[ -z "${prev_state[$name]:-}" || "${prev_state[$name]}" == "queued/" ]]; then
            emit "JOB ${name} -> in_progress"
          fi ;;
      esac
      prev_state[$name]="$cur"
    fi
  done < <(echo "$jobs_json" | jq -r '.jobs[]? | [.name, .status, (.conclusion // "")] | @tsv')

  if [[ "$run_status" == "completed" ]]; then
    emit "RUN ${run_id} COMPLETED conclusion=${run_conclusion}"
  fi

  sleep 60
done
```

### Arming the watchdog

```text
Monitor(
  description="CICD NeMo RL run state changes on PR <N>",
  command="bash /tmp/watchdog-<N>.sh",
  persistent=true,
  timeout_ms=3600000
)
```

`persistent: true` keeps it alive across re-runs (you'll push more
commits when quarantining flakes). Stop it with `TaskStop(<task-id>)`
once the run is green.

### Why never a cronjob / scheduled wakeup

- Cronjobs run blind — they fire on a clock, not on an event. You'll
  either over-poll (cache miss every wake-up) or miss long stalls.
- Wakeups can't easily fan out to "tell me whenever a job transitions"
  — they only resume the agent on a fixed interval.
- A persistent Monitor surfaces every job edge in real time and exits
  cleanly when the work is done.

## Step 7 — Quarantine on red, then iterate

When a `JOB <name> -> failure` event fires:

1. **Triage the failure — is it the bump or a flake?** Pull logs per
   @skills/cicd/SKILL.md "CI Failure Investigation". Only quarantine if
   the failure reproduces on `main` or is clearly unrelated
   infrastructure. If it's caused by the bump itself, **stop
   quarantining** — fix or revert. Quarantining a real regression hides
   the very signal the bump PR exists to surface.

2. **Quarantine via `tests/test_suites/disabled.txt`.** NeMo-RL has no
   active/flaky directory split — the disabled list is the canonical
   quarantine surface, matched by exact path against the entries in
   `nightly.txt` / `release.txt` / `nightly_gb200.txt`:

   ```bash
   cat >> tests/test_suites/disabled.txt <<EOF

   # Disabled for <package> bump PR #<N>: <one-line reason / upstream issue link>
   tests/test_suites/<llm|vlm>/<script-name>.sh
   EOF
   ```

   Copy the path verbatim from the suite list — extra whitespace or a
   trailing `/` will miss.

3. **Append to the PR body's Quarantined tests section** with a one-line
   reason and a follow-up tracking link if you have one. This is the
   durable record of what this bump deferred.

4. **Commit, push, retrigger:**

   ```bash
   git add tests/test_suites/disabled.txt
   git commit -S -s -m "ci: quarantine flaky <test> for <package> bump"
   git push
   gh pr comment <N> --repo NVIDIA-NeMo/RL --body "/ok to test $(git rev-parse HEAD)"
   ```

5. **Update the PR body** via `gh api PATCH` so the quarantine list
   stays current.

The watchdog is persistent — it picks up the new run automatically and
emits `RUN <id> STARTED` for the new attempt. Loop back to step 1.

## Step 8 — Stop when green

`RUN <id> COMPLETED conclusion=success` is the exit condition. Then:

```bash
gh pr checks <N> --repo NVIDIA-NeMo/RL | awk '{print $2}' | sort | uniq -c
TaskStop(<watchdog-task-id>)
gh api -X PATCH "repos/NVIDIA-NeMo/RL/pulls/<N>" -F "body=@/tmp/pr-body.md"
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Quality-check job is red but no test failures | `test_level=none` — PR has no `CI:*` label, so the matrix never expanded | Attach `CI:L2` (or the tier the bump warrants) |
| `semantic-pull-request` rejects the title | Bracket-prefixed `[build]` style — NeMo-RL uses bare Conventional Commits | Rewrite to `build: bump …` (no `[area]`) |
| TE / vLLM SHA still pinned somewhere after edit | `pyproject.toml` lists the same git ref in multiple places (override-dependencies + helper list) | `grep -n "<package>.*git@" pyproject.toml` and update every occurrence |
| Quarantined script still runs in CI | Disabled list path doesn't match the path used in `nightly*.txt` exactly | Copy the path verbatim — extra whitespace or trailing `/` will miss |
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` before locking |
| `uv lock` errors with "not a Python project" on a `3rdparty/*` path | Submodule not initialised in the worktree | `git submodule update --init --recursive` |

## Anti-patterns

- **Cron / scheduled wakeups for this loop.** Always Monitor.
- **Polling all workflows.** Filter to `CICD NeMo RL` — the rest are
  noise for a bump.
- **Quarantining a real regression** to "make CI green." That defeats
  the purpose of the bump PR. Only quarantine if the failure reproduces
  on `main` or is clearly unrelated infrastructure.
- **`gh pr edit`** for title/body. Use `gh api PATCH`.
- **HEREDOC in `gh pr create --body`.** Always go through a tmpfile +
  `--body-file`.
- **`[area]`-prefixed PR titles.** NeMo-RL's `semantic-pull-request`
  check rejects them — use bare Conventional Commits (`build:`,
  `chore:`, `ci:`).
- **Bundling unrelated changes** (feature work, refactors) into a bump
  PR. Bumps should stay surgical so CI failures attribute cleanly.
- **Removing entries from `disabled.txt`** as part of the bump PR
  cleanup. Removal is a separate PR — re-enabling a quarantined test
  should be its own reviewable change.
