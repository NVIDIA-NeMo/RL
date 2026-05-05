---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, vLLM, SGLang, Megatron-LM / Megatron-Bridge / Automodel / Gym submodules, etc.), regenerate the lockfile, open a PR, and drive it to green by attaching a watchdog to the "CICD NeMo RL" workflow and quarantining failing functional tests in `tests/test_suites/disabled.txt` until the run is green.
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

- Bumping a git-source pin in `pyproject.toml` (e.g. the
  `transformer-engine[pytorch,core_cu12] @ git+...@<ref>` line, or
  `vllm`, `sglang`, etc.).
- Bumping any of the `3rdparty/*` submodules — `Megatron-LM`,
  `Megatron-Bridge`, `Automodel`, `Gym`.
- Any change that touches `uv.lock` and needs the full L1 (functional) +
  L2 (convergence) matrix to prove out before merge.

For pure dep additions/removals without a CI loop, the
`build-and-dependency` skill is enough.

## Required context

Read first, then follow the steps below:

- @skills/contributing/SKILL.md — Conventional Commits PR title, DCO sign-off
- @skills/build-and-dependency/SKILL.md — `uv lock` mechanics, container choice
- @skills/cicd/SKILL.md — how `copy-pr-bot`, `/ok to test`, and the `CI:*` labels work
- @skills/testing/SKILL.md — recipe naming and the `tests/test_suites/` layout

## Step 1 — Worktree and edit

```bash
# From the NeMo-RL repo root
git worktree add .claude/worktrees/<slug> -b <branch-name> origin/main
git submodule update --init --recursive   # required before `uv lock` for submodule bumps
```

Edit the pin. Two common shapes:

**Git-source override** (TE, etc.) — edit the matching line in
`pyproject.toml`. There are usually two places to keep in sync (the
runtime override-dependencies block and the build-time helper list near
the bottom of the file). Search for the package name and update *every*
occurrence of the SHA so they don't drift:

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

**Submodule bump** (`3rdparty/Megatron-LM-workspace/Megatron-LM`,
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`,
`3rdparty/Automodel-workspace/Automodel`,
`3rdparty/Gym-workspace/Gym`):

```bash
cd 3rdparty/<name>-workspace/<name>
git fetch origin
git checkout <new-ref>
cd -
git add 3rdparty/<name>-workspace/<name>
```

## Step 2 — Regenerate the lockfile

`uv.lock` is Linux + CUDA only. Run inside the project image (see
@skills/build-and-dependency/SKILL.md for image build):

```bash
docker run --rm \
  -v $(pwd):/opt/nemo-rl \
  -v $HOME/.cache/uv:/root/.cache/uv \
  -w /opt/nemo-rl \
  nemo-rl:latest \
  bash -c 'uv lock'
```

Confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock
git diff uv.lock | grep -E '^\+\+\+|^---|name = ' | head -50
```

If the diff carries changes you didn't ask for (transitive movements you
can't explain), stop and investigate before pushing.

## Step 3 — Commit and push

```bash
git add pyproject.toml uv.lock 3rdparty/  # 3rdparty/ only if a submodule moved
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin <branch-name>
```

PR title format per @skills/contributing/SKILL.md: Conventional Commits,
`build:` (or `chore:`) — **no `[area]` prefix** (NeMo-RL's
`semantic-pull-request` job rejects bracket-prefixed titles). Sign-off
(`-s`) is mandatory; signed commits (`-S`) let `copy-pr-bot` mirror the
push to `pull-request/<N>` without a maintainer needing to post
`/ok to test` for every push.

## Step 4 — Open the PR

PR body goes through a tmpfile to preserve formatting. Wrap it in a
`<details>` block:

```bash
cat > /tmp/pr-body.md <<'EOF'
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
EOF

gh pr create \
  --repo NVIDIA-NeMo/RL \
  --base main \
  --head <branch-name> \
  --title "build: bump <package> to <ref>" \
  --body-file /tmp/pr-body.md \
  --label "CI:L2"
```

The `CI:L2` label is **mandatory** for a bump — it is the only way to
expand the matrix to L1 (functional) + L2 (convergence). Without a `CI:*`
label, `pre-flight` resolves `test_level=none` and the quality-check job
stays red. See @skills/cicd/SKILL.md for the full label table.

`gh pr edit` is unreliable. To update a PR's title or body later, use the
REST API directly:

```bash
gh api -X PATCH "repos/NVIDIA-NeMo/RL/pulls/<N>" \
  -F "body=@/tmp/pr-body.md"

gh api -X PATCH "repos/NVIDIA-NeMo/RL/pulls/<N>" \
  -f "title=build: bump <package> to <ref>"
```

## Step 5 — Trigger CI on the exact SHA

Even with a signed commit, post `/ok to test` on the SHA you actually
want exercised so any cached / cancelled run is re-fired. The
`pull-request/<N>` mirror branch is what `CICD NeMo RL` watches:

```bash
SHA=$(git rev-parse HEAD)
gh pr comment <N> --repo NVIDIA-NeMo/RL --body "/ok to test $SHA"
```

Use the **full** SHA (`git rev-parse HEAD`), never the short form.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **CICD NeMo RL** workflow only. Other workflows
(copyright check, semantic-pull-request, secrets scan, automodel
submodule check, wheel build) are noise here — the gate that decides
green-or-red for a bump is `CICD NeMo RL`.

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

1. Skim the logs to confirm it's a flake / pre-existing issue, not the
   bump itself:

   ```bash
   RUN_ID=<from "RUN ... STARTED" event>
   gh run view "$RUN_ID" --repo NVIDIA-NeMo/RL --log-failed > /tmp/run.log
   wc -l /tmp/run.log
   tail -200 /tmp/run.log
   ```

   If the failure is caused by the bump (real regression, not a flake),
   **stop quarantining** — fix the underlying issue or revert the bump.
   Quarantining a real regression hides the very signal the bump PR
   exists to surface.

2. Quarantine the failing test by appending its driver-script path to
   @tests/test_suites/disabled.txt (NeMo-RL does not have an
   active/flaky directory split — the disabled list is the canonical
   quarantine surface; see @skills/testing/SKILL.md for the
   `tests/test_suites/` layout):

   ```bash
   cat >> tests/test_suites/disabled.txt <<EOF

   # Disabled for <package> bump PR #<N>: <one-line reason / upstream issue link>
   tests/test_suites/<llm|vlm>/<script-name>.sh
   EOF
   ```

   Use the same path that appears in `nightly.txt` / `release.txt` /
   `nightly_gb200.txt` etc. — the disabled list is matched by exact
   path and applies to every suite that references the script.

3. Append the test to the PR description's **Quarantined tests**
   section, with a one-line reason and a follow-up tracking link if you
   have one. This is the durable record of what this bump deferred.

4. Commit, push, retrigger:

   ```bash
   git add tests/test_suites/disabled.txt
   git commit -S -s -m "ci: quarantine flaky <test> for <package> bump"
   git push
   SHA=$(git rev-parse HEAD)
   gh pr comment <N> --repo NVIDIA-NeMo/RL --body "/ok to test $SHA"
   ```

5. Update the PR body via `gh api PATCH` so the quarantine list stays
   current.

The watchdog is persistent — it will pick up the new run automatically
and emit `RUN <id> STARTED` for the new attempt.

## Step 8 — Stop when green

`RUN <id> COMPLETED conclusion=success` is the exit condition. Then:

```bash
# Sanity check
gh pr checks <N> --repo NVIDIA-NeMo/RL | awk '{print $2}' | sort | uniq -c

# Tear down
TaskStop(<watchdog-task-id>)

# Tick the boxes in the PR body
gh api -X PATCH "repos/NVIDIA-NeMo/RL/pulls/<N>" -F "body=@/tmp/pr-body.md"
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `uv lock` errors with "not a Python project" on a `3rdparty/*` path | Submodule not initialised in the worktree | `git submodule update --init --recursive` |
| CI never starts on a new push | Commit not GPG-signed and no `/ok to test` for the new SHA, or no `CI:*` label | Post `/ok to test $(git rev-parse HEAD)`; ensure exactly one `CI:*` label is attached |
| Quality-check job is red but no test failures | `test_level=none` — PR has no `CI:*` label, so the matrix never expanded | Attach `CI:L2` (or whatever tier the bump warrants) |
| `semantic-pull-request` rejects the title | Bracket-prefixed `[build]` style — NeMo-RL uses bare Conventional Commits | Rewrite to `build: bump …` (no `[area]`) |
| Watchdog goes silent for 30+ min | `gh` rate-limited or auth expired | Bump poll interval; `gh auth status`; restart Monitor |
| Quarantine commit doesn't trigger a new run | Pushed but didn't post `/ok to test` for the new SHA | Always re-post on the new SHA |
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` before locking |
| TE / vLLM SHA still pinned somewhere after edit | `pyproject.toml` lists the same git ref in multiple places (override-dependencies + helper list) | `grep -n "<package>.*git@" pyproject.toml` and update every occurrence |
| Quarantined script still runs in CI | Disabled list path doesn't match exactly the path used in `nightly*.txt` | Copy the path verbatim from the suite list — extra whitespace or trailing `/` will miss |

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
