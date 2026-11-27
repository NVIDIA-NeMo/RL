# CI/CD

NeMo RL uses GitHub Actions for continuous integration, testing, and release automation. The CI pipeline implements a tiered testing system that balances thoroughness with resource efficiency.

## Test Levels

Tests are organized into levels of increasing scope and cost:

| Level | What runs | When |
|-------|-----------|------|
| **docs** | Lint + Sphinx docs build only; no container, no GPU | `CI:docs` label |
| **L0** | CPU config checks + unit tests (one parallel GPU job per `tests/unit/L0_Unit_Tests_*.sh` script) | `CI:L0` label |
| **L1** | Unit tests + functional tests (GPU) | `CI:L1` label, push to main/merge-group |
| **L2** | Same as L1 today (plan to add convergence tests) | `CI:L2` label |
| **Lfast** | Fast unit + functional tests, reuses pre-built main container (skips build); PRs to a release branch (e.g. `r0.8.0`) reuse that branch's container, which is published only by builds of the release branch itself (pushes after merges), never by PR builds; until the first such build exists, Lfast on that branch cannot pull its image | `CI:Lfast` label |

**Defaults:**
- PRs do not run tests unless a CI label is applied.
- Pushes to `main` and merge-group events force **L1**.
- Nightly scheduled runs (09:00 UTC) run the full suite.
- Doc-only changes are auto-detected and skip unnecessary tests.

### When Lfast missed a failure

`CI:Lfast` runs a subset of `CI:L1`. If L1 catches a failure that Lfast missed, promote the test that caught it into Lfast so the next PR sees it earlier:

- Unit test: narrow or remove the entry that excludes it in `tests/unit/excluded_unit_tests.sh`. Tests under `research/*/tests/unit` never run in Lfast.
- Functional test: change its `run_test <cmd>` line in `tests/functional/L1_Functional_Tests_*.sh` to `run_test fast <cmd>`. Lfast runs only the H100 scripts, so a test in an `L1_Functional_Tests_GB200_*.sh` script, or a failure that reproduces only on GB200, cannot be promoted this way.

## Shared Build Caches

- The prebuilt `main` container and the TRT-LLM ccache/wheel cache images are written only by builds of `main` itself. PR builds read them but never push to them.
- Release branches (`r<major>.<minor>.<patch>`) get their own container tag (`:<branch>`), `<branch>-buildcache` and branch-scoped TRT-LLM caches, written only by pushes to that branch (manual dispatches on a release branch publish nothing shared). They never write `main`'s caches or the `uv-cache` image.
- A PR's own image is tagged with its run ID and PR number only.

## Triggering CI on Pull Requests

1. **The CI label is set for you.** On PRs to `main`, the `Set CI label` workflow picks `CI:docs`, `CI:Lfast` or `CI:L1` from the changed files (rules in `skills/cicd/SKILL.md`). Only repo maintainers can change it, for example to `CI:L1` for a large refactor or broad changes to core code; ask one if the label is wrong.
2. **Comment** `/ok to test <commit-sha>` — a bot will acknowledge with a thumbs-up and start CI.
   - If you are an external contributor, you will need an internal NVIDIA developer to comment this on your PR to trigger CI.
3. The **`Skip CICD`** label bypasses tests entirely (except on `main`/merge-group). Only repo maintainers can set it.

## Nightly CI with `/nightly-ci`

Members of the `automation` or `rl_autobot_approvers` GitHub teams can comment `/nightly-ci test-case=<test name>` on a PR to run NVIDIA's nightly CI on the PR head. The pipeline builds the RL image from the PR head. For example:

```text
/nightly-ci test-case=llm_grpo_llama3_2_1b_instruct_1n8g_megatron
```

- **Enroll once (NVIDIA employees only):** `run_as: requester` runs the pipeline as the commenter. Enroll once as described in the internal Autobot `/nightly-ci` docs. Without enrollment, Autobot refuses the command with a neutral check explaining what to do.
- **Fork PRs:** first have the PR head vetted with `/ok to test <full-sha>` and wait for the copy-pr-bot mirror to match that head.
- **Settings:** `test-suites` defaults to `nightly`; the default cluster is configured in [`.github/autobot/nightly-ci.yml`](https://github.com/NVIDIA-NeMo/RL/blob/main/.github/autobot/nightly-ci.yml). A command may override `test-cluster` (`h100-2`, the default, or `gb200-1`) and `test-suites`.
- **Results:** the **`Autobot nightly CI`** check shows the applied settings, including `run as | requester`, and the pipeline result. A new push supersedes the run; the running pipeline is not cancelled.

## Required Checks

All PRs must pass these checks before merging:

- **Lint**: ruff + pyrefly via pre-commit
- **Branch freshness**: PR branch must be at most 10 commits behind the base branch
- **Semantic PR title**: must follow conventional commit format
- **DCO sign-off**: all commits must be signed with `--signoff` (see [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md))
- **Secrets detection**: scans for accidentally committed secrets
- **Submodule validation**: Automodel submodule must be fast-forwarded from the base branch

## CI Pipeline Architecture

The main pipeline (`cicd-main.yml`) runs through these stages:

1. **Pre-flight**: determines test level from PR labels, changed files, and event type
2. **Container build**: Docker image built on GPU runners (skipped for `Lfast` or when a pre-built `image_tag` is provided via `workflow_dispatch`)
3. **Tests**: unit and functional tests run in containers on GPU runners using the custom `test-template` action. Doctests no longer run in CI; run them locally with `tests/docs/Docs_Tests.sh` (see [documentation](documentation.md)). The CPU config checks (`test_recipes_and_test_suites.py`, `test_config_validation.py`) run on a GitHub-hosted CPU runner after pre-flight and, on PR runs, lint and the test approval queue pass, alongside the container build when there is one. No test job waits for them; the QA gate requires them, and on a PR run a failure cancels the rest of the run.
4. **Coverage**: aggregated from unit-tests and e2e; uploaded to Codecov
5. **QA Gate**: aggregates all job results into a single pass/fail status

## Re-running a Failed Run

To retry the same commit, re-run only the failed jobs (**Re-run jobs → Re-run failed jobs**, or `gh run rerun <run-id> --failed`). This also re-runs cancelled jobs, so it recovers a run that the CPU config checks cancelled. **Re-run all jobs** repeats every GPU job that already passed. See [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md).

## Nightly Runs

- Full test suite runs daily at **09:00 UTC** on `main`. Failures send Slack alerts.
- Nightly docs are published at **10:00 UTC** to a separate "nightly" version (does not overwrite stable "latest" docs).

## Release Process

All release workflows are manual (`workflow_dispatch`) with dry-run defaults:

| Workflow | Purpose |
|----------|---------|
| `release-freeze.yml` | Create release branch and version bump |
| `release.yaml` | Build wheel, create GitHub release, generate changelog |
| `release-docs.yml` | Publish docs to S3 + Akamai CDN (versioned and/or "latest") |
| `build-test-publish-wheel.yml` | Auto-publish to TestPyPI on main/release pushes (dry-run by default) |

## Infrastructure

- **VM health checks** (`healthcheck_vms.yml`): daily GPU health checks (07:00 UTC) on self-hosted runners. Auto-reboots degraded VMs and alerts via Slack on persistent failures.
- **Merge queue retry** (`merge-queue-retry.yml`): auto-retries PRs dequeued due to CI timeout (max 3 retries before alerting).
- **Stale cleanup** (`close-inactive-issue-pr.yml`): daily auto-close of inactive issues and PRs.
- **Backports**: comment `/cherrypick to=<branch>` on the source PR; Autobot opens the backport PR (not a workflow in this repo).
- **Community bot** (`community-bot.yml`): syncs issues and comments to a GitHub Project board for tracking.

## Workflow Reference

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `cicd-main.yml` | push, PR, schedule, dispatch | Main CI pipeline |
| `build-test-publish-wheel.yml` | push main/r** | Wheel build + TestPyPI |
| `release.yaml` | dispatch | Full release |
| `release-freeze.yml` | dispatch | Code freeze |
| `release-docs.yml` | dispatch, callable | Publish docs to S3/CDN |
| `release-nightly-docs.yml` | schedule (10:00 UTC) | Nightly docs publish |
| `detect-secrets.yml` | PR | Secrets scanning |
| `semantic-pull-request.yml` | PR | PR title validation |
| `labeler.yaml` | PR | Auto-label by file path |
| `lockfile-check.yml` | PR (dependency paths) | uv.lock freshness vs submodule pyprojects |
| `healthcheck_vms.yml` | schedule (07:00 UTC), dispatch | GPU runner health |
| `automodel-submodule-checks.yml` | PR | Submodule validation |
| `merge-queue-retry.yml` | PR dequeued (timeout) | Auto-retry merge queue |
| `close-inactive-issue-pr.yml` | schedule (01:30 UTC) | Stale issue/PR cleanup |
| `community-bot.yml` | issues, comments | Project board sync |
| `pr-checks-comment.yml` | PR | Post submodule check results |
