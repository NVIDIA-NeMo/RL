# CI/CD

NeMo RL uses GitHub Actions for continuous integration, testing, and release automation. The CI pipeline implements a tiered testing system that balances thoroughness with resource efficiency.

## Test Levels

Tests are organized into levels of increasing scope and cost:

| Level | What runs | When |
|-------|-----------|------|
| **docs** | Doctests only | `CI:docs` label |
| **L0** | Doctests + unit tests (3 parallel suites: Generation, Policy, Other) | `CI:L0` label |
| **L1** | Doctests + unit tests + functional tests (GPU) | `CI:L1` label, push to main/merge-group |
| **L2** | Full suite including convergence tests | `CI:L2` label |
| **Lfast** | Fast unit + functional tests, reuses pre-built main container (skips build) | `CI:Lfast` label |

**Defaults:**
- PRs do not run tests unless a CI label is applied.
- Pushes to `main` and merge-group events force **L1**.
- Nightly scheduled runs (09:00 UTC) run the full suite.
- Doc-only changes are auto-detected and skip unnecessary tests.

## Triggering CI on Pull Requests

1. **Apply a CI label** to your PR: `CI:docs`, `CI:L0`, `CI:L1`, `CI:L2`, or `CI:Lfast`.
2. **Comment** `/ok to test <commit-sha>` — a bot will acknowledge with a thumbs-up and start CI.
   - If you are an external contributor, you will need an internal NVIDIA developer to comment this on your PR to trigger CI.
3. The **`Skip CICD`** label bypasses tests entirely (except on `main`/merge-group).

## Required Checks

All PRs must pass these checks before merging:

- **Lint**: ruff + pyrefly via pre-commit
- **Branch freshness**: PR branch must be at most 10 commits behind the base branch
- **Semantic PR title**: must follow conventional commit format
- **DCO sign-off**: all commits must be signed with `--signoff` (see [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md))
- **Secrets detection**: scans for accidentally committed secrets
- **Submodule validation**: Automodel submodule must be fast-forwarded from the base branch
- **Megatron-Bridge dependency sync**: `pyproject.toml` dependencies must match the Megatron-Bridge submodule metadata

## CI Pipeline Architecture

The main pipeline (`cicd-main.yml`) runs through these stages:

1. **Pre-flight**: determines test level from PR labels, changed files, and event type
2. **Container build**: Docker image built on GPU runners (skipped for `Lfast` or when a pre-built `image_tag` is provided via `workflow_dispatch`)
3. **Tests**: run in containers on GPU runners using the custom `test-template` action
4. **Coverage**: aggregated from doc-tests, unit-tests, and e2e; uploaded to Codecov
5. **QA Gate**: aggregates all job results into a single pass/fail status

## Code Review

Commenting `/claude-review` on a PR triggers an AI-powered code review. This is restricted to org members.

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
- **Cherry-pick** (`cherry-pick-release-commit.yml`): auto-creates cherry-pick PRs from release branches back to main.
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
| `claude-review.yml` | `/claude-review` comment | AI code review |
| `healthcheck_vms.yml` | schedule (07:00 UTC), dispatch | GPU runner health |
| `automodel-submodule-checks.yml` | PR | Submodule validation |
| `mbridge-deps-sync.yml` | PR (specific paths) | Dependency sync check |
| `merge-queue-retry.yml` | PR dequeued (timeout) | Auto-retry merge queue |
| `cherry-pick-release-commit.yml` | push main | Release cherry-picks |
| `close-inactive-issue-pr.yml` | schedule (01:30 UTC) | Stale issue/PR cleanup |
| `community-bot.yml` | issues, comments | Project board sync |
| `pr-checks-comment.yml` | PR | Post submodule check results |
