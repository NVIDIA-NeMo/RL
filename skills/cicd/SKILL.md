---
name: cicd
description: CI/CD reference for NeMo-RL. Covers GitHub Actions pipeline structure, CI triggering via /ok to test, and CI failure investigation.
when_to_use: Investigating a CI failure; understanding the pipeline structure; triggering CI; re-running CI; 'CI is red', 'how do I trigger CI', '/ok to test', 'PR workflow', 'where are the logs', 'CI did not run', 'copy-pr-bot', 're-run', 'rerun', 'retry CI', 'flaky test'.
---

# CI/CD Guide

---

## How CI Works

NeMo-RL CI runs on GitHub Actions. Workflows live in `.github/workflows/`.

The main test workflow triggers on pushes to `pull-request/<number>` branches.
These branches are created automatically by **copy-pr-bot** when a contributor
pushes to their fork and the PR receives a trust signal:
- The commit is GPG-signed by a maintainer, **or**
- A maintainer posts `/ok to test <full-sha>` as a PR comment.

---

## Triggering CI

After pushing a new commit to your PR, trigger CI with:

```
/ok to test <full-commit-sha>
```

Use `git rev-parse HEAD` (not the short form) to get the full SHA.

> **Re-triggering after new commits**: each `/ok to test <sha>` is specific to
> that SHA. After pushing additional commits, post a new comment with the
> updated SHA.

---

## CI Labels

Every PR **must** have exactly one `CI:*` label — the quality-check job stays
red until one is attached. Labels control which test tier runs and whether a
new container image is built.

| Label | What runs | Container |
|-------|-----------|-----------|
| `CI:docs` | Doc tests only | Reuses main container (PRs to a release branch such as `r0.8.0` reuse that branch's container, same as `CI:Lfast`) |
| `CI:Lfast` | Fast test subset | Reuses main container (PRs to a release branch such as `r0.8.0` reuse that branch's container, published only by pushes to that branch after merges) |
| `CI:L0` | Unit tests + docs + lint | Builds new image |
| `CI:L1` | L0 + functional tests | Builds new image |
| `CI:L2` | L1 + convergence tests | Builds new image |
| `Skip CICD` | Nothing (skips all tests) | — |

**Default on merge group / push to main**: L1.

**Which label to attach when opening a PR:**

| Changed paths / nature of change | Label |
|----------------------------------|-------|
| Docs only (`docs/`, `*.md`, docstrings) | `CI:docs` |
| Trivial fix, no logic change | `CI:Lfast` |
| New code, bug fix, refactor | `CI:L0` |
| Changes that could affect model behaviour | `CI:L1` |
| Changes that could affect convergence | `CI:L2` |

---

## CI Failure Investigation

```bash
# List recent workflow runs for the PR
gh run list --repo NVIDIA-NeMo/RL --branch "pull-request/<pr-number>"

# View failing run summary
gh run view <run-id> --repo NVIDIA-NeMo/RL

# Stream failing job output
gh run view <run-id> --repo NVIDIA-NeMo/RL --log-failed
```

Common failure patterns:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CI never starts | No trust signal or copy-pr-bot not triggered | Post `/ok to test <sha>` |
| `semantic-pull-request` fails | PR title doesn't follow Conventional Commits | Fix PR title; see `contributing` skill |
| Linting fails | Style violation | Run `uv run ruff check --fix . && uv run ruff format .` |
| Unit test failure | Code regression or missing dependency | Reproduce locally; see `testing` skill |
| `CPU config checks` fails (on a PR, the rest of the run is cancelled) | A recipe config, test-suite list or nightly GPU-hour budget check failed (`tests/unit/test_recipes_and_test_suites.py`, `tests/unit/test_config_validation.py`) | Reproduce with `cd tests && uv run --group test pytest unit/test_recipes_and_test_suites.py unit/test_config_validation.py`; the job runs on a GitHub-hosted CPU runner |

---

## Re-running CI

To retry a failed run on the **same commit**, re-run only the failed jobs:

```bash
gh run rerun <run-id> --failed --repo NVIDIA-NeMo/RL
```

In the UI this is **Re-run jobs → Re-run failed jobs**. It keeps every job that
already passed and also re-runs jobs that were skipped because they depended on
a failed job.

Rules for agents:

- **Never run `gh run rerun <run-id>` without `--failed`**, and never suggest
  **Re-run all jobs**. A full re-run repeats every GPU job on a commit that
  already passed most of them.
- **Read the failure first** (`gh run view <run-id> --log-failed`). Re-run only
  when the failure is unrelated to the change: a known flaky test, a download or
  network error, a runner or container problem.
- **If the failure comes from the change, fix it and push.** A new commit needs a
  new `/ok to test <full-sha>` comment; do not re-run the old commit.
- **Do not push empty commits** to re-trigger CI; use `--failed` on the existing run.

