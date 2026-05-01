---
name: cicd
description: CI/CD reference for NeMo-RL. Covers GitHub Actions pipeline structure, CI triggering via /ok to test, and CI failure investigation.
when_to_use: Investigating a CI failure; understanding the pipeline structure; triggering CI; 'CI is red', 'how do I trigger CI', '/ok to test', 'PR workflow', 'where are the logs', 'CI did not run', 'copy-pr-bot'.
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
