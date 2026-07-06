---
name: contributing
description: Contribution and Git workflow conventions for NeMo-RL. Use when switching or checking out branches, running git switch or git checkout, synchronizing submodules, opening a PR, writing a commit message, triggering CI, or reviewing the PR process. Covers mandatory recursive submodule updates, PR title format, commit sign-off, and CI triggering.
---

# Contributing Conventions

## Branch Switching and Submodules

After every successful NeMo-RL branch switch or checkout, immediately synchronize
all registered submodules, including nested submodules, before inspecting files,
building, testing, or launching jobs:

```bash
git submodule update --init --recursive
git submodule status --recursive
```

Treat this as a mandatory part of `git switch` and `git checkout`, not an optional
cleanup step. A top-level branch change updates recorded gitlinks but does not
guarantee that submodule worktrees or their nested submodules moved to those SHAs.

If Git refuses because a submodule has local changes, stop and report the dirty
paths. Do not force checkout, clean, reset, or delete those changes without the
user's explicit approval.

## PR Title Format

PR titles **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) spec. This is enforced by the `semantic-pull-request` CI check.

```
<type>[optional scope]: <description>
```

Allowed types:

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `ci` | CI/CD changes |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or fixing tests |
| `chore` | Maintenance (deps, configs, tooling) |
| `perf` | Performance improvement |
| `build` | Build system changes |
| `revert` | Reverts a previous commit |

**Do:**
```
ci: retry apt-get installs to handle mirror sync failures
feat(grpo): add dataclass config defaults infrastructure
fix: preserve RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
```

**Don't:**
```
[ci] fix: retry apt-get installs   ← area tags are not part of this convention
Update stuff
Fix bug
```

## Commit Sign-off

All commits must be signed off with `-s`:

```bash
git commit -s -m "fix: correct reward normalization"
```

## CI Triggering

After pushing, trigger CI with:

```
/ok to test <full-commit-sha>
```

Use `git rev-parse HEAD` (not the short form) to get the full SHA.
