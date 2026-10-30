---
name: contributing
description: Contribution conventions for NeMo-RL. Covers the rules every code change follows, PR title format, commit sign-off, and CI triggering.
when_to_use: Writing or changing code; removing, renaming or reshaping a config key; opening a PR; writing a commit message; triggering CI; 'PR title format', 'sign-off', 'conventional commits', 'how do I trigger CI', during code review.
---

# Contributing Conventions

## Rules for Every Code Change

The rules live in `references/`, one file per theme. Before writing or changing code, read all of them. Reviews check against the same files. A rule that ends with `Contributing Details:` has step-by-step instructions in `references/details/`; read them when the rule applies. A `Review Details:` link is the review lesson a rule came from and is not needed to write code.

| File | Covers |
|------|--------|
| [comments-docstrings-and-docs](references/comments-docstrings-and-docs.md) | comments, docstrings and docs a change adds, edits or moves |
| [config](references/config.md) | config a change adds, reads or overrides, and config models built in tests |
| [dead-code](references/dead-code.md) | names a change adds, removes or stops using |
| [error-handling](references/error-handling.md) | exceptions a change catches, and misconfigurations it must reject |
| [linting-and-formatting](references/linting-and-formatting.md) | style of the Python and shell a change writes: naming, signatures, types, imports |

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
