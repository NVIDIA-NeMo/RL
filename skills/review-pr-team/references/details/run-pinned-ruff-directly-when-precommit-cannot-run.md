# run-pinned-ruff-directly-when-precommit-cannot-run

> On macOS `pre-commit` cannot run (linux-only uv.lock) — use `uvx ruff@<pinned>` and run all THREE hooks, because a bare `ruff check` misses I001 entirely

The skill runs `uv run --group dev pre-commit run --all-files` first. On macOS that can fail outright — `uv.lock` only supports
`sys_platform == 'linux'`, so the venv cannot even be built. Reporting "linter not run" throws
away a real finding.

The right fallback is `uvx ruff@<version pinned in .pre-commit-config.yaml>`, which gives the
same version and the same config as CI. NeMo-RL runs ruff as **three** separate hooks, and you
must run all three:

```
uvx ruff@0.9.9 check .              # F rules
uvx ruff@0.9.9 check --select I .   # import sorting — SEPARATE hook
uvx ruff@0.9.9 format --check .
```

`[tool.ruff.lint] select = ["D", "F"]` in `pyproject.toml` does **not** include `I`, so a plain
`ruff check` reports zero import-order errors while CI fails on them. Also required: attribute
findings only to files in the PR diff, and re-run each file's base-SHA version through ruff to
prove the failure is PR-introduced rather than pre-existing.

**Why:** On PR #2957 a teammate ran a bare `ruff check` and found 2 failures; running all three
hooks found 6, including a dead `BatchedDataDict` import in production code and an `I001` caused
by a stray blank line that *also* created the merge conflict with main. Reporting 2 would have
sent the author back for a second lint round.

**How to apply:** Try `pre-commit` first (it may work in a container). On the "not compatible with
the lockfile's supported environments" error, switch to `uvx ruff@<pinned>` immediately and say in
the review which route you took and why — the author should know the check was real, not skipped.
Note that `pre-commit` also needs submodules initialized; a `nemo-gym ... is not a workspace
member` build error means run `git submodule update --init --recursive` in the worktree first.
Related: `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)).
