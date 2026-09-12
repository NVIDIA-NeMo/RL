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

## CI Labels

Every PR **must** have exactly one `CI:*` label — the quality-check job stays
red until one is attached. Labels control which test tier runs and whether a
new container image is built.

| Label | What runs | Container |
|-------|-----------|-----------|
| `CI:docs` | Doc tests only | Reuses main container |
| `CI:Lfast` | Fast test subset | Reuses main container |
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

---

## Adding Tests to CI

See `docs/ci-cd.md` → "Adding Tests to CI" for the full guide with code links.
Key decision tree for common cases:

### Adding a single test to an existing L1 suite

1. Write `tests/functional/<scenario>.sh`
2. Add `run_test [fast] uv run --no-sync bash ./tests/functional/<scenario>.sh` to the suite orchestrator `tests/functional/L1_Functional_Tests_<Suite>.sh`
3. Use `run_test fast` to also include it in `Lfast`; plain `run_test` for L1 only

Reference: [L1_Functional_Tests_SGLang.sh](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_SGLang.sh)

### Adding a single unit test to an existing L0 shard

Add the test file to the appropriate `tests/unit/` subdirectory with the correct
`@pytest.mark.<backend>` decorator. It is picked up automatically by the shard
script that runs `--<backend>-only`.

Reference: [tests/unit/models/generation/trtllm/](https://github.com/NVIDIA-NeMo/RL/tree/main/tests/unit/models/generation/trtllm)

### Adding a brand-new suite (new backend)

Eight touch-points, in order:

1. **`tests/unit/conftest.py`** — add `--<backend>-only` option + filter block
2. **`pyproject.toml`** — add marker under `[tool.pytest.ini_options] markers` and optional dep under `[project.optional-dependencies]`
3. **Unit test files** — `tests/unit/models/generation/<backend>/` with `@pytest.mark.<backend>`
4. **`tests/unit/L0_Unit_Tests_<Backend>.sh`** — sources `run_unit_shard_common.sh`, runs `uv run --extra <backend> ... --<backend>-only`
5. **`tests/functional/<scenario>.sh`** files — individual functional scenarios
6. **`tests/functional/L1_Functional_Tests_<Backend>.sh`** — suite orchestrator; add `run_test fast` for Lfast-eligible tests
7. **`.github/workflows/cicd-main.yml`** — add to L0 unit matrix, L1 functional matrix (standard), and L1 functional matrix (GB200)
8. **`examples/configs/`** — at least one exemplar YAML

Reference suites: [TRT-LLM](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_Trtllm.sh) · [SGLang](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_SGLang.sh)
