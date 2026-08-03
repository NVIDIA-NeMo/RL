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

## CI Pipeline Architecture

The main pipeline (`cicd-main.yml`) runs through these stages:

1. **Pre-flight**: determines test level from PR labels, changed files, and event type
2. **Container build**: Docker image built on GPU runners (skipped for `Lfast` or when a pre-built `image_tag` is provided via `workflow_dispatch`)
3. **Tests**: run in containers on GPU runners using the custom `test-template` action
4. **Coverage**: aggregated from doc-tests, unit-tests, and e2e; uploaded to Codecov
5. **QA Gate**: aggregates all job results into a single pass/fail status

## Adding Tests to CI

### Adding a single test to an existing suite

Each L1 suite has a corresponding orchestrator script in
[`tests/functional/`](https://github.com/NVIDIA-NeMo/RL/tree/main/tests/functional/)
that lists which scenario scripts to run. To add one test:

1. **Write a scenario script** in `tests/functional/`, following the pattern of an
   existing script in the same suite. For example,
   [`tests/functional/grpo_sglang_sync.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/grpo_sglang_sync.sh)
   is a scenario in the SGLang suite.

2. **Register it in the suite orchestrator**. Open the suite's
   `tests/functional/L1_Functional_Tests_<Suite>.sh` and add a `run_test` call:

   ```bash
   run_test      uv run --no-sync bash ./tests/functional/<your_scenario>.sh   # full L1 only
   run_test fast uv run --no-sync bash ./tests/functional/<your_scenario>.sh   # also runs in Lfast
   ```

   Use `run_test fast` if the test is short enough to include in `Lfast` (reuses
   the pre-built `main` container). Use plain `run_test` if it should run at L1
   only. See
   [`tests/functional/L1_Functional_Tests_SGLang.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_SGLang.sh)
   for an example with both.

3. **For unit tests**: add the test file under
   `tests/unit/models/generation/<backend>/` or the appropriate subdirectory,
   decorate it with the backend's pytest marker (e.g. `@pytest.mark.vllm`), and
   it will be picked up automatically by the existing shard script (e.g.
   [`tests/unit/L0_Unit_Tests_Vllm_1.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/L0_Unit_Tests_Vllm_1.sh)).

---

### Adding a new test suite (e.g. a new backend)

Adding a first-class suite (like the TRT-LLM or SGLang suites) requires changes
in **eight places**. Use the TRT-LLM and SGLang suites as reference throughout.

#### 1. pytest marker and filtering logic — `tests/unit/conftest.py`

Register a `--<backend>-only` CLI flag and the corresponding
filter in `collection_modifyitems`. Every new backend needs:

- An `addoption` entry (see the existing
  [`--trtllm-only`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/conftest.py#L67)
  and
  [`--vllm-only`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/conftest.py#L55)
  entries).
- A variable in `collection_modifyitems` and a filter block that
  validates the import and selects/excludes tests by marker (see
  [the trtllm block](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/conftest.py#L191)).

By default, unmarked tests in the backend's own directory also run — your
`--<backend>-only` flag handles the explicit pass; tests without any marker run
in the baseline shard.

#### 2. pytest marker declaration — `pyproject.toml`

Add `<backend> = "..."` under `[tool.pytest.ini_options] markers` so pytest
does not warn about an unknown marker.

#### 3. Unit test files

Place backend-specific tests under
`tests/unit/models/generation/<backend>/` and decorate them with
`@pytest.mark.<backend>`. See
[`tests/unit/models/generation/trtllm/`](https://github.com/NVIDIA-NeMo/RL/tree/main/tests/unit/models/generation/trtllm)
and
[`tests/unit/models/generation/sglang/`](https://github.com/NVIDIA-NeMo/RL/tree/main/tests/unit/models/generation/sglang)
for examples.

#### 4. L0 unit shard script — `tests/unit/L0_Unit_Tests_<Backend>.sh`

Source the shared boilerplate, then run the backend's tests via `uv run --extra <backend>`:

```bash
source "$(dirname "${BASH_SOURCE[0]}")/run_unit_shard_common.sh"

uv run --extra <backend> bash -x ./tests/run_unit.sh \
    "unit/" \
    "${EXCLUDED_UNIT_TESTS[@]}" \
    --cov=nemo_rl --cov-report=term-missing --cov-report=json \
    --hf-gated \
    --<backend>-only
```

See
[`tests/unit/L0_Unit_Tests_Trtllm.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/L0_Unit_Tests_Trtllm.sh)
and
[`tests/unit/L0_Unit_Tests_Sglang.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/L0_Unit_Tests_Sglang.sh)
for full examples. SGLang also does a separate pass on its own directory
to catch tests without an explicit marker — follow that pattern if needed.

#### 5. L1 functional suite orchestrator — `tests/functional/L1_Functional_Tests_<Backend>.sh`

Create the suite orchestrator that calls each scenario script. Use
`run_test fast` for tests short enough to include in `Lfast`:

```bash
run_test fast uv run --no-sync bash ./tests/functional/<scenario_a>.sh
run_test      uv run --no-sync bash ./tests/functional/<scenario_b>.sh
```

See
[`tests/functional/L1_Functional_Tests_SGLang.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_SGLang.sh)
(uses `run_test fast` for Lfast inclusion) and
[`tests/functional/L1_Functional_Tests_Trtllm.sh`](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/functional/L1_Functional_Tests_Trtllm.sh)
(L1 only, no `run_test fast`).

#### 6. CI matrix entries — `.github/workflows/cicd-main.yml`

Add the new suite to **three places** in the workflow:

| Location | What to add |
|----------|-------------|
| L0 unit matrix | `- script: L0_Unit_Tests_<Backend>` (see [existing entries](https://github.com/NVIDIA-NeMo/RL/blob/main/github/workflows/cicd-main.yml#L523-L567)) |
| L1 functional matrix — standard runners | `- script: L1_Functional_Tests_<Backend>` with runner variable (see [line ~708](https://github.com/NVIDIA-NeMo/RL/blob/main/github/workflows/cicd-main.yml#L708)) |
| L1 functional matrix — GB200 runners | Same script with `runner: ${{ vars.GB200_RUNNER }}` (see [line ~781](https://github.com/NVIDIA-NeMo/RL/blob/main/github/workflows/cicd-main.yml#L781)) |

If your backend requires special build-time caches (like TRT-LLM's
`trtllm-ccache-tag`), add those to the container build step as well.

**Lfast guard (temporary)**: if the backend is not yet in the `main` container,
add a conditional to skip the unit shard in Lfast:

```yaml
if: ${{ needs.pre-flight.outputs.test_level != 'Lfast' || matrix.script != 'L0_Unit_Tests_<Backend>' }}
```

Remove this guard once the backend ships on main. See
[PR #3479](https://github.com/NVIDIA-NeMo/RL/pull/3479) which removed this
guard for TRT-LLM once it landed.

#### 7. uv optional extra — `pyproject.toml`

Declare the backend as an optional dependency so `uv run --extra <backend>`
resolves correctly:

```toml
[project.optional-dependencies]
<backend> = ["<backend-package>>=<version>"]
```

#### 8. Example configs (optional but standard)

Add at least one exemplar YAML under `examples/configs/` and a recipe under
`examples/configs/recipes/`. See
[`examples/configs/grpo_math_1B_trtllm.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B_trtllm.yaml)
and
[`examples/configs/grpo_math_1B_sglang.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B_sglang.yaml)
as examples.

---

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
| `lockfile-check.yml` | PR (dependency paths) | uv.lock freshness vs submodule pyprojects |
| `claude-review.yml` | `/claude-review` comment | AI code review |
| `healthcheck_vms.yml` | schedule (07:00 UTC), dispatch | GPU runner health |
| `automodel-submodule-checks.yml` | PR | Submodule validation |
| `merge-queue-retry.yml` | PR dequeued (timeout) | Auto-retry merge queue |
| `cherry-pick-release-commit.yml` | push main | Release cherry-picks |
| `close-inactive-issue-pr.yml` | schedule (01:30 UTC) | Stale issue/PR cleanup |
| `community-bot.yml` | issues, comments | Project board sync |
| `pr-checks-comment.yml` | PR | Post submodule check results |
