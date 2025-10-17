# Test NeMo RL

This guide outlines how to test NeMo RL using unit and functional tests, detailing steps for local or Docker-based execution, dependency setup, and metric tracking to ensure effective and reliable testing.

## Unit Tests

> [!IMPORTANT]
> Unit tests require 2 GPUs to test the full suite.

> [!TIP]
> Some unit tests require setting up test assets which you can download with 
> ```sh
> uv run tests/unit/prepare_unit_test_assets.py
> ```


```sh
# Run the unit tests using local GPUs

# Configuration 1: Default tests only - excludes both hf_gated and mcore tests
uv run --group test bash tests/run_unit.sh

# Configuration 2: Default + HF gated tests, excluding mcore tests
uv run --group test bash tests/run_unit.sh --hf-gated

# Configuration 3: ONLY mcore tests, excluding ones with hf_gated
uv run --extra mcore --group test bash tests/run_unit.sh --mcore-only

# Configuration 4: ONLY mcore tests, including ones with hf_gated
uv run --extra mcore --group test bash tests/run_unit.sh --mcore-only --hf-gated
```

### Experimental: Faster local test iteration with pytest-testmon

We support `pytest-testmon` to speed up local unit test runs by re-running only impacted tests. This works for both regular in-process code and out-of-process `@ray.remote` workers via a lightweight, test-only selection helper.

Usage:
```sh
# Re-run only impacted unit tests
uv run --group test pytest --testmon tests/unit

# You can also combine with markers/paths
uv run --group test pytest --hf-gated --testmon tests/unit/models/policy/test_dtensor_worker.py
```

What to expect:
- On the first run in a fresh workspace, testmon may run a broader set (or deselect everything if nothing was executed yet) to build its dependency cache.
- On subsequent runs, editing non-remote code narrows selection to only the tests that import/use those modules.
- Editing code inside `@ray.remote` actors also retriggers impacted tests. We maintain a static mapping from test modules to transitive `nemo_rl` modules they import and intersect that with changed files when `--testmon` is present.
- After a successful impacted run, a second `--testmon` invocation (with no further edits) will deselect all tests.
- Running `pytest` with `-k some_substring_in_test_name` will always run tests that match even if `--testmon` is passed.

Limitations and tips:
- Selection is based on Python imports and file mtimes; non-Python assets (YAML/JSON/shell) are not tracked. When editing those, re-run target tests explicitly.
- The remote-aware selection uses a conservative static import map (no dynamic import resolution). If a test loads code dynamically that isn’t visible via imports, you may need to run it explicitly once to seed the map.
- The helper is test-only and does not alter library behavior. It activates automatically when you pass `--testmon`.

Refreshing remote-selection artifacts
-------------------------------------
If you change test layout or significantly refactor imports, the remote-selection artifacts may become stale.
To rebuild them, delete the following files at the repo root and re-run with `--testmon` to seed again:

```sh
# At the root of nemo-rl
rm .nrl_remote_map.json .nrl_remote_state.json
```


### Run Unit Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in Docker with this script:

```sh
CONTAINER=... bash tests/run_unit_in_docker.sh
```

The required `CONTAINER` can be built by following the instructions in the [Docker documentation](docker.md).

### Track Metrics in Unit Tests

Unit tests may also log metrics to a fixture. The fixture is called `tracker` and has the following API:

```python
# Track an arbitrary metric (must be json serializable)
tracker.track(metric_name, metric_value)
# Log the maximum memory across the entire cluster. Okay for tests since they are run serially.
tracker.log_max_mem(metric_name)
# Returns the maximum memory. Useful if you are measuring changes in memory.
tracker.get_max_mem()
```

Including the `tracker` fixture also tracks the elapsed time for the test implicitly.

Here is an example test:

```python
def test_exponentiate(tracker):
    starting_mem = tracker.get_max_mem()
    base = 2
    exponent = 4
    result = base ** exponent
    tracker.track("result", result)
    tracker.log_max_mem("memory_after_exponentiating")
    change_in_mem = tracker.get_max_mem() - starting_mem
    tracker.track("change_in_mem", change_in_mem)
    assert result == 16
```

Which would produce this file in `tests/unit/unit_results.json`:

```json
{
  "exit_status": 0,
  "git_commit": "f1062bd3fd95fc64443e2d9ee4a35fc654ba897e",
  "start_time": "2025-03-24 23:34:12",
  "metrics": {
    "test_hf_ray_policy::test_lm_policy_generation": {
      "avg_prob_mult_error": 1.0000039339065552,
      "mean_lps": -1.5399343967437744,
      "_elapsed": 17.323044061660767
    }
  },
  "gpu_types": [
    "NVIDIA H100 80GB HBM3"
  ],
  "coverage": 24.55897613282601
}
```

> [!TIP]
> Past unit test results are logged in `tests/unit/unit_results/`. These are helpful to view trends over time and commits.
>
> ```sh
> jq -r '[.start_time, .git_commit, .metrics["test_hf_ray_policy::test_lm_policy_generation"].avg_prob_mult_error] | @tsv' tests/unit/unit_results/*
>
> # Example output:
> #2025-03-24 23:35:39     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
> #2025-03-24 23:36:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
> #2025-03-24 23:37:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
> #2025-03-24 23:38:14     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
> #2025-03-24 23:38:50     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
> ```

## Functional Tests

> [!IMPORTANT]
> Functional tests may require multiple GPUs to run. See each script to understand the requirements.

Functional tests are located under `tests/functional/`.

```sh
# Run the functional test for sft
uv run bash tests/functional/sft.sh
```

At the end of each functional test, the metric checks will be printed as well as
whether they pass or fail. Here is an example:

```text
                              Metric Checks
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Status ┃ Check                          ┃ Value             ┃ Message ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ PASS   │ data["train/loss"]["9"] < 1500 │ 817.4517822265625 │         │
└────────┴────────────────────────────────┴───────────────────┴─────────┘
```

### Run Functional Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in Docker with this script:

```sh
CONTAINER=... bash run_functional_in_docker.sh functional/sft.sh
```

The required `CONTAINER` can be built by following the instructions in the [Docker documentation](docker.md).

## Bisecting Failing Tests

> [!IMPORTANT]
> Always rsync the `tools/` directory to `tools.bisect/` before starting a bisect:
>
> ```sh
> rsync -ahP --delete tools/ tools.bisect/
> ```
>
> This creates a stable copy of the bisect scripts that won't change as git checks out different commits during the bisect process. Without this, the scripts themselves may change mid-bisect, leading to inconsistent behavior or failures. All examples below reference `tools.bisect/` to ensure you use the stable copy.

### Bisecting Unit/Functional Tests

Use `tools.bisect/bisect-run.sh` to automatically run your test command across a commit range and find the first bad commit. It forces venv rebuilds so dependencies match each commit.

Basic usage:

```sh
GOOD=<good_ref> BAD=<bad_ref> \
  tools.bisect/bisect-run.sh uv run --group test pytest tests/unit/test_foobar.py::test_case
```

Examples:

```sh
GOOD=56a6225 BAD=32faafa \
  tools.bisect/bisect-run.sh uv run --group dev pre-commit run --all-files

GOOD=464ed38 BAD=c843f1b \
  tools.bisect/bisect-run.sh uv run --group test pytest tests/unit/test_foobar.py
```

Notes:

- Exit codes drive the classification: 0=good, non-zero=bad, 125=skip.
- The script pre-verifies that `GOOD` is actually good by running your command on it.
- On failure or interruption, it saves a timestamped `git bisect log` to `<repo>/bisect-logs/`. You can resume later with `BISECT_REPLAY_LOG` (see below).
- Set `BISECT_NO_RESET=1` to keep the bisect state after the script exits.

Resume from a saved bisect log:

```sh
BISECT_REPLAY_LOG=/abs/path/to/bisect-2025....log \
  tools.bisect/bisect-run.sh uv run --group test pytest tests/unit/test_foobar.py
```

### Bisecting nightlies

Nightly training scripts can be bisected using the same driver plus a helper that sets up hermetic runs on Slurm.

Vanilla flow:

```sh
# Copy bisect utilities outside of VCS to ensure a stable runner
rsync -ahP --delete tools/ tools.bisect/

TEST_CASE=tests/test_suites/llm/sft-llama3.2-1b-1n8g-fsdp2tp1.v3.sh

HF_HOME=... \
HF_DATASETS_CACHE=... \
CONTAINER=... \
MOUNTS=... \
ACCOUNT=... \
PARTITION=... \
GOOD=$(git log --format="%h" --diff-filter=A -- "$TEST_CASE") \
BAD=HEAD \
  tools.bisect/bisect-run.sh tools.bisect/launch-bisect.sh "$TEST_CASE"
```

::::{note}
The command `GOOD=$(git log --format="%h" --diff-filter=A -- "$TEST_CASE")` selects the commit that introduced the test script. Because the path is typically added only once, this yields the introduction commit to use as the known good baseline.
::::

- `tools.bisect/launch-bisect-helper.sh` ensures each commit runs in a fresh venv, creates an isolated code snapshot per commit, blocks until metrics are checked, and returns a suitable exit code for bisect.

Progressively more advanced cases:

1) Adjusting the test case on the fly with `SED_CLAUSES`

- If a test script needs small textual edits during bisect (e.g., relax a threshold; drop a noisy metric you don’t care to bisect over when focusing on convergence vs. perf), provide a sed script via `SED_CLAUSES`. You can also use this to adjust runtime controls like `MAX_STEPS`, `STEPS_PER_RUN`, or `NUM_MINUTES` when a perf regression slows runs down so they still complete and emit metrics. The helper applies it and automatically restores the test script after the run.

```sh
SED_CLAUSES=$(cat <<'SED'
s#mean(data\["timing/train/total_step_time"\], -6, -1) < 0\.6#mean(data["timing/train/total_step_time"], -6, -1) < 0.63#
/ray\/node\.0\.gpu\.0\.mem_gb/d
SED
) \
GOOD=$(git log --format="%h" --diff-filter=A -- "$TEST_CASE") \
BAD=HEAD \
  tools.bisect/bisect-run.sh tools.bisect/launch-bisect.sh "$TEST_CASE"
```

1) Passing extra script arguments

- If the nightly script supports Hydra/CLI overrides, pass them via `EXTRA_SCRIPT_ARGS` so each run adopts those overrides (e.g., fix a transient incompatibility):

:::{important}
Changing script arguments can materially affect performance characteristics and/or convergence behavior. This may influence the validity of the bisect outcome relative to your baseline configuration. Prefer the smallest, clearly-justified overrides, keep them consistent across all commits, and document them alongside your results so conclusions are interpreted correctly.
:::

```sh
EXTRA_SCRIPT_ARGS="++data.num_workers=1" \
GOOD=$(git log --format="%h" --diff-filter=A -- "$TEST_CASE") \
BAD=HEAD \
  tools.bisect/bisect-run.sh tools.bisect/launch-bisect.sh "$TEST_CASE"
```

1) Resuming from an earlier interrupted or misclassified session

- Use `BISECT_REPLAY_LOG` with the bisect driver to replay prior markings and continue running. This is handy if a run failed for an unrelated reason or you manually edited a log to change `bad` → `skip` or to drop an incorrect line.

```sh
BISECT_REPLAY_LOG=/abs/path/to/bisect-logs/bisect-YYYYmmdd-HHMMSS-<sha>.log \
HF_HOME=... HF_DATASETS_CACHE=... CONTAINER=... MOUNTS=... ACCOUNT=... PARTITION=... \
  tools.bisect/bisect-run.sh tools.bisect/launch-bisect.sh "$TEST_CASE"
```

Tips and conventions:

- Exit code 125 means "skip this commit" in git bisect; our helper returns 125 if required env is missing or if it needs to abort safely.
- Submodules must be clean. The bisect script enforces `submodule.recurse=true` and `fetch.recurseSubmodules=on-demand` so submodules follow commit checkouts.
- The bisect script automatically unshallows all submodules at the start to ensure any submodule commit can be checked out during the bisect process. This is important because bisecting may need to jump to arbitrary commits in submodule history.
- Each commit uses a fresh code snapshot directory and a separate Megatron checkpoint dir to avoid cross-commit contamination.
- On failure/interrupt, a timestamped bisect log is saved under `<repo>/bisect-logs/`. Use it with `BISECT_REPLAY_LOG` to resume.
- In some unusual cases, the bisect may fail while updating a submodule because it references a commit that is orphaned or deleted. Git will typically print the commit hash it was unable to find (e.g., `fatal: remote error: upload-pack: not our ref <commit>`). If the commit is simply orphaned, you can try to manually fetch it:

  ```sh
  # Assuming Automodel is the submodule with the missing commit
  cd 3rdparty/Automodel-workspace/Automodel/
  git fetch origin $the_automodel_commit_that_it_could_not_find
  ```

  If the manual fetch fails, the commit has likely been deleted from the remote. In this case, skip the problematic commit:

  ```sh
  git bisect skip $the_nemorl_commit_that_has_the_broken_automodel_commit
  ```

  After skipping, add the skip command to your `BISECT_REPLAY_LOG` file (located in `<repo>/bisect-logs/`) so the bisect will continue from where it left off and skip that commit when you relaunch `tools.bisect/bisect-run.sh`:

  ```sh
  echo "git bisect skip $the_nemorl_commit_that_has_the_broken_automodel_commit" >> bisect-logs/bisect-<timestamp>-<sha>.log
  ```
