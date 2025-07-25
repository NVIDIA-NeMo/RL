# Test NeMo RL

This guide outlines how to test NeMo RL using unit and functional tests, detailing steps for local or Docker-based execution, dependency setup, and metric tracking to ensure effective and reliable testing.

## Unit Tests

:::{important}
Unit tests require 2 GPUs to test the full suite.
:::

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

:::{tip}
Past unit test results are logged in `tests/unit/unit_results/`. These are helpful to view trends over time and commits.

Here's an example `jq` command to view trends:

```sh
jq -r '[.start_time, .git_commit, .metrics["test_hf_ray_policy::test_lm_policy_generation"].avg_prob_mult_error] | @tsv' tests/unit/unit_results/*

# Example output:
#2025-03-24 23:35:39     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:36:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:37:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:38:14     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:38:50     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
```
:::

## Functional Tests

:::{important}
Functional tests may require multiple GPUs to run. See each script to understand the requirements.
:::

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


## Static Type Checking with [MyPy](https://mypy-lang.org/)
Static type checking can be run with no GPU resources:

```sh
uv run --group test mypy {program}.py
```

For example,
```sh
uv run --group test mypy examples/run_grpo_math.py
uv run --group test mypy examples/run_sft.py
```

mypy.ini controls the configuration of mypy.