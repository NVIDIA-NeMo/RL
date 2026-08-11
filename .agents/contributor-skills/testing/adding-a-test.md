# Adding a Suite Test — Mechanics

Read @SKILL.md first for whether the test should exist. This file is the how.

A test is a **pair**: a recipe YAML and a driver script at mirrored paths with
the same basename. `common.env` derives one from the other, so the pairing is
structural, not a convention you can drift from.

```
examples/configs/recipes/<family>/<name>.yaml
tests/test_suites/<family>/<name>.sh
```

`<family>` is the model *generation*, taken from the resolved
`policy.model_name` — `qwen3/`, `llama3.1/`, `nemotron3-super/`. Performance
recipes go one level deeper in `<family>/performance/`. Recipes built on a model
outside any official family live in `reproduce/`.

Hardware is **not** a directory. A GB200 variant is a filename token and a
`defaults:` child of its H100 counterpart.

## 1. The recipe YAML

Inherit from the algorithm's base config rather than restating it:

```yaml
defaults: ../../grpo_math_1B.yaml
```

The path is **relative to the recipe file**, so it changes if the recipe moves
between depths. Only override what your test actually varies — every key you
restate is a key that stops tracking the base config.

## 2. The driver script

```bash
#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "${SCRIPT_DIR%%/tests/test_suites/*}/tests/test_suites/common.env"
# ===== BEGIN CONFIG =====
SUITE=nightly
SKU=h100
NUM_NODES=1
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/token_mult_prob_error"]["450"] < 1.1'
fi
```

### The source line

Copy it verbatim. `${SCRIPT_DIR%%/tests/test_suites/*}` strips everything from
the `tests/test_suites` path component onward, which finds the project root at
**any** driver depth. A counted `../..` breaks the moment a recipe moves between
a family directory and a `performance/` subdirectory, and it breaks silently.

### The CONFIG block

`tools/launch`, `tools/list-suites`, and the unit tests all parse this block as
text between the `BEGIN`/`END` markers. Keep declarations there — a variable set
after `END CONFIG` is invisible to all three.

| Variable | Meaning |
|---|---|
| `SUITE` | `nightly`, `release`, `performance`, or `disabled` |
| `TAGS` | Optional. Slice within the suite, e.g. `"mcore"` → the `nightly_mcore` gate |
| `SKU` | `h100` (8 GPUs/node) or `gb200` (4). Must agree with `GPUS_PER_NODE` |
| `DISABLED_REASON` | Required when `SUITE=disabled` |
| `NUM_NODES` | Slurm allocation. Must match the `<N>n` in the filename |
| `GPUS_PER_NODE` | Defaults to 8. Set to 4 for GB200 |
| `MAX_STEPS` | Total steps across all runs |
| `STEPS_PER_RUN` | Steps per Slurm job — set below the partition's time limit |
| `NUM_RUNS` | Jobs needed to reach `MAX_STEPS`; use the ceiling-division idiom |
| `NUM_MINUTES` | Wall-clock per job. GPU-hours = `NUM_NODES × GPUS_PER_NODE × NUM_MINUTES / 60 × NUM_RUNS` |

`NUM_MINUTES` is what the budget is computed from, so an inflated value costs
the lane real headroom whether or not the job uses the time.

### What `common.env` gives you

`PROJECT_ROOT`, `EXP_NAME` (basename of the driver), `EXP_DIR`, `LOG_DIR`,
`CKPT_DIR`, `JSON_METRICS`, `RUN_LOG`, and `CONFIG_PATH` — derived from the
driver's own path by swapping `tests/test_suites` → `examples/configs/recipes`,
which is what makes the YAML/driver pairing mandatory. It errors out if the
config is missing.

It also provides `exit_if_max_steps_reached` (skip the job if a previous run
already reached `MAX_STEPS`) and `assert_not_grep`, sets `PYTHONPATH`, installs
audio deps for `omni`/`audio`/`avqa`/`vlm` drivers, and exits early under
`TEST_DRYRUN`.

## 3. The suite manifest

Add the driver path, relative to the project root, to the matching
`tests/test_suites/<suite>.txt`. This must agree with the driver's `SUITE`/`SKU`
declaration — `test_declarations_reproduce_the_manifest` compares the two and
fails on any disagreement.

The manifests are on their way out: once nemo-ci reads the declarations, they
and that test are deleted together, and this step disappears.

## GB200 mirrors

The YAML inherits from its H100 sibling and overrides only what the SKU forces:

```yaml
defaults: ./grpo-qwen3-8b-1n8g-megatron.yaml
cluster:
  gpus_per_node: 4
policy:
  megatron_cfg:
    tensor_model_parallel_size: 4   # halved from 8
```

The driver sets `SKU=gb200` and `GPUS_PER_NODE=4`, and the filename says `1n4g`.

## TQ wrappers

A `-tq_simple` / `-tq_mooncake` variant re-runs an existing recipe with a
different transfer backend. The wrapper delegates rather than duplicating:

```bash
source "${SCRIPT_DIR%%/tests/test_suites/*}/tests/test_suites/common-tq.env"
export EXP_NAME="$TQ_EXP_NAME"
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"
```

`EXP_NAME` is exported so the wrapper gets its own log directory, checkpoint
directory, and W&B run name while executing the base driver's body — including
its metric assertions. The wrapper's CONFIG block must mirror the base's
`NUM_NODES`/`MAX_STEPS`/`NUM_MINUTES` so GPU-hour accounting stays right.

## Verifying locally

You cannot run the training itself without a cluster, but everything else is
checkable:

```sh
# The driver resolves its config and sources cleanly (no GPU needed)
TEST_DRYRUN=1 ./tests/test_suites/<family>/<name>.sh

# GPU-hour cost of your test and of the lane it joins
DRYRUN=1 CONTAINER= ACCOUNT= PARTITION= ./tools/launch ./tests/test_suites/<family>/<name>.sh
tools/list-suites --gpu-hours nightly | tail -1

# The declaration lands in the suite you expect
tools/list-suites nightly | grep <name>

# Naming, declarations, manifests, budgets
uv run pytest tests/unit/test_recipe_naming.py tests/unit/test_recipes_and_test_suites.py
```

`tools/launch` needs GNU `sed` and will not run on macOS.

## Running on the head node

A driver is directly executable, and results land beside it in a directory named
after the script:

```sh
uv run ./tests/test_suites/llama3.1/sft-llama3.1-8b-1n8g-megatron-seqpack.sh

ls -lh tests/test_suites/llama3.1/sft-llama3.1-8b-1n8g-megatron-seqpack/
# drwxr-xr-x 2 user dip 4.0K Apr 23 18:07 ckpts
# drwxr-xr-x 3 user dip 4.0K Apr 23 18:07 logs
# -rw-r--r-- 1 user dip 142K Apr 23 18:23 metrics.json
# -rw-r--r-- 1 user dip  94K Apr 23 18:23 run.log
```

## Launching with code snapshots

`tools/launch` copies the tree to a code snapshot and submits `NUM_RUNS` Slurm
jobs against **that** copy, so an experiment keeps running against the code as it
was at launch even as the repo moves on.

```sh
# Launch
CONTAINER=... ACCOUNT=... PARTITION=... ./tools/launch ./tests/test_suites/<family>/<name>.sh

# Print estimated GPU hours, then exit
DRYRUN=1 CONTAINER=... ACCOUNT=... PARTITION=... ./tools/launch ./tests/test_suites/<family>/<name>.sh

# Print estimated GPU hours, create the snapshot, then exit
DRYRUN=2 CONTAINER=... ACCOUNT=... PARTITION=... ./tools/launch ./tests/test_suites/<family>/<name>.sh

# Launch with extra env vars
EXTRA_ENV="NRL_FORCE_REBUILD_VENVS=true NRL_DEEPSCALER_8K_CKPT=/8k-ckpt" \
CONTAINER=... ACCOUNT=... PARTITION=... ./tools/launch ./tests/test_suites/<family>/<name>.sh
```

`tools/launch` reads `NUM_NODES`, `GPUS_PER_NODE`, `NUM_RUNS` and `NUM_MINUTES`
straight out of the CONFIG block to size the allocation and estimate the cost.
Results live under the snapshot:

```sh
ls -lh code_snapshots/<name>/recipes/<family>/<name>/
```

Each snapshot also gets a `continue.sh` that launches another run with the same
arguments — useful if a job was cancelled or you want to run longer:

```sh
code_snapshots/<name>/continue.sh
```
