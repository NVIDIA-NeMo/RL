# Recipes

## Test Suites

Test suites are defined in `.txt` files that list the test scripts to run:

- `nightly.txt` - H100 tests for nightly CI (8 GPUs per node)
- `release.txt` - H100 tests for release CI (8 GPUs per node)
- `nightly_gb200.txt` - GB200 tests for nightly CI (4 GPUs per node)
- `release_gb200.txt` - GB200 tests for release CI (4 GPUs per node)
- `performance_h100.txt` - Performance benchmarks for H100 (8 GPUs per node)
- `performance_gb200.txt` - Performance benchmarks for GB200 (4 GPUs per node)

## Naming

Base pattern (LLM):

```
<algo>-<model>-<nodes>n<gpus>g-<strategy-and-params>[-modifiers][-long][.vN].sh
```

VLM pattern:

```
vlm_<algo>-<model>-<nodes>n<gpus>g-<strategy>[-modifiers][.vN].sh
```

- **algo**: task or algorithm, e.g., `sft`, `dpo`, `grpo`.
- **model**: model identifier, e.g., `llama3.1-8b-instruct`, `qwen2.5-7b-instruct`.
- **nodes/gpus**: cluster allocation, e.g., `1n8g`, `4n8g`, `8n8g`.
- **strategy-and-params**: parallelism or framework detail, e.g., `fsdp2tp1`, `tp4pp2`, `megatron`, `dtensor2tp1`.
- **modifiers** (optional): short flags like `sp` (sequence packing), `actckpt` (activation checkpointing), `fp8`, `noncolocated`, `quick`.
- **-long** (optional): indicates long-running recipe.
- **.vN** (optional): version suffix (e.g., `.v2`, `.v3`) reserved for convergence-impacting changes. Use when the recipe's convergence behavior changes (dataset, loss, convergence bug fix). Pure performance changes do not require a version bump.

Examples:

```
sft-llama3.1-8b-1n8g-fsdp2tp1-long.sh
dpo-llama3.1-8b-instruct-4n8g-fsdp2tp4.sh
grpo-llama3.1-8b-instruct-1n8g-megatron-fp8.sh
grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4sp.v3.sh
```

Known exceptions currently present:
- Deepscaler recipes encode context length in place of the cluster tuple, e.g., `grpo-deepscaler-1.5b-8K.sh`. These are allowed but should document the intended hardware in the script body.
- Some recipes include additional short flags in the strategy token (e.g., `fsdp2tp8sp`). Treat these as modifiers appended to the strategy.

Directory placement and naming parity:
- Place driver scripts under `tests/test_suites/llm/` or `tests/test_suites/vlm/`.
- The script filename should mirror the YAML recipe filename under `examples/configs/recipes/**` but with a `.sh` suffix.
- Add the relative script path to `tests/test_suites/nightly.txt` for nightly execution.

## Running manually

Each recipe can be run on the head node:

```sh
uv run ./llm/sft-llama3.2-1b-1n8g-fsdp2tp1.sh
```

and the result directory can be found at the same level of the script (w/o `.sh` prefix):

```sh
ls -lh llm/sft-llama3.2-1b-1n8g-fsdp2tp1/
# drwxr-xr-x 2 terryk dip 4.0K Apr 23 18:07 ckpts
# drwxr-xr-x 3 terryk dip 4.0K Apr 23 18:07 logs
# -rw-r--r-- 1 terryk dip 142K Apr 23 18:23 metrics.json
# -rw-r--r-- 1 terryk dip  94K Apr 23 18:23 run.log
```

## GB200 Variants

For GB200 systems with 4 GPUs per node, test scripts should include `GPUS_PER_NODE=4` in the CONFIG section. This ensures the launch script uses the correct GPU count for slurm allocation and GPU hour calculations:

```sh
# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4    # 4 for GB200, 8 for H100 (default)
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
# ===== END CONFIG =====
```

GB200 YAML configs should inherit from their 8g counterparts and override:
- `cluster.gpus_per_node: 4`
- Any parallelism settings that need to change (e.g., halving `tensor_parallel_size`)
- Directory/name references updated to reflect the 4g naming

## Launching with code snapshots

We provide a convenience script that will create a code snapshot and launch `NUM_RUNS` number of slurm jobs (`NUM_RUNS` is defined in the script itself). We create a code snapshot to
ensure that even as the master repo changes its code, you can always run your experiment with
the snapshot of the code at the time the experiment was initially launched.

```sh
# Launch
CONTAINER=... ACCOUNT=... PARTITION=... ../tools/launch ./llm/sft-llama3.2-1b-1n8g-fsdp2tp1.sh

# Prints Estimated GPUhrs and then exits
DRYRUN=1 CONTAINER=... ACCOUNT=... PARTITION=... ../tools/launch ./llm/sft-llama3.2-1b-1n8g-fsdp2tp1.sh

# Prints Estimated GPUhrs, creates code snapshot, then exits
DRYRUN=2 CONTAINER=... ACCOUNT=... PARTITION=... ../tools/launch ./llm/sft-llama3.2-1b-1n8g-fsdp2tp1.sh

# Launch but set extra env vars
EXTRA_ENV="NRL_FORCE_REBUILD_VENVS=true NRL_DEEPSCALER_8K_CKPT=/8k-ckpt NRL_DEEPSCALER_16K_CKPT=/16k-ckpt" \
CONTAINER=... ACCOUNT=... PARTITION=... ../tools/launch ./llm/sft-llama3.2-1b-1n8g-fsdp2tp1.sh
```

After this completes, you can find the result under

```sh
ls -lh ../code_snapshots/sft-llama3.2-1b-1n8g-fsdp2tp1/recipes/llm/sft-llama3.2-1b-1n8g-fsdp2tp1/
# drwxr-xr-x 2 terryk dip 4.0K Apr 23 18:07 ckpts
# drwxr-xr-x 3 terryk dip 4.0K Apr 23 18:07 logs
# -rw-r--r-- 1 terryk dip 142K Apr 23 18:23 metrics.json
# -rw-r--r-- 1 terryk dip  94K Apr 23 18:23 run.log
```

As a convenience, there's also a `continue.sh` script under that will launch
another run using the same arguments. This is helpful if your job was
unexpectedly cancelled or you want to run it for a little longer.

```sh
# This launches one more run of the same experiment
../code_snapshots/sft-llama3.2-1b-1n8g-fsdp2tp1/continue.sh
```
