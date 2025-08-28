# Recipes

## Naming

Each test is named:
```
<algo>-<model>-#n#g-<parallelism>-<opt:long><opt:v$N>.sh
```

Examples:
* sft-llama3.2-1b-1n8g-fsdp2tp1.sh
* grpo-qwen2-1.5B-instruct-4n8g-fsdp2tp2.sh
* grpo-qwen2-1.5B-instruct-4n8g-fsdp2tp2-long.sh
* grpo-qwen2-1.5B-instruct-4n8g-fsdp2tp2-long.v2.sh
    * The final verison suffix (starts with `.v2`, `.v3`, ...), is reserved for cases contributors believe the recipe's 
      convergence has changed due to their commit. Versioning signals that this recipe should not be compared to its
      predecessor due to a change in convergence behavior. Examples of this change include: changing dataset, changing loss,
      convergence bug fix. Changes affecting performance do not need a version change. 

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
