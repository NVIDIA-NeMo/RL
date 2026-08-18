# Heterogeneous-job external GenRM pool

These helpers run independent vLLM servers beside NeMo RL in one two-component
Slurm heterogeneous job. The servers are exposed through one OpenAI-compatible
load-balancer URL.

The topology is intentionally opt-in:

```text
one Slurm heterogeneous job
├── hetgroup 0: NeMo RL Ray cluster started by ray.sub
└── hetgroup 1: independent private Ray/vLLM replicas
```

`run_in_allocation.sh` performs the lifecycle:

1. Validate the launcher configuration before starting any service.
2. Read the NeMo RL nodes from hetgroup 0 and GenRM nodes from hetgroup 1.
3. Start each GenRM replica in its own private Ray cluster.
4. Start the load balancer and wait for every backend to become healthy.
5. Replace `GENRM_URL_PLACEHOLDER` in `COMMAND` with the load-balancer URL.
6. Normalize the component-0 Slurm environment and start `ray.sub`.
7. Stop the whole job if any required service exits unexpectedly.

The placeholder replacement is textual; neither Gym nor NeMo RL consumes the
configured token directly. Put `GENRM_URL_PLACEHOLDER` (default
`__GENRM_BASE_URL__`) in the Gym `base_url` override in `COMMAND`. The served
model defaults to the literal name `model`, so the Gym model setting must also
be `model` unless `GENRM_SERVED_MODEL_NAME` is changed.

## Filesystem and container requirements

The GenRM replicas mount `/lustre:/lustre`. Consequently, `BASE_LOG_DIR`,
`GENRM_TOOLS_DIR_HOST`, an optional `GENRM_REASONING_PARSER`, and an absolute
`GENRM_MODEL` path must be under `/lustre`. `GENRM_MODEL` may instead be a
Hugging Face model ID such as `Qwen/Qwen3-32B`.

`GENRM_CONTAINER` must contain:

- the `GENRM_VLLM_PYTHON` executable;
- importable `nemo_rl`, `ray`, and `vllm` packages in that Python environment;
- the `ray` command on `PATH`.

`serve_vllm_on_ray.py` imports NeMo RL's vLLM compatibility patches, so a
standalone vLLM-only container is insufficient.

`CONTAINER` must provide `GENRM_LB_PYTHON` with `aiohttp` installed.
`MOUNTS` remains the mount list for the NeMo RL container and must expose the
Slurm submission directory and any paths used by `COMMAND`. The launcher adds
read-only access to `GENRM_TOOLS_DIR_HOST` and read-write access to
`GENRM_STATE_DIR` explicitly for the load-balancer container; callers do not
need to duplicate those two mounts.

## Launcher environment

Required variables:

| Variable | Purpose |
| --- | --- |
| `BASE_LOG_DIR` | Shared `/lustre` parent directory for `<job-id>-logs`. |
| `COMMAND` | NeMo RL command containing `GENRM_URL_PLACEHOLDER` (default `__GENRM_BASE_URL__`) in its Gym `base_url` override. |
| `CONTAINER` | NeMo RL container used by `ray.sub` and the load balancer. |
| `GENRM_CONTAINER` | Container used by the native vLLM replicas; see the import requirements above. |
| `GENRM_MODEL` | `/lustre` model path or Hugging Face model ID passed to vLLM. |
| `GENRM_TOOLS_DIR_HOST` | `/lustre` path to this directory. |
| `GENRM_VLLM_PYTHON` | Python executable inside `GENRM_CONTAINER`. |
| `MOUNTS` | Mounts required by `ray.sub` and `COMMAND`. |

The launcher requires exactly two Slurm hetgroups. Slurm provides
`SLURM_HET_SIZE`, `SLURM_JOB_NODELIST_HET_GROUP_0`, and
`SLURM_JOB_NODELIST_HET_GROUP_1`; the script fails before starting services if
those variables are absent or the component count is not two. Submit it with
the two-component `sbatch` form shown below rather than running it on a login
node.

Common optional variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPUS_PER_NODE` | `4` | GPUs claimed on every GenRM and NeMo RL Ray node. |
| `GENRM_REPLICAS` | `8` | Number of independent vLLM replicas. |
| `GENRM_TENSOR_PARALLEL_SIZE` | `8` | Tensor parallel size of each replica. |
| `GENRM_SERVED_MODEL_NAME` | `model` | Model name accepted by the OpenAI-compatible API. |
| `GENRM_REASONING_PARSER` | unset | Shared path to a reasoning-parser plugin. Must be set with `GENRM_REASONING_PARSER_NAME`. |
| `GENRM_REASONING_PARSER_NAME` | unset | Parser name registered by `GENRM_REASONING_PARSER`. |
| `GENRM_TOOL_CALL_PARSER` | unset | vLLM tool-call parser; also enables automatic tool choice when set. |
| `GENRM_ENABLE_EXPERT_PARALLEL` | `0` | Pass `--enable-expert-parallel` when set to `1`. |
| `GENRM_COMPILATION_CONFIG` | unset | JSON passed to vLLM `--compilation-config`. |
| `GENRM_MODEL_LOADER_EXTRA_CONFIG` | unset | JSON passed to vLLM `--model-loader-extra-config`; set node-specific thread counts here. |
| `GENRM_GROUP_ID` | `inline-$SLURM_JOB_ID` | Registry namespace for this job's backend pool. |
| `GENRM_VLLM_PORT` | `8000` | Backend HTTP port. |
| `GENRM_LB_PORT` | `9213` | Load-balancer HTTP port. |
| `GENRM_LB_PYTHON` | `/opt/nemo_rl_venv/bin/python` | Python executable in `CONTAINER` used for the load balancer. |
| `GENRM_STARTUP_TIMEOUT` | `3600` | Seconds allowed for the entire GenRM pool to become healthy. |
| `GENRM_URL_PLACEHOLDER` | `__GENRM_BASE_URL__` | Token replaced in `COMMAND`. |
| `RAY_SUB` | `$SLURM_SUBMIT_DIR/ray.sub` | NeMo RL Slurm launcher. |

The topology must satisfy:

```text
nodes per replica = GENRM_TENSOR_PARALLEL_SIZE / GPUS_PER_NODE
hetgroup 1 nodes = GENRM_REPLICAS * nodes per replica
```

The default topology is eight TP=8 replicas on four-GPU nodes, so hetgroup 1
must contain exactly 16 nodes. Hetgroup 0 must contain all nodes used by NeMo
RL and its in-cluster Gym services.

## Example

This example co-schedules 64 four-GPU NeMo RL nodes in hetgroup 0 and 16
four-GPU nodes for eight TP=8 GenRM replicas in hetgroup 1:

```bash
cd /lustre/path/to/RL

BASE_LOG_DIR=/lustre/path/to/results/ray_logs \
CONTAINER=/lustre/path/to/nemo-rl.sqsh \
MOUNTS="/lustre:/lustre,$PWD:/opt/nemo-rl" \
GENRM_CONTAINER=/lustre/path/to/nemo-rl-with-vllm.sqsh \
GENRM_MODEL=/lustre/path/to/genrm-checkpoint \
GENRM_TOOLS_DIR_HOST="$PWD/tools/external_genrm" \
GENRM_VLLM_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python \
GENRM_REASONING_PARSER=/lustre/path/to/ultra_v3_reasoning_parser.py \
GENRM_REASONING_PARSER_NAME=ultra_v3 \
GENRM_TOOL_CALL_PARSER=qwen3_coder \
GENRM_ENABLE_EXPERT_PARALLEL=1 \
GENRM_COMPILATION_CONFIG='{"pass_config":{"fuse_allreduce_rms":false}}' \
GENRM_MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":true,"num_threads":96}' \
GPUS_PER_NODE=4 \
COMMAND='uv run examples/nemo_gym/run_grpo_nemo_gym.py --config /lustre/path/to/recipe.yaml ++env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=__GENRM_BASE_URL__ ++env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=model' \
sbatch \
  --account=<account> \
  --partition=<partition> \
  --nodes=64 \
  --exclusive \
  --gres=gpu:4 \
  --time=04:00:00 \
  --export=ALL \
  : \
  --account=<account> \
  --partition=<partition> \
  --nodes=16 \
  --exclusive \
  --gres=gpu:4 \
  --time=04:00:00 \
  tools/external_genrm/run_in_allocation.sh
```

The batch script runs in hetgroup 0. Every GenRM `srun` explicitly targets
hetgroup 1; the load balancer and `ray.sub` remain in hetgroup 0. Slurm
co-schedules both components and cancellation of the heterogeneous-job leader
cancels the complete deployment.

The `++` prefix makes these overrides work whether or not the recipe already
declares the keys.

The resolved URL is also written to
`$BASE_LOG_DIR/<job-id>-logs/genrm_url`.
