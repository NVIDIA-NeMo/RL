# Heterogeneous-job external Gym vLLM pools

These helpers run arbitrary fixed-model vLLM pools beside NeMo RL in one
two-component Slurm heterogeneous job:

```text
one Slurm heterogeneous job
├── hetgroup 0: NeMo RL Ray cluster and per-pool load balancers
└── hetgroup 1: independent private Ray/vLLM replicas
```

`run_in_allocation.sh` is service-agnostic. A recipe launcher defines named
pools such as GenRM, NL2Bash, or safety, including each model's environment and
vLLM arguments. The wrapper then:

1. validates every pool and splits hetgroup 1 into disjoint node slices;
2. starts every replica in its own private Ray cluster;
3. starts one OpenAI-compatible load balancer per pool;
4. waits for every backend and load balancer to become healthy;
5. replaces each pool's URL placeholder in `COMMAND`;
6. starts `ray.sub` on hetgroup 0 only; and
7. stops training if any required external service exits.

Recipe launchers keep their concrete model and serving settings outside these
helpers. A launcher can add another service without adding another server body
or lifecycle path to this wrapper.

## Pool registration interface

Launchers source `pool_config.sh` and use the same three functions for every
model. `register_external_vllm_pool` declares topology and endpoint metadata;
the other functions append environment assignments and argv entries:

```bash
source "$PROJECT_ROOT/tools/external_gym_vllm/pool_config.sh"
EXTERNAL_VLLM_POOLS=""
export GPUS_PER_NODE=4
NUM_EXTERNAL_SERVICE_NODES=2

register_external_vllm_pool SAFETY \
  --display-name "Safety judge" \
  --model "$SAFETY_MODEL" \
  --container "$SERVICE_CONTAINER" \
  --python /opt/vllm/bin/python \
  --replicas 2 \
  --tensor-parallel-size 4 \
  --lb-port 9215 \
  --served-model-name "$SAFETY_GYM_MODEL_NAME" \
  --url-placeholder __SAFETY_BASE_URL__ \
  --shared-path "$SAFETY_REASONING_PARSER"

external_vllm_pool_env SAFETY \
  FLASHINFER_WORKSPACE_BASE=/tmp \
  NCCL_MNNVL_ENABLE=0

external_vllm_pool_args SAFETY \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN

validate_external_vllm_submission "$COMMAND" "$NUM_EXTERNAL_SERVICE_NODES"
```

Registration appends `SAFETY` to `EXTERNAL_VLLM_POOLS` and exports the
normalized `SAFETY_*` contract inherited by `sbatch`. Adding this pool requires
no change to `run_in_allocation.sh`.

The generated fields are:

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `POOL_MODEL` | yes | — | Checkpoint path under `EXTERNAL_VLLM_SHARED_ROOT` or Hugging Face model ID. |
| `POOL_CONTAINER` | yes | — | Container used by this pool's replicas. |
| `POOL_VLLM_PYTHON` | yes | — | Python executable containing vLLM, Ray, and NeMo RL's compatibility patch. |
| `POOL_REPLICAS` | yes | — | Number of independent DP=1 servers. |
| `POOL_TENSOR_PARALLEL_SIZE` | yes | — | Tensor parallel size per server. |
| `POOL_LB_PORT` | yes | — | Unique load-balancer port on the Ray head node. |
| `POOL_URL_PLACEHOLDER` | yes | — | Token in `COMMAND` replaced by this pool's `/v1` URL. |
| `POOL_GROUP_ID` | no | `inline-<pool>-<job-id>` | Registry namespace; set with `--group-id` only when an explicit stable namespace is needed. |
| `POOL_DISPLAY_NAME` | no | pool name | Human-readable log label. |
| `POOL_SERVED_MODEL_NAME` | no | `model` | OpenAI API model name. Must equal the calling Gym server's `model` field because Gym overwrites the request model. |
| `POOL_VLLM_PORT` | no | `8000` | Backend HTTP port; replicas use disjoint private clusters. |
| `POOL_STARTUP_TIMEOUT` | no | `3600` | Seconds allowed for startup. |
| `POOL_SHARED_PATHS` | no | empty | Newline-separated absolute paths under `EXTERNAL_VLLM_SHARED_ROOT` that the pool container must access. |
| `POOL_ENV_VARS` | no | empty | Newline-separated `NAME=value` assignments applied before vLLM starts. |
| `POOL_VLLM_ARGS` | no | empty | Newline-separated vLLM CLI arguments, one argv entry per line. |

Registration validates required fields, positive topology values, TCP port
ranges, TP divisibility, and duplicate ports/placeholders before `sbatch`.
`EXTERNAL_VLLM_NUM_NODES` is exported as the node total computed from all
registered pools. Call `validate_external_vllm_submission` after constructing
`COMMAND` to check its placeholders, shared paths, tool files, and requested
external node count before submitting the allocation.

The interface uses one-argument-per-line encoding internally, preserving JSON
configs and paths containing spaces without `eval`. The wrapper itself supplies `--tensor-parallel-size`,
`--distributed-executor-backend ray`, `--port`, and `--served-model-name`.
Everything model-specific—including attention, reasoning/tool parsers, expert
parallelism, MoE backend, cache settings, and loader settings—belongs in the
launcher's pool definition. A pool's reasoning-parser setting must also agree
with the consuming Gym server's `uses_reasoning_parser` setting; in particular,
do not pass `--reasoning-parser` when Gym explicitly disables it.

## Global contract

Required variables:

| Variable | Purpose |
|---|---|
| `BASE_LOG_DIR` | Parent under `EXTERNAL_VLLM_SHARED_ROOT` for `<job-id>-logs`. |
| `COMMAND` | NeMo RL command containing every pool's URL placeholder. |
| `CONTAINER` | NeMo RL and load-balancer container. |
| `MOUNTS` | Mount list required by `ray.sub` and `COMMAND`. |
| `EXTERNAL_VLLM_POOLS` | Ordered pool names; this order determines node slicing. |
| `EXTERNAL_VLLM_TOOLS_DIR_HOST` | Path to this directory under `EXTERNAL_VLLM_SHARED_ROOT`. |

Optional globals:

| Variable | Default | Purpose |
|---|---|---|
| `GPUS_PER_NODE` | `4` | GPUs claimed per node. Set this before registering pools; it sizes hetgroup 1 and is exported to `ray.sub`, so it must match the physical GPUs per node for both components. |
| `NUM_EXTERNAL_SERVICE_NODES` | empty | Expected hetgroup 1 node count. Pass it to `validate_external_vllm_submission` to fail before `sbatch` on a topology mismatch; validation warns and skips this check when unset. |
| `EXTERNAL_VLLM_LB_PYTHON` | `/opt/nemo_rl_venv/bin/python` | Python with `aiohttp` in `CONTAINER`. |
| `RAY_SUB` | `$SLURM_SUBMIT_DIR/ray.sub` | Normal NeMo RL Slurm entrypoint. |
| `EXTERNAL_VLLM_SHARED_ROOT` | `/lustre` | Shared host path mounted at the same path in every external-service container. |
| `DEDICATED_RAY_HEAD` | unset | Passed through to `ray.sub`. With `1`, include one extra node in hetgroup 0 while keeping `cluster.num_nodes` equal to the GPU worker-node count; the head node's GPUs remain allocated but idle. |

The number of nodes in hetgroup 1 must equal:

```text
sum(POOL_REPLICAS * POOL_TENSOR_PARALLEL_SIZE / GPUS_PER_NODE)
```

Every TP value must be divisible by `GPUS_PER_NODE`. The wrapper records the
actual split in `$BASE_LOG_DIR/<job-id>[-<restart-count>]-logs/node-allocation.txt`
and writes each resolved URL to `<pool-name-lowercase>_url` in the same
directory.

### Private-cluster port layout

Each replica uses the same fixed port layout on its disjoint node set. This
matches `ray.sub` and stays below the `9000` ephemeral-port floor observed on
GB200 nodes:

| Range | Purpose |
|---|---|
| `1200-1201` | Private Ray GCS and client server. |
| `1301-1312` | Private Ray management services. |
| `2000-2999` | Private Ray worker gRPC ports. |
| `7000-7999` | vLLM engine rendezvous, anchored by `VLLM_PORT=7000`. |
| `8000` by default | Per-pool vLLM HTTP endpoint (`POOL_VLLM_PORT`). |

The old `10002-19999` Ray worker default overlaps the `9000-65000` ephemeral
range on these nodes. A vLLM TCPStore probe could therefore select a Ray worker
port and later fail with `EADDRINUSE`. Keeping both Ray and vLLM internal ports
in disjoint sub-ephemeral bands removes that race; ports above `19999` are not
reserved for this helper.

## Filesystem and container requirements

External replicas mount `EXTERNAL_VLLM_SHARED_ROOT` at the same path inside the
container. Therefore `BASE_LOG_DIR`, `EXTERNAL_VLLM_TOOLS_DIR_HOST`, and
absolute local model paths must be under that root. A Hugging Face model ID is
also accepted.

Each pool container must provide:

- its configured `POOL_VLLM_PYTHON`;
- importable `nemo_rl`, `ray`, and `vllm` packages in that environment; and
- the `ray` command on `PATH`.

`serve_vllm_on_ray.py` applies NeMo RL's vLLM compatibility patches before it
imports the vLLM API server. `CONTAINER` must provide
`EXTERNAL_VLLM_LB_PYTHON` with `aiohttp` installed.

## Slurm submission

The wrapper requires exactly two hetgroups. Submit the NeMo RL nodes first and
the sum of all external-pool nodes second:

```bash
sbatch \
  --account=<account> \
  --partition=<partition> \
  --nodes=<nemo-rl-nodes> \
  --exclusive \
  --gres=gpu:4 \
  --time=04:00:00 \
  --export=ALL \
  : \
  --account=<account> \
  --partition=<partition> \
  --nodes=<external-pool-nodes> \
  --exclusive \
  --gres=gpu:4 \
  --time=04:00:00 \
  tools/external_gym_vllm/run_in_allocation.sh
```

Slurm gang-schedules the two components. Replica `srun` steps explicitly use
hetgroup 1 and load-balancer steps explicitly use hetgroup 0. Before starting
`ray.sub`, the wrapper scopes its unsuffixed Slurm nodelist and node-count
variables to component 0; Slurm then uses component 0 by default for its steps.
External nodes therefore cannot accidentally join the training Ray cluster.
