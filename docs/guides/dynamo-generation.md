# Managed Dynamo generation on Slurm

NeMo RL can launch and own Dynamo's control plane, frontend, and a fixed vLLM
worker fleet inside a Slurm-backed Ray allocation. This mode supports direct
GRPO and NeMo-Gym rollouts, NCCL weight refits, cache invalidation, and
`generation_metrics/*` telemetry sent to enabled loggers such as W&B.

This integration is managed and vLLM-only. It does not connect to an external
Dynamo deployment and does not support Kubernetes, DGD, SGLang, TensorRT-LLM,
speculative decoding, quantized generation, or model-parallel engine groups
that span nodes.

## Build the image

The normal image is unchanged unless `BUILD_DYNAMO` is set:

```bash
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg BUILD_DYNAMO=1 \
  --target release \
  --file docker/Dockerfile \
  --tag registry.example.com/nemo-rl:dynamo \
  .
```

The opt-in layer installs `ai-dynamo[vllm]==1.3.0.post1` in isolated Python
3.12 under `/opt/dynamo_venv`, along with etcd v3.5.21 and NATS Server v2.11.6.
It does not replace NeMo RL's normal Ray or vLLM dependencies. For a local
source checkout, the same environment can be installed under
`venvs/dynamo`:

```bash
bash docker/dynamo/install.sh
```

Set `NEMO_RL_DYNAMO_VENV_DIR` to choose another location. The installer checks
that Dynamo resolved vLLM 0.23.0, applies the vLLM PR #44814 backport only after
`git apply --check`, and writes the upstream marker to `VLLM_BACKPORTS`.

## Configure Dynamo

Start with [`examples/configs/grpo_math_1B_dynamo.yaml`](../../examples/configs/grpo_math_1B_dynamo.yaml).
The important boundary is:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      engine: vllm
      frontend_args:
        router_mode: kv
    vllm_cfg:
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      expert_parallel_size: 1
    colocated:
      enabled: false
      resources:
        gpus_per_node: 1
        num_nodes: 1
```

NeMo RL derives each engine's world size from TP times PP. EP must be one or
equal to TP. Parser settings belong under `dynamo_cfg.worker_args`; inherited
vLLM HTTP-parser settings are rejected with the corresponding Dynamo field.
Service ports and the namespace are runtime-owned rather than public config.

`vllm_cfg` settings are handled in four explicit classes:

| Class | Behavior | Examples |
| --- | --- | --- |
| Translated | Forwarded to `dynamo.vllm` | TP, PP, EP, dtype, model length |
| Moved | Startup error naming the Dynamo replacement | tool and reasoning parsers, HTTP serving chat kwargs |
| Unsupported | Warning, or an error when it requests unsupported low precision | tokenizer skipping, MX and mixed BF16/FP8 helpers |
| Inapplicable | Ignored because the managed path owns that behavior | async mode, progress bars, NeMo RL HTTP/ZMQ refit ports |

The NCCL sender also selects vLLM's peer protocol: the policy publishes both
the raw NeMo RL unique ID and vLLM's pickled `ncclUniqueId`, then uses the
all-reduce warmup expected by `PyNcclCommunicator`. This protocol choice and
the packed 1-GiB/two-buffer geometry come from the generation backend rather
than GRPO-specific branches.

The fixed port layout is:

- `1313-1399`: driver-local etcd and NATS control plane
- `3000-3999`: frontend and token-wrapper HTTP endpoints
- `4000-4099`: node-local `DYN_SYSTEM_PORT`
- `7000 + slot * 100`: node-local vLLM rendezvous ports

## Run the two-GPU smoke

Convert the image to the format required by the Slurm site, then submit from
the repository root:

```bash
export CONTAINER=/shared/images/nemo-rl-dynamo.sqsh
export MOUNTS="$PWD:$PWD"
export GPUS_PER_NODE=2
export BASE_LOG_DIR="$PWD/results/dynamo-smoke/logs"
printf -v COMMAND '%q ' \
  /opt/nemo_rl_venv/bin/python -u "$PWD/examples/run_grpo.py" \
  --config "$PWD/examples/configs/grpo_math_1B_dynamo.yaml"
export COMMAND

sbatch \
  --nodes=1 \
  --gres=gpu:2 \
  --exclusive \
  --account=<account> \
  --partition=<partition> \
  ray.sub
```

The recipe assigns one GPU to training and one to a TP1 Dynamo worker. Its two
steps exercise generation, refit, post-refit cache invalidation, telemetry,
and cleanup. For a matched control, run the same seed/model/batch settings with
the standard non-colocated vLLM backend and compare post-refit output validity.

## Run SWE with W&B

The public six-node recipe targets
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. Supply site data, sandbox images,
Slurm routing, and W&B credentials to its launcher:

```bash
CONTAINER=/shared/images/nemo-rl-dynamo.sqsh \
TRAIN_PATH=/shared/data/swe-train.jsonl \
VAL_PATH=/shared/data/swe-validation.jsonl \
SIF_FORMATTERS='["/shared/swe/{instance_id}.sif"]' \
SANDBOX_CONTAINER=/shared/images/nemo-skills-sandbox.sqsh \
SLURM_ACCOUNT=<account> \
SLURM_PARTITION=<partition> \
WANDB_API_KEY=<key> \
EXTRA_MOUNTS=/shared:/shared \
bash examples/swe_bench/run_grpo_nanov3_30ba3b_swe_dynamo_hsg_r2_wandb.sh
```

Use `DRY_RUN=1` to inspect the command without submitting. A successful
acceptance run completes four training steps, produces valid generations after
refit, and records worker timelines under `generation_metrics/*` in W&B.

## Operational notes

- The driver owns all services. Do not start a separate etcd, NATS, frontend,
  or worker fleet for this mode.
- Startup validates fixed worker membership; a dead or replaced worker fails
  refit instead of serving mixed model versions.
- Shutdown is idempotent and terminates whole subprocess groups, including
  partial-startup failures.
- Fault tolerance and a multi-controller architecture remain follow-up work.
