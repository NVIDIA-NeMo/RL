# Managed Dynamo generation on Slurm

NeMo RL can launch and own Dynamo's control plane, frontend, and a fixed vLLM
worker fleet inside a Slurm-backed Ray allocation. This mode supports direct
GRPO and NeMo-Gym rollouts, NCCL weight refits, cache invalidation, and
`generation_metrics/*` telemetry sent to enabled loggers such as W&B.
See the [Dynamo integration design](../design-docs/dynamo-integration.md) for
the service ownership, startup, and weight-refit architecture.

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
It does not replace NeMo RL's normal Ray or vLLM dependencies: the standard
NeMo RL vLLM environment currently uses vLLM 0.25.1, while this isolated
Dynamo environment uses Dynamo's vLLM 0.23.0 pin. Both environments pin
`nvidia-nccl-cu13==2.30.7` so their NCCL communicators use the same release.
For a local source checkout, the same environment can be installed under
`venvs/dynamo`:

```bash
bash docker/dynamo/install.sh
```

Set `NEMO_RL_DYNAMO_VENV_DIR` to choose another location. The installer checks
that Dynamo resolved vLLM 0.23.0 and NCCL 2.30.7, applies the vLLM PR #44814
backport only after `git apply --check`, and writes the upstream marker to
`VLLM_BACKPORTS`.

Treat the isolated dependency pin and backport as one update. A Dynamo upgrade
must update and reverify these coupled locations:

- `docker/dynamo/pyproject.toml` and `docker/dynamo/uv.lock`: the Dynamo pin
  and resolved dependency set
- root `pyproject.toml` and `uv.lock`, plus the isolated Dynamo project and
  lockfile: the `nvidia-nccl-cu13` pins must remain identical
- `docker/dynamo/install.sh` and `tests/functional/grpo_dynamo.sh`: the vLLM
  version, backport marker, and runtime assertions
- `docker/dynamo/patches/vllm-0.23.0-layerwise-reload-composed-loader.patch`:
  the version-specific #44814 backport
- `tests/unit/distributed/test_stateless_process_group.py`: vLLM's
  `broadcast_from/0/0` weight-transfer wire key
- `nemo_rl/models/generation/dynamo/token_wrapper.py`: the real Dynamo response
  keys `nvext.engine_data.{prompt_token_ids,completion_token_ids,completion_logprobs}`;
  reverify them against real Dynamo output because unit tests validate only the
  expected local response shape
- `nemo_rl/models/generation/dynamo/managed_runtime.py`: the managed
  `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=1` setting. Dynamo 1.3.0's legacy tool
  jail removes `nvext.engine_data`; remove this setting only after an upgraded
  Dynamo preserves the token metadata for `tool_choice=auto`
- this guide and the Dynamo design document: the stated versions and backport
  behavior

If the new Dynamo vLLM pin contains PR #44814, delete the patch file,
patch-application block, marker assertion, and explanatory backport text. Do
not rebase the patch onto the newer vLLM release.

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

`vllm_cfg` settings are handled in five explicit classes:

| Class | Behavior | Examples |
| --- | --- | --- |
| Translated | Forwarded to `dynamo.vllm` | TP, PP, EP, dtype, model length |
| Moved | Startup error naming the Dynamo replacement | tool and reasoning parsers, HTTP serving chat kwargs |
| Unsupported | Warning when active, or an error when it requests unsupported low precision | tokenizer skipping, MX and mixed BF16/FP8 helpers |
| Managed runtime | Consumed or enforced by NeMo RL rather than forwarded | HTTP-wrapper enablement, metrics sampling, processed rollout logprobs |
| Inapplicable | Ignored because the managed path owns that behavior | async mode, progress bars, NeMo RL HTTP/ZMQ refit ports |

The shared GRPO base config also supplies `mcore_generation_config` and
`refit_cfg`. Dynamo accepts these inherited sections but does not use them;
worker arguments come from `vllm_cfg`, and refit uses the collective weight
synchronizer.

Enable and filter worker telemetry with both managed configuration sections:

```yaml
policy:
  generation:
    vllm_cfg:
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 1.0
    dynamo_cfg:
      metrics_include_prefixes: null  # null selects the curated defaults
      metrics_exclude_prefixes: null  # null excludes python_ and process_
```

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

## Run SWE1 with W&B

The three-node nightly recipe targets
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. It uses two 8-GPU training
nodes and one 8-GPU inference node. The inference node runs two TP4/EP4 Dynamo
engines. Download the standard SWE1 split under
`${HF_HOME}/superv3_data/swe1`, then run the registered test-suite driver:

```bash
HF_HOME=/shared/huggingface \
WANDB_API_KEY=<key> \
bash tests/test_suites/llm/grpo-nanov3-30ba3b-3n8g-megatron-dynamo-swe1.sh
```

A successful acceptance run completes four training steps, produces valid
generations after refit, and records worker timelines under
`generation_metrics/*` in TensorBoard and W&B.

## Operational notes

- The driver owns all services. Do not start a separate etcd, NATS, frontend,
  or worker fleet for this mode.
- Startup validates fixed worker membership; a dead or replaced worker fails
  refit instead of serving mixed model versions.
- Shutdown is idempotent and terminates whole subprocess groups, including
  partial-startup failures.
- Fault tolerance and a multi-controller architecture remain follow-up work.
