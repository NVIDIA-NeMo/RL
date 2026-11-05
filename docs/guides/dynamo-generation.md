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

The opt-in layer installs `ai-dynamo[vllm]==1.5.1` in isolated Python
3.12 under `/opt/dynamo_venv`, along with etcd v3.5.21 and NATS Server v2.11.6.
It does not replace NeMo RL's normal Ray or vLLM dependencies: the standard
NeMo RL vLLM environment currently uses vLLM 0.29.0, while this isolated
Dynamo environment uses Dynamo's vLLM 0.28.0 pin. Both environments pin
`nvidia-nccl-cu13==2.30.7` so their NCCL communicators use the same release.
For a local source checkout, the same environment can be installed under
`venvs/dynamo`:

```bash
bash docker/dynamo/install.sh
```

Set `NEMO_RL_DYNAMO_VENV_DIR` to choose another location. The installer checks
that Dynamo resolved vLLM 0.28.0 and NCCL 2.30.7, and that the installed vLLM
contains the PR #44814 layerwise-reload fix.

A Dynamo upgrade must update and reverify these coupled locations:

- `docker/dynamo/pyproject.toml` and `docker/dynamo/uv.lock`: the Dynamo pin
  and resolved dependency set
- root `pyproject.toml` and `uv.lock`, plus the isolated Dynamo project and
  lockfile: the `nvidia-nccl-cu13` pins must remain identical
- `docker/dynamo/install.sh`, `tests/functional/grpo_dynamo.sh`, and
  `tests/functional/grpo_dynamo_single_controller.sh`: the vLLM version, the
  #44814 fix check, and runtime assertions
- `nemo_rl/models/generation/dynamo/refit.py` and
  `validate_dynamo_vllm_args.py`: the vLLM weight-transfer payloads (packing is
  declared in the init info; update rounds carry names, dtypes, and shapes;
  `start_weight_update` takes no arguments)
- `tests/unit/distributed/test_stateless_process_group.py`: vLLM's
  `broadcast_from/0/0` weight-transfer wire key
- `nemo_rl/models/generation/dynamo/token_wrapper.py`: the real Dynamo response
  keys `nvext.engine_data.{prompt_token_ids,completion_token_ids,completion_logprobs}`;
  reverify them against real Dynamo output because unit tests validate only the
  expected local response shape
- `nemo_rl/models/generation/dynamo/legacy_managed_runtime.py`: the managed
  `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=1` setting. Dynamo 1.3.0's legacy tool
  jail removes `nvext.engine_data`; remove this setting only after an upgraded
  Dynamo preserves the token metadata for `tool_choice=auto`
- this guide and the Dynamo design document: the stated versions

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
synchronizer. Any other active top-level `policy.generation` setting that
Dynamo does not honor, such as `bad_words` or `ignore_eos`, draws a startup
warning.

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

## Run under the SingleController

The async SingleController trainer can also use managed Dynamo. It follows the
same data path as the native vLLM backend: the SingleController sends each turn
to the Dynamo frontend, builds the rollout record from the response, and commits
it to the TransferQueue data plane, from which the policy workers read their
shards. No Dynamo-side component writes to TransferQueue.

Start from
[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
and launch it with `examples/run_grpo_single_controller.py`. Before every weight
update the SingleController drains and pauses each Dynamo worker (requests the
engine is running finish on the old weights; queued and new requests wait and
run on the new weights), runs the NCCL refit, and then resumes the workers. With
`async_rl.recompute_kv_cache_after_weight_updates` (the default), the drain also
clears each worker's KV and prefix cache. The drain is bounded by
`dynamo_cfg.control_timeout_s`.

Current limits on this path:

- Native rollouts only. NeMo-Gym under the SingleController supports only
  `backend: vllm` and `backend: megatron`.
- Text policies only, and no MoE router replay: `policy.is_vlm` and
  `policy.router_replay.enabled` must be false.
- Fleet health is not supported: keep `async_rl.generation_fleet_health.enabled`
  false (the Dynamo frontend owns request routing) and set
  `async_rl.generation_fleet_health.refit_timeout_s: null` (default 300), since
  Dynamo workers cannot abort their refit collective.
- `dynamo_cfg.control_timeout_s` must be at least `dynamo_cfg.request_timeout_s`,
  because it bounds the drain.
- `vllm_cfg.enable_vllm_metrics_logger` requires
  `rollout_checkpointing.telemetry_interval_s`: the rollout telemetry pump drains
  the Dynamo metrics sampler and reports each engine's generated tokens, running
  and waiting requests, and KV-cache usage.

Setup rejects any of these before allocating clusters.

## Run Megatron prefill/decode engines under the SingleController

The SingleController can also put NeMo RL's own Megatron inference engines
behind the Dynamo frontend, split into a prefill pool and a decode pool. This
is the Megatron backend with one extra switch, not the managed vLLM fleet
above. Start from
[`examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml)
and launch it with `examples/run_grpo_single_controller.py`. The config
boundary is:

```yaml
async_rl:
  recompute_kv_cache_after_weight_updates: false   # the engines keep their caches
policy:
  megatron_cfg:
    enabled: true            # the engines receive logical weights from the Megatron trainer
  generation:
    backend: megatron
    dynamo: true             # serve the engines through a managed Dynamo frontend
    refit_transport: null    # collective refit only
    dynamo_cfg: {}           # frontend args, timeouts, per-role layouts; all defaulted
    disaggregation:
      enabled: true
      num_prefill_engines: 1
      num_decode_engines: 1
    mcore_generation_config:
      enable_prefix_caching: true
      kv_cache_management_mode: persist
      expose_http_server: false
    colocated:
      enabled: false
```

What runs:

- **Engines.** `disaggregation` tiles the inference GPUs into replicas of
  `num_prefill_engines` prefill engines followed by `num_decode_engines` decode
  engines. Each engine slot becomes one dedicated Megatron inference policy
  (its own worker group, data parallel one), the same object `backend:
  megatron` builds when `colocated.enabled` is false. Every engine has the
  layout of `policy.megatron_cfg` with `mcore_generation_config` applied, plus
  its role's `dynamo_cfg.prefill_engine` / `decode_engine` overrides (below);
  each engine's width (TP x PP) must fit on one node.
- **Dynamo endpoint on rank zero.** Once an engine is built, its rank zero
  also hosts Megatron-LM's Dynamo adapter
  (`megatron.inference.integrations.dynamo`), an implementation of Dynamo's
  `LLMEngine`, driven by a Dynamo `Worker` task on the worker's inference loop,
  as the vLLM backend hosts its engines. The engine's role comes from its
  config. In external launch mode the adapter starts nothing: it waits for the
  engine's ready message, connects an `InferenceClient` to the coordinator,
  registers the engine with the frontend,
  forwards requests to the coordinator, and carries the KV handoff between
  prefill and decode over NIXL. The `mcore` extra carries `ai-dynamo` for this.
- **Frontend, etcd, NATS.** Driver-owned `ManagedDynamoRuntime`
  (`dynamo/managed_runtime.py`; the vLLM `DynamoGeneration` still uses
  `LegacyManagedDynamoRuntime`), started by
  `MegatronDynamoGeneration` once the engines' workers exist. The frontend runs from
  the Megatron workers' environment. Prefill
  engines register as the `prefill` component and decode engines as `backend`;
  the frontend's prefill router splits each request between them. Rollouts are
  posted to the frontend as token-ID completions; the sampled token IDs and
  log-probabilities come back as on the direct path.

Prefill and decode engines may use different layouts. `dynamo_cfg.prefill_engine`
and `dynamo_cfg.decode_engine` override `tensor_model_parallel_size` and
`pipeline_model_parallel_size` for one role (a `null` field keeps the shared
layout); Megatron reshards the KV cache during the handoff, so the two sides
need not match. For example, one prefill engine at TP 4 next to two decode
engines at TP 2 is an 8-GPU replica:

```yaml
policy:
  generation:
    dynamo_cfg:
      prefill_engine:
        tensor_model_parallel_size: 4
    disaggregation:
      enabled: true
      num_prefill_engines: 1
      num_decode_engines: 2
    mcore_generation_config:
      tensor_model_parallel_size: 2
```

With `T` trainer ranks, the refit collective places the trainer ranks first,
then the engines in launch order, each starting where the previous one ended:
the prefill engine at `T..T+3`, the decode engines at `T+4..T+5` and
`T+6..T+7`.

Refit and startup follow the Megatron weight synchronizer's sequence
(suspend, prepare the weights, transfer, prepare the KV cache, resume), with
the fleet standing in for one dedicated engine:

- The engines are built at the first weight update (they start without
  weights), so their Dynamo endpoints start and register right after that
  first refit, and the first resume waits for the frontend to list every
  engine and the model before rollouts start. `dynamo_cfg.startup_timeout_s`
  bounds the services' start-up and that registration.
- A request spans two engines, and a prefill engine must not be suspended while
  a decode engine may still be pulling one of its KV blocks (a "handoff pin").
  So the suspend stops the prefill engines from scheduling first, keeps decode
  running until every pin on prefill has cleared, then suspends prefill, then
  decode, all within `dynamo_cfg.control_timeout_s`. If the pins do not clear
  in time the prefill engines are unpaused again and the refit fails rather
  than leaving the fleet half stopped. Resume brings decode back first, then
  prefill, because decode is the reader of prefill's KV.
- The collective is one NCCL group over the trainers plus all engines. The
  engines receive logical (unfused) weights, as the Megatron backend does.
- Shutdown stops decode engines first, confirms their termination to every
  prefill coordinator so pinned blocks are released, then stops prefill, then
  the frontend, NATS and etcd.

Limits:

- Prefill/decode only: aggregated Megatron engines behind Dynamo are rejected
  (leave `dynamo` false for that).
- Native rollouts only; NeMo-Gym through the frontend is follow-up work, so
  `env.should_use_nemo_gym` must be false.
- Text prompts only (`policy.is_vlm` false); no router replay.
- Each engine fits on one node; `refit_transport` must be null;
  `recompute_kv_cache_after_weight_updates` must be false (the engines keep
  their caches, mode `persist`); setup rejects these otherwise.
- Data-plane storage units cannot be placed on the `inference` cluster
  (`storage_unit_placement: all` or `[inference]`): the engines reserve those
  GPUs through their own clusters. Place them on `train`.
- The Megatron environment needs the `nixl` package for the KV handoff and a
  Megatron-LM with the Dynamo integration
  (`megatron/inference/integrations/dynamo`). etcd and nats-server are looked
  up in `NEMO_RL_DYNAMO_BIN_DIR`, then on `PATH`, then in the image's Dynamo
  virtual environment.

The functional test is
`tests/functional/grpo_dynamo_pd_megatron_single_controller.sh` (three GPUs:
trainer, prefill, decode). It is not in the L1 Dynamo suite until the pinned
Megatron-LM includes the Dynamo integration.

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
