# Managed Dynamo generation on Slurm

NeMo RL can launch and own Dynamo's control plane, frontend, and a fixed vLLM
worker fleet inside a Slurm-backed Ray allocation. This mode supports direct
GRPO and NeMo-Gym rollouts, NCCL and `nccl_reshard` weight refits, cache invalidation, and
`generation_metrics/*` telemetry sent to enabled loggers such as W&B.
See the [Dynamo integration design](../design-docs/dynamo-integration.md) for
the service ownership, startup, and weight-refit architecture.

This integration is managed and vLLM-only. It does not connect to an external
Dynamo deployment and does not support Kubernetes, DGD, SGLang, TensorRT-LLM,
FP8 KV caches, ModelOpt quantization, or model-parallel engine groups that span
nodes. FP8 and MXFP8 weights (`vllm_cfg.precision: fp8`) and speculative
decoding are supported; see [FP8 generation](#fp8-generation) and
[Speculative decoding](#speculative-decoding).

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

`BUILD_DYNAMO=1` prebuilds the `DynamoVllmWorker` actor venv from the root
project's `dynamo` extra, and installs etcd v3.5.21 and NATS Server v2.11.6 into
`/usr/local/bin` (`docker/install_dynamo_services.sh`). The `dynamo` extra is
the `vllm` extra plus the base `ai-dynamo==1.5.1` package, so the Dynamo
workers run the same vLLM (0.29.0), NCCL and `nemo_rl` code as NeMo RL's own
vLLM workers. That is what lets each engine load NeMo RL's vLLM worker
extension for refit (see [Weight refit](#weight-refit)). Only the base
`ai-dynamo` package is installed: its `[vllm]` extra pins vLLM 0.28.0.

From a source checkout, the venv is built on first use under
`venvs/nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker`, like
every other actor venv. etcd and nats-server are looked up in
`NEMO_RL_DYNAMO_BIN_DIR` when it is set, otherwise on `PATH`;
`NEMO_RL_DYNAMO_BIN_DIR=$PWD/bin bash docker/install_dynamo_services.sh`
installs them.

A Dynamo upgrade must update and reverify these coupled locations:

- root `pyproject.toml` and `uv.lock`: the `ai-dynamo` and
  `ai-dynamo-runtime` pins in the `dynamo` extra, which must otherwise stay
  identical to the `vllm` extra (`tests/unit/test_dependency_pins.py`)
- `tests/functional/grpo_dynamo*.sh`: the asserted Dynamo and vLLM versions
- `nemo_rl/models/generation/dynamo/refit.py`: the `/engine/*` RL routes it
  posts to, which must still forward any `engine_rpc` to vLLM `collective_rpc`
  with the request body as keyword arguments
  (`test_every_engine_rpc_the_driver_sends_binds_to_the_extension`)
- `nemo_rl/models/generation/dynamo/token_wrapper.py`: the real Dynamo response
  keys `nvext.engine_data.{prompt_token_ids,completion_token_ids,completion_logprobs}`;
  reverify them against real Dynamo output because unit tests validate only the
  expected local response shape
- `nemo_rl/models/generation/dynamo/managed_runtime.py`: the managed
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

`vllm_cfg` settings are handled in seven explicit classes:

| Class | Behavior | Examples |
| --- | --- | --- |
| Translated | Forwarded to `dynamo.vllm` | TP, PP, EP, dtype, model length |
| Refit | Read by NeMo RL's worker extension inside each engine | `refit_with_reload_api` |
| Moved | Startup error naming the Dynamo replacement | tool and reasoning parsers, HTTP serving chat kwargs |
| FP8 | Read by the vLLM backend's FP8 builder when `precision: fp8`; an error otherwise | `is_mx`, mixed BF16/FP8 layers, DeepGEMM, pow2 scales |
| Unsupported | Warning when active | tokenizer skipping, context-length token capping |
| Managed runtime | Consumed or enforced by NeMo RL rather than forwarded | HTTP-wrapper enablement, metrics sampling, processed rollout logprobs |
| Inapplicable | Ignored because the managed path owns that behavior | async mode, progress bars, NeMo RL HTTP/ZMQ refit ports |

The shared GRPO base config also supplies `mcore_generation_config` and
`refit_cfg`. Dynamo accepts these inherited sections but does not use them;
worker arguments come from `vllm_cfg`. Any other active top-level
`policy.generation` setting that Dynamo does not honor, such as `bad_words` or
`ignore_eos`, draws a startup warning.

Set `policy.generation.refit_transport` to `null` for the packed NCCL broadcast
(`CollectiveWeightSynchronizer`), or to `nccl_reshard` to send each Megatron
training shard straight into the Dynamo engines' shards
([NCCL reshard refit](../design-docs/nccl-reshard-refit.md)). `nccl_reshard`
has the same requirements as with the vLLM backend: a Megatron policy, train
ETP 1, generation PP 1, and EP equal to 1 or TP. Other transports are rejected.

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

## FP8 generation

`vllm_cfg.precision: fp8` (optionally with `is_mx`, mixed BF16 layers, DeepGEMM
or pow2 scales) quantizes generation weights exactly as the vLLM backend does.
The Dynamo worker actor runs the vLLM backend's FP8 builder and passes the
result to `dynamo.vllm` as `--quantization fp8` and `--hf-overrides`; NeMo RL's
worker extension applies the FP8 refit patches inside each engine before it
loads the model. Every refit then sends BF16 weights that the engine quantizes
on receive. As with the vLLM backend, importance-sampling correction is
required. FP8 KV caches stay unsupported, because their scales are refit only
with synchronous rollouts.

## Speculative decoding

Set `policy.generation.vllm_kwargs.speculative_config` exactly as for the vLLM
backend; it is passed to `dynamo.vllm` as `--speculative-config`, and
`num_speculative_tokens: 0` turns it off. Drafter weights follow the vLLM
backend's rules, because the same worker extension receives every refit:

- A co-trained MTP head is fed by every refit.
- Otherwise an MTP drafter is read from the checkpoint once at startup
  (`load_mtp_weights_from_disk`), so dummy-loaded engines still draft correctly.
- Other methods (for example Eagle3) without a refit-fed draft force
  `load_format: auto`.

Request budgets stop `num_speculative_tokens + 1` tokens short of
`max_model_len`, so the drafter never reads past the context.

## Weight refit

Every managed engine loads NeMo RL's vLLM worker extension
(`--worker-extension-cls`), the same receiver the vLLM backend uses. The driver
calls it through Dynamo's per-worker `/engine/*` RL routes, which forward an
`engine_rpc` name to vLLM `collective_rpc`. HTTP carries only control calls;
the weights move over NCCL, using NeMo RL's own communicator protocol and
packing on both sides. Each refit has a deadline: GRPO uses
`dynamo_cfg.control_timeout_s`, and the SingleController uses
`async_rl.generation_fleet_health.refit_timeout_s` (300 s by default; `null`
disables it). When it passes, the extension and the trainers each abort their
own communicators and raise instead of hanging.

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
- `async_rl.generation_fleet_health.enabled` must stay false, because the
  Dynamo frontend owns request routing. Its `refit_timeout_s` is honored: the
  worker extension in each engine enforces it.
- `dynamo_cfg.control_timeout_s` must be at least `dynamo_cfg.request_timeout_s`,
  because it bounds the drain.
- `vllm_cfg.enable_vllm_metrics_logger` requires
  `rollout_checkpointing.telemetry_interval_s`: the rollout telemetry pump drains
  the Dynamo metrics sampler and reports each engine's generated tokens, running
  and waiting requests, and KV-cache usage.

Setup rejects any of these before allocating clusters.

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
