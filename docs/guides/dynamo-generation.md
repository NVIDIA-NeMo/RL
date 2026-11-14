# Dynamo serving for vLLM and Megatron generation

SingleController runs (`examples/run_grpo_single_controller.py`) with non-colocated
vLLM generation can serve their rollouts through
[Dynamo](https://github.com/ai-dynamo/dynamo)'s OpenAI frontend and KV-aware
router. Add one line:

```yaml
policy:
  generation:
    backend: vllm
    dynamo: true
```

The same NeMo RL vLLM workers run the same engines, with the same refit
transports, pause and resume, quantization, speculative decoding and metrics.
Each engine also registers with a managed `dynamo.frontend`, and NeMo RL's own
rollouts (`generate_async`) go through that frontend, so each request lands on
the engine that already holds most of its prompt's KV cache. The few settings
Dynamo cannot serve fail at startup; see [Not supported](#not-supported).

The Megatron backend can also serve its rollouts through the same frontend,
split into prefill and decode engines; see
[Run Megatron prefill/decode engines under the SingleController](#run-megatron-prefilldecode-engines-under-the-singlecontroller).

See the [Dynamo integration design](../design-docs/dynamo-integration.md) for
how the engines are hosted and why Dynamo upgrades and vLLM bumps stay
independent.

## What happens at startup

With `dynamo: true`:

- `vllm_cfg.async_engine` is turned on, because Dynamo drives the engine's
  `AsyncLLM`. Settings Dynamo does not serve are rejected at startup; see
  [Not supported](#not-supported).
- The driver starts etcd, NATS and `dynamo.frontend`. The frontend runs from the
  vLLM workers' own venv, so it is the same ai-dynamo release.
- Each engine turns on vLLM's KV-cache events for `router_mode: kv` and
  registers with the frontend from inside its own worker process. Setup waits
  until the frontend routes to every engine.
- vLLM publishes KV-cache events only for blocks its prefix cache stores. With
  `vllm_cfg.enable_prefix_caching: false`, the engines publish none, and the
  KV router routes on load alone.
- Shutdown stops the services after the engines.

## Configure

`dynamo: true` is a complete configuration. `policy.generation.dynamo_cfg` only
tunes the frontend and registration. Its fields and defaults live in
`nemo_rl/models/generation/dynamo/config.py`, and
[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
shows them all:

| Key | Default | Meaning |
|---|---|---|
| `startup_timeout_s` | 600 | Bounds service start-up and engine registration |
| `request_timeout_s` | null | Deadline for one NeMo RL rollout through the frontend. null waits as long as generation takes, as native does; on the SingleController, `async_rl.rollout_failure.native.generation_timeout_s` bounds both paths |
| `frontend_args.router_mode` | `kv` | Frontend routing policy |
| `frontend_args.extra_cli_args` | `[]` | Extra `dynamo.frontend` flags |

The etcd and nats-server binaries are found in `NEMO_RL_DYNAMO_BIN_DIR` when it
is set (the image sets it to `/opt/dynamo/bin`), and on `PATH` otherwise.

## Which requests go through the frontend

| Request | Path |
|---|---|
| `generate_async` on text token prompts, router replay (R3) included | Frontend (KV-aware); the engine samples and assembles it exactly as native `generate_async` does |
| NeMo RL's own multimodal rollouts | Native shard dispatch. Those rollouts send media to the engine directly rather than as OpenAI content |
| `generate_text_async`, which evaluation uses | Native shard dispatch |
| NeMo-Gym's OpenAI requests | Not served yet; see [Not supported](#not-supported) |

A rollout through the frontend has the same `GenerationOutputSpec` as the
native path: token IDs, log probabilities, truncation, routed experts with
their `r3_routed_experts_*` counts, and `gen_leader_worker_idx`. Both paths
build it with `nemo_rl/models/generation/vllm/request_pipeline.py`.

The engines serve only these rollouts; the frontend refuses other requests.

## Not supported

Startup rejects these with `dynamo: true`:

- NeMo-Gym (`env.should_use_nemo_gym`): the frontend does not serve Gym's
  calls yet. Run Gym recipes with `dynamo: false`.
- `colocated.enabled: true`: a colocated engine sleeps while the policy trains,
  and the frontend would keep routing to it.
- `async_rl.generation_fleet_health.enabled`: the frontend picks engines itself,
  and a restarted shard does not register with it yet.
- `vllm_cfg.expert_parallel_size > tensor_parallel_size`: that runs data
  parallelism inside vLLM, which Dynamo does not serve yet. Above data-parallel
  rank 0, vLLM rejects each engine's `ipc://` KV-event endpoint.
- `vllm_cfg.http_server_api_key_required`: the frontend serves the same engines
  on 0.0.0.0 and checks no bearer token.
- `frontend_args.router_mode: direct`: it needs a worker ID on every request, and
  NeMo RL's rollouts name none.
- `dynamo: true` on a backend other than vLLM or Megatron, and a `dynamo_cfg`
  block without `dynamo: true`. The Megatron backend has its own limits; see
  [Run Megatron prefill/decode engines under the SingleController](#run-megatron-prefilldecode-engines-under-the-singlecontroller).

## Build the image

ai-dynamo is part of the `vllm` extra, so every vLLM worker venv carries it.
`BUILD_DYNAMO=1` adds only the etcd and nats-server binaries, under
`/opt/dynamo/bin`:

```bash
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg BUILD_DYNAMO=1 \
  --target release \
  --file docker/Dockerfile \
  --tag registry.example.com/nemo-rl:dynamo \
  .
```

For a local checkout, install the binaries and keep the variable set for the
runs that follow. The runtime looks for etcd and nats-server there, or else on `PATH`:

```bash
export NEMO_RL_DYNAMO_BIN_DIR="$PWD/venvs/dynamo-bin"
bash docker/dynamo/install.sh
```

## Run

Use any non-colocated SingleController vLLM recipe that does not use NeMo-Gym,
and add the switch:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
  ++policy.generation.dynamo=true
```

[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
is the same run with every `dynamo_cfg` default spelled out.

## Run Megatron prefill/decode engines under the SingleController

The SingleController can also put NeMo RL's own Megatron inference engines
behind the Dynamo frontend, split into a prefill pool and a decode pool. This
is the Megatron backend with the same `dynamo: true` switch as the vLLM path
above; the vLLM-only sections above do not apply to it. Start from
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
  (`dynamo/managed_runtime.py`, the same runtime the vLLM path uses), started by
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
  than leaving the fleet half stopped. Resume brings every engine back in one
  wave: the engines keep their KV on the GPU across suspend and resume, and a
  handoff that reaches a decode engine still waking is queued, so no role
  order is needed.
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
  up in `NEMO_RL_DYNAMO_BIN_DIR` when it is set (only there); otherwise on
  `PATH`, then in the image's Dynamo virtual environment.

The functional test is
`tests/functional/grpo_dynamo_pd_megatron_single_controller.sh` (three GPUs:
trainer, prefill, decode). It is not in the L1 Dynamo suite until the pinned
Megatron-LM includes the Dynamo integration.

## Migrating from `backend: dynamo`

`backend: dynamo` launched Dynamo's own vLLM engine and was removed. Set
`backend: vllm` and `dynamo: true`; `vllm_cfg` and `vllm_kwargs` keep their
meaning. Startup rejects the old backend and each removed `dynamo_cfg` key
(`engine`, `control_timeout_s`, `metrics_*_prefixes`, `worker_args`,
`frontend_args.tokenizer*`) with what replaces it.
