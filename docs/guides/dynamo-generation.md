# Dynamo serving for vLLM generation

Any non-colocated vLLM generation config can serve its rollouts through
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
  and the frontend would keep routing to it. The SingleController never runs
  colocated vLLM; the other entrypoints need non-colocated generation.
- `async_rl.generation_fleet_health.enabled`: the frontend picks engines itself,
  and a restarted shard does not register with it yet.
- `vllm_cfg.expert_parallel_size > tensor_parallel_size`: that runs data
  parallelism inside vLLM, which Dynamo does not serve yet. Above data-parallel
  rank 0, vLLM rejects each engine's `ipc://` KV-event endpoint.
- `vllm_cfg.http_server_api_key_required`: the frontend serves the same engines
  on 0.0.0.0 and checks no bearer token.
- `frontend_args.router_mode: direct`: it needs a worker ID on every request, and
  NeMo RL's rollouts name none.
- `dynamo: true` on a backend other than vLLM, and a `dynamo_cfg` block without
  `dynamo: true`.

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

Use any non-colocated vLLM recipe that does not use NeMo-Gym, and add the switch:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
  ++policy.generation.dynamo=true
```

[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
is the same run with every `dynamo_cfg` default spelled out.

## Migrating from `backend: dynamo`

`backend: dynamo` launched Dynamo's own vLLM engine and was removed. Startup
rejects it with the migration:

- Set `backend: vllm` and `dynamo: true`. `vllm_cfg` and `vllm_kwargs` keep their
  meaning, and every vLLM feature is now available.
- Remove these from `dynamo_cfg`:
  - `engine`;
  - `control_timeout_s`: refit and pause use the vLLM backend's controls;
  - `metrics_include_prefixes` / `metrics_exclude_prefixes`: use
    `vllm_cfg.enable_vllm_metrics_logger`;
  - `worker_args`: the engines serve only NeMo RL's rollouts, which need no chat
    parsers or templates; engines are configured through `vllm_cfg` and
    `vllm_kwargs`;
  - `frontend_args.tokenizer`, `tokenizer_cache` and `tokenizer_cache_bytes`:
    rollouts send token IDs, which the frontend never tokenizes.
