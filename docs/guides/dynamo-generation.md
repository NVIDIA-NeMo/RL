# Dynamo serving for vLLM and Megatron generation

SingleController runs (`examples/run_grpo_single_controller.py`) with non-colocated vLLM generation can serve their rollouts through [Dynamo](https://github.com/ai-dynamo/dynamo)'s OpenAI frontend and KV-aware router. Add one line:

```yaml
policy:
  generation:
    backend: vllm
    dynamo: true
```

The same NeMo RL vLLM workers run the same engines, with the same refit transports, pause and resume, quantization, speculative decoding and metrics.
Each engine also registers with a managed `dynamo.frontend`, and NeMo RL's own rollouts (`generate_async`) go through that frontend, so each request lands on the engine that already holds most of its prompt's KV cache. The few settings Dynamo cannot serve fail at startup; see [Not supported](#not-supported).

The Megatron backend can also serve its rollouts through the same frontend, split into prefill and decode engines; see [Run Megatron prefill/decode engines under the SingleController](#run-megatron-prefilldecode-engines-under-the-singlecontroller).

See the [Dynamo integration design](../design-docs/dynamo-integration.md) for how the engines are hosted and why Dynamo upgrades and vLLM bumps stay independent.

## What happens at startup

With `dynamo: true`:

- `vllm_cfg.async_engine` is turned on, because Dynamo drives the engine's `AsyncLLM`. Settings Dynamo does not serve are rejected at startup; see [Not supported](#not-supported).
- The driver starts etcd, NATS and `dynamo.frontend`. The frontend runs from the vLLM workers' own venv, so it is the same ai-dynamo release.
- Each engine turns on vLLM's KV-cache events for `router_mode: kv` and registers with the frontend from inside its own worker process. Setup waits until the frontend routes to every engine.
- With NeMo-Gym (`vllm_cfg.expose_http_server: true`), the frontend starts its render workers at launch and builds them while the engines load, from the settings one engine sends before it loads. Gym's first calls then do not wait for them. Other runs start none.
- vLLM publishes KV-cache events only for blocks its prefix cache stores. With `vllm_cfg.enable_prefix_caching: false`, the engines publish none, and the KV router routes on load alone.
- Shutdown stops the services after the engines.

## Configure

`dynamo: true` is a complete configuration. `policy.generation.dynamo_cfg` only tunes the frontend and registration. Its fields and defaults live in `nemo_rl/models/generation/dynamo/config.py`, and [`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml) shows them all:

| Key | Default | Meaning |
|---|---|---|
| `startup_timeout_s` | 600 | Bounds service start-up and engine registration |
| `request_timeout_s` | null | Deadline for one NeMo RL rollout through the frontend. null waits as long as generation takes, as native does; on the SingleController, `async_rl.rollout_failure.native.generation_timeout_s` bounds both paths |
| `frontend_args.router_mode` | `kv` | Frontend routing policy |
| `frontend_args.session_affinity_ttl_secs` | 300 | Keeps a NeMo-Gym token-capture rollout on one engine while it is idle for up to this many seconds; null disables it. Takes effect from Dynamo 1.7: Dynamo 1.5.1 drops the session id of a chat call served by NeMo RL's chat engine. Dynamo keeps at most 65,536 sessions and answers a new one past that with HTTP 529 |
| `frontend_args.render_workers` | 8 | Processes in the frontend that render NeMo-Gym's chat calls and build their replies; every Gym call in the fleet shares them. Each holds its own tokenizer and processor, about 1.7–2.3 GB. On one Grace core of a GB200, one serves about 640 single-turn calls/s, 45 for a 21K-token agentic history (22 with its prefix splice), or 200 with 1–4 images |
| `frontend_args.extra_cli_args` | `[]` | Extra `dynamo.frontend` flags. `--dyn-chat-processor` and `--custom-jinja-template` are rejected: NeMo RL renders chat itself |

The etcd and nats-server binaries are found in `NEMO_RL_DYNAMO_BIN_DIR` when it is set (the image sets it to `/opt/dynamo/bin`), and on `PATH` otherwise.

## Which requests go through the frontend

| Request | Path |
|---|---|
| `generate_async` on text token prompts, router replay (R3) included | Frontend (KV-aware); the engine samples and assembles it exactly as native `generate_async` does |
| NeMo RL's own multimodal rollouts | Native shard dispatch. Those rollouts send media to the engine directly rather than as OpenAI content |
| `generate_text_async`, which evaluation uses | Native shard dispatch |
| NeMo-Gym's chat calls | Frontend, rendered by NeMo RL's chat engine with the native server's code. See [NeMo-Gym](#nemo-gym) |

A rollout through the frontend has the same `GenerationOutputSpec` as the native path: token IDs, log probabilities, truncation, routed experts with their `r3_routed_experts_*` counts, and `gen_leader_worker_idx`. Both paths build it with `nemo_rl/models/generation/vllm/request_pipeline.py`.

## NeMo-Gym

NeMo-Gym's policy model calls the frontend. Setup turns on Gym's Dynamo dialect (`dynamo_frontend` in Gym's [`vllm_model`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_models/vllm_model) config). The frontend takes only OpenAI fields plus `nvext`, so Gym sends the native request's extra fields (`ng_capture`, `required_prefix_token_ids`), and `ng_measure` for the measure call it makes in place of `/tokenize`, in one `nemo_rl:gym:` annotation, and reads the call's token fields (`prompt_token_ids`, `generation_token_ids`, `generation_log_probs`, `routed_experts`, `ng_commit_coords`) from `nvext.nemo_rl`.

The frontend serves chat with NeMo RL's chat engine instead of Dynamo's own renderer and parsers (Dynamo's `chat_engine_factory` hook). Each call takes the native server's steps, with the native server's code:

1. In the frontend, a render-pool process renders the call with the engine's own tokenizer, template, processor and `vllm_cfg` chat settings (parser plugins included), which each engine publishes when it registers. It also decides how a token-capture call's prefix is served.
2. The chat engine checks on-policy sampling, reads a captured prefix from TQ, splices the prefix into the render, clamps the output budget to the context, and routes the final token IDs through the KV router.
3. On the engine, the adapter snapshots the call's media, begins its capture, generates with the `SamplingParams` the native server builds, and stages the call in one TQ write. Only the commit coordinates travel back.
4. A render-pool process builds the reply with vLLM's reasoning and tool parsers.

`tests/unit/models/generation/test_gym_chat_path.py` sends the same calls down both paths and checks they run the same prompt and `SamplingParams`, stage the same record and return the same reply.

Where the behavior differs from the native endpoint:

- The frontend refuses more than 32 `stop` strings, and vLLM request fields its own request type lacks (for example `structured_outputs`, `return_token_ids`, `truncate_prompt_tokens`, `priority`, `chat_template`), with HTTP 400.
- gpt-oss (Harmony) chat calls are refused with HTTP 400.
- A call that streams (`stream: true`) or asks for more than one choice (`n`) is refused with HTTP 400. The native endpoint serves both, and fails the capture of a streamed token-capture call. NeMo-Gym sends neither.
- A token-capture call refused for its context length acknowledges the refusal at the end of the error message instead of in the native endpoint's `x-ng-capture-refused` header, which Dynamo's errors cannot carry. Gym's dialect moves it back to the header, so the harness sees the native error.
- A call's processed media travels to the engine inside the request.
- A call NeMo-Gym marks as untrained, such as an LLM judge's on the policy model, keeps its own sampling on the native endpoint, which reads the mark from the `x-ng-sampling: null` header. Dynamo's frontend does not forward that header, so NeMo RL's chat engine reads the mark from the `nemo_rl:gym:` annotation (`ng_sampling`). The pinned NeMo-Gym does not send it there yet: until it does, such a call is refused with HTTP 400 unless it matches the training or validation sampling. Give a judge that samples differently a model server of its own.

### Rollout recovery

`rollout_recovery.target_level: turn` and `prefix` work through Dynamo with no extra setting. With `prefix`, NeMo-Gym asks the generation backend to cut its running calls at each checkpoint, and re-sends each cut call after a restore.
Two things differ from the native endpoint:

- **Where the cut goes.** Gym sends the cut beside the URL its calls go to, which is the frontend, but the calls run in the engines. The frontend process takes the cut on a port of its own, from the same range as etcd and NATS, and sends it to every engine over Dynamo's request plane; each engine cuts the calls it holds, up to the frontend's deadline (`token_capture.control_timeout_s` from the cut's arrival there). Setup tells Gym the port (`generation_cut_control_url` in Gym's `vllm_model` config). The port is needed until a Dynamo frontend route can take a POST ([ai-dynamo/dynamo#18486](https://github.com/ai-dynamo/dynamo/issues/18486)).
- **What the router sees for a resumed call.** The engine replaces the prompt the frontend sent with the call's saved tokens and generates the rest, and the frontend puts the saved text back in front of the reply. So the KV router scores that one call on a prompt without the saved tokens. A cut the engine cannot use is dropped and the call regenerates, as natively.

## Not supported

Startup rejects these with `dynamo: true`:

- `colocated.enabled: true`: a colocated engine sleeps while the policy trains, and the frontend would keep routing to it.
- `async_rl.generation_fleet_health.enabled`: the frontend picks engines itself, and a restarted shard does not register with it yet.
- `async_rl.generation_router`: the frontend already routes the fleet behind one URL.
- `bad_words` in `extra_body` or `sampling_overrides` on the policy model (`env.nemo_gym.policy_model.responses_api_models.vllm_model`): the frontend answers every request carrying it with HTTP 400. Startup checks only this overlay, not Gym's own YAMLs.
- `stop_token_ids`, `allowed_token_ids`, `bad_words_token_ids`, `cache_salt`, `logprob_token_ids` or `detokenize` in either block on the policy model: the frontend accepts them but drops them before NeMo RL's chat engine sees them.
- `vllm_cfg.skip_tokenizer_init: true` with NeMo-Gym: the engine detokenizes the replies Gym's calls get through the frontend.
- `vllm_cfg.expert_parallel_size > tensor_parallel_size`: that runs data parallelism inside vLLM, which Dynamo does not serve yet. Above data-parallel rank 0, vLLM rejects each engine's `ipc://` KV-event endpoint.
- `vllm_cfg.http_server_api_key_required`: the frontend serves the same engines on 0.0.0.0 and checks no bearer token.
- `vllm_cfg.http_api_server_count` above 1: NeMo-Gym's calls go through the Dynamo frontend, so the extra vLLM HTTP frontends would serve none of them. `frontend_args.render_workers` sets how many processes render chat calls behind Dynamo.
- `frontend_args.router_mode: direct`: it needs a worker ID on every request, and NeMo RL's rollouts name none.
- `dynamo: true` on a backend other than vLLM or Megatron, and a `dynamo_cfg` block without `dynamo: true`. The Megatron backend has its own limits; see [Run Megatron prefill/decode engines under the SingleController](#run-megatron-prefilldecode-engines-under-the-singlecontroller).

## Build the image

ai-dynamo is part of the `vllm` extra, so every vLLM worker venv carries it.
`BUILD_DYNAMO=1` adds only the etcd and nats-server binaries, under `/opt/dynamo/bin`:

```bash
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg BUILD_DYNAMO=1 \
  --target release \
  --file docker/Dockerfile \
  --tag registry.example.com/nemo-rl:dynamo \
  .
```

For a local checkout, install the binaries and keep the variable set for the runs that follow. The runtime looks for etcd and nats-server there, or else on `PATH`:

```bash
export NEMO_RL_DYNAMO_BIN_DIR="$PWD/venvs/dynamo-bin"
bash docker/dynamo/install.sh
```

## Run

Use any non-colocated SingleController vLLM recipe and add the switch:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
  ++policy.generation.dynamo=true
```

[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml) is the same run with every `dynamo_cfg` default spelled out.

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
    refit_transport: null    # packed collective; or mcore (see below)
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
need not match. They also take `max_requests`, the cap on requests one engine
holds at once: a prefill engine keeps each finished prompt in a request slot
until a decode engine claims it, so while decode is full a shared cap also
throttles prefill. For example, one prefill engine at TP 4 next to two decode
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

For hybrid (Mamba) models, `mcore_generation_config.prefix_caching_mamba_gb`
sets the Mamba prefix-cache budget of the prefill engines only. Decode engines
always run without one: they import each handoff straight into live SSM state.

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
- The suspend takes the prefill engines down first, then decode, within
  `dynamo_cfg.control_timeout_s`; if a step fails, every engine it reached is
  resumed and the refit fails rather than leaving the fleet half stopped. It
  does not wait for decode to pull the KV blocks prefill still holds for
  handoffs (the "handoff pins"): the engines run the `persist` KV cache, so
  suspend and resume leave the KV memory and the pinned blocks untouched and
  the transfer writes only parameters, and a decode engine reads the same
  bytes whether it pulls a block before, during or after the refit. Waiting
  for the pins would tie the refit to decode capacity. Resume brings every
  engine back in one wave: the engines keep their KV on the GPU across suspend
  and resume, and a handoff that reaches a decode engine still waking is
  queued, so no role order is needed.
- The collective is one group over the trainers plus all engines.
  `refit_transport: null` broadcasts HF tensors from train rank zero and every
  engine rank imports them into Megatron. `refit_transport: mcore` with
  `mcore_generation_config.refit_backend: nccl` (or `gloo`) uses
  Megatron-Core's native refit instead, as the aggregated Megatron engines
  do: each training rank sends its own shards straight to the engine ranks
  that hold them. One transfer covers every engine even when prefill and
  decode use different tensor-parallel sizes, because each engine reports its
  layout at its own ranks in the group. Like the aggregated native refit, it
  needs `async_rl.generation_fleet_health.refit_timeout_s: null`.
- Shutdown stops decode engines first, confirms their termination to every
  prefill coordinator so pinned blocks are released, then stops prefill, then
  the frontend, NATS and etcd.

Limits:

- Prefill/decode only: aggregated Megatron engines behind Dynamo are rejected
  (leave `dynamo` false for that).
- Native rollouts only; NeMo-Gym through the frontend is follow-up work, so
  `env.should_use_nemo_gym` must be false.
- Text prompts only (`policy.is_vlm` false); no router replay.
- Each engine fits on one node; `refit_transport` must be null or `mcore`
  (with `refit_backend` `nccl` or `gloo`);
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

`backend: dynamo` launched Dynamo's own vLLM engine and was removed. Set `backend: vllm` and `dynamo: true`; `vllm_cfg` and `vllm_kwargs` keep their meaning. Startup rejects the old backend and each removed `dynamo_cfg` key (`engine`, `control_timeout_s`, `metrics_*_prefixes`, `worker_args`, `frontend_args.tokenizer*`) with what replaces it.
