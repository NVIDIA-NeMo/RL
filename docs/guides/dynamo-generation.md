# Dynamo serving for vLLM generation

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

See the [Dynamo integration design](../design-docs/dynamo-integration.md) for
how the engines are hosted and why Dynamo upgrades and vLLM bumps stay
independent.

## What happens at startup

With `dynamo: true`:

- `vllm_cfg.async_engine` is turned on, because Dynamo drives the engine's
  `AsyncLLM`. Settings Dynamo does not serve are rejected at startup; see
  [Not supported](#not-supported).
- The driver starts etcd and NATS. Each `dynamo.frontend` runs on a generation
  node, from a small Ray actor, in the vLLM workers' own venv, so it is the same
  ai-dynamo release.
- Each engine turns on vLLM's KV-cache events for `router_mode: kv` and
  registers with the frontends from inside its own worker process. Setup waits
  until every frontend routes to every engine.
- With NeMo-Gym (`vllm_cfg.expose_http_server: true`), each frontend starts its
  render workers when the first engine registers, while the trainer loads, so
  Gym's first calls do not wait for them. Other runs start none.
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
| `frontend_args.session_affinity_ttl_secs` | 300 | Keeps a NeMo-Gym token-capture rollout on one engine while it is idle for up to this many seconds; null disables it |
| `frontend_args.session_affinity_max_entries` | 16,777,216 | Most sessions the frontends hold at once, across all of them; a new one past it gets HTTP 529. Entries are allocated as sessions start, so a high cap costs nothing until it is used |
| `frontend_args.num_frontends` | 1 | Frontends, each with its own router, spread round-robin over the generation nodes. More than one adds `--router-replica-sync`, so each router sees the requests the others have in flight; NeMo-Gym keeps each session on one frontend, and NeMo RL's own rollouts pick one at random |
| `frontend_args.render_workers` | 8 | Processes in each frontend that render NeMo-Gym's chat calls and build their replies; every Gym call that frontend serves shares them. Each holds its own tokenizer and processor, about 1.7–2.3 GB. On one Grace core of a GB200, one serves about 640 single-turn calls/s, 45 for a 21K-token agentic history (22 with its prefix splice), or 200 with 1–4 images |
| `frontend_args.extra_cli_args` | `[]` | Extra `dynamo.frontend` flags. `--dyn-chat-processor` and `--custom-jinja-template` are rejected: NeMo RL renders chat itself |
| `frontend_args.extra_env` | `{}` | Extra frontend environment, for settings Dynamo reads only from the environment, such as `DYN_TCP_LISTEN_BACKLOG`. Variables the runtime manages are rejected |

The etcd and nats-server binaries are found in `NEMO_RL_DYNAMO_BIN_DIR` when it
is set (the image sets it to `/opt/dynamo/bin`), and on `PATH` otherwise.

## Which requests go through the frontend

| Request | Path |
|---|---|
| `generate_async` on text token prompts, router replay (R3) included | Frontend (KV-aware); the engine samples and assembles it exactly as native `generate_async` does |
| NeMo RL's own multimodal rollouts | Native shard dispatch. Those rollouts send media to the engine directly rather than as OpenAI content |
| `generate_text_async`, which evaluation uses | Native shard dispatch |
| NeMo-Gym's chat calls | Frontend, rendered by NeMo RL's chat engine with the native server's code. See [NeMo-Gym](#nemo-gym) |

A rollout through the frontend has the same `GenerationOutputSpec` as the
native path: token IDs, log probabilities, truncation, routed experts with
their `r3_routed_experts_*` counts, and `gen_leader_worker_idx`. Both paths
build it with `nemo_rl/models/generation/vllm/request_pipeline.py`.

## NeMo-Gym

NeMo-Gym's policy model calls the frontend. Setup turns on Gym's Dynamo dialect
(`dynamo_frontend` in Gym's
[`vllm_model`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_models/vllm_model)
config). The frontend takes only OpenAI fields plus `nvext`, so Gym sends the native
request's extra fields (`ng_capture`, `required_prefix_token_ids`), and
`ng_measure` for the measure call it makes in place of `/tokenize`, in one
`nemo_rl:gym:` annotation, and reads the call's token fields
(`prompt_token_ids`, `generation_token_ids`, `generation_log_probs`,
`routed_experts`, `ng_commit_coords`) from `nvext.nemo_rl`.

The frontend serves chat with NeMo RL's chat engine instead of Dynamo's own
renderer and parsers (Dynamo's `chat_engine_factory` hook). Each call takes the
native server's steps, with the native server's code:

1. In the frontend, a render-pool process renders the call with the engine's
   own tokenizer, template, processor and `vllm_cfg` chat settings (parser
   plugins included), which each engine publishes when it registers. It also
   decides how a token-capture call's prefix is served.
2. The chat engine checks on-policy sampling, reads a captured prefix from TQ,
   splices the prefix into the render, clamps the output budget to the
   context, and routes the final token IDs through the KV router.
3. On the engine, the adapter snapshots the call's media, begins its capture,
   generates with the `SamplingParams` the native server builds, and stages the
   call in one TQ write. Only the commit coordinates travel back.
4. A render-pool process builds the reply with vLLM's reasoning and tool parsers.

`tests/unit/models/generation/test_gym_chat_path.py` sends the same calls down
both paths and checks they run the same prompt and `SamplingParams`, stage the
same record and return the same reply.

Both paths refuse, with HTTP 400 and before anything runs, a call that streams
(`stream: true`), asks for more than one choice (`n`), or sets `logprobs`
without `top_logprobs`. NeMo-Gym sends none of these.

Where the behavior differs from the native endpoint:

- The frontend refuses more than 32 `stop` strings, and vLLM request fields its
  own request type lacks (for example `structured_outputs`, `return_token_ids`,
  `truncate_prompt_tokens`, `priority`, `chat_template`), with HTTP 400.
- gpt-oss (Harmony) chat calls are refused with HTTP 400.
- A token-capture call refused for its context length acknowledges the refusal
  at the end of the error message instead of in the native endpoint's
  `x-ng-capture-refused` header, which Dynamo's errors cannot carry. Gym's
  dialect moves it back to the header, so the harness sees the native error.
- A call's processed media travels to the engine inside the request.

### Rollout recovery

`rollout_recovery.target_level: turn` and `prefix` work through Dynamo with no
extra setting. With `prefix`, NeMo-Gym asks the generation backend to cut its
running calls at each checkpoint, and re-sends each cut call after a restore.
Two things differ from the native endpoint:

- **Where the cut goes.** Gym sends the cut beside the URL its calls go to,
  which is a frontend, but the calls run in the engines. Each frontend process
  takes the cut on a port of its own, from the same range as etcd and NATS, and
  sends it to every engine over Dynamo's request plane; each engine cuts the
  calls it holds, up to that frontend's deadline
  (`token_capture.control_timeout_s` from the cut's arrival there). Setup tells
  Gym each frontend's port (`generation_cut_control_url` in Gym's `vllm_model`
  config, one per `base_url`). The port is needed until a Dynamo frontend route
  can take a POST
  ([ai-dynamo/dynamo#18486](https://github.com/ai-dynamo/dynamo/issues/18486)).
- **What the router sees for a resumed call.** The engine replaces the prompt
  the frontend sent with the call's saved tokens and generates the rest, and
  the frontend puts the saved text back in front of the reply. So the KV router
  scores that one call on a prompt without the saved tokens. A cut the engine
  cannot use is dropped and the call regenerates, as natively.

## Not supported

Startup rejects these with `dynamo: true`:

- `colocated.enabled: true`: a colocated engine sleeps while the policy trains,
  and the frontend would keep routing to it.
- `async_rl.generation_fleet_health.enabled`: the frontend picks engines itself,
  and a restarted shard does not register with it yet.
- `async_rl.generation_router`: the frontend already routes the fleet behind one
  URL.
- `extra_body.bad_words` on the policy model
  (`env.nemo_gym.policy_model.responses_api_models.vllm_model`): the frontend
  answers every request carrying it with HTTP 400. Startup checks only this
  overlay, not Gym's own YAMLs.
- `extra_body` `stop_token_ids`, `allowed_token_ids`, `bad_words_token_ids`,
  `cache_salt`, `logprob_token_ids` or `detokenize` on the policy model: the
  frontend accepts them but drops them before NeMo RL's chat engine sees them.
- `vllm_cfg.skip_tokenizer_init: true` with NeMo-Gym: the engine detokenizes
  the replies Gym's calls get through the frontend.
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

Use any non-colocated SingleController vLLM recipe and add the switch:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
  ++policy.generation.dynamo=true
```

[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
is the same run with every `dynamo_cfg` default spelled out.

## Migrating from `backend: dynamo`

`backend: dynamo` launched Dynamo's own vLLM engine and was removed. Set
`backend: vllm` and `dynamo: true`; `vllm_cfg` and `vllm_kwargs` keep their
meaning. Startup rejects the old backend and each removed `dynamo_cfg` key
(`engine`, `control_timeout_s`, `metrics_*_prefixes`, `worker_args`,
`frontend_args.tokenizer*`) with what replaces it.
