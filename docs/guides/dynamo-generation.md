# Dynamo serving for vLLM generation

Any vLLM generation config can serve its rollouts through
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
  `AsyncLLM`. Settings the async engine does not run, such as data parallelism
  inside vLLM (`expert_parallel_size > tensor_parallel_size`), are rejected at
  startup.
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
| `request_timeout_s` | 900 | Timeout for one NeMo RL rollout through the frontend, in place of `NRL_VLLM_ASYNC_TIMEOUT_SECONDS`. A timeout fails the attempt as a native timeout does, and the rollout retry policy applies; NeMo-Gym's calls keep Gym's timeouts |
| `worker_args.tool_call_parser`, `reasoning_parser` | null | Dynamo's built-in parser names. null follows the native server's settings; see [NeMo-Gym](#nemo-gym) |
| `worker_args.custom_jinja_template` | null | A chat template that replaces the model's own. null follows the native server's |
| `frontend_args.router_mode` | `kv` | Frontend routing policy |
| `frontend_args.session_affinity_ttl_secs` | 1800 | Keeps a NeMo-Gym token-capture rollout on one engine while it is idle for up to this many seconds; null disables it |
| `frontend_args.tokenizer`, `tokenizer_cache*` | `default`, off | Frontend tokenizer |
| `frontend_args.extra_cli_args` | `[]` | Extra `dynamo.frontend` flags |

The etcd and nats-server binaries are found in `NEMO_RL_DYNAMO_BIN_DIR` when it
is set (the image sets it to `/opt/dynamo/bin`), and on `PATH` otherwise.

## Which requests go through the frontend

| Request | Path |
|---|---|
| `generate_async` on text token prompts, router replay (R3) included | Frontend (KV-aware); the engine samples and assembles it exactly as native `generate_async` does |
| NeMo RL's own multimodal rollouts | Native shard dispatch. Those rollouts send media to the engine directly rather than as OpenAI content |
| `generate_text_async`, which evaluation uses | Native shard dispatch |
| NeMo-Gym's OpenAI requests | Frontend, in Gym's Dynamo dialect; the engine serves them as the native chat endpoint does. See [NeMo-Gym](#nemo-gym) |

A rollout through the frontend has the same `GenerationOutputSpec` as the
native path: token IDs, log probabilities, truncation, routed experts with
their `r3_routed_experts_*` counts, and `gen_leader_worker_idx`. Both paths
build it with `nemo_rl/models/generation/vllm/request_pipeline.py`.

OpenAI clients of the frontend get what vLLM's own OpenAI server applies:
- the model's generation-config sampling defaults;
- `n > 1`;
- guided decoding (`guided_*`, `response_format`);
- `bad_words_token_ids` and `allowed_token_ids`;
- prompt log probabilities (`nvext.extra_fields: ["prompt_logprobs"]`);
- vLLM's queue limits, rejected with HTTP 503;
- a 400 naming the "maximum context length" for a prompt that leaves no room
  for output. The engine, not the frontend, checks the context, so it checks
  the prompt it actually runs.

## NeMo-Gym

NeMo-Gym's policy model calls the frontend. Setup turns on Gym's Dynamo dialect
(`vllm_model.dynamo_frontend`, from
[NVIDIA-NeMo/Gym#4817](https://github.com/NVIDIA-NeMo/Gym/pull/4817)). Gym marks
every call with an annotation, and the engine serves a marked call as the native
chat endpoint does:

- Sampling must match `temperature`/`top_p` or `val_temperature`/`val_top_p`,
  with `top_k` unset. vLLM samples with the configured values.
- The token prefix Gym echoes replaces the history the frontend rendered.
- Output is clamped to the context, and a prompt that leaves no room is refused
  with the native endpoint's "maximum context length" error.
- Stop strings stop vLLM itself, so the token IDs match the native endpoint's.
- The response carries `prompt_token_ids`, `completion_token_ids` and
  `completion_logprobs` in `nvext.engine_data`, which Gym lifts onto the message.
  With router replay on, it also carries the call's `routed_experts`, as the
  native endpoint attaches them.
- Images, video and audio in the messages are processed as on the native
  endpoint. The frontend passes each media URL through with one unexpanded
  placeholder per item. The engine loads the media with vLLM's media connector
  and runs vLLM's multimodal processor on the frontend's tokens. The splice,
  placeholder remap and media capture then work on that processed prompt.

With `token_capture.enabled`, Gym uses its `dynamo_worker` capture backend. The
same vLLM workers stage each call's tokens, log probabilities, routed experts and
processed media from their own process, exactly as the native endpoint does, and
only the commit coordinates travel back through the frontend. The frontend keeps a rollout on
one engine (`session_affinity_ttl_secs`), as the native backend pins each Gym
session to one server.

The native server's chat settings carry over:

| Native setting (`vllm_cfg`) | Under Dynamo |
|---|---|
| `http_server_serving_chat_kwargs.tool_parser`, `reasoning_parser` | The frontend's parsers, under Dynamo's names: NeMo RL's `nano_v3` plugin is `nemotron_v3`, vLLM's `openai` tool parser is `harmony` |
| `tool_parser_plugin`, `reasoning_parser_plugin` | Python plugins cannot run in the Rust frontend. Startup asks for `dynamo_cfg.worker_args` to name a Dynamo parser, unless the plugin has a rename above |
| `http_server_serving_chat_kwargs.chat_template` | The frontend's template, as a file, a vLLM template name or literal Jinja |
| `http_server_serving_chat_kwargs.default_chat_template_kwargs` | Sent by Gym under its own `chat_template_kwargs` |
| `http_server_serving_chat_kwargs.enable_auto_tools`, `chat_template_content_format` | The frontend always parses `tool_choice: auto` calls and has no content-format setting, so only the defaults (`true`, `auto`) are accepted |

Where the behavior differs from the native endpoint:

- The engine sees the rendered prompt, not the messages. It finds where the
  echoed prefix ends by counting the prefix's EOS tokens, not by rendering the
  history.
- When a request sets `chat_template_kwargs.force_nonempty_content`, Dynamo
  reports a reasoning-only turn as content.

## Not supported

Startup rejects these with `dynamo: true`:

- `colocated.enabled: true`: a colocated engine sleeps while the policy trains,
  and the frontend would keep routing to it. The SingleController never runs
  colocated vLLM; the other entrypoints need non-colocated generation.
- `async_rl.generation_fleet_health.enabled`: the frontend picks engines itself,
  and a restarted shard does not register with it yet.
- `async_rl.generation_router`: the frontend already routes the fleet behind one
  URL.
- `vllm_cfg.http_server_api_key_required`: the frontend checks no bearer token.
- `vllm_cfg.expert_parallel_size > tensor_parallel_size`: that runs data
  parallelism inside vLLM, which the async engine Dynamo serves does not.
- Parser plugins with no Dynamo parser, and the chat settings above that the
  frontend has no counterpart for.
- `dynamo: true` on a backend other than vLLM, and a `dynamo_cfg` block without
  `dynamo: true`.

A Gym call with `n > 1` is refused when it arrives.

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

For a local checkout, install the binaries with:

```bash
NEMO_RL_DYNAMO_BIN_DIR=$PWD/venvs/dynamo-bin bash docker/dynamo/install.sh
```

## Run

Use any non-colocated vLLM recipe and add the switch:

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
  - `worker_args.extra_cli_args`: engines are configured through `vllm_cfg` and
    `vllm_kwargs`.
