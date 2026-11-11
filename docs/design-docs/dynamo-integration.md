# Dynamo integration design

`policy.generation.dynamo: true` serves NeMo RL's own vLLM engines through
Dynamo's frontend and KV-aware router. Two requirements shape the design:

- A vLLM bump must never be blocked by Dynamo.
- Dynamo CI coverage must stay consistent with native vLLM's.

Both hold only if no process runs Dynamo-owned vLLM code. A vLLM bump can then
break Dynamo serving only in NeMo RL's own `dynamo/vllm/` package:

- the engine adapter (`adapter.py`) calls a few vLLM APIs native NeMo RL does
  not: the `get_kv_cache_group_metadata` engine utility and `KVCacheSpecKind`
  at registration, and `StatLoggerBase` for the router's load signal;
- the frontend's chat engine (`chat_engine.py`, `render_pool.py`) builds the
  same vLLM renderer and chat serving the native server builds
  (`vllm/chat_server.py`), from the engine's own engine args.

A renamed import fails the vLLM-lane unit tests at collection, and the Dynamo
functional overlays run registration and NeMo-Gym's chat calls.

## Engine hosting

`VllmGeneration` builds its workers exactly as without Dynamo. Each data-parallel
leader (`VllmAsyncGenerationWorkerImpl`) owns an `AsyncLLM`. With Dynamo on, it
also runs `dynamo.common.backend.Worker` as a task on its own actor event loop,
which is the loop that owns the `AsyncLLM`.

`dynamo.common.backend` is Dynamo's engine-agnostic backend SDK and imports no
vLLM. NeMo RL implements its `LLMEngine` in two parts:

- `dynamo/engine_adapter.py`: `DynamoEngineAdapter`, the Dynamo side shared by
  every generation backend. It serves request kinds as handlers keyed by their
  annotation, maps engine errors to a 400 that keeps the cause, and registers the
  endpoint for `/v1/completions` (NeMo RL's rollouts) and chat (NeMo-Gym). It
  imports no backend, and the engine only ever receives token IDs.
- `dynamo/vllm/adapter.py`: `DynamoVllmAdapter`, the vLLM backend over the
  worker's `AsyncLLM`. A backend implements `serve_rollout`, `start`, `abort` and
  `kv_event_sources`, and may add request handlers.

The two together:

- `generate` maps Dynamo's token request onto vLLM `SamplingParams` the way
  vLLM's OpenAI server does, and streams token deltas back. Rollouts that NeMo RL
  itself sends carry an annotation. For those, the engine uses the same code as
  native `generate_async`:
  - the worker's `_rollout_max_new_tokens` and `_rollout_sampling_params` build
    the request;
  - `request_pipeline.rollout_result` builds the result, routed experts
    included.

  The result travels in `engine_data`, and the driver assembles the same
  output with `request_pipeline.rollout_output`.

  NeMo-Gym's calls arrive rendered and spliced by NeMo RL's chat engine (see
  [NeMo-Gym chat](#nemo-gym-chat)), annotated for the adapter's Gym handler,
  with the capture plan in `extra_args`. The handler runs the native server's
  steps after the render, with the worker's own helpers: the media snapshot
  (`_capture_request_media`), `begin_call` (which stamps the weight version),
  the generation with the `SamplingParams` vLLM's chat serving builds from the
  request, and the one TQ write. Its `engine_data` carries the output the
  frontend builds the reply from, and the call's `GymCallResult`. A call the
  engine refuses before it runs (a vLLM request error, a full queue, media
  capture refused) comes back in `engine_data` too, with the status the native
  server answers it with: Dynamo hands a raised error to the chat engine
  without its status.
- `abort` cancels the vLLM request on client disconnect.
- Registration advertises the context length, the KV block size (from vLLM's
  KV-cache group metadata), the block count and the scheduler limits. It
  publishes no token budget. The engine owns the context check and the output
  cap, as the native server does. With a budget, the frontend would fill an
  omitted `max_tokens` from the prompt it rendered, which a prefix splice then
  changes, and would refuse NeMo RL's own rollouts that fill the context.
- `kv_event_sources` points Dynamo at the engine's own vLLM KV-event ZMQ
  publisher. It is set only when the engine caches prefixes. Each engine binds
  its own Linux abstract-namespace socket (`ipc://@...`), so engines on one node
  need no port and cannot collide. The name is random rather than the pid,
  because containers that share the host network share these names. vLLM refuses
  `ipc://` only above data-parallel rank 0, and vLLM data parallelism is rejected
  at startup with Dynamo. A vLLM stat logger publishes the router's per-engine
  load signal.

Because the Worker runs on the actor's own loop, the worker's other methods
interleave with Dynamo requests exactly as with native requests:
- refit (every transport);
- `pause_generation_async` / `resume_generation_async`.

Colocated generation is rejected at startup: a sleeping engine stays
registered, so the frontend would keep routing to it.

No HTTP control routes are involved. The Rust Worker installs a SIGTERM handler
that chains to Ray's, so actor shutdown is unchanged, and discovery drops an
engine when its process exits.

## NeMo-Gym chat

Dynamo's frontend hands chat calls to a Python `chat_engine_factory` when one is
given. `dynamo/vllm/frontend.py` starts `dynamo.frontend`'s own `main` with
NeMo RL's factory added to the `EntrypointArgs` it builds, so the frontend
keeps its HTTP server, router and `/v1/completions` path, and only chat goes to
NeMo RL's chat engine (`dynamo/vllm/chat_engine.py`).

Each engine publishes, in its registration's `runtime_data`, its own engine
args and the `policy.generation` chat settings. The chat engine builds a
render pool (`dynamo/vllm/render_pool.py`) from them: `render_workers` spawned
processes, each holding the vLLM renderer and chat serving the native server
builds. Rendering is CPU-bound Python, and the frontend serves every call on
one event loop, so it runs in the pool, first in first out. Per call:

1. the pool renders it with the native server's render step
   (`_render_chat_call`, shared with `vllm/chat_server.py`), including the
   token-capture decision for a Gym candidate and the history render the
   splice cuts against;
2. the chat engine checks on-policy sampling (`on_policy_sampling`, shared
   with the native route), reads a token-in prefix from TQ through a read-only
   client, splices it in with the model's turn-end tokens, and clamps the
   output budget (`prepare_prompt` in `vllm/prompt_and_capture.py`, shared
   with the native server);
3. the final token IDs go through the KV router to one engine, whose adapter
   runs the native server's `start_capture` and `save_capture`;
4. the pool builds the reply with vLLM's own response builder and parsers,
   through the same reply fixes for restored calls as the native server.

`GymCallExtras` and `GymCallResult` (`models/generation/gym_call.py`) define
the fields both paths exchange with Gym. `tests/unit/models/generation/test_gym_chat_path.py`
sends the same calls through the native routes and through this path and
checks they match step for step.

### Prefix recovery

With `rollout_recovery.target_level: prefix`, a checkpoint cuts each running
call (its decoded tokens are staged in TQ), and a restore resumes it from
those tokens. The engines hold the calls and the worker-side state
(`models/generation/prefix_recovery`), so both halves run there, with the
native server's code:

- **The cut.** NeMo-Gym posts it beside the URL its calls go to, the
  frontend. A Dynamo frontend route can only be a synchronous GET, so
  `dynamo/vllm/frontend.py` also wraps `DistributedRuntime` and serves the
  native route (`vllm/generation_cut_route.py`) on a second port, in the
  frontend process (`dynamo/vllm/generation_cut.py`). It sends the cut to
  every engine with `Client.direct`; the adapter's `nemo_rl:generation_cut`
  handler runs the worker's own cut and answers with the calls it cut; and
  the answers are merged into the one receipt Gym expects. A call no engine
  cut gets Gym's failure ack and regenerates after a restore.
- **The resume.** The frontend renders and routes a re-sent call like any
  other. The adapter then runs the steps the native server runs
  (`resolve_restored_call` and its helpers in `vllm/chat_server.py`): it
  reads and checks the saved rows, replaces the prompt with them, lowers the
  output budget, and begins capture with the cut. A cut it cannot use (rows
  gone, or newer than its weights) is dropped, and the frontend's prompt
  runs. The engine's reply names what it resumed from, and the render pool
  puts the saved text back before vLLM's parsers run and counts the saved
  tokens as output (`prefix_recovery/restore.py`).

The resolve is in the engine, not the frontend, because the check needs the
engine's weight version and its capture state, and a cut the engine drops then
needs no second pass through the frontend. The cost is that the KV router
scores a resumed call on the prompt without its saved tokens.

## Driver

`VllmGeneration` starts `ManagedDynamoRuntime`, which runs etcd, NATS and
`dynamo.frontend`. After the engines load, it asks every DP leader to register
and waits until the frontend routes to all of them. Each poll also asks the
leaders whether their endpoint task stopped, so a failed registration fails
with its cause instead of a timeout.

`generate_async` on text prompts posts a token-prompt completion to the frontend
(`dynamo/rollout_client.py`). It sets top-level `ignore_eos` and a context-sized
`max_tokens`, so the frontend's own stop rules never end a stream before the
engine reports its result. Everything else keeps native dispatch; the guide
lists what goes where.

## Dependency coupling

The vLLM workers' venv (the `vllm` extra) carries the released `ai-dynamo` and
`ai-dynamo-runtime` wheels, never `ai-dynamo[vllm]`. NeMo RL imports only the
Dynamo modules `tests/unit/models/generation/test_dynamo_vllm_decoupling.py`
lists: the backend SDK, `dynamo.llm` and the frontend's `main`, none of which
import vLLM.

What a Dynamo upgrade has to re-check:
- the `dynamo.common.backend` contract (`LLMEngine`, `Worker`, `WorkerConfig`,
  publisher and logprob helpers);
- the request fields of Dynamo's preprocessed request, and `extra_args`
  reaching the engine;
- the frontend's `chat_engine_factory` hook and the `EntrypointArgs` name
  `dynamo/vllm/frontend.py` wraps, which are internal API;
- the Python chat request the hook receives, its `nvext` and annotation
  fields, and `EngineConfig.runtime_data` reaching the frontend.

The one vLLM↔Dynamo wire coupling is vLLM's KV-event msgpack, which Dynamo's
router decodes. A change there degrades routing, not serving. The managed
runtime logs the router's applied-event counts at shutdown, and the Dynamo
functional overlays fail if the engines published events but none were
applied (`tests/functional/check_dynamo_overlay.sh`).

ai-dynamo's own dependency bounds (`ai-dynamo-runtime` caps pydantic, for
example) can still block a vLLM bump, and the import guard does not see them.
Until a Dynamo release lifts a bound, override it under
`[tool.uv] override-dependencies` (as `aisimulate` and `kubernetes` are), and
let the vLLM-lane Dynamo unit tests catch a real break.
