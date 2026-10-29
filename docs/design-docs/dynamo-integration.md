# Dynamo integration design

`policy.generation.dynamo: true` serves NeMo RL's own vLLM engines through
Dynamo's frontend and KV-aware router. Two requirements shape the design:

- A vLLM bump must never be blocked by Dynamo.
- Dynamo CI coverage must stay consistent with native vLLM's.

Both hold only if the engine process contains no Dynamo-owned vLLM code. A vLLM
bump can then break Dynamo serving only in NeMo RL's own adapter
(`native_engine.py`), which calls a few vLLM APIs native NeMo RL does not: the
`get_kv_cache_group_metadata` engine utility and `KVCacheSpecKind` at
registration, and `StatLoggerBase` for the router's load signal. A renamed import
fails the vLLM-lane unit tests at collection, and the Dynamo functional overlays
run registration.

## Engine hosting

`VllmGeneration` builds its workers exactly as without Dynamo. Each data-parallel
leader (`VllmAsyncGenerationWorkerImpl`) owns an `AsyncLLM`. With Dynamo on, it
also runs `dynamo.common.backend.Worker` as a task on its own actor event loop,
which is the loop that owns the `AsyncLLM`.

`dynamo.common.backend` is Dynamo's engine-agnostic backend SDK and imports no
vLLM. The adapter, `nemo_rl/models/generation/dynamo/native_engine.py`,
implements its `LLMEngine` over the worker's `AsyncLLM`:

- `generate` serves only the rollouts NeMo RL itself sends, which carry an
  annotation, and refuses other requests. The worker's own `_serve_rollout` runs
  each rollout exactly as native `generate_async` does, and
  `request_pipeline.rollout_result` builds its result, routed experts included.
  The result travels in `engine_data`, and the driver assembles the same output
  with `request_pipeline.rollout_output`.
- `abort` cancels the vLLM request on client disconnect.
- Registration advertises the context length, the KV block size (from vLLM's
  KV-cache group metadata), the block count and the scheduler limits. It
  publishes no token budget. The engine owns the context check and the output
  cap, as the native server does. With a budget, the frontend would refuse NeMo
  RL's own rollouts that fill the context.
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
`ai-dynamo-runtime` wheels, never `ai-dynamo[vllm]`. The engine process only
imports `dynamo.common.backend`, `dynamo.llm.exceptions` and `dynamo._core`.

What a Dynamo upgrade has to re-check:
- the `dynamo.common.backend` contract (`LLMEngine`, `Worker`, `WorkerConfig`,
  publisher);
- the request fields of Dynamo's preprocessed request;
- the frontend's `nvext` extensions and annotation forwarding.

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
