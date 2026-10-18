# Porting guide

## Start from what the target already has

Instrumentation is revision-specific. Search the target first: recent NeMo-RL
already writes the training trace (`nemo_rl/utils/timer.py`,
`nemo_rl/algorithms/grpo.py`), the rollout lifecycle
(`ReplayBuffer(lifecycle_log_path=...)`), and, for the TRT-LLM backend, the
request milestones (`nemo_llm_*_ts_us` in
`nemo_rl/models/generation/trtllm/trtllm_http_server.py`). Port only what is
missing, onto fresh copies of the target revision's files, using the semantic
insertion points below. Never replace a whole target file with a patched copy
from another revision.

For the rollout side, `../patches/` holds reference NeMo Gym and OpenHands
patches with the sha256 of the image files they apply to
(`../patches/README.md`). Apply them directly only when every listed file
matches; otherwise read their hunks as the reference implementation and port
them.

## Instrumentation map

Port changes onto fresh copies of these files from the target container. Source
roots may differ by image; discover them rather than assuming the paths.

| Role | Typical container path | Required change |
|---|---|---|
| Training timers | `/opt/nemo-rl/nemo_rl/utils/timer.py` | When `NRL_TRAINING_TIMELINE_FPATH` is set, append a complete event for every stopped timer. |
| Rollout identity | `/opt/nemo-rl/nemo_rl/algorithms/async_utils/trajectory_collector.py` | Assign one stable ID per rollout, propagate it through Gym request metadata, and retain it in the replay-buffer group. |
| Rollout lifecycle | `/opt/nemo-rl/nemo_rl/algorithms/async_utils/replay_buffer.py` | Append targeting metadata when buffered and actual consumption metadata when sampled. |
| Lifecycle wiring | `/opt/nemo-rl/nemo_rl/algorithms/grpo.py` | Pass the persistent lifecycle JSONL path to the replay-buffer actor. |
| vLLM request metadata | `/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py` | Enable prompt-token details and per-request metrics; copy vLLM arrival, first-scheduled, first-token, and last-token timestamps into the live response. |
| TRT-LLM request metadata | `/opt/nemo-rl/nemo_rl/models/generation/trtllm/trtllm_http_server.py` (and `trtllm_disagg_server.py` under PD disaggregation) | Same `nemo_llm_*_ts_us` fields from the request perf metrics; under disaggregation, relay the context leg's `nemo_ctx_*` fields onto the final response. |
| Gym orchestration | `/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py` | Create the rollout trace, pass its path/rollout ID into OpenHands, and span queue, container, OpenHands, and evaluation phases. |
| Gym response schema | `/opt/nemo-rl/3rdparty/Gym-workspace/Gym/nemo_gym/openai_utils.py` | Add optional generation-replica and LLM request-milestone fields to the proxied chat-completion response. |
| Gym model routing | `/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py` | Preserve sticky session routing and return the selected endpoint index with each completion. |
| OpenHands helper | `.../OpenHands/openhands/utils/nemo_gym_timeline.py` | Add a best-effort context manager that emits complete events. |
| Model calls | `.../OpenHands/openhands/agenthub/nemo_gym_client.py` | Span each actual completion request as `model_call`; attach turn index, live usage token counts, and LLM request milestones. |
| Tool actions | `.../OpenHands/openhands/runtime/base.py` | Span handling of each concrete `Action`; preserve action/result semantics. |
| Agent lifecycle | `.../OpenHands/evaluation/benchmarks/swe_bench/run_infer.py` | Span runtime creation/connect/init, instruction preparation, controller, patch extraction, and cleanup. |

Read the target implementations and locate semantic insertion points. Function
names and line numbers can change. Prefer adding a small helper/import and a
context manager around the actual operation over duplicating logic.

Replay-buffer instrumentation must not change which entries are eligible for a
training step. Preserve the target revision's comparison and ordering exactly.
In particular, do not change `>=` to `==` or `==` to `>=` as part of this
work; that controls future-rollout stealing and is a separate behavioral
experiment.

## Overlay construction

Use a revision-specific directory, for example:

```text
timeline_overlay/
  nemo_rl/algorithms/async_utils/trajectory_collector.py
  nemo_rl/algorithms/async_utils/replay_buffer.py
  nemo_rl/algorithms/grpo.py
  nemo_rl/utils/timer.py
  nemo_rl/models/generation/vllm/vllm_worker_async.py
  3rdparty/Gym-workspace/Gym/nemo_gym/openai_utils.py
  3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py
  3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py
  3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands/
    openhands/agenthub/nemo_gym_client.py
    openhands/runtime/base.py
    openhands/utils/nemo_gym_timeline.py
    evaluation/benchmarks/swe_bench/run_infer.py
```

Copy each original from the target image first. Keep its relative path in the
overlay, apply the timeline edits there, and review the diff against that same
revision. Never reuse whole patched files from another revision merely because
their filenames match.

Include only the files the target still needs. Mount each overlay file
individually and read-only at its container path, so the overlay never hides
other files the image supplies, and fail the launch if a listed overlay file is
missing. The vLLM file contains only the cached-token usage-field change; keep
worker/executor profiler patches separate unless the user requests Nsight.
A single-file bind mount pins the file as it was at job start: an overlay file
edited afterwards is not seen by the running job (on Lustre it reads as a
stale file handle), so stage edits as new files for the next submission.

## Runtime wiring

Set:

```bash
NRL_TRAINING_TIMELINE=1
NRL_TRAINING_TIMELINE_FPATH=/logs/training_timeline.jsonl
NRL_ROLLOUT_LIFECYCLE_FPATH=/logs/rollout_lifecycle.jsonl
```

`/logs` must already be a persistent host-backed job log mount. The Gym actor
should create one `timeline.jsonl` in each rollout's persistent directory and
pass its mounted path to the OpenHands process as:

```bash
NEMO_GYM_TIMELINE_FPATH=<agent-container-visible-path>/timeline.jsonl
NEMO_GYM_ROLLOUT_ID=<stable-agent-run-id>
```

Quote mount lists carefully and retain the recipe/config mount used by the job.
The exact launch command is site-specific; derive it from the target repo's
current launcher rather than copying an old Slurm command.

## Minimal implementation pattern

Training `Timer`:

1. At start, capture `time.time_ns()` only when the trace path is configured.
2. At stop, retain the existing monotonic elapsed calculation.
3. Append one event using the timer name, elapsed duration, process/thread IDs,
   hostname, and a stable source label.
4. Reset timeline state along with the timer's normal reset state.

OpenHands:

1. Implement `nemo_gym_timeline_span(name, category, **args)` as a context
   manager.
2. Yield normally when no path is configured.
3. Preserve and re-raise application exceptions after recording error status.
4. Instrument the narrowest operation that represents the named phase.
5. Let the `model_call` span accept numeric metadata after the request returns.
6. Read `usage.prompt_tokens` and `usage.completion_tokens` from the live
   chat-completion response and emit them as `prompt_tokens` and
   `decode_tokens`. Do not scan persisted rollout `output.jsonl` files.
7. Read `usage.prompt_tokens_details.cached_tokens` without parsing prompts or
   responses. Emit `cached_prompt_tokens`, `uncached_prompt_tokens`, and
   `kv_cache_hit_rate` when that cache detail is available. Do not make
   `decode_tokens` conditional on cache-detail availability.
8. Preserve zero-valued metrics. Test field presence/type, not truthiness:

   ```python
   value = provider_fields.get(field)
   if value is not None:
       timeline_args[field] = value
   ```

   In particular, `cached_tokens == 0` is a valid full cache miss and must
   still produce all four token fields, including `kv_cache_hit_rate == 0.0`.
9. Read the proxy's `nemo_generation_replica_idx`, validate it as a
   nonnegative integer, and emit it as `generation_replica_idx`.
10. Read the proxy's positive integer timing fields and emit
    `llm_arrival_ts_us`, `queued_ts_us`, `first_scheduled_ts_us`,
    `first_token_ts_us`, `last_token_ts_us`, `llm_response_ready_ts_us`,
    `gym_llm_response_received_ts_us`, and `gym_response_ready_ts_us`. Record
    `openhands_response_received_ts_us` after the Gym body is received and
    parsed. Measure `gym_tokenize_duration_us` and
    `openhands_parse_validate_duration_us` with local monotonic clocks. Do not
    derive these fields from the outer span boundaries.

vLLM:

1. Verify the exact installed API before patching. In vLLM 0.20,
   `OpenAIServingChat(..., enable_prompt_tokens_details=True)` copies
   `RequestOutput.num_cached_tokens` into the OpenAI usage object.
2. Prefer enabling that supported constructor flag. If the revision lacks it,
   port a minimal equivalent at the full chat-completion response boundary.
3. Do not infer cache reuse from global `kv_cache_usage_perc`; occupancy is not
   a per-request hit rate.
4. Define `uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens` and
   `kv_cache_hit_rate = cached_prompt_tokens / prompt_tokens`, using `0.0` for
   an empty prompt.
5. When tracing is enabled and the installed API supports it, pass
   `enable_per_request_metrics=True` to the chat-serving constructor. Do not
   turn it on unconditionally when tracing is disabled.
6. Capture `final_res.metrics` at the full-response boundary. Treat
   `arrival_time` as epoch seconds and `queued_ts`, `scheduled_ts`,
   `first_token_ts`, and `last_token_ts` as monotonic seconds. In the generation server process,
   convert a positive monotonic timestamp with
   `time.time_ns() - time.monotonic_ns()` and export epoch microseconds.
7. Record `llm_response_ready_ts_us` immediately after the full-response
   generator returns and before returning the response to Gym.
8. Keep the timing export best effort. A missing/zero metric must omit that
   field rather than fabricating an epoch timestamp from zero.

Gym:

1. Put the trace in the rollout's persistent directory.
2. Touch/create it before starting the agent container so bind mounting works.
3. Pass both trace path and rollout ID into that container.
4. Instrument orchestration outside the container with the same event schema.
5. Preserve the existing session-to-client routing policy. Record the selected
   client-list index and return it as `nemo_generation_replica_idx`; do not
   infer it later from timestamps or session admission order.
6. Interpret the client-list index as a DP generation replica. Maintain a
   separate endpoint/topology map if TP ranks, hosts, or GPU IDs are required.
7. Declare every optional `nemo_llm_*_ts_us` and `nemo_gym_*` timing field in
   the Gym response model so validation and reserialization preserve it.
8. In the Gym model proxy (`vllm_model`), timestamp receipt after the upstream body is fully
   parsed, measure any fallback `/tokenize` call with `perf_counter_ns()`, and
   timestamp response readiness immediately before returning.

## Smoke test and diagnosis

Before the synthetic smoke test, validate that the completed overlay contains
the real Gym proxy-to-OpenHands attribution path:

```bash
python scripts/validate_replica_wiring.py \
  --source-root /path/to/revision-specific-overlay \
  --generation-backend vllm   # or trtllm, with --nemo-rl-root when the
                              # TRT-LLM server lives outside the overlay
```

This check exercises the deployed `_resolve_client` implementation with fake
clients and checks the response schema, response propagation, and OpenHands
consumer. The bundled `smoke_timeline.py` writes a synthetic replica index; it
tests trace serialization but cannot detect a missing proxy/schema hunk.

Use one training step, one prompt, one rollout, and a low max-turn limit.
Confirm:

1. the job sees the overlay mounts;
2. `/logs/training_timeline.jsonl` is nonempty;
3. rollout directories contain nonempty `timeline.jsonl`;
4. JSONL validation succeeds;
5. timestamps overlap plausibly across host, trainer, and OpenHands processes;
6. HTML renders training and rollout rows.
7. successful `model_call` events include complete, internally consistent
   input/decode token counts, and cache metrics when the backend reports cached-token
   details.
8. the smoke test exercises both a nonzero cache hit and an explicit
   `cached_tokens == 0` miss, and the miss retains its token counts.
9. the smoke test exercises valid completion counts both with and without
   `prompt_tokens_details`.
10. successful model calls contain a nonnegative `generation_replica_idx`,
    and all turns in the smoke rollout use the same index.
11. successful model calls contain internally ordered LLM request timestamps:
    arrival <= first scheduled <= first token <= last token.

After the representative job, run `validate_timeline.py` on its real rollout
tree with `--require-generation-replica`. Do not infer successful integration
from the synthetic smoke trace.

If the training trace is absent, check the environment reached the trainer and
that `/logs` is writable. If rollout traces are absent, inspect the Gym-created
container command and path mapping. If only host-side rollout events appear,
the OpenHands environment/path was not propagated. If spans have wrong order or
duration, verify epoch timestamps are used only for starts and monotonic clocks
for elapsed time. If cache fields are absent, inspect the raw chat-completion
`usage.prompt_tokens_details` object. A null object means the backend did not report cached tokens (vLLM: the serving flag
was not enabled or that revision did not expose `num_cached_tokens`). If token
metrics arrive but `generation_replica_idx` does not, verify all three links:
`NeMoGymChatCompletion` declares `nemo_generation_replica_idx`, the proxy saves
the selected client index and attaches it to every response path, and OpenHands
copies that field into the `model_call` event.
If `first_scheduled_ts_us` is absent, verify that the backend exports the engine scheduling timestamps (vLLM: engine statistics are
enabled, the chat-serving API accepts `enable_per_request_metrics`, the
full-response generator retains `final_res.metrics`; TRT-LLM: `return_perf_metrics` is on), Gym declares the timing
fields, and OpenHands copies the live top-level response fields.
If `queued_ts_us` is absent, verify the backend exposes it (vLLM: `RequestStateStats`
has `queued_ts`; TRT-LLM maps it to the request arrival time). If Gym or OpenHands boundary fields are absent,
verify the response schema preserves the fields and that each client records
receipt only after reading/parsing the full HTTP response body.
