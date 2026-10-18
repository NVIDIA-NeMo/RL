# Timeline event contract

Every trace file is newline-delimited JSON. Each nonblank line is one Chrome
trace complete event:

```json
{"name":"model_call","cat":"model","ph":"X","ts":123456789.0,"dur":42000.0,"pid":123,"tid":456,"args":{"rollout_id":"R1","turn":2,"status":"ok"}}
```

Required fields:

| Field | Contract |
|---|---|
| `name` | Non-empty event name |
| `cat` | Non-empty category |
| `ph` | Exactly `"X"` |
| `ts` | Epoch timestamp in microseconds |
| `dur` | Nonnegative duration in microseconds |
| `pid` | Process identifier |
| `tid` | Thread identifier |
| `args` | JSON object; include hostname/source/status when available |

Generate `ts` from `time.time_ns()` so independently running processes can be
aligned. Measure elapsed time with `time.perf_counter_ns()` or
`time.perf_counter()` so clock adjustments cannot create bad durations.

## Training events

Training events use category `training_timer`. Preserve the timer names emitted
by the current NeMo-RL workflow. The plotter recognizes at least:

- `total_step_time`
- `exposed_generation`
- `reward_calculation`
- `data_processing`
- `logprob_inference_prep`
- `policy_and_reference_logprobs`
- `advantage_calculation`
- `training_prep`
- `policy_training`
- `weight_sync`
- `total_validation_time`
- `checkpointing`

The training trace is selected by `NRL_TRAINING_TIMELINE_FPATH`. Also set
`NRL_TRAINING_TIMELINE=1` as the explicit user-facing opt-in even if the target
revision currently gates only on the path.

## Rollout events

The host-side NeMo Gym actor should emit:

- `ray_queue`
- `agent_container_spinup`
- `openhands_run`
- `eval_container_spinup`
- `final_eval`

The OpenHands container should emit:

- runtime: `create_runtime`, `connect_runtime`, `initialize_runtime`,
  `runtime_cleanup`;
- agent: `prepare_instruction`, `run_controller`, `extract_patch`;
- model: `model_call`, with a zero-based or one-based `turn` used consistently;
- tool: the concrete action class name, plus `action_type` and `event_id`.

Use `NEMO_GYM_TIMELINE_FPATH` for the file inside the agent container and
`NEMO_GYM_ROLLOUT_ID` to correlate events. Add `status: "ok"` on success and
`status: "error"` plus `error_type` on exceptions, then re-raise the original
exception unchanged.

Successful `model_call` events include token counts copied from the live
OpenAI-compatible response:

| Argument | Meaning |
|---|---|
| `prompt_tokens` | Total input/context tokens for this request |
| `decode_tokens` | Output tokens decoded for this request; copied from `usage.completion_tokens` |

Collect these values at response time. Do not parse prompts, responses, token
ID arrays, or persisted rollout `output.jsonl` files. Keep `decode_tokens`
independent of cache-detail availability.

When the Gym model proxy exposes multiple generation endpoints, successful
calls also include:

| Argument | Meaning |
|---|---|
| `generation_replica_idx` | Zero-based index of the selected DP generation endpoint |

The index identifies a complete generation replica, not one TP rank. Sticky
session routing means all turns in one rollout should normally use the same
index.

Successful calls may also include LLM-server (vLLM or TRT-LLM) request milestones copied from the live
response:

| Argument | Meaning |
|---|---|
| `llm_arrival_ts_us` | Epoch microseconds when the LLM server frontend accepted the request |
| `queued_ts_us` | Epoch microseconds when the request entered the LLM engine scheduler queue |
| `first_scheduled_ts_us` | Epoch microseconds for the request's first engine scheduling event (vLLM `SCHEDULED`) |
| `first_token_ts_us` | Epoch microseconds for the first output-bearing engine iteration |
| `last_token_ts_us` | Epoch microseconds for the final output-bearing engine iteration |
| `llm_response_ready_ts_us` | Epoch microseconds when the LLM server finished constructing the full chat response |
| `gym_llm_response_received_ts_us` | Epoch microseconds when the Gym proxy finished receiving/parsing the LLM server response |
| `gym_tokenize_duration_us` | Gym-local duration of the optional secondary `/tokenize` request; zero means it was not needed |
| `gym_response_ready_ts_us` | Epoch microseconds when the Gym proxy response was ready to return |
| `openhands_response_received_ts_us` | Epoch microseconds when OpenHands finished receiving/parsing the Gym response |
| `openhands_parse_validate_duration_us` | OpenHands-local response validation and metadata-extraction duration |

Require positive integer timestamps and nonnegative integer durations. Preserve
owner-local ordering: `llm_arrival_ts_us <= queued_ts_us <=
first_scheduled_ts_us <= first_token_ts_us <= last_token_ts_us <=
llm_response_ready_ts_us`, and `gym_llm_response_received_ts_us <=
gym_response_ready_ts_us`. Cross-host timestamp differences can contain small
wall-clock skew and must not be treated as local durations. The true LLM-engine
scheduler queue is `first_scheduled_ts_us - queued_ts_us`; arrival-to-queued is
frontend preprocessing and engine submission. `first_scheduled_ts_us` remains
a low-overhead proxy for GPU start, not the first CUDA-kernel timestamp. Convert
engine monotonic timestamps to epoch time in the generation server process; do
not compare or convert raw monotonic values in OpenHands on another host.

When the backend reports per-request cached-token usage, successful
`model_call` events also include:

| Argument | Meaning |
|---|---|
| `prompt_tokens` | Total prompt tokens for this request |
| `cached_prompt_tokens` | Prompt tokens reused from the LLM engine prefix/KV cache |
| `uncached_prompt_tokens` | Prompt tokens computed for this request |
| `kv_cache_hit_rate` | `cached_prompt_tokens / prompt_tokens`, from 0 to 1 |

The cache-specific fields and `prompt_tokens` must be present together. Token
counts are nonnegative integers,
`cached_prompt_tokens + uncached_prompt_tokens == prompt_tokens`, and the rate
must agree with the counts. These are token-level prefix-cache metrics for one
turn, not global cache occupancy and not a count of cache-block lookup calls.
Zero is a valid value, not a missing value: when the backend reports
`cached_prompt_tokens == 0`, retain `prompt_tokens`, set
`uncached_prompt_tokens == prompt_tokens`, and emit `kv_cache_hit_rate == 0.0`.
Implementations must use presence/`is not None` checks rather than truthiness
checks when copying these numeric fields into an event.

When the request went through a PD-disaggregated TRT-LLM server, the disagg
server relays the frontend and context-engine (prefill) leg onto the final
response (`nemo_fe_*`, `nemo_ctx_*`), and successful `model_call` events also
include:

| Argument | Meaning |
|---|---|
| `frontend_recv_ts_us` | Epoch microseconds when the disagg frontend received the request |
| `frontend_fwd_ts_us` | Epoch microseconds when the disagg frontend forwarded it to the context engine |
| `ctx_arrival_ts_us` | Epoch microseconds when the context-engine adapter accepted the request |
| `ctx_queued_ts_us` | Epoch microseconds when the request entered the context engine's scheduler queue |
| `ctx_first_scheduled_ts_us` | Epoch microseconds of the request's first context-engine schedule |
| `ctx_done_ts_us` | Epoch microseconds when the context-engine response (prefill done) was ready |
| `ctx_cached_prompt_tokens` | Prompt tokens reused from the context engine's prefix/KV cache for this request |
| `ctx_computed_tokens` | Prompt tokens the context engine actually pushed through the model, over all chunks |
| `ctx_first_begin` | Prompt position at which the first prefill chunk started |
| `ctx_num_chunks` | Number of prefill chunks |

Under context-first disaggregation these, not `cached_prompt_tokens` (which is
copied from the generation leg's `usage`), describe the prefill's prefix reuse.
`ctx_cached_prompt_tokens` is block-granular when the engine reports reuse in
KV blocks, so it is not clamped against `prompt_tokens`. Timestamps are ordered
per owner (`frontend_recv_ts_us <= frontend_fwd_ts_us`;
`ctx_arrival_ts_us <= ctx_queued_ts_us <= ctx_first_scheduled_ts_us <=
ctx_done_ts_us`); differences across owners can carry wall-clock skew.
Zero is a valid value for `ctx_cached_prompt_tokens` and `ctx_first_begin`.
Validate a disaggregated run with `validate_timeline.py
--require-disagg-ctx-metrics`.

## Rollout lifecycle records

When async rollout lifecycle logging is enabled, the trainer writes
`rollout_lifecycle.jsonl` beside `training_timeline.jsonl`. These records are
not Chrome trace events. They are point-in-time join records keyed by the same
`rollout_id` carried in each Gym timeline:

```json
{"event":"rollout_consumed","ts":123456789,"rollout_id":"nrl-abc","prompt_idx":4,"sample_idx":7,"generation_weight_version":1,"target_weight_version":2,"target_display_step":3,"consumed_weight_version":1,"consumed_display_step":2,"disposition":"consumed_early"}
```

`rollout_buffered` records contain the rollout identity, generation version,
and target version. `rollout_consumed` records repeat those fields and add the
actual consuming version and one of:

- `consumed_as_targeted`
- `consumed_early`
- `consumed_late`

Weight versions are zero-based internal coordinates. Display steps are
one-based and must be written explicitly to avoid off-by-one interpretation.
The plotter treats a buffered rollout without a consumption record as pending
or unconsumed.

## Writing rules

- Open with `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)`.
- Serialize compact JSON plus one newline and issue one `os.write`.
- Create the parent directory where appropriate.
- Catch tracing-only exceptions. Never swallow application exceptions.
- Keep arguments small and stable. Token counts may be added when already
  available without parsing or copying model text.
