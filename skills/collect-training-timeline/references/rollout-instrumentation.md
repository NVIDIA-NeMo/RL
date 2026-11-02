# Rollout instrumentation: NeMo Gym and OpenHands

The per-rollout `timeline.jsonl` is written by two processes:

- the NeMo Gym SWE agent on the host, which writes orchestration spans;
- OpenHands inside the agent container, which writes runtime, agent, model and
  tool spans.

Per-request metrics travel on the live chat-completion response, through
three hops:

```
LLM server (nemo_llm_*, nemo_ctx_*, nemo_fe_*, usage)
  -> Gym model proxy (adds nemo_generation_replica_idx, nemo_gym_*)
  -> OpenHands NeMo Gym client (copies them onto the model_call span)
```

Make the changes below against the Gym and OpenHands revisions you actually
run. OpenHands is cloned by Gym from `agent_framework_repo` /
`agent_framework_commit` (see `responses_api_agents/swe_agents/configs/`), so
patch that checkout. [`../patches/`](../patches/README.md) holds these changes
as ready-to-apply patches for one baseline.

## Shared span writer

Both sides need the same best-effort helper. It appends one Chrome complete
event per call, and on an exception it records `status: "error"` and re-raises.

```python
def append_event(path, name, cat, rollout_id, start_epoch_ns, dur_ns, **args):
    event = {"name": name, "cat": cat, "ph": "X",
             "ts": start_epoch_ns / 1_000, "dur": max(dur_ns, 0) / 1_000,
             "pid": os.getpid(), "tid": threading.get_native_id(),
             "args": {"hostname": socket.gethostname(), "rollout_id": rollout_id, **args}}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (json.dumps(event, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        pass

@contextmanager
def timeline_span(path, name, cat, rollout_id, **args):
    t0, p0 = time.time_ns(), time.perf_counter_ns()
    try:
        yield args          # callers may add numeric fields while inside
    except BaseException as e:
        append_event(path, name, cat, rollout_id, t0, time.perf_counter_ns() - p0,
                     **args, status="error", error_type=type(e).__name__)
        raise
    append_event(path, name, cat, rollout_id, t0, time.perf_counter_ns() - p0,
                 **args, status="ok")
```

Use wall-clock `time_ns()` for `ts`, because it is compared across hosts. Use
`perf_counter_ns()` for durations.

## NeMo Gym

1. **Response model** (`nemo_gym/openai_utils.py`). Declare every forwarded
   field as `Optional[int] = None` on `NeMoGymChatCompletion`; otherwise
   validation drops them. The fields are:
   - `nemo_generation_replica_idx`;
   - `nemo_llm_{arrival,queued,first_scheduled,first_token,last_token,response_ready}_ts_us`;
   - `nemo_gym_{llm_response_received_ts_us,tokenize_duration_us,response_ready_ts_us}`;
   - `nemo_fe_{recv,fwd}_ts_us`;
   - `nemo_ctx_{arrival,queued,first_scheduled,done}_ts_us`;
   - `nemo_ctx_{cached_tokens,computed_tokens,first_begin,num_chunks}`.

2. **Model proxy** (`responses_api_models/vllm_model/app.py`), on both the
   chat-completions and completions paths:
   - In `_resolve_client`, remember the chosen client index per session, in a
     `session_id -> idx` map that is cleared together with the session-to-client
     map. Do not change the routing policy.
   - Record `nemo_gym_llm_response_received_ts_us = time_ns() // 1000` right
     after the upstream call returns.
   - Time any fallback `/tokenize` call with `perf_counter_ns()` into
     `nemo_gym_tokenize_duration_us`; it is `0` when no tokenize call is made.
   - On the validated response, set `nemo_generation_replica_idx` and
     `nemo_gym_response_ready_ts_us = time_ns() // 1000` just before
     returning. Set the replica index on the context-length-exceeded empty
     response too.

3. **SWE agent** (`responses_api_agents/swe_agents/app.py`):
   - Only when `NRL_COLLECT_TIMELINE_FIELDS=1`, `touch` `<persistent_dir>/timeline.jsonl`
     before the agent container starts, so it exists to be bind mounted.
   - Carry a `timeline_rollout_id` on the run config. Use NeMo Gym's run-body
     correlation id `"{_ng_task_index}-{_ng_rollout_index}"` when present, else
     `agent_run_id`. The replay buffer's `rollout_lifecycle.jsonl` uses the same
     id, which is what joins the two files.
   - In the agent container command, export
     `NEMO_GYM_TIMELINE_FPATH=<base_mounted_dir>/timeline.jsonl` and
     `NEMO_GYM_ROLLOUT_ID=<shlex-quoted id>`.
   - Emit these `orchestration` spans into the host-side `timeline.jsonl`:
     - `ray_queue`, from `ray_queue_timestamp` to run start;
     - `agent_container_spinup` and `eval_container_spinup`, from the start
       command to the spin-up timestamp file the container writes;
     - `openhands_run` and `final_eval`, by wrapping the matching
       `_finish_container_command`.

## OpenHands

1. **Helper** (new `openhands/utils/nemo_gym_timeline.py`):
   - `nemo_gym_timeline_span(name, cat, **args)` is the span writer above. It
     reads `NEMO_GYM_TIMELINE_FPATH` and `NEMO_GYM_ROLLOUT_ID`, and is a no-op
     when the path is unset.
   - Add three pure functions that turn a response dict into timeline fields:

     | Function | Output |
     |---|---|
     | `cache_metrics(usage)` | `prompt_tokens`, `decode_tokens` (= `completion_tokens`); `cached_prompt_tokens`, `uncached_prompt_tokens` and `kv_cache_hit_rate` only when `0 <= cached <= prompt` |
     | `llm_timing_metrics(resp)` | the `nemo_llm_*` and `nemo_gym_*` fields, with prefixes stripped: `nemo_llm_queued_ts_us -> queued_ts_us`, `nemo_llm_arrival_ts_us -> llm_arrival_ts_us`, `nemo_llm_response_ready_ts_us -> llm_response_ready_ts_us`, `nemo_gym_X -> gym_X` |
     | `disagg_metrics(resp)` | `nemo_fe_* -> frontend_*`, `nemo_ctx_* -> ctx_*`, and `nemo_ctx_cached_tokens -> ctx_cached_prompt_tokens` |

   - Accept only non-bool ints. Timestamps must be `> 0`; counts may be `0`.

2. **NeMo Gym client** (`openhands/agenthub/nemo_gym_client.py`):
   - In `_post_completion`, record `openhands_response_received_ts_us` after the
     body is parsed, then merge the three helpers' output and
     `generation_replica_idx` into `provider_specific_fields`.
   - Time the parse and validate step into
     `openhands_parse_validate_duration_us`.
   - Always assign `response._provider_specific_fields`. The token-id fields
     are still only added when present.
   - In `model_call`, wrap the request in
     `nemo_gym_timeline_span("model_call", "model", turn=n)` and copy those
     provider fields onto the span args.

3. **Tool actions** (`openhands/runtime/base.py`): wrap
   `_handle_action(event)` in a span named `type(event).__name__` with category
   `tool` and args `action_type` and `event_id`.

4. **Agent lifecycle** (`evaluation/benchmarks/swe_bench/run_infer.py`): wrap
   these steps.

   | Step | Span | Category |
   |---|---|---|
   | `create_runtime` | `create_runtime` | `runtime` |
   | `runtime.connect` | `connect_runtime` | `runtime` |
   | `initialize_runtime` | `initialize_runtime` | `runtime` |
   | `get_instruction` | `prepare_instruction` | `agent` |
   | `run_controller` | `run_controller` | `agent` |
   | `complete_runtime_fn` | `extract_patch` | `agent` |
   | `runtime.close` | `runtime_cleanup` | `runtime` |

## Rules

- Do not change control flow, return values, routing or exception types; only
  wrap.
- Never put prompts, responses or token-id arrays into span args.
- Every new response field must be declared at every hop: the Gym response
  model, then the OpenHands helper mapping. A field added on the LLM server
  alone is silently dropped.

## Verify

Run one short rollout with `NRL_COLLECT_TIMELINE_FIELDS=1`, then check its
`timeline.jsonl`:

```bash
f=$(find $GYM_RESULTS_DIR -name timeline.jsonl -size +0 | head -1)
jq -r '.name' "$f" | sort | uniq -c        # orchestration, runtime, agent, model_call, tool spans
jq -c 'select(.name=="model_call" and .args.status=="ok") | .args
       | {turn, generation_replica_idx, prompt_tokens, decode_tokens,
          cached_prompt_tokens, queued_ts_us, first_scheduled_ts_us,
          last_token_ts_us}' "$f" | head -3
```

If `model_call` spans exist but the LLM fields are missing, check the hops in
order:

1. Does the raw server response carry `nemo_llm_*`? If not, the flag is not
   set on that node.
2. Does Gym's response model declare the field?
3. Does the OpenHands mapping include it?

If the file is empty, the container did not receive `NEMO_GYM_TIMELINE_FPATH`,
or the path is not bind mounted.
