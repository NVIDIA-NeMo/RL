---
name: collect-training-timeline
license: Apache-2.0
description: Port, enable, validate, analyze, package, and visualize end-to-end NeMo-RL training timelines, including trainer phases, rollout queueing, NeMo Gym/OpenHands model calls, generation-replica attribution, per-turn token/KV-cache metrics, LLM-server scheduling milestones, tool actions, runtime lifecycle, and final evaluation. Use when adding timeline tracing to a NeMo-RL container or revision, diagnosing missing timeline events, extracting metrics, preparing a restorable timeline-data archive, or generating an interactive HTML timeline from JSONL logs.
when_to_use:
  - "collect a training timeline"
  - "add timeline tracing to this image"
  - "why is the rollout timeline empty"
  - "plot the timeline for this run"
  - "validate the timeline JSONL"
  - "archive the timeline data"
  - "what is the prefix hit rate / TPOT per turn"
allowed-tools: Bash Read Grep Glob Edit Write
---

# Collect Training Timeline

## Objective

Collect low-risk Chrome-trace JSONL from a NeMo-RL GRPO run (training or
generation-only) with NeMo Gym/OpenHands rollouts, then validate, analyze and
render it. A complete collection produces:

- a training trace, conventionally `/logs/training_timeline.jsonl`;
- rollout targeting and consumption records, conventionally
  `/logs/rollout_lifecycle.jsonl`;
- one `timeline.jsonl` per rollout under the NeMo Gym result tree;
- an interactive HTML that aligns training and rollout activity and shows
  per-turn generation-replica attribution, input/decoded-output counts,
  prompt-token cache reuse, and LLM-server, Gym-proxy, and OpenHands response
  milestones taken from the live model response.

The trace contract is in [references/event-schema.md](references/event-schema.md).
It is backend-agnostic: vLLM and TRT-LLM both export `nemo_llm_*_ts_us`
request milestones that end up as `llm_*_ts_us` and the shared
`queued/first_scheduled/first_token/last_token_ts_us` fields.

## Required workflow

1. Inspect the target before reasoning or editing:
   - identify the image/container, the NeMo-RL source actually used (image
     copy or mounted checkout) and the Gym/OpenHands roots;
   - search for existing support with
     `rg -n "TIMELINE|timeline_span|training_timeline|lifecycle_log_path|nemo_llm_"`.
     Recent NeMo-RL already writes the training trace, the rollout lifecycle
     and (TRT-LLM backend) the request milestones; port only what is missing;
   - for the backend in use, inspect how per-request cached-token counts and
     scheduler timestamps are exposed before enabling cache and scheduling
     metrics.
2. Read [references/porting-guide.md](references/porting-guide.md) and
   [references/event-schema.md](references/event-schema.md) completely.
3. Build a revision-specific overlay outside the immutable container: copy
   fresh target-revision files, make minimal semantic edits, and review each
   diff against that same revision. Never reuse whole patched files from
   another revision because the filenames match. For NeMo Gym and OpenHands,
   apply the patches in [patches/](patches/README.md) directly only when the
   target files match `patches/image-originals.sha256`; otherwise port their
   hunks.
4. Wire the launcher: mount each overlay file read-only at its container path,
   export `NRL_TRAINING_TIMELINE=1`, `NRL_TRAINING_TIMELINE_FPATH` and
   `NRL_ROLLOUT_LIFECYCLE_FPATH` into the driver, and make `/logs` a persistent
   writable mount. A generation-only launcher must accept the same overlay and
   emit step spans without requiring Nsight.
5. Preserve all existing APIs, control flow, exception behavior, and return
   values. Timeline collection must remain best effort.
6. Compile the changed Python files, run `scripts/validate_replica_wiring.py`
   against the completed overlay, then the synthetic smoke test. The smoke test
   writes a synthetic replica index and does not validate proxy-to-agent
   propagation by itself.
7. Launch the smallest representative job, verify both trace layers, and only
   then scale up. Add `--require-model-cache-metrics` when every successful
   call of a real run is expected to expose cache details; the smoke test
   intentionally includes one response without them.
8. Run `scripts/validate_timeline.py`, then render with
   `scripts/plot_rl_timeline.py`.
9. When extracting metrics or preparing a portable archive, read
   [references/archive-timeline-data.md](references/archive-timeline-data.md)
   completely. Package canonical trace JSONL plus an explicit README and
   manifest; never package rollout response or sandbox data.

## Non-negotiable properties

- Disabled tracing behaves as before.
- Trace-write failures never fail or delay training materially.
- Use wall-clock epoch time for cross-process timestamps and a monotonic clock
  for durations.
- Write one complete JSON object per line using append semantics.
- Emit Chrome complete events (`ph: "X"`) with microsecond `ts` and `dur`.
- Do not record prompts, responses, credentials, environment dumps, or other
  sensitive payloads. Record IDs, phase names, status, and numeric metadata.
- Treat `cached_prompt_tokens` as token-level prefix reuse for one request, not
  global KV-cache occupancy or a count of cache blocks. Under PD
  disaggregation the context engine's reuse is `ctx_cached_prompt_tokens` /
  `ctx_computed_tokens`; the final response's usage describes the decode leg.
- Treat `cached_prompt_tokens == 0` as a valid reported cache miss. When the
  backend reports zero, emit the complete token-metric group; never drop token
  fields by testing their numeric values or dictionaries for truthiness.
- Record `usage.completion_tokens` as `decode_tokens` directly while handling
  the live chat-completion response. Do not recover it later from rollout
  `output.jsonl` files, and keep it independent of cache metadata.
- Record the Gym model proxy's zero-based `generation_replica_idx` on every
  successful model call. It identifies a data-parallel generation replica; all
  ranks of that replica execute the request.
- Record `first_scheduled_ts_us` from the engine's first scheduling event, not
  from the OpenHands request start or the first returned token, and `queued_ts_us`
  separately: their difference is the engine scheduler queue. Convert engine
  monotonic timestamps to epoch microseconds in the generation server process
  before they cross hosts. Keep these metrics independent of Nsight.
- Record response readiness/receipt timestamps at the LLM-server, Gym, and
  OpenHands boundaries plus the local `gym_tokenize_duration_us` and
  `openhands_parse_validate_duration_us`. A zero tokenize duration means no
  secondary tokenize request was needed.
- Generation-only collection must work without Nsight; do not mount timeline
  and profiler overlays of the same file unless they were merged and validated.
- A generation-only run has no trainer, optimizer, or weight-refit phases. Its
  training-level trace identifies each step with `total_step_time` and
  `exposed_generation` (plus `policy_training` when training time is
  simulated), alongside the normal rollout lifecycle and per-rollout events.
- Do not change replay-buffer selection policy while adding instrumentation.
  Preserve the target revision's `==`, `>=`, or other eligibility condition.
- Use the earliest included rollout as time zero in the final HTML.
- An all-rollouts timeline of every step includes every discovered rollout
  trace without lane sampling; verify that the displayed rollout count equals
  the number of discovered `timeline.jsonl` files. With `--step`, it includes
  every rollout overlapping the selected steps.
- A restorable data archive includes `training_timeline.jsonl`,
  `rollout_lifecycle.jsonl`, every discovered per-rollout `timeline.jsonl`, an
  archive README, and a checksum manifest. Exclude rollout `output.jsonl`,
  prompts, responses, sandbox contents, credentials, and environment dumps.

## Validation commands

```bash
python -m py_compile <all-changed-python-files>
python scripts/validate_replica_wiring.py \
  --source-root /path/to/revision-specific-overlay \
  --generation-backend vllm        # or trtllm [--nemo-rl-root /path/to/RL]
python scripts/smoke_timeline.py \
  --source-root /path/to/revision-specific-overlay \
  --output-dir /tmp/timeline-smoke
python scripts/validate_timeline.py \
  --training-timeline /tmp/timeline-smoke/training_timeline.jsonl \
  --rollout-root /tmp/timeline-smoke/nemo_gym \
  --rollout-lifecycle /tmp/timeline-smoke/rollout_lifecycle.jsonl \
  --require-generation-replica \
  --require-model-token-counts \
  --require-model-scheduling-timestamps
python scripts/plot_rl_timeline.py \
  --training-timeline /path/to/training_timeline.jsonl \
  --rollout-root /path/to/nemo_gym \
  --rollout-lifecycle /path/to/rollout_lifecycle.jsonl \
  --output /path/to/all_rollouts_timeline.html \
  --zero-at-first-rollout \
  --include-all-rollouts
```

`smoke_timeline.py` loads `nemo_rl/utils/timer.py` and the OpenHands timeline
helper from `--source-root`; point it at a root that holds both (for example a
directory of symlinks to the NeMo-RL checkout and the overlay's `3rdparty/`).

If only one trace layer is available during an early smoke test, pass
`--allow-partial` to the validator. Do not call the port complete until both
training and rollout traces validate. On the representative real run, pass
`--require-generation-replica`; a synthetic smoke result alone is insufficient
evidence that replica attribution works end to end.

For a large run, prefer `--max-rollouts` / `--rollouts-per-step` sampling or
one `--step N --include-all-rollouts` plot per step: an all-rollouts HTML of
thousands of rollouts can reach hundreds of MB.

## Completion report

Report:

- target revision or image identifier and generation backend;
- overlay location and changed files;
- exact environment variables and mounts;
- compile/smoke-test results;
- trace paths, event counts, and HTML path;
- archive path, member counts, checksum result, and documented exclusions when
  packaging timeline data;
- whether per-turn generation-replica attribution, input/decode counts, cache
  metrics, scheduling milestones, and detailed response-path timing were
  available and their exact semantics;
- any event types unavailable in that revision and why.
