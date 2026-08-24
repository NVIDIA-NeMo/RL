# Nano SWE RL with Ledger-Authoritative Token Capture

A reproducible 6-node recipe that runs agentic SWE RL on
Nemotron-3-Nano-30B-A3B with **exact-token capture** enabled: the vLLM worker
stages each model call's token delta durably into the TransferQueue data
plane, the Gym capture ledger serves verified prefix token ids back on every follow-up
call (token-in), and the trainer consumes rows rebuilt from the staged deltas.
No token echo over HTTP, no re-tokenization of agent history — the tokens the
engine sampled are byte-for-byte the tokens the trainer sees.

It builds directly on the [Nano SWE TransferQueue
recipe](nano-swe-transferqueue.md); read that first for the cluster shape,
`swe_nano.env` setup, and the SingleController constraints. This guide covers
only what token capture adds.

## Verified result

Pool-only smoke run on 6 GB200 NVL72 nodes (Slurm job `6294776`,
2026-08-18):

| | |
|---|---|
| Entrypoint | `examples/run_grpo_single_controller.py` |
| Config | `examples/configs/ultra/nano_swe_teacher_sc.yaml` |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Shape | train 4 nodes (TP2·PP2·CP4, EP1) / gen 2 nodes (vLLM TP2 → 4 engines) |
| Smoke sizing | 2 prompts × 4 generations = GBS 8, one training step |
| Finalization | fixed pool of 2 CPU Ray actors; routed-expert assembly deferred to policy workers |
| Outcome | exit 0, step 1/1, `global_valid_seqs=8`, `invalid_row_rate=0`, routed-expert coverage 1.0, zero capture failures |
| Timing | 33m29s wall clock; 765.4s setup on warm shared caches |
| Launched via | `swe_nano_sc_capture.sh` (batch) |

## How it works

Legacy SWE runs recover training tokens by echoing token ids in every model
response and re-tokenizing the agent's rendered history each turn. Both are
lossy for a reasoning model — the chat template strips `<think>` content from
history, and re-tokenizing an assistant turn can split differently than the
tokens the model actually sampled. Token capture replaces that path.

### Anatomy of one rollout

The call flow for a single SWE rollout, end to end:

1. **Dispatch.** The `SingleControllerActor`'s rollout pump pulls a prompt
   group from the dataset and hands it to the NemoGym environment actor,
   which POSTs `/run` to the `swe_agents_train` Gym server. The run body
   carries a fresh `ng_rollout_id`. There is no registration step: the
   rollout's ledger file is created lazily by its first committed call.
2. **Sandbox + agent.** The SWE harness materializes the SWE-bench instance
   in a sandbox and starts the OpenHands agent. The rollout id is snapshotted
   into the agent's config so every LLM request uses the rollout-prefixed
   token-capture route.
3. **Agent turn loop.** Each turn, OpenHands POSTs `/v1/chat/completions`
   with the *full rendered history* to the policy model server, which runs
   in **ledger mode**:
   - the server fingerprints the incoming history and resolves which
     committed ledger row this call continues (its *parent*); admission is a
     pure function of that lineage result;
   - on a match it attaches the parent's **exact cumulative token ids**
     (`required_prefix_token_ids`) plus a capture context
     (`rollout_id`, `model_call_id`, `parent_call_id`, `prev_len`), and
     forwards the request to a vLLM engine;
   - no assistant-authored history is a true text root; assistant history
     without one verified committed parent is `UNRESOLVED` and fails closed
     instead of silently starting a new chain.
   Lineage and capture custody are stored together in a per-rollout
   append-only JSONL file under the shared capture directory, so consecutive
   calls may land on different serving workers without losing ancestry or
   serializing unrelated rollouts.
4. **Generate + stage.** The vLLM worker splices the supplied prefix
   verbatim, renders only the new tail through the chat template, and
   generates. Before acknowledging the call it **stages the call's token
   delta** — `rendered_prompt[prev_len:] + generated` ids, a loss mask
   (0.0 on carried prompt, 1.0 on generated), and per-token logprobs — to
   the TransferQueue staging partition (synchronous `tq_put`: bytes are
   durable before the response leaves the worker). The response back to the
   ledger carries only text plus token-light `CommitCoords` (~4 B/token).
5. **Commit.** Gym reconstructs cumulative tokens from the coordinates and
   appends one ledger row carrying both lineage and staging custody
   (including the call's `logical_request_id`). This is the authoritative
   commit the next turn resolves against; Gym strips the coordinates and
   returns a plain OpenAI-shaped completion to the agent. Steps 3-5 repeat for every tool call the agent
   makes (tool execution happens agent-side between turns).
6. **Verify + assemble.** When the agent finishes, the harness runs the
   SWE-bench verifier to score the patch (reward 0/1) and reports the
   terminal response id. RL fetches the rollout's **token-free manifest**
   (`GET /training-token-capture/rollouts/{id}/manifest`) and assembles the
   `RolloutReceipt` locally: the manifest of committed calls (model-call ids,
   staging keys, digests, weight versions), a `terminal_model_call_id`
   selected by the terminal logical request id, and fail-closed poisoning
   (any failure row, or a missing terminal row, masks the rollout).
   Harnesses that report no terminal id (declared > heuristic > mask
   precedence) fall back to Gym's `select_terminal_call`, which infers the
   terminal from the manifest's parent links and masks on any ambiguity; the
   per-group `finalize/heuristic_terminal_fraction` metric meters that
   fallback and should stay 0 on the SWE recipe, whose harness declares.
7. **Finalize.** The controller constructs a metadata-only
   `FinalizationRequest`, releases the rollout concurrency permit, and submits
   it to a fixed pool of CPU Ray actors. Each actor owns a connect-only TQ
   client and a `BlackboxFinalizer`; it fetches the staged deltas named by the
   manifest, re-verifies them (digest, lengths, mask shape, weight-version
   tags), linearizes the terminal chain into one exact token row, and publishes
   it to the training partition. A rejected rollout becomes a masked
   placeholder so the GRPO group keeps its shape. The pool is the only
   token-capture finalization path; there is no inline controller finalizer.
   In the deferred-route posture below, staged rows remain until policy workers
   assemble routes and the training step consumes them; direct mode clears them
   immediately after publication.
8. **Train.** Once a global batch of rows is buffered, the SC takes an
   optimizer step and syncs weights to the engines; the bumped
   `weight_version` is stamped on subsequent calls so refit boundaries are
   visible in the data.

The token path in that flow, compressed:

```
agent (nv-OpenHands)                     ledger (vllm_model, external staging)
  rollout-prefixed capture path    ───►  fingerprints incoming history →
                                         resolves the parent call → sends the
                                         parent's exact prefix token ids
                                              │  required_prefix_token_ids
                                              ▼
vLLM worker: splices the prefix verbatim, renders only the new tail,
  generates, then STAGES the call's token delta (ids + mask + logprobs)
  to TransferQueue — a synchronous tq_put, durable before the call is
  acked — and returns token-light CommitCoords on the response
                                              │  coords (≈4 B/token)
                                              ▼
ledger atomically publishes coords + lineage; when the rollout ends RL
fetches the token-free manifest (model_call_ids and staging keys) from the
control route and assembles the RolloutReceipt itself
                                              ▼
fixed CPU Ray finalizer pool: accepts metadata only, fetches staged deltas
by key, verifies digests, rebuilds the exact row, and publishes it to TQ
```

The heavy bytes (token arrays, logprobs) move exactly once, worker→TQ,
node-locally. The ledger hop and the `/run` response stay token-light.

The pieces, by repo:

| Component | Where |
|---|---|
| Ledger mode, prefix serving, lineage | Gym `responses_api_models/vllm_model/app.py` + `nemo_gym/token_id_capture/lineage.py` |
| Rollout attribution | Gym capture middleware + `swe_agents` rollout-prefixed routing |
| Wire schema (deltas, coords, receipts) | Gym `nemo_gym/token_id_capture/staging/records.py` |
| Worker-side capture + prefix splice | `nemo_rl/models/generation/vllm/vllm_worker_async.py` |
| Staging sink/source over TransferQueue | `nemo_rl/data_plane/tq_token_sink.py` |
| Receipt → training row | `nemo_rl/experience/blackbox_finalizer.py` |
| Metadata-only finalizer actor pool | `nemo_rl/experience/finalizer_actor.py` |

## Quick start

### Prepare the checkout

Run from a networked shell at this repository's root. Before submitting,
change every per-user write setting in `swe_nano.env`:

| Variable | Required value |
|---|---|
| `CODE_DIR` | Absolute path to **this checkout**; the launcher mounts it into the container |
| `WORKSPACE_DIR` | Writable results, Ray-log, and checkpoint root |
| `HF_HOME` | Writable Hugging Face cache (~60 GB for the model) |
| `PERSISTENT_CACHE` | Writable vLLM, Triton, and Inductor cache |
| `NRL_MEGATRON_CHECKPOINT_DIR` | Writable Megatron conversion cache, normally below `PERSISTENT_CACHE` |
| `SLURM_ACCOUNT` | Slurm account you can charge |

The container, SWE data, sandbox SIFs, and model name in the shared read-only
block can be reused. Keep `USE_SNAPSHOT=0` to execute the live files in
`CODE_DIR`; set a unique `SC_EXP_NAME` and staging partition for every active
run. Export `HF_TOKEN` if the model is not already cached.

### Reproduce the one-step, six-node smoke

The following is the pool-only posture validated by job `6294776`. The batch
wrapper itself supplies `token_capture.enabled=true` and pins Gym rollout
attempts to one:

```bash
CAPTURE_OVERRIDES=(
  grpo.max_num_steps=1
  grpo.num_prompts_per_step=2
  policy.train_global_batch_size=8
  token_capture.num_finalizer_workers=2
  token_capture.defer_routed_experts_to_policy=true
  token_capture.staging_partition=rollout_staging_my_capture_smoke
  +env.nemo_gym.policy_model.responses_api_models.vllm_model.num_workers=2
  +policy.router_replay.enabled=true
  async_rl.sampler.name=windowed
  +async_rl.sampler.max_staleness_versions=1
  +env.nemo_gym.model_endpoint_readiness_timeout_seconds=1800
  policy.generation.vllm_cfg.reasoning_parser_plugin=/opt/nemo-rl/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py
)

# Resolve and inspect the six-node driver command without submitting.
DRY_RUN=1 SC_EXP_NAME=my-capture-smoke NG_TIC_FP_CANONICAL=1 \
  WALLTIME=1:49:00 bash swe_nano_sc_capture.sh "${CAPTURE_OVERRIDES[@]}"

# Submit the same command.
DRY_RUN=0 SC_EXP_NAME=my-capture-smoke NG_TIC_FP_CANONICAL=1 \
  WALLTIME=1:49:00 bash swe_nano_sc_capture.sh "${CAPTURE_OVERRIDES[@]}"
```

`num_prompts_per_step × num_generations_per_prompt` must equal
`train_global_batch_size`; this config has four generations per prompt, hence
`2 × 4 = 8`. The finalizer pool is mandatory whenever capture is enabled,
and `num_finalizer_workers` must be positive. There is no pool enable/disable
or legacy-inline-finalizer flag. The smoke deliberately runs two Gym policy
model workers; their per-rollout ledger files are shared, so it does not rely
on single-worker request affinity. Different rollouts use different locks.

The 1h49 allocation is suitable when the model and compilation caches are
warm and automatically selects the `short` QOS in `ultra_launch.sh`. For a
cold cache, or when the short-QOS node quota is unavailable, use
`WALLTIME=3:59:00`; `swe_nano.env` leaves `SLURM_QOS` empty for that case.
`WALLTIME` must be a Slurm time string—a value such as `3h` is invalid.

For a longer run, change `grpo.max_num_steps` and use the longer walltime. Do
not reuse a staging partition concurrently with another run.

### Interactive iteration

The interactive launcher accepts the same override array:

```bash
SC_EXP_NAME=my-capture-smoke-interactive NG_TIC_FP_CANONICAL=1 \
  WALLTIME=3:59:00 bash swe_nano_sc_capture_interactive.sh \
  "${CAPTURE_OVERRIDES[@]}"
```

It allocates six nodes, keeps Ray alive, and prints commands equivalent to:

```bash
bash <jobid>-attach.sh
source <jobid>-run-cmd.sh
```

Attach to the head node and source the generated run command after Ray is up;
edit and re-source it to iterate in the same allocation. The non-capture
baseline remains `swe_nano_sc.sh` / `swe_nano_sc_interactive.sh`. If batch and
interactive runs overlap, first change the staging-partition value in
`CAPTURE_OVERRIDES` so each live run has its own partition.

## What the capture launchers add, and why

Every line of the capture posture in `swe_nano_sc_capture*.sh` exists because
its absence broke a run:

| Setting | Without it |
|---|---|
| `token_capture.enabled=true` | Capture never engages. The launcher flips the config's default. Enabling it always constructs the fixed CPU finalizer actor pool sized by `token_capture.num_finalizer_workers`; there is no inline fallback. |
| `token_capture.defer_routed_experts_to_policy=true` + `+policy.router_replay.enabled=true` | The validated R3 posture keeps routed-expert tensors out of controller RPCs and canonical rows, then reconstructs them on policy workers from staged-fragment plans. Set both together. |
| `async_rl.sampler.name=windowed` + `max_staleness_versions=1` | The smoke does not exercise the intended bounded-staleness sampling policy. |
| `NG_TIC_FP_CANONICAL=1` | Reasoning models otherwise echo history with `<think>` blocks stripped, so the ledger cannot verify a unique parent and fails the call as `UNRESOLVED`. With canonical fingerprints, `token_in_rate ≈ 0.9999`. |
| `NRL_DRIVER_PYTHONPATH=/opt/nemo-rl/3rdparty/Gym-workspace/Gym` | Driver `ModuleNotFoundError: nemo_gym` — the driver imports the staging record schema, and the baked driver venv has no nemo_gym. |
| `NRL_DRIVER_PIP_INSTALL=orjson` | Driver `ModuleNotFoundError: orjson` — Gym's `token_id_capture/__init__` eagerly imports the store. |
| `NRL_DRIVER_UV_RUN_FLAGS="--locked --no-sync"` | `uv run` otherwise replaces the prefetched driver environment and can give the driver a different Python/Ray version from the already-running Ray cluster. Lock mutation is forbidden; worker-specific environments are still rebuilt. |
| `VllmAsyncGenerationWorker` in `NRL_FORCE_REBUILD_VENVS_LIST` | Worker `ModuleNotFoundError: orjson` — venv caching is spec-unaware and silently reuses a non-capture worker venv built by an earlier job. |
| capture env set *after* sourcing `swe_nano.env` | `swe_nano.env` exports `NRL_FORCE_REBUILD_VENVS_LIST` unconditionally and clobbers an env-prefix value — which is why these are dedicated launchers rather than an env prefix on `swe_nano_sc.sh`. |
| `CALL_TIMING=0` (optional, batch) | Per-call latency JSONL is on by default in the batch launcher (`NRL_CALL_TIMING_DIR`/`NG_CALL_TIMING_DIR`); set 0 to disable. All probes are env-gated and dormant without the dir. |

## Verifying capture is really engaged

Config echo is not evidence. Check, in order:

1. **Ledger-derived admission counters in the finalize metrics.** Each
   manifest row records its admission mode, and the finalizer aggregates them
   per group into `step_metrics` (W&B prefix `train`):

   ```
   finalize/token_in_calls, finalize/text_root_calls,
   finalize/token_in_rate, finalize/capture_poisoned_rollouts
   ```

   `finalize/token_in_rate` should be ≥ 0.99 after the root calls (each chain
   opens with exactly one `text` root). A rate near 0 with a large
   `text_root_calls` count means canonical fingerprints are off (see above).
   Nonzero `capture_poisoned_rollouts` means calls are failing admission or
   commit — check the model-server logs for `unresolved_parent` /
   `worker_capture_failed` poison reasons.

2. **Finalizer-pool health.** In `step_metrics`, require
   `finalize/invalid_row_rate=0` and, for the R3 command above,
   `finalize/routed_experts_row_coverage=1`. The one-step smoke reported
   `finalize/queue_depth=0` and `finalize/active_actor_count=1`; queue depth can
   be nonzero under heavier load. A `finalizer actor RPC failed after
   submission` message is fatal because the publication outcome is unknown
   and actors are deliberately not retried.

3. **TQ staging traffic**: `PUT_DATA` on the staging partition fires per
   model call (tens of thousands per run), not just per training batch.

4. **Training equivalence**: `token_mult_prob_error` should sit near 1.0
   (max ≲ 3), `gen_kl_error` in the same band as a non-capture run (~0.004).

## Known limits

- **Grade runs from the SC worker `.out` or W&B, never the driver log** —
  Ray's driver-log stdout forwarding dropped entire actors in testing (runs
  looked stalled while training normally).
- **Receipt-mode W&B rollout metrics are not yet comparable to legacy**:
  `gen_tokens_per_sample` counts the carried prompt tail and
  `truncation_rate` is constant on the capture arm.
- **Weight-version mixing** across a spliced chain is tag-checked per call.
  The recipe defaults to `mixed_weight_version_policy=allow` and stamps the
  row with the group's oldest version for staleness accounting; set it to
  `reject` to emit a placeholder for a mixed-version rollout.
- **Router replay (R3)** is enabled in the verified pool-only smoke. Routed
  experts are staged beside token deltas and reconstructed on policy workers;
  keep `defer_routed_experts_to_policy=true` paired with
  `policy.router_replay.enabled=true`.
- **Shutdown noise after a completed smoke** can include forced Ray/Gym actor
  teardown because the asynchronous rollout pump may have work beyond the
  final requested train step. Grade the run from the completed `train step
  1/1`, metrics, and Slurm exit code rather than teardown warnings alone.

## Related

- [Nano SWE with TransferQueue](nano-swe-transferqueue.md) — base recipe,
  cluster shape, SingleController constraints
- [Router Replay](router-replay.md) — R3 background and trainer-side replay
- `nemo_rl/data_plane/tq_token_sink.py` — the staging sink/source over TQ
- `nemo_rl/experience/blackbox_finalizer.py` — receipt → training row
- Gym `nemo_gym/token_id_capture/staging/records.py` — the wire schema
