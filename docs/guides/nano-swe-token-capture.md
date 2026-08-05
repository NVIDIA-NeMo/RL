# Nano SWE RL with Gate-Authoritative Token Capture

A reproducible 6-node recipe that runs agentic SWE RL on
Nemotron-3-Nano-30B-A3B with **exact-token capture**: the vLLM worker stages
each model call's token delta durably into the TransferQueue data plane, the
Gym gate serves verified prefix token ids back on every follow-up call
(token-in), and the trainer consumes rows rebuilt from staged deltas — no
token echo over HTTP, no re-tokenization of agent history.

It builds directly on the [Nano SWE TransferQueue
recipe](nano-swe-transferqueue.md); read that first for the cluster shape,
`swe_nano.env` setup, and SingleController constraints. This guide covers only
what capture adds. Code under test: RL PR #3456 + Gym PR #2278 (plus the
attribution and canonical-fingerprint fixes staged for #2278).

## Verified result

Measured across four legacy-vs-capture A/B pairs on 6× GB200 nodes
(15-step pair: jobs 5845431/5845433; three 12-step pairs: 5860015-22):

| | legacy (token echo) | capture (token-in) |
|---|---|---|
| Total step time (3 pairs, steps 1-12) | 7,582 ± 292 s | 7,590 ± 220 s (**+0.1 %**) |
| exposed_generation | 5,896 ± 291 s | 5,884 ± 229 s (−0.2 %) |
| HTTP bytes / 15-step run | 69.96 GB | 21.11 GB (**−70 %**) |
| worker preprocess CPU | 17-19 ks (2 renders/call) | 8-10 ks (1/call) |
| gate event-loop lag p99 | 22 ms | 4-5 ms |
| token_in_rate | n/a | **0.9999** |
| token_mult_prob_error max (3 runs) | 121 / 1.8e9 / 301 | **1.13 / 1.13 / 2.87** |
| capture staging cost | — | ~39 ms/call (tq_put p50 38 ms; 0.8 % of engine) |

Wall-clock parity, −70 % bytes, half the worker CPU, and — decisively —
clean training rows: the legacy path's re-tokenization drift produces
pathological probability-ratio errors at 20k+-token contexts; exact-token
rows do not.

## Quick start

Batch, one command from a networked shell at the repo root:

```bash
DRY_RUN=0 SC_EXP_NAME=my-capture-run NG_TIC_FP_CANONICAL=1 \
  WALLTIME=3:59:00 bash swe_nano_sc_capture.sh
```

Interactive (allocate once, iterate on the driver by hand — see the
TransferQueue guide for the attach/run-cmd workflow):

```bash
NG_TIC_FP_CANONICAL=1 bash swe_nano_sc_capture_interactive.sh
```

The legacy comparison arm is the plain `swe_nano_sc.sh` /
`swe_nano_sc_interactive.sh` from the TransferQueue recipe. Batch and
interactive build byte-for-byte the same driver command.

Two flags matter:

- `NG_TIC_FP_CANONICAL=1` — canonical text fingerprints in the gate.
  **Without it token-in silently degrades to ~0**: reasoning models echo
  history with `<think>` blocks stripped, the gate's fingerprint of the served
  turn never matches, every call falls back to text mode, and the run develops
  a +38 % generated-token divergence. With it, `token_in_rate ≈ 0.9999`.
- `WALLTIME` must be a Slurm time string (`3:59:00`). `3h` fails with
  `sbatch: error: Invalid --time specification` printed at the very *end* of
  the launch output — easy to miss. Budget ~1 h 40 of setup (venv rebuild +
  checkpoint load) before the first step.

## What the capture launcher adds, and why

Every line of the capture posture in `swe_nano_sc_capture.sh` exists because
its absence crashed a run:

| Setting | Without it |
|---|---|
| `+token_capture.enabled=true` (hydra append) | Capture never engages. It is a `+` append because no yaml in the nano chain defines the key; a plain override is rejected. |
| `NRL_DRIVER_PYTHONPATH=/opt/nemo-rl/3rdparty/Gym-workspace/Gym` | Driver `ModuleNotFoundError: nemo_gym` — the driver imports the staging record schema, and the baked driver venv has no nemo_gym. |
| `NRL_DRIVER_PIP_INSTALL=orjson` | Driver `ModuleNotFoundError: orjson` — Gym's `token_id_capture/__init__` eagerly imports the store. |
| `VllmAsyncGenerationWorker` in `NRL_FORCE_REBUILD_VENVS_LIST` | Worker `ModuleNotFoundError: orjson` — venv caching is spec-unaware and silently reuses the legacy arm's venv (RL #3456 known issue). |
| capture env set *after* sourcing `swe_nano.env` | `swe_nano.env` exports `NRL_FORCE_REBUILD_VENVS_LIST` unconditionally and clobbers an env-prefix value. |

## How the data flows

```
agent (nv-OpenHands)                     gate (vllm_model, gate mode)
  /run body carries ng_rollout_id  ───►  registers rollout, admits calls
                                          fingerprints history → resolves
                                          parent → sends exact prefix ids
                                              │  required_prefix_token_ids
                                              ▼
vLLM worker: splices prefix verbatim, renders only the new tail,
  generates, then STAGES the delta (ids+mask+logprobs) to TransferQueue
  (synchronous tq_put — durable before the call is acked) and returns
  token-light CommitCoords on the response
                                              │  coords (≈4 B/token)
                                              ▼
gate ingests coords into lineage; at /run end returns a token-free
RolloutReceipt (manifest of call_ids + staging keys) to the trainer
                                              ▼
finalizer: fetches staged deltas by key, digest-verifies, rebuilds the
exact training row, clears the staged rows
```

The heavy bytes (token arrays, logprobs) move exactly once, worker→TQ,
node-locally. The gate and the `/run` response stay token-light.

## Verifying capture is really engaged

Config echo is not evidence. Check, in order:

1. **Gate metrics in the SC step output** (grade from the SC worker
   `.out` under `<job>-logs/ray/session_*/logs/worker-*-<pid>.out` — the
   driver log's actor-stdout forwarding is not reliable):

   ```
   gate_metrics={'registered': 128.0, 'token_in': 7042.0,
                 'fallback_no_match': 0.0, 'capture_failed': 0.0, ...}
   ```

   `token_in / (token_in + fallback_*)` should be ≥ 0.99. A rate near 0 with
   large `fallback_no_match` means canonical fingerprints are off (see above).
   `unattributed_calls` > 0 means the harness is not forwarding
   `ng_rollout_id` (attribution fix missing).

2. **No finalize rejections.** `finalize: ... rejected (empty_manifest)` on
   every rollout means calls are reaching the worker without gate context —
   the run generates but nothing is trainable.

3. **TQ staging traffic**: `PUT_DATA` on the staging partition per model call
   (tens of thousands per run), not just per training batch.

4. **Training equivalence**: `token_mult_prob_error` should sit near 1.0
   (max ≲ 3). `gen_kl_error` ~0.004 matches the legacy arm.

## Perf instrumentation (optional)

`CALL_TIMING=1` (default in these launchers) sets `NRL_CALL_TIMING_DIR` /
`NG_CALL_TIMING_DIR` and emits per-call JSONL segments from every server:
worker engine/preprocess/splice, `tq_put`, gate prepare/ingest, per-path HTTP
timings, event-loop lag. Aggregate with `swe/aggregate_call_timing.py`.
`cached_tokens` on each `worker_engine` record reports vLLM prefix-cache hits
(~94 % of prompt tokens on this workload). All probes are env-gated and
dormant unless the dir is set.

## Known limits

- **Grade runs from the SC worker `.out` or W&B, never the driver log** —
  Ray's driver-log stdout forwarding dropped entire actors in testing (runs
  looked stalled while training normally).
- **Receipt-mode W&B rollout metrics are not yet comparable**:
  `gen_tokens_per_sample` counts the carried prompt tail and
  `truncation_rate` is constant on the capture arm. Compare token counts via
  the call-timing JSONL (engine `usage`) instead.
- **`global_valid_toks` runs ~15 % lower on capture** at equal work —
  masking-semantics difference under investigation; do not quote
  convergence-per-wallclock until resolved.
- **Router replay (R3)**: capture stages routed experts as TQ extras beside
  the token delta (`routed_experts_delta`), keeping the multi-MB arrays off
  HTTP — but vLLM 0.20.0's engine crashes with
  `enable_return_routed_experts=true` on this model (`IndexError` in
  `sample_tokens`); blocked on an engine fix.
- **Weight-version mixing** across spliced chains at a refit boundary is
  guarded conservatively; a per-row `weight_version` check in the finalizer
  is the hardening item.

## Related

- [Nano SWE with TransferQueue](nano-swe-transferqueue.md) — base recipe,
  cluster shape, SingleController constraints
- [Router Replay](router-replay.md) — R3 background and trainer-side replay
- `nemo_rl/data_plane/tq_token_sink.py` — the staging sink/source over TQ
- `nemo_rl/experience/blackbox_finalizer.py` — receipt → training row
- Gym `nemo_gym/token_id_capture/staging/records.py` — the wire schema
