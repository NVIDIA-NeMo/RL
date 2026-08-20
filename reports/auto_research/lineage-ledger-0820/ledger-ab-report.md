# Lineage-Ledger Token Capture — 5-Step Smoke A/B (2026-08-20)

**Goal:** validate the gate→ledger replacement (`token-capture-lineage-ledger-approach.md`)
end to end: external token capture ON (ledger) vs OFF (legacy token echo), both with
router replay (R3) enabled and the Gym policy model server at `num_workers=2`.

**Setup** (identical across arms, per `docs/guides/nano-swe-token-capture.md`):
Nemotron-3-Nano-30B-A3B on 6 GB200 nodes (train 4 / gen 2), `grpo.seed=43`,
`num_prompts_per_step=2`, GBS 8, 5 steps, `+policy.router_replay.enabled=true`,
windowed sampler `max_staleness_versions=1`. Capture arm additionally:
`token_capture.enabled=true`, `defer_routed_experts_to_policy=true`,
`num_finalizer_workers=2`, `NG_TIC_FP_CANONICAL=1`.
Code: RL `autoresearch/2026-08-20-lineage-ledger/impl`, Gym `token-capture-lineage-ledger`.
W&B: `PR3456-Ledger-AB-0820`.

| Arm | Job | Result |
|---|---|---|
| capture (ledger) | 6359951 | COMPLETED, 5/5 steps, 58:44 |
| legacy (token echo) | 6359952 | COMPLETED, 5/5 steps, 1:07:03 |

## Training dynamics (per step, 1→5)

| Metric | Capture (ledger) | Legacy (echo) |
|---|---|---|
| `token_mult_prob_error` | 1.0147, 1.0145, 1.0137, 1.0144, 1.0147 | 1.0199, 1.0319, 1.0657, 1.0347, 1.0283 |
| `gen_kl_error` | 0.00090, 0.00092, 0.00100, 0.00117, 0.00106 | 0.00216, 0.00403, 0.00371, 0.00377, 0.00384 |
| `reward` (mean) | 0, 0, 0.25, 0, 0 | 0, 0, 0.375, 0, 0 |
| `probs_ratio` | 1.0 all steps | 1.0 all steps |

**Verdict: training dynamics are similar and the capture arm is strictly tighter** —
the same exact-token + exact-route property the gate-based runs showed
(prior campaign bands: capture 1.012–1.017 / ~0.001 vs legacy 1.023–1.027 / ~0.003).
Both arms are far below the `seq_logprob_error_threshold=2.0` health gate, rewards
spike on the same step (3), and the ratio stays pinned at 1.0.

## Ledger health (capture arm)

- **2,821 ledger rows across 44 rollouts; zero `unresolved_parent`, zero
  `worker_capture_failed`, zero `invalid_worker_commit_coordinates`.** Lineage
  resolution and tri-state admission worked flawlessly across 2 uvicorn workers
  (no request affinity): `finalize/token_in_rate` 0.954–0.987 per group
  (last step: 306 token-in calls vs 4 text roots), `routed_experts_row_coverage=1.0`.
- **16 poison rows, all `request_finished_without_staged_coordinates`** →
  16/40 trained rollouts masked as placeholders (`invalid_row_rate` 0–0.625 per
  group; `num_valid_samples` 3–8 of 8). Root cause: ~0.57% of model calls
  (16/2,821) finish without a worker ack — agent-side aborts/timeouts plus two
  engine `ClientOSError` retry bursts — and one such call fail-closes its whole
  rollout. With ~64 calls/rollout this amplifies to ~36% rollout loss. This is
  the *designed* fail-closed behavior (identical semantics to the gate's
  `fail_call`), and the documented retry-idempotency follow-up (harness-minted
  logical ids + deterministic `model_call_id`) is the recovery path. The legacy
  arm keeps such rollouts because re-tokenized echo text doesn't need per-call
  acks — at the cost of the looser dynamics above.

## Perf

| Metric (step 5 / summary) | Capture | Legacy |
|---|---|---|
| `timing/train/total_step_time` (s) | 392.4 | 573.3 |
| `timing/train/exposed_generation` (s) | 368.6 | 542.3 |
| `timing/train/policy_training` (s) | 9.5 | 14.4 |
| `timing/train/valid_tokens_per_sec_per_gpu` | 6.84 | 8.08 |
| Row assembly (finalizer, per group, ms) | fetch+verify+linearize 250–650; tensorize 11–21; tq_put 41–44; total 313–725 | n/a (inline echo path) |
| `finalize/queue_wait_ms` | ≤0.013 | n/a |
| Wall clock, 5 steps e2e | 58:44 | 1:07:03 |

Reading: the capture arm's off-hot-path finalizer costs are sub-second per group
and its queue never backs up. Its lower `valid_tokens_per_sec_per_gpu` is an
artifact of masked placeholder rows (fewer valid tokens per identical step), not
slower machinery — total step time is *shorter* than legacy, partly because
poisoned rollouts contribute less generation. At smoke scale (2 prompts/step)
these arms are not a rigorous throughput comparison; the prior campaign's
15-step pairs remain the perf reference.

## Fixes landed to get here (all committed on the impl branch)

1. `venvs.py`: `NRL_FORCE_REBUILD_VENVS_LIST` was never consumed in this tree —
   every 08-19/08-20 capture smoke (8 jobs) died on stale worker venvs
   (`orjson` missing in `setup_token_capture`).
2. `nemo_gym.py`: tool-call-only assistant items (`content: None`) crashed the
   legacy postprocess (`TypeError`) — killed every legacy run on this branch;
   regression tests added.
3. `swe_nano.env`: `UV_CACHE_DIR_OVERRIDE` must not mount over the prefetch-venvs
   container's `/root/.cache/uv` (severs baked-venv hardlinks). Driver uv cache
   pinned to `/lustre/fsw/portfolios/llmservice/users/pthombre/uv` (never /tmp).
4. `transfer_queue.py`: ported `NRL_TQ_SKIP_RUNTIME_ENV_PIN`.
5. `swe_nano_sc.sh`: honour `SC_EXP_NAME`.

## Follow-ups

- Retry idempotency (plan's explicit deferral): recover the ~36% rollout
  poisoning from per-call aborts.
- Investigate the vLLM engine `ClientOSError` bursts (engine 13015) — connection
  drops under concurrent capture traffic.
- CI (`/ok to test`) to run the Ray-heavy RL suites (blackbox_finalizer,
  tq_token_sink, vllm hosting) that cannot run outside the container.

## Addendum: r3 rerun with the off-chain carve-out (job 6365594)

`_assemble_receipt` now ignores `request_finished_without_staged_coordinates`
failure rows off the terminal chain (commit b39dbd8cd). Capture arm rerun,
same posture, short QOS, COMPLETED 5/5 in 1:11:49.

| Metric | r2 (blanket poison) | r3 (carve-out) |
|---|---|---|
| valid rows / 40 | 24 (60%) | **32 (80%)** |
| rejection reason | 16× failure-row poison | 9× missing_terminal_row only |
| `token_mult_prob_error` | 1.0137–1.0147 | 1.0137–1.0171 |
| `gen_kl_error` | ~0.001 | ~0.001 |
| `finalize/token_in_rate` | 0.95–0.99 | 0.98–0.99 |
| ledger census | 2,821 rows / 16 uncommitted failures | 3,525 rows / 19 uncommitted failures |

The carve-out worked exactly as designed: zero rollouts were rejected for
failure rows. The remaining 9 `missing_terminal_row` rejections split into:

- 2 rollouts whose *first* call died (empty ledger — nothing to train;
  correctly masked).
- ~7 overflow rollouts where the terminal id OpenHands reported does not
  match any committed row: the harness derives it from the last llm_completion
  file (`swe_agents/app.py:3101`), which for these episodes corresponds to the
  doomed final attempt rather than the last successful completion. (In the
  other ~9 overflow rollouts this run, the last file was the last successful
  call and they trained — the carve-out's win.)

Options for the residue (not taken here): (a) harness-side — report the last
*successful* response id when the final call errors; (b) receipt-side — fall
back to the deepest committed row when the terminal is missing and all
failures are uncommitted. (b) weakens the agent-kept-response attestation, so
(a) is the better follow-up alongside retry idempotency.
