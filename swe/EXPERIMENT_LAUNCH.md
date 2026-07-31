# Experiment Launch Plan — SWE Token-Capture A/B (2026-07-28)

Concrete launch record for the run described generically in `SWE_RUN.md`.
Everything here is pinned to what will actually execute; update this file if
any pin changes before submission. **Status: READY — not yet submitted.**

## 1. Hypothesis and expected outcome

The gate-authoritative token-capture pipeline
(`docs/design-docs/tq-gym-gate-authoritative.md`) reduces HTTP bytes per
trained token and wall-clock step time vs the legacy token-echo path on a
real multi-turn agentic workload, at equal training quality.

| Expectation | Basis |
|---|---|
| Bytes/token reduction ≥ 35.6 % (the single-turn floor), toward/beyond −46.9 % (sync-prototype multi-turn) | S5 instrumented A/B; legacy pays per-turn history re-echo (~quadratic in turns), capture stays O(generated tokens) |
| `exposed_generation` and `total_step_time` at or below legacy | S5 saw −16.8 % / −9.0 % single-turn; expect relative timing win to compress on a 30B MoE (GPU decode dominates) while the bytes win grows |
| `gate/token_in_rate` → 1.0 for continuation calls | First real measurement — OpenHands marker survival is the biggest unknown (risk #1); a low rate is a correctness-preserving perf bug, not a training bug |
| Reward / KL curves in the same band on both arms | S5: identical rewards 10/10, same KL band |

## 2. Exact pins

| What | Value |
|---|---|
| RL repo | branch `yukih/sc-entrypoint` @ `e71055650` (local-only; no remote has it) |
| Gym fork (submodule) | branch `tq-gate-capture` @ `e3b3eac6` (= upstream #2124 head `32b555f04` + S1–S3 capture commits + S5 byte-counter middleware; local-only) |
| Code snapshot | `code_snapshots/sc-swe-capture` @ `e710556` (immutable run tree; both arms run from it) |
| Container | `nemo-rl:sc-swe-baked.sqsh` (biguo; nightly-062526 + baked venvs — **stale vs branch lock**, hence forced venv rebuild below) |
| Model | Qwen3-30B-A3B-Thinking-2507, from SWE1 `step_230_hf` checkpoint (bihu, run dc3m70us lineage) |
| Train data | R2E-Gym subset jsonl (sdevare), 4,518 samples |
| Site config | `test_assets/SWE/grpo_qwen3_30b_async_swe.yaml` (real sandbox-image paths; snapshot copy is authoritative at run time) |
| Derived arm configs | `test_assets/SWE/derived_configs/grpo_swe_ab_{legacy,capture}.yaml` — regenerated at each submit by `make_capture_config.py` |

Verified diff between derived arm configs: **only** `token_capture.enabled:
true` on the capture arm; `rollout_max_attempts_to_avoid_lp_nan: 1` pinned on
**both** arms (capture hard-errors without it; pinning both removes the
NaN-retry confound).

## 3. Compute shape (per job; one job per arm)

- 16 nodes × 8 GPUs (128 GPUs), SLURM `batch` partition, account
  `coreai_dlalgo_genai`, 4 h wall time, `--exclusive`,
  `--dependency=singleton` (arms queue back-to-back on comparable
  allocations), idle-GPU-reaper exemption (180 min) in the job comment.
- 8 training nodes: Megatron-Core, TP=4 EP=8 CP=4 PP=2, pad 32.
- 8 generation nodes: async vLLM, TP=2, non-colocated.
- Single-controller entrypoint (`SC_MODE=1`,
  `examples/run_grpo_single_controller.py`) + TransferQueue data plane.
- Sequence budget 131,072; GBS=64 (8 prompts/step × 8 generations/prompt);
  LR 1e-6; staleness `age=1`, `force_in_order=true`, `over_sampling=false`.
- Agent: NeMo-Gym OpenHands, per-instance Singularity sandboxes (`.sif`
  under igitman/sdevare lustre trees), up to ~100–200 turns/rollout.

## 4. Environment posture (identical on both arms)

- `NRL_FORCE_REBUILD_VENVS=true` — container venvs predate the branch's
  `uv.lock`; capture arm additionally needs the unbaked `VLLM_GYM` worker
  venv. First node-local build may take ~30 min (reaper exemption covers it).
- `GYM_VENV_DIR=/tmp/nemo_gym_venvs` — baked `/opt/gym_venvs` deps predate
  the Gym fork's floors (aiohttp bump).
- `BYTES=1` (capture-arm measurement run): `NG_HTTP_BYTES_DIR` +
  `NRL_HTTP_BYTES_DIR` → per-server JSON under `http_bytes/<exp>/`;
  forwarding into the job env added in `e71055650`.
- Secrets: `profiles/env.sh` (mode 600) sourced by the site wrapper; wrapper
  fails loudly if `WANDB_API_KEY`/`HF_TOKEN` are unset.

## 5. Run sequence and gates

All submissions via (from repo root):

```bash
SITE_WRAPPER=$PWD/test_assets/SWE/grpo_swe_tests.sh MAX_NUM_STEPS=<N> ARM=<arm> [BYTES=1] bash swe/launch_swe_ab.sh
```

| # | Run | Purpose | Gate to proceed |
|---|---|---|---|
| 1 | `ARM=legacy MAX_NUM_STEPS=3` | Flag-off smoke: branch + Gym pin reproduce the known-good legacy SWE run | Job completes 3 steps; reward/KL sane; no venv/import failures |
| 2 | `ARM=legacy MAX_NUM_STEPS=25` | Measurement baseline | Completes ≥ 20 steps cleanly |
| 3 | `ARM=capture MAX_NUM_STEPS=25 BYTES=1` | Measurement + mechanism + byte accounting | — |
| 4 | (optional) repeat pair ×2 | Variance bars; S5 assessment puts run-to-run variance at ±5–10 % | 1 pair = directional, 3 pairs = quotable |

W&B: project `pthombre-swe-capture-ab`, runs named
`swe-ab-<arm>-<MMDDHHMM>`. Checkpoints: `<repo>/results/<exp>/`.

## 6. Analysis plan

- Medians over steps 5..N (skip warm-up): `timing/train/total_step_time`,
  `timing/train/exposed_generation`, `timing/train/valid_tokens_per_sec_per_gpu`.
- Mechanism: `gate/token_in_rate` (the headline first-time measurement),
  `gate/fallback_{no_marker,fingerprint_miss,unknown_marker}` — sustained
  `fingerprint_miss` ⇒ OpenHands rewrites history (marker-survival risk #1;
  report back to the design doc §11).
- Health: `gate/capture_failed`, `train/global_valid_seqs` == GBS both arms;
  `finalize/*` latency (if it dominates dispatch → argues for H5 finalizer pool).
- Equivalence: overlay `train/reward` + `train/gen_kl_error` across arms.
- Bytes: `python swe/aggregate_perf.py <legacy_http_bytes> <capture_http_bytes>`
  → total bytes, bytes/trained-token, per-hop table.
- Results recorded in
  `docs/design-docs/tq-gym-gate-authoritative-implementation-log.md` (post-MVP
  perf report per design § 10).

## 7. Known risks (accepted, watched)

- **Gate death = silent stall** (S5 chaos finding; H1 fix queued): control
  plane retries a dead gate unboundedly → job hangs, doesn't fail.
  Supervise via `squeue` + driver log in
  `code_snapshots/sc-swe-capture/logs/slurm/`.
- **Gate buffer growth uncapped** (H5): ids-only delta forests; at 131k ×
  in-flight concurrency raw ids are ~100s MB but Python overhead is real —
  watch policy-model server RSS.
- **OpenHands marker survival unknown** — measured, not assumed
  (`token_in_rate`); low rate degrades perf, never correctness.
- **Secrets in job env**: the launcher embeds `WANDB_API_KEY`/HF token in
  the sbatch command (visible in SLURM logs; pre-existing behavior).
  Rotate tokens if logs are shared.
- Snapshot is immutable but the **derived configs are re-generated per
  submit** — do not edit the site yaml between arms of a pair.

## 8. Pre-flight checklist (all verified 2026-07-28)

- [x] S1–S5 + launcher committed (RL `e71055650`, Gym `e3b3eac6`)
- [x] Snapshot `sc-swe-capture` @ `e710556`: S5 files, `examples/swe_bench/`
      launcher, site yaml, Gym gate + byte middleware present
- [x] Dry-run both arms end-to-end (`DRY_RUN=1`): correct account, snapshot
      paths, singleton, reaper comment; derived-config diff = capture flag only
- [x] Container / model / data / sandbox images readable by pthombre
- [x] SLURM account `coreai_dlalgo_genai` valid for user (nemotron_sw_post is not)
- [x] Secrets profile in place; wrapper verified with tokens stripped from env
- [ ] SUBMIT (awaiting explicit go)
