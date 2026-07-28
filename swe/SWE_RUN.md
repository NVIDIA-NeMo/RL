# SWE Run — Token-Capture vs Legacy Perf A/B at Scale

Runbook for measuring the performance of the gate-authoritative token-capture
pipeline (`token_capture.enabled=true`, design:
`docs/design-docs/tq-gym-gate-authoritative.md`) against the legacy token-echo
path on a real workload: **async GRPO on SWE-bench** (Qwen3-30B-A3B thinking,
NeMo-Gym OpenHands agent, 16 nodes). This is the multi-turn run the 2-GPU S5
experiments could not provide — SWE agents run up to 100–200 turns per
rollout, which is exactly where the legacy echo pays its per-turn token
re-transmission and where capture's `token_in_rate` and prefix serving are
finally exercised for real.

Small-scale S5 results this run extends (implementation log,
`docs/design-docs/tq-gym-gate-authoritative-implementation-log.md` § S5):
−35.6 % HTTP bytes/trained token, −9 % step time, −16.8 % generation time on
a *single-turn* workload — the floor. The sync prototype measured −46.9 %
bytes/token multi-turn.

---

## 1. The stack being launched

```
swe/launch_swe_ab.sh  (this folder: picks the arm, derives the capture yaml)
  └─ site wrapper      e.g. .../nemo_rl-sc-test/test_assets/SWE/grpo_swe_tests.sh
       (account, container, secrets, lustre paths — user-specific, gitignored)
       └─ snapshot launcher   <snapshot>/examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh
            (env-agnostic: parallelism, async_rl/data_plane overrides, sbatch)
            └─ ray.sub → SLURM (16 nodes: 8 train TP4/EP8/CP4/PP2 + 8 gen vLLM TP2,
                                seqlen 131072, SC_MODE=1 single-controller + TQ)
```

Jobs run from an immutable **code snapshot**
(`test_assets/SWE/sync_code.sh`). The snapshot rsyncs `3rdparty/` wholesale,
so the **local-only Gym fork commits ride along** — the open gitlink/CI
question does not block this experiment.

## 2. The A/B design

One independent variable. Everything else — snapshot, container, data, model,
parallelism, async_rl knobs, node count — identical between arms.

| | Arm A: `legacy` | Arm B: `capture` |
|---|---|---|
| Config | site yaml as-is | site yaml + `token_capture.enabled: true` (derived by `make_capture_config.py`) |
| Token path | gate string-parses ids + `/tokenize` per call; ids/logprobs echoed through agent messages every turn; tokens ride the Ray return | gate custodies lineage; worker stages delta ids+logprobs to TQ once; markers + token-free receipts |
| Rows into training | `TQReplayBuffer.commit` tensorize | `BlackboxFinalizer` verify → publish |

Both arms use the launcher's defaults, which sit inside the capture MVP
matrix: `batch_selection_strategy=staleness_window`, age 1,
`mixed_weight_version_policy=allow` (default), async vLLM engine,
non-colocated generation. `make_capture_config.py` also pins
`rollout_max_attempts_to_avoid_lp_nan: 1` — capture setup hard-errors
otherwise (NaN-retry would re-register create-only rollout ids); it is set on
**both** arms' derived configs so retry behavior is not a confound.

### Run protocol (perf reading)

1. **Flag-off smoke first** (`MAX_NUM_STEPS=3`): proves the branch + Gym pin
   reproduce the legacy SWE run before anything is compared. The standing
   dormant-by-default discipline.
2. **Measurement runs**: `MAX_NUM_STEPS=20`–`30` per arm (long enough that
   medians beat setup noise; short enough to iterate). `--dependency=singleton`
   is already in the launcher — submit both arms back-to-back under the same
   job name family so they land on comparable allocations.
3. **Repeats**: 1 pair = directional; 3 pairs = quotable (the S5 assessment:
   run-to-run variance on this stack is plausibly ±5–10 %).

## 3. Launching

### Prerequisites (once)

- The branch state you want measured is **committed** (RL repo and Gym
  submodule) — snapshots are cut from a checkout.
- A checkout of `yukih/sc-entrypoint` (with the `tq-gate-capture` submodule
  state) on lustre to snapshot from, e.g. clone + check out, or reuse an
  existing sc-test workspace updated to this branch.
- A personal site wrapper (copy
  `test_assets/SWE/grpo_swe_tests.sh` and edit ACCOUNT / CONTAINER /
  secrets-profile / cache / W&B paths for your user).
- Snapshot it:

```bash
cd <your lustre checkout>          # repo root, branch checked out
bash test_assets/SWE/sync_code.sh sc-swe-capture
```

### Submitting the arms

```bash
# from this repo's swe/ folder
SITE_WRAPPER=/lustre/.../your_grpo_swe_tests.sh \
  MAX_NUM_STEPS=3  ARM=legacy  bash swe/launch_swe_ab.sh     # flag-off smoke

SITE_WRAPPER=... MAX_NUM_STEPS=25 ARM=legacy  bash swe/launch_swe_ab.sh
SITE_WRAPPER=... MAX_NUM_STEPS=25 ARM=capture bash swe/launch_swe_ab.sh

# optional: HTTP byte accounting (writes per-server JSON to lustre)
SITE_WRAPPER=... MAX_NUM_STEPS=25 ARM=capture BYTES=1 bash swe/launch_swe_ab.sh
```

`launch_swe_ab.sh` does four things: derives the arm's config
(`make_capture_config.py` for the capture arm), stamps
`EXP_SUFFIX=swe-ab-<arm>-...` so W&B separates the arms, forces the venv
posture below, and `exec`s your site wrapper (all launcher knobs still pass
through the environment: `TP`, `PPS`, `OVER_SAMPLING`, `DRY_RUN=1`, …).

### Environment posture (why the wrapper forces these)

- **`NRL_FORCE_REBUILD_VENVS=true` on BOTH arms.** The baked container's
  `/opt/ray_venvs` predate this branch's `uv.lock` (S1 regenerated it), and
  the capture arm additionally needs the `VLLM_GYM` worker venv
  (`--extra vllm --extra nemo_gym`) that no image bakes yet. Forcing rebuild
  on both arms keeps setup cost out of the A/B. Pre-warm `LUSTRE_UV_CACHE`
  once (see the launcher header) or the first node-local build is ~30 min —
  the 180-min idle-GPU reaper exemption in the site wrapper covers it.
- **`GYM_VENV_DIR=/tmp/nemo_gym_venvs` (node-local rebuild) on BOTH arms.**
  The baked `/opt/gym_venvs` were built against the old Gym pin; the fork
  bumped dependency floors (e.g. aiohttp). The editable install means *code*
  comes from the mounted fork either way, but stale *deps* would crash the
  policy-model server on import. Rebaking the image
  (`test_assets/SWE/prebuild_gym_venvs.sh`) removes this cost permanently.

## 4. What to compare (the perf read)

All of these land in W&B (both arms) — the capture-arm `gate/*` block comes
from the per-step SC gate-metrics logging added in S5.

**Headline perf (medians over steps 5..N, skipping warm-up):**

| Metric (W&B key) | Expectation |
|---|---|
| `timing/train/total_step_time` | capture ≤ legacy; S5 saw −9 % single-turn |
| `timing/train/exposed_generation` | the mechanism lives here (no `/tokenize` round-trip, ~40 % smaller per-call payloads, no echo growth); S5 saw −16.8 % |
| `timing/train/valid_tokens_per_sec_per_gpu` | inverse of the above |

**Capture-arm health / mechanism metrics:**

| Metric | What it tells you |
|---|---|
| `gate/token_in_rate` | THE number this run finally measures. High (→1.0 for continuations) = OpenHands echoes markers faithfully and exact-prefix serving works at depth. Low = fallbacks (see next row) — runs stay *correct* but pay full re-renders. |
| `gate/fallback_no_marker` / `fallback_fingerprint_miss` / `fallback_unknown_marker` | Which § 3.3 fallback is firing if token_in_rate is low. `fingerprint_miss` at scale ⇒ the OpenHands history pipeline rewrites messages (report back — this is marker-survival risk #1 for agent frameworks). |
| `gate/capture_failed`, `train/global_valid_seqs` | staging failures → placeholder rows. valid_seqs should equal GBS on both arms. |
| `finalize/*` (rollout metrics) | finalizer latency rides the dispatch task (MVP placement) — SWE's 131k-token multi-call rollouts make this worth watching; if it dominates dispatch, that argues for the H5 finalizer pool. |

**Training equivalence (guards the perf claim):** overlay `train/reward`
curves and `train/gen_kl_error` between arms — same band = the perf delta is
not bought with training drift. (S5: identical rewards 10/10, same KL band.)

**Optional byte accounting (`BYTES=1`):** env-gated ASGI counters
(`NG_HTTP_BYTES_DIR` on every Gym server — the SWE config runs
`num_workers: 1`, which the counter supports — and `NRL_HTTP_BYTES_DIR` on
the vLLM workers) write per-server, per-route JSON to
`<checkpoint dir>/http_bytes/`. Aggregate with `swe/aggregate_perf.py
<legacy_dir> <capture_dir>` → total bytes, bytes per trained token, per-hop
table. Expect the multi-turn reduction to land between the S5 floor
(−35.6 %) and beyond the prototype's −46.9 % as turn count grows.

## 5. Known risks at this scale (accepted, watched)

- **Gate death = silent stall, not loud failure** (S5 chaos finding, fix
  queued for H1): Gym's control-plane client retries a dead gate unboundedly,
  so if the policy-model server dies mid-run the job hangs rather than
  failing. Supervise; `squeue` + driver log tell you which.
- **Gate buffer growth is uncapped** (H5 adds caps): per-rollout delta
  forests, ids only. At seqlen 131k × concurrency 768 the raw ids are modest
  (~100s of MB), but Python-object overhead is real — watch the policy-model
  server RSS.
- **Marker survival through OpenHands** is the biggest unknown — it uses the
  same message carrier the legacy token echo uses (so it *should* survive
  wherever legacy works today), but `token_in_rate` is the proof either way,
  and a low rate is a correctness-preserving perf bug, not a training bug.
- The idle-GPU reaper exemption and `checkpoint_must_save_by` are already
  handled by the site wrapper / launcher.

## 6. Files in this folder

| File | Purpose |
|---|---|
| `SWE_RUN.md` | this runbook |
| `launch_swe_ab.sh` | arm selector: derives config, names the run, forces venv posture, delegates to your site wrapper |
| `make_capture_config.py` | derives the arm configs from the site yaml (`token_capture.enabled` + NaN-retry pin) |
| `aggregate_perf.py` | offline aggregation: per-hop HTTP bytes, bytes/trained-token, timing medians, token_in_rate |
