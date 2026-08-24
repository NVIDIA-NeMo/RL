# Handoff

## Current Final State (2026-08-24 ~10:00 PDT)
Token-free ledger implementation committed (Gym `6061dadb`, NeMo-RL `6d12d9663`).
Gym unit tests 217/218 (1 pre-existing timeout). NeMo-RL hosting tests untestable
on login node (vllm/torch need GPU); changes correct by inspection.
Slurm smoke **6490178** (PENDING) submitted for end-to-end ledger row verification.
Chain-in-request validation campaign complete (rows 13–14). Sections below preserve history.

## Resume From Here
Multi-worker ledger revalidation job **6368085** (het 6368085-6368086) submitted 2026-08-20 11:24, PENDING on batch/short. Branch `autoresearch/2026-08-20-lineage-ledger/mw-smoke-revalidation` (3 commits: launcher port 53713adeb, wrapper cbadca28b, ultra_launch fixes 02d700023). Mirrors failed job 6300221 with num_workers=16/4 intact.

## Next Actions (2026-08-24)
- Poll: `squeue -j 6490178` (token-free ledger smoke); once done:
  ```bash
  OUTPUT_LOG=$(find /lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3456-delta-smoke-workspace/results/swe-token-free-ledger-s43-0824 -path '*/wandb/run-*/files/output.log' -print | head -1)
  grep -c '^step_metrics=' "$OUTPUT_LOG"   # must be 5
  grep -nE 'chain_hash_mismatch|cumulative_hash_mismatch|invalid_worker_commit_coordinates|worker_capture_failed|409 Conflict|UnknownRolloutError' /lustre/fsw/.../ray_logs/swe-token-free-ledger-s43-0824/6490178-logs/ray-driver.log  # must be 0
  ```
  JSONL custody row check (key new verification):
  ```bash
  python3 -c "
  import json, glob
  for f in glob.glob('/lustre/fsw/.../results/swe-token-free-ledger-s43-0824/*/gym_token_capture/lineage/*.lineage.jsonl'):
      rows = [json.loads(l) for l in open(f) if l.strip() and '{' in l]
      ext = [r for r in rows if r.get('staging_key')]
      if ext:
          print(f, len(ext), 'ext rows')
          print('  cumulative_token_ids:', sum(1 for r in ext if r.get('cumulative_token_ids')))
          print('  chain_hash:', sum(1 for r in ext if r.get('chain_hash')))
  "
  ```
  Expected: cumulative_token_ids count = 0, chain_hash count > 0 for every ledger file.
- Update experiments.tsv row 15 with result and mark keep/discard/crash.
- Update session/20260820_111240/token_free_ledger_progress.md with Slurm verdict.

## OLD Next Actions (from lineage-ledger campaign — superseded)
- Poll: `squeue -j 6368085`; once running, watch
  `/lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3456-delta-smoke-workspace/ray_logs/nano35-rlvr-sc-tc-smoke-ledger/6368085-logs/ray-driver.log`
- Verdict greps on driver log: `grep -c '409 Conflict'` (must be 0), `grep -c UnknownRolloutError` (must be 0), `Collecting rollouts` reaches 16/16, >=2 steps + nccl_reshard sync, `capture_dir/lineage/*.lineage.jsonl` populated, no `manifest(...) fetch failed`.
- On verdict: add row to reports/auto_research/lineage-ledger-0820/experiments.tsv.

## Watch Outs
- UV cache must stay /lustre/fsw/portfolios/llmservice/users/pthombre/uv — never /tmp, never UV_CACHE_DIR_OVERRIDE with prefetch containers.
- Do NOT pin num_workers=1 — 16/4 is the regression trigger under test.
- rollout_checkpointing.* overrides are ignored (extra-allow) in this tree — expected.
- Secrets in swe_nano.secrets.env (untracked, mode 600).

## 2026-08-23 Chain Validation Resume Point
Run focused preflight checks for the uncommitted chain-in-request feature, then
commit it on a dedicated experiment branch and launch matched five-step
`swe_nano_sc_capture.sh` and `swe_nano_sc.sh` arms. The user explicitly wants a
real training smoke, not `test_chain_prefix_smoke.py`. Append every attempt and
failure/fix to `reports/auto_research/lineage-ledger-0820/experiments.tsv`.

Jobs submitted: baseline `6465030`, capture `6465058`. Poll both with `squeue`.
Ray logs are under `.../ray_logs/token-chain-{baseline,capture}-s43-0823/<job>-logs`.
Do not modify tracked source while these live-checkout jobs are pending/running.

Final verdict at 04:44 PDT: baseline `6465030` (W&B `dyb1em3g`) and capture
`6465058` (W&B `wiyqqjau`) both passed 5/5 and exited 0:0. Capture produced
3,440 committed nodes across 44 ledgers, 98%+ token-in activity, chains up to
length 200, and zero chain/TQ/staging/commit integrity errors. It trained 31/40
rows; nine max-context terminal calls were explicitly fail-closed. Experiment
rows 13–14 and the report addendum contain the full comparison. No relaunch or
code fix remains.
