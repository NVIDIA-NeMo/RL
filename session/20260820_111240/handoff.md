# Handoff

## Resume From Here
Multi-worker ledger revalidation job **6368085** (het 6368085-6368086) submitted 2026-08-20 11:24, PENDING on batch/short. Branch `autoresearch/2026-08-20-lineage-ledger/mw-smoke-revalidation` (3 commits: launcher port 53713adeb, wrapper cbadca28b, ultra_launch fixes 02d700023). Mirrors failed job 6300221 with num_workers=16/4 intact.

## Next Actions
- Poll: `squeue -j 6368085`; once running, watch
  `/lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3456-delta-smoke-workspace/ray_logs/nano35-rlvr-sc-tc-smoke-ledger/6368085-logs/ray-driver.log`
- Verdict greps on driver log: `grep -c '409 Conflict'` (must be 0), `grep -c UnknownRolloutError` (must be 0), `Collecting rollouts` reaches 16/16, >=2 steps + nccl_reshard sync, `capture_dir/lineage/*.lineage.jsonl` populated, no `manifest(...) fetch failed`.
- On verdict: add row to reports/auto_research/lineage-ledger-0820/experiments.tsv.

## Watch Outs
- UV cache must stay /lustre/fsw/portfolios/llmservice/users/pthombre/uv — never /tmp, never UV_CACHE_DIR_OVERRIDE with prefetch containers.
- Do NOT pin num_workers=1 — 16/4 is the regression trigger under test.
- rollout_checkpointing.* overrides are ignored (extra-allow) in this tree — expected.
- Secrets in swe_nano.secrets.env (untracked, mode 600).
