# Session State

- Session: 20260820_111240
- Repo: /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/pthombre/sweRun/RL-pr3456-delta-staging (fsw twin: /lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3456-delta-staging)
- Branch: autoresearch/2026-08-20-lineage-ledger/mw-smoke-revalidation
- Started: 2026-08-20 11:12
- Updated: 2026-08-20 11:25

## Goal
Rerun the failed nano-35-rlvr-sc-tc-smoke shape (job 6300221, amahishi) with identical params — critically policy_model num_workers=16 / reasoning_off 4 — on the lineage-ledger token-capture code, to prove the multi-worker gate failure (409 Conflict + UnknownRolloutError from the per-process LineageRegistry) cannot recur.

## Current Subtask
Submit the smoke job and monitor to verdict.

## Loaded Skills
- `launch-nemo-rl` — debugging playbook (k8s-focused; this run is Slurm).
- `nemo-rl-auto-research` — campaign workflow, TSV ledger, branching.
- `nemo-rl-session-memory` — this record.

## Current Status
- Root cause confirmed: job 6300221's gate registry was per-uvicorn-process; register PUT hit 1 of 16 workers, data-plane calls load-balanced → 409 at prepare, UnknownRolloutError at ingest after fail_rollouts cleanup. Current code (Gym 1e1a16cb ledger) has no registration and uses FileLineageStore under a shared root with fcntl locks.
- Ported examples/nemo_gym/nemotron-3.5-nano/ (5 files) from nemo-rl-partial-rollout-recovery-refresh (commit 53713adeb).
- Wrote nano35_ledger_smoke.sh batch wrapper (commit cbadca28b): pthombre paths, zhiyul nightly-gym.2026-08-10 container, proven 0820 launch plumbing, UV_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/pthombre/uv (user directive: NEVER /tmp).
- Fixed examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh (commit 02d700023): NRL_ENTRYPOINT, NRL_DRIVER_PIP_INSTALL/PYTHONPATH, NRL_DRIVER_UV_RUN_FLAGS, external UV_CACHE_DIR precedence.
- Dry run clean: SC driver + --locked --no-sync, lustre UV cache, num_workers 16/4 from student_rlvr1.yaml defaults, all model/data/judge paths readable.

## Plan
- [x] Port launchers, write wrapper, dry-run
- [ ] DRY_RUN=0 submit; record job id
- [ ] Monitor ray-driver.log; verdict greps
- [ ] Record row in reports/auto_research/lineage-ledger-0820/experiments.tsv

## Pass Criteria
0x "409 Conflict", 0x UnknownRolloutError in driver log; Collecting rollouts reaches 16/16; >=2 steps incl. nccl_reshard sync; lineage/*.lineage.jsonl populated; no manifest fetch failures.

## Assumptions
- rollout_checkpointing.* overrides are extra-allowed no-ops in this tree (MasterConfig extra="allow").
- Container deviation (nightly-gym.2026-08-10 vs amahishi's bake) is required for this branch's validated venv plumbing.

## Blockers
- None known.
