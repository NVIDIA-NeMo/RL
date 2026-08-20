# Handoff

## Resume From Here
CAMPAIGN COMPLETE. The lineage-ledger plan is implemented, unit-tested, and validated by a passing 5-step smoke A/B.

- Gym branch `token-capture-lineage-ledger` (head: lint fix on 1e1a16cb): gate deleted, ledger implemented, 436 tests green.
- RL branch `autoresearch/2026-08-20-lineage-ledger/impl` (head 5bc2b50d1): receipt assembly from manifest, gate plumbing removed, ledger metrics in finalizer, launch-infra fixes (venv rebuild-list, UV cache, content-None legacy fix, TQ pin skip, SC_EXP_NAME), docs + design doc, A/B report.
- Smoke A/B (seed 43, R3 on, num_workers=2, 5 steps): capture 6359951 PASS (58:44), legacy 6359952 PASS (1:07:03). Capture dynamics strictly tighter (tmpe 1.0137-1.0147 vs 1.020-1.066; gen_kl ~0.001 vs ~0.004). Ledger: 2821 rows, 0 unresolved/worker failures, token_in_rate 0.95-0.99.
- Full report: reports/auto_research/lineage-ledger-0820/ledger-ab-report.md; ledger TSV in the same dir.

## Next Actions
- Push both branches and update Gym PR #2278 / RL PR #3456; run CI (/ok to test) — CI covers the Ray-heavy RL suites that cannot run outside the container.
- Follow-up items (report §Follow-ups): retry idempotency (recovers ~36% fail-closed rollout loss from per-call aborts), vLLM engine ClientOSError bursts, ledger-file drop on publish.

## Watch Outs
- swe_nano.env now targets pthombre paths + nightly-gym 08-10; never set UV_CACHE_DIR_OVERRIDE with prefetch-venvs containers; uv cache = /lustre/fsw/portfolios/llmservice/users/pthombre/uv (never /tmp).
- swe_nano.secrets.env is untracked (git-excluded via .git/info/exclude), mode 600.
- Lustre git operations intermittently time out (~2 min) — retry with timeout.
