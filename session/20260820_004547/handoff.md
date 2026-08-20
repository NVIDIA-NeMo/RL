# Handoff

## Resume From Here
Implementing token-capture-lineage-ledger-approach.md. Gym side DONE and committed (Gym branch `token-capture-lineage-ledger`, commit 1e1a16cb; gate deleted, ledger implemented, 436 Gym tests green). RL side code changes DONE but UNCOMMITTED on RL branch `autoresearch/2026-08-20-lineage-ledger/impl`: nemo_gym.py (external_staging config, manifest fetch + local receipt assembly via `_assemble_receipt`), rollout_manager.py (fail_rollouts/gate_metrics removed), single_controller.py (_log_gate_metrics removed), blackbox_finalizer.py (finalize/token_in_rate + capture_poisoned_rollouts metrics), config.py (TTL fields dropped), exemplar YAML, docs (design-docs/token-capture-ledger.md created; guide updated), rewritten tests (test_nemo_gym_token_capture.py, test_rollout_manager.py updates). RL unit tests running in background task baao872q9.

## Next Actions
- Confirm RL unit tests green (env tests, blackbox_finalizer, rollout_manager, tq_token_sink, vllm_token_capture_hosting); fix failures; then commit RL side (sign-off + Co-Authored-By Claude Fable 5).
- Launch 5-step smoke A/B from repo root (fsw path):
  - capture arm: `DRY_RUN=0 SC_EXP_NAME=ledger-capture-s43-0820 NG_TIC_FP_CANONICAL=1 WALLTIME=3:59:00 WANDB_PROJ=PR3456-Ledger-AB-0820 bash swe_nano_sc_capture.sh grpo.max_num_steps=5 grpo.seed=43 grpo.num_prompts_per_step=2 policy.train_global_batch_size=8 token_capture.num_finalizer_workers=2 token_capture.defer_routed_experts_to_policy=true token_capture.staging_partition=rollout_staging_ledger_smoke_0820 +env.nemo_gym.policy_model.responses_api_models.vllm_model.num_workers=2 +policy.router_replay.enabled=true async_rl.sampler.name=windowed +async_rl.sampler.max_staleness_versions=1 +env.nemo_gym.model_endpoint_readiness_timeout_seconds=1800 policy.generation.vllm_cfg.reasoning_parser_plugin=/opt/nemo-rl/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py`
  - legacy arm: same via `swe_nano_sc.sh` (no NG_TIC_FP_CANONICAL, no token_capture overrides, keep num_workers=2 + R3 + sampler + seed 43, SC_EXP_NAME=ledger-legacy-s43-0820).
  - dry-run of capture arm validated: UV cache resolves to /lustre/fsw/portfolios/llmservice/users/pthombre/uv (was /tmp; fixed via UV_CACHE_DIR/UV_CACHE_DIR_OVERRIDE in swe_nano.env).
- Compare token_mult_prob_error, gen_kl_error, reward across 5 steps + perf (row_assembly/* timings, step time, finalize/token_in_rate ≥0.99) per docs/guides/nano-swe-token-capture.md; write TSV ledger in reports/auto_research/lineage-ledger-0820/.

## Watch Outs
- swe_nano.env was retargeted to pthombre paths + nightly-gym 08-10 container (commit with the work). swe_nano.secrets.env copied, chmod 600, git-excluded via .git/info/exclude.
- Prior 08-19 ledger smoke runs (pr3456-ledger-*-s43-0819*) in RL-pr3456-delta-smoke-workspace came from a DIFFERENT since-reverted implementation — reference only for launch shape, not results.
- Slurm submits from a networked shell; jobs monitored via slurm-broker MCP or squeue.
- Do not stop before both arms complete 5 steps and comparison is written (user stop rule).
