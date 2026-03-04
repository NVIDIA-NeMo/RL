# Spec-Decoding Zero-Acceptance Debug Memory (2026-02-17)

This note preserves the debugging outcome for speculative decoding acceptance staying at `0` and the validated rerun that fixed it.

## What was observed

- Pair used in the run:
  - verifier/target model: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
  - draft model: same path as verifier
- Zero-acceptance runs:
  - `/home/scratch.shaunakj_other/logs/specdecode-1step-greedydebug-14b/run.log` reported `Early Token Acceptance Rate: 0.0000 (0/315)`
  - `/home/scratch.shaunakj_other/logs/specdecode-1step-earlydebug-14b/run.log` reported `Early Token Acceptance Rate: 0.0000 (0/315)`
- Metrics were wired correctly:
  - selected counters were `accepted=vllm:spec_decode_num_accepted_tokens`, `proposed=vllm:spec_decode_num_draft_tokens`

## Root cause

Even with explicit CLI override, training flow effectively ended up with `load_format=dummy` in generation worker setup. Then GRPO refit updated only the verifier/target weights before generation, while the draft model stayed on stale/dummy-loaded state. That made target and draft diverge in live weights despite identical model path strings, which drove acceptance to zero.

## Code change that fixed it

File changed:
- `nemo_rl/models/generation/__init__.py`

Change made:
- only set `vllm_cfg.load_format` default when user did not provide one
- preserve explicit user override (for this repro: `++policy.generation.vllm_cfg.load_format=auto`)

## Validation run after fix

Run:
- `/home/scratch.shaunakj_other/logs/specdecode-1step-greedy-autoformat-fixed-skipvenv-14b/run.log`

Key metric:
- `Early Token Acceptance Rate: 1.0000 (55/55)`

Note:
- this run later failed in policy training with CUDA OOM, but acceptance metrics were emitted before that failure, so speculative-decoding validation was completed.

## Environment notes that mattered

- Use scratch-backed caches (`UV_CACHE_DIR`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_DATASETS_CACHE`, `VLLM_CACHE_ROOT`, `RAY_TMPDIR`)
- Use `NRL_SKIP_LOCAL_VENV=true` to avoid per-worker uv rebuild path issues for this environment
