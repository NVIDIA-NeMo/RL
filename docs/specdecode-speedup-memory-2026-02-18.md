# Spec-Decoding + GRPO Session Memory (2026-02-18)

This memory note captures the final working setup, results, and exact restart points from this session.

## Primary objective completed

- Reproducible GRPO benchmark comparing:
  - spec decode: 32B target + 14B draft (`num_speculative_tokens=2`)
  - non-spec baseline: 32B only
- Runs were executed for `5` GRPO steps on the same dataset/config family.

## Critical fix that made TAR non-zero

File:
- `nemo_rl/models/generation/__init__.py`

Required behavior:
- Preserve user-provided `vllm_cfg.load_format` in training runs.
- Only set a default if `load_format` is absent.

Expected code pattern:

```python
if "load_format" not in config["vllm_cfg"]:
    config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
```

Why this mattered:
- Without this, training path could effectively run with dummy/stale draft behavior, collapsing acceptance.
- With this fix and non-dummy run setup, TAR recovered to high values.

## Final run settings used

Shared:
- TP=8
- `grpo.max_num_steps=5`
- `max_new_tokens=256`
- `temperature=0.0`, `top_p=1.0`, `top_k=null`
- `policy.generation.vllm_cfg.load_format=auto`
- `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
- Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`

Spec-only:
- Draft model: Qwen3-14B
- `num_speculative_tokens=2`
- `draft_tensor_parallel_size=1`

Non-spec:
- `policy.generation.vllm_kwargs.speculative_config=null`

## Key outcomes

Spec run (32B+14B, 5 steps):
- TAR by step: `0.9302`, `0.9893`, `0.9833`, `0.9940`, `0.9832`
- Aggregate TAR: `0.9752` (`6536/6702`)

Spec vs non-spec (steady-state, steps 2-5):
- Generation speedup: `1.161x`
- E2E throughput: near parity (`0.996x`)
- Total step time: near parity (`0.997x`)

Interpretation:
- Spec decode clearly speeds generation phase at these settings.
- End-to-end GRPO remains near parity because non-generation phases dominate total step time.

## Canonical artifacts

Full reproducibility guide (fresh pull -> env prep -> both runs):
- `docs/repro-specdec-vs-nospec-grpo-5steps-2026-02-18.md`

Full metrics report (all step timings + token/s + TAR):
- `docs/specdec-vs-nospec-grpo-5steps-2026-02-18.md`

Run logs:
- Spec: `/home/scratch.shaunakj_other/logs/grpo-32b-spec14b-s2-l256-loadfmtauto-2026-02-18/run-5steps-spec14b-flashattn-scratchpaths-policyvenv-loadfmtauto.log`
- Non-spec: `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-l256-loadfmtauto-2026-02-18/run-5steps-nospec-flashattn-scratchpaths-policyvenv-loadfmtauto.log`

## Quick resume checklist

1. Start from `docs/repro-specdec-vs-nospec-grpo-5steps-2026-02-18.md` section 2.
2. Re-verify `load_format` patch in `nemo_rl/models/generation/__init__.py`.
3. Ensure policy-worker numpy fix is present (as documented in repro guide).
4. Run spec and non-spec commands from sections 8 and 9 in repro guide.
5. Use report doc extraction commands to recompute step metrics.

## Notes for next extension

If you want bigger end-to-end gains than parity, sweep the phases that dominate GRPO wall time (prep/logprob/train-transfer), not only speculative generation knobs.
