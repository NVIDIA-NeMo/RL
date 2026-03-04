# Session Memory (2026-02-19): SpecDec 0.6B Sweep

## What was done

- Completed speculative decode sweep with smaller draft model (`Qwen3-0.6B`) at `temp=1`, `max_new_tokens=4096`, `max_num_steps=5`.
- Sweep windows: `num_speculative_tokens in {1,2,4,7,10}`.
- Added no-spec comparison against existing matched baseline.

## Key config

- Target: `Qwen3-32B` snapshot
  `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
- Draft: `Qwen3-0.6B` snapshot
  `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- Shared overrides:
  - `grpo.max_num_steps=5`
  - `policy.generation.max_new_tokens=4096`
  - `policy.generation.temperature=1.0`
  - `policy.generation.top_p=1.0`
  - `policy.generation.top_k=null`
  - `policy.generation.vllm_cfg.load_format=auto`
  - `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
  - `policy.dtensor_cfg.activation_checkpointing=true`
- Dataset:
  `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- Important runtime controls:
  - `VLLM_DISABLE_COMPILE_CACHE=1`
  - unique `TORCHINDUCTOR_CACHE_DIR` per sweep point

## Completion status

- Sweep status file:
  `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp1-0p6b-1-2-4-7-10-nocache-2026-02-18-233624.status.txt`
- Final status: all completed with `rc=0` for `{1,2,4,7,10}`.

## Main outcomes (steady-state steps 2-5)

- Best point: `s=2`
- Relative to no-spec:
  - `s=1`: gen `1.189x`, step `1.096x`, e2e `1.098x`
  - `s=2`: gen `1.424x`, step `1.211x`, e2e `1.203x`
  - `s=4`: gen `1.202x`, step `1.112x`, e2e `1.122x`
  - `s=7`: gen `0.841x`, step `0.892x`, e2e `0.911x`
  - `s=10`: gen `0.637x`, step `0.747x`, e2e `0.761x`
- TAR trend (steps 2-5):
  - `s=1: 76.98%`
  - `s=2: 67.77%`
  - `s=4: 52.30%`
  - `s=7: 38.08%`
  - `s=10: 30.47%`

## Canonical docs produced

- Report:
  `docs/specdec-window-sweep-temp1-0p6b-vs-nospec-grpo-5steps-4k-2026-02-19.md`
- Repro:
  `docs/repro-specdec-window-sweep-temp1-0p6b-vs-nospec-grpo-5steps-4k-2026-02-19.md`
