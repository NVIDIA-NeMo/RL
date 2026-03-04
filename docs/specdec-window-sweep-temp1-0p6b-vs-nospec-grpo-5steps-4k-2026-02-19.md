# SpecDec Window Sweep (0.6B Draft) vs No-Spec (GRPO 5-step, 4k, temp=1) Report

Date: 2026-02-19

## 1) Experimental setup

1. Goal: repeat the temp=1, 4k speculative-window sweep with a smaller draft model (`Qwen3-0.6B`) and compare against no-spec baseline.
2. Hardware: 1 node, 8 GPUs.
3. Target model: Qwen3-32B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
4. Draft model (spec runs): Qwen3-0.6B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
5. Shared settings:
   - `grpo.max_num_steps=5`
   - `policy.generation.max_new_tokens=4096`
   - `temperature=1.0`, `top_p=1.0`, `top_k=null`
   - `policy.generation.vllm_cfg.load_format=auto`
   - `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
   - `policy.dtensor_cfg.activation_checkpointing=true`
   - Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
6. Sweep values: `num_speculative_tokens in {1, 2, 4, 7, 10}`.
7. Compile/cache setting used for this sweep:
   - `VLLM_DISABLE_COMPILE_CACHE=1`
   - Unique `TORCHINDUCTOR_CACHE_DIR` per sweep point.
8. Baseline: matched no-spec run at same temp=1/4k settings from:
   `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-temp1-l4096-actckpt-loadfmtauto-2026-02-18-202012/run-5steps-nospec-temp1-l4096-actckpt-flashattn-loadfmtauto.log`

## 2) Canonical artifacts

- Sweep status file:
  `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp1-0p6b-1-2-4-7-10-nocache-2026-02-18-233624.status.txt`
- Spec logs:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s1-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-18-233624/run-5steps-spec0p6b-temp1-s1-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-18-234906/run-5steps-spec0p6b-temp1-s2-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s4-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-000121/run-5steps-spec0p6b-temp1-s4-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s7-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-001416/run-5steps-spec0p6b-temp1-s7-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s10-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-002848/run-5steps-spec0p6b-temp1-s10-l4096-actckpt-flashattn-loadfmtauto-nocache.log`

All five spec runs completed with `rc=0`.

## 3) Metric definitions

1. Steady-state comparison uses steps `2-5` (step 1 excluded).
2. `Avg generation (s)`: mean of step generation times (steps 2-5).
3. `Avg total step (s)`: mean of total step time (steps 2-5).
4. `Avg E2E tok/s (group)`: mean of `E2E (Tokens/sec)` (steps 2-5).
5. `TAR (steps 2-5)`: aggregated accepted/proposed draft tokens over steps 2-5.

## 4) Primary results (steady-state: steps 2-5)

| Run | Setup (s) | Avg generation (s) | Gen speedup vs no-spec | Avg total step (s) | Step-time speedup vs no-spec | Avg E2E tok/s (group) | E2E speedup vs no-spec | TAR (steps 2-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-spec | 253.2 | 56.53 | 1.000x | 98.18 | 1.000x | 288.23 | 1.000x | n/a |
| Spec s=1 (0.6B) | 310.1 | 47.53 | 1.189x | 89.58 | 1.096x | 316.50 | 1.098x | 76.98% |
| Spec s=2 (0.6B) | 315.0 | 39.70 | 1.424x | 81.06 | 1.211x | 346.76 | 1.203x | 67.77% |
| Spec s=4 (0.6B) | 329.9 | 47.03 | 1.202x | 88.32 | 1.112x | 323.31 | 1.122x | 52.30% |
| Spec s=7 (0.6B) | 313.1 | 67.23 | 0.841x | 110.08 | 0.892x | 262.58 | 0.911x | 38.08% |
| Spec s=10 (0.6B) | 313.1 | 88.71 | 0.637x | 131.37 | 0.747x | 219.27 | 0.761x | 30.47% |

## 5) Per-step generation timing (s)

| Run | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|
| No-spec | 56.55 | 56.53 | 56.54 | 56.51 |
| Spec s=1 | 48.24 | 46.48 | 47.72 | 47.68 |
| Spec s=2 | 40.39 | 36.11 | 41.94 | 40.38 |
| Spec s=4 | 49.00 | 38.19 | 49.29 | 51.66 |
| Spec s=7 | 68.72 | 56.04 | 71.94 | 72.23 |
| Spec s=10 | 94.80 | 64.74 | 105.48 | 89.83 |

## 6) Step-1 cold-start context

| Run | Step 1 total (s) | Step 1 generation (s) | Step 1 E2E tok/s | Step 1 TAR |
|---|---:|---:|---:|---:|
| No-spec | 44.31 | 18.04 | 198.43 | n/a |
| Spec s=1 | 50.38 | 24.46 | 229.73 | 77.95% |
| Spec s=2 | 40.30 | 13.68 | 208.53 | 75.11% |
| Spec s=4 | 45.12 | 19.28 | 195.04 | 61.78% |
| Spec s=7 | 46.66 | 20.17 | 184.58 | 48.27% |
| Spec s=10 | 55.86 | 29.28 | 174.31 | 36.22% |

## 7) Summary

1. Best setting in this 0.6B temp=1 sweep is `s=2`.
2. `s=1`, `s=2`, and `s=4` beat no-spec on generation time and E2E throughput.
3. `s=7` and `s=10` are slower than no-spec.
4. Larger windows show strong TAR collapse (`76.98% -> 67.77% -> 52.30% -> 38.08% -> 30.47%`) and matching generation-time degradation.
5. Compared with the prior 1.7B draft temp=1 sweep, the qualitative shape is the same: clear local optimum, then large-window regression.

## 8) Notes

1. Teardown `.nfs*` cleanup traceback (`OSError: [Errno 16] Device or resource busy`) appears but does not prevent completion (`rc=0`).
2. This sweep intentionally used `VLLM_DISABLE_COMPILE_CACHE=1`; setup times are therefore higher and should not be compared as pure decode-performance signal.
