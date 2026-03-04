# SpecDec Window Sweep vs No-Spec (GRPO 5-step, 4k, temp=1) Report

Date: 2026-02-18

## 1) Experimental setup

1. Goal: measure how speculative window size affects generation time and throughput at `temperature=1.0` with `max_new_tokens=4096`, and compare directly against a matched non-spec baseline.
2. Hardware: 1 node, 8 GPUs.
3. Target model: Qwen3-32B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
4. Draft model (spec runs): Qwen3-1.7B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
5. Shared settings:
   - `grpo.max_num_steps=5`
   - `policy.generation.max_new_tokens=4096`
   - `temperature=1.0`, `top_p=1.0`, `top_k=null`
   - `policy.generation.vllm_cfg.load_format=auto`
   - `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
   - `policy.dtensor_cfg.activation_checkpointing=true`
   - Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
6. Sweep values: `num_speculative_tokens in {1, 2, 4, 7, 10}`.
7. Baseline: non-spec (`speculative_config=null`) with the same settings.

## 2) Canonical artifacts

- Sweep status file:
  `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp1-1-2-4-7-10-retry-2026-02-18-191514.status.txt`
- Spec logs:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-temp1-s1-l4096-actckpt-loadfmtauto-sweep-a1-2026-02-18-191514/run-5steps-spec1p7b-temp1-s1-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-temp1-s2-l4096-actckpt-loadfmtauto-sweep-a1-2026-02-18-192722/run-5steps-spec1p7b-temp1-s2-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-temp1-s4-l4096-actckpt-loadfmtauto-sweep-a1-2026-02-18-193846/run-5steps-spec1p7b-temp1-s4-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-temp1-s7-l4096-actckpt-loadfmtauto-sweep-a1-2026-02-18-195028/run-5steps-spec1p7b-temp1-s7-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-temp1-s10-l4096-actckpt-loadfmtauto-sweep-a1-2026-02-18-200350/run-5steps-spec1p7b-temp1-s10-l4096-actckpt-flashattn-loadfmtauto.log`
- No-spec log:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-temp1-l4096-actckpt-loadfmtauto-2026-02-18-202012/run-5steps-nospec-temp1-l4096-actckpt-flashattn-loadfmtauto.log`

All runs completed with `rc=0`.

## 3) Metric definitions

1. Steady-state comparisons use steps `2-5` (step 1 excluded as warmup/cold start).
2. `Avg generation (s)`: arithmetic mean of per-step `generation` time for steps 2-5.
3. `Avg total step (s)`: arithmetic mean of per-step `Total step time` for steps 2-5.
4. `Avg E2E tok/s (group)`: arithmetic mean of per-step `E2E (Tokens/sec)` for steps 2-5.
5. `TAR (steps 2-5)`: aggregated accepted/proposed tokens across steps 2-5.

## 4) Primary results (steady-state: steps 2-5)

| Run | Setup (s) | Avg generation (s) | Gen speedup vs no-spec | Avg total step (s) | Step-time speedup vs no-spec | Avg E2E tok/s (group) | E2E speedup vs no-spec | TAR (steps 2-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-spec | 253.2 | 56.53 | 1.000x | 98.18 | 1.000x | 288.23 | 1.000x | n/a |
| Spec s=1 | 273.5 | 48.17 | 1.174x | 90.34 | 1.087x | 317.26 | 1.101x | 79.04% |
| Spec s=2 | 267.3 | 41.27 | 1.370x | 84.07 | 1.168x | 340.50 | 1.181x | 70.93% |
| Spec s=4 | 271.3 | 45.74 | 1.236x | 87.62 | 1.121x | 323.14 | 1.121x | 55.87% |
| Spec s=7 | 271.3 | 67.88 | 0.833x | 109.98 | 0.893x | 258.85 | 0.898x | 42.92% |
| Spec s=10 | 268.4 | 82.20 | 0.688x | 125.24 | 0.784x | 226.06 | 0.784x | 33.29% |

## 5) Per-step generation timing (s)

| Run | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|
| No-spec | 56.55 | 56.53 | 56.54 | 56.51 |
| Spec s=1 | 48.98 | 46.24 | 48.78 | 48.69 |
| Spec s=2 | 41.87 | 38.85 | 42.38 | 42.00 |
| Spec s=4 | 47.37 | 38.11 | 49.90 | 47.60 |
| Spec s=7 | 68.26 | 59.53 | 72.23 | 71.50 |
| Spec s=10 | 81.30 | 60.95 | 95.53 | 91.04 |

## 6) Step-1 cold-start context

| Run | Step 1 total (s) | Step 1 generation (s) | Step 1 TAR |
|---|---:|---:|---:|
| No-spec | 44.31 | 18.04 | n/a |
| Spec s=1 | 41.42 | 14.66 | 85.42% |
| Spec s=2 | 38.58 | 12.00 | 77.44% |
| Spec s=4 | 38.21 | 12.10 | 67.15% |
| Spec s=7 | 47.87 | 19.72 | 54.94% |
| Spec s=10 | 74.91 | 47.86 | 37.70% |

## 7) Summary

1. Best setting in this temp=1 sweep is `s=2`.
2. `s=1`, `s=2`, and `s=4` outperform non-spec on generation time and E2E throughput.
3. `s=7` and `s=10` are slower than non-spec, with large TAR drops.
4. As speculative window increases in this regime, TAR falls (`79.04% -> 70.93% -> 55.87% -> 42.92% -> 33.29%`), and generation time degrades correspondingly.

## 8) Notes

1. The `.nfs*` cleanup traceback (`OSError: [Errno 16] Device or resource busy`) appeared at teardown in logs but did not affect run completion (`rc=0`).
2. This report records parsed metrics from run logs and uses the same steady-state methodology as prior 4k sweep reports.
