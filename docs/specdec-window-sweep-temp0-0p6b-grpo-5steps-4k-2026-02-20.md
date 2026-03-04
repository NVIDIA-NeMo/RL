# SpecDec Window Sweep (0.6B Draft, temp=0) Report

## 1) Experiment summary

This report covers the completed GRPO speculative-window sweep where all settings from the prior 0.6B run were kept the same, except:

- `policy.generation.temperature=0.0` (changed from `1.0`)

Shared setup used:

- `grpo.max_num_steps=5`
- `policy.generation.max_new_tokens=4096`
- `policy.generation.top_p=1.0`
- `policy.generation.top_k=null`
- `policy.generation.vllm_cfg.load_format=auto`
- `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
- `policy.dtensor_cfg.activation_checkpointing=true`
- Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- Sweep: `num_speculative_tokens in {1,2,4,7,10}`
- `VLLM_DISABLE_COMPILE_CACHE=1` and unique `TORCHINDUCTOR_CACHE_DIR` per sweep point

## 2) Artifacts

- Sweep status file:
  - `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp0-0p6b-1-2-4-7-10-nocache-2026-02-19-181452.status.txt`
- Per-point logs:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0-s1-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-181452/run-5steps-spec0p6b-temp0-s1-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0-s2-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-183402/run-5steps-spec0p6b-temp0-s2-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0-s4-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-184639/run-5steps-spec0p6b-temp0-s4-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0-s7-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-185905/run-5steps-spec0p6b-temp0-s7-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0-s10-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-19-191148/run-5steps-spec0p6b-temp0-s10-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
- Matched no-spec baseline log:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-temp0-l4096-actckpt-loadfmtauto-nocache-2026-02-19-193030/run-5steps-nospec-temp0-l4096-actckpt-flashattn-loadfmtauto-nocache.log`

All 5 points completed with `rc=0`.

## 3) Temp=0 sweep results (steady-state: steps 2-5)

| spec tokens | avg setup (s) | avg total step (s) | avg generation (s) | avg E2E tok/s | token acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1  | 674.20 | 84.19 | 40.91 | 397.96 | 0.9986 |
| 2  | 363.60 | 72.80 | 30.95 | 460.14 | 0.9978 |
| 4  | 357.50 | 71.94 | 28.85 | 465.75 | 0.9948 |
| 7  | 367.70 | 72.68 | 30.82 | 460.89 | 0.9897 |
| 10 | 365.70 | 74.81 | 33.13 | 448.29 | 0.9817 |

Notes:

- Setup for `s=1` is materially higher than the other points; this is consistent with first-run cold-start overhead.
- `s=4` is the best point in this temp=0 sweep by both generation latency and E2E throughput.

## 4) Matched temp=0 no-spec baseline (steady-state: steps 2-5)

| Run | Setup (s) | Avg total step (s) | Avg generation (s) | Avg E2E tok/s |
|---|---:|---:|---:|---:|
| No-spec (temp=0) | 335.80 | 98.24 | 56.54 | 340.96 |

No-spec per-step timing reference:

| Step | Total step (s) | Generation (s) | E2E tok/s |
|---:|---:|---:|---:|
| 2 | 100.71 | 56.59 | 332.93 |
| 3 | 99.13 | 56.51 | 336.85 |
| 4 | 97.67 | 56.49 | 343.56 |
| 5 | 95.43 | 56.57 | 350.50 |

## 5) Direct speedups vs temp=0 no-spec baseline (steps 2-5)

| spec tokens | avg generation (s) | Gen speedup vs no-spec | avg total step (s) | Step-time speedup vs no-spec | avg E2E tok/s | E2E speedup vs no-spec | TAR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 40.91 | 1.382x | 84.19 | 1.167x | 397.96 | 1.167x | 0.9986 |
| 2  | 30.95 | 1.827x | 72.80 | 1.349x | 460.14 | 1.350x | 0.9978 |
| 4  | 28.85 | 1.960x | 71.94 | 1.366x | 465.75 | 1.366x | 0.9948 |
| 7  | 30.82 | 1.835x | 72.68 | 1.352x | 460.89 | 1.352x | 0.9897 |
| 10 | 33.13 | 1.707x | 74.81 | 1.313x | 448.29 | 1.315x | 0.9817 |

## 6) Comparison vs prior temp=1 sweep (same 0.6B draft setup)

Steady-state (steps 2-5) comparison:

| spec tokens | temp=0 avg gen (s) | temp=1 avg gen (s) | temp=0 avg E2E tok/s | temp=1 avg E2E tok/s | temp=0 TAR | temp=1 TAR |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 40.91 | 47.53 | 397.96 | 316.50 | 0.9986 | 0.7698 |
| 2  | 30.95 | 39.70 | 460.14 | 346.76 | 0.9978 | 0.6777 |
| 4  | 28.85 | 47.03 | 465.75 | 323.31 | 0.9948 | 0.5230 |
| 7  | 30.82 | 67.23 | 460.89 | 262.58 | 0.9897 | 0.3808 |
| 10 | 33.13 | 88.71 | 448.29 | 219.27 | 0.9817 | 0.3047 |

## 7) Findings

1. Best speculative window at `temp=0` is `s=4`.
2. All tested speculative windows beat matched no-spec at `temp=0` on both generation and E2E.
3. `s=4` is best overall vs no-spec at `temp=0`:
   - generation speedup: `1.960x`
   - step-time speedup: `1.366x`
   - E2E throughput speedup: `1.366x`
4. Unlike temp=1, larger windows at temp=0 degrade more gradually:
   - acceptance remains high (`~0.98-0.999`) across all tested windows
   - performance drops at `s=10`, but still remains above no-spec
5. Relative to temp=1, temp=0 is consistently stronger for this setup at every tested window.

## 8) Notes

1. The matched temp=0 no-spec baseline run completed with `rc=0`.
2. A teardown `.nfs*` cleanup traceback (`OSError: [Errno 16] Device or resource busy`) appeared at process shutdown, but final status was successful (`rc=0`) and metrics were fully logged.
3. This sweep and baseline used `VLLM_DISABLE_COMPILE_CACHE=1`; setup times include cold-start and compile effects.
