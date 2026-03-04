# SpecDec Window Sweep (0.6B Draft, Thinking Mode, temp=0.6) Report

Date: 2026-02-23

## 1) Experiment summary

This report covers the completed GRPO speculative-window sweep rerun with the same base 0.6B draft setup and these verifier changes enabled:

- `text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)`
- `++policy.tokenizer.chat_template_kwargs.enable_thinking=true`
- `++policy.generation.temperature=0.6`
- `++policy.generation.top_p=0.95`
- `++policy.generation.top_k=20`
- `++policy.generation.min_p=0`

Shared setup used:

- `grpo.max_num_steps=5`
- `policy.generation.max_new_tokens=4096`
- `policy.generation.vllm_cfg.load_format=auto`
- `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
- `policy.dtensor_cfg.activation_checkpointing=true`
- Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- Sweep: `num_speculative_tokens in {1,2,4,7,10}`
- `VLLM_DISABLE_COMPILE_CACHE=1` and unique `TORCHINDUCTOR_CACHE_DIR` per sweep point
- Runtime launcher patch used to permit requested verifier sampling knobs:
  - `nemo_rl.models.generation.vllm.vllm_generation.TOP_K_THRESHOLD = -1`
  - `nemo_rl.models.generation.vllm.vllm_generation.TOP_P_THRESHOLD = 0.0`

## 2) Artifacts

- Sweep status file:
  - `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp0p6-thinking-0p6b-1-2-4-7-10-nocache-2026-02-22-195701.status.txt`
- Sweep master log:
  - `/home/scratch.shaunakj_other/logs/spec-window-sweep-temp0p6-thinking-0p6b-1-2-4-7-10-nocache-2026-02-22-195701.master.log`
- Per-point logs:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s1-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-22-195702/run-5steps-spec0p6b-think-t0p6-p0p95-k20-s1-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s2-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-22-201255/run-5steps-spec0p6b-think-t0p6-p0p95-k20-s2-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s4-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-22-202650/run-5steps-spec0p6b-think-t0p6-p0p95-k20-s4-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s7-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-22-204101/run-5steps-spec0p6b-think-t0p6-p0p95-k20-s7-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s10-l4096-actckpt-loadfmtauto-sweep-nocache-2026-02-22-205610/run-5steps-spec0p6b-think-t0p6-p0p95-k20-s10-l4096-actckpt-flashattn-loadfmtauto-nocache.log`
- Matched no-spec status file:
  - `/home/scratch.shaunakj_other/logs/nospec-thinking-temp0p6-k20-0p95-0p6b-nocache-2026-02-22-221846.status.txt`
- Matched no-spec run log:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-think-t0p6-p0p95-k20-l4096-actckpt-loadfmtauto-nocache-2026-02-22-221846/run-5steps-nospec-think-t0p6-p0p95-k20-l4096-actckpt-flashattn-loadfmtauto-nocache.log`

All five sweep points completed with `rc=0` (`SWEEP_DONE`), and the matched no-spec run also completed with `rc=0` (`NOSPEC_DONE`).

## 3) Thinking-mode sweep results (steady-state: steps 2-5)

| spec tokens | avg setup (s) | avg total step (s) | avg generation (s) | avg E2E tok/s | token acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1  | 466.40 | 89.09 | 47.86 | 325.23 | 0.8019 |
| 2  | 400.60 | 80.80 | 38.20 | 345.87 | 0.7179 |
| 4  | 394.30 | 86.51 | 44.16 | 330.11 | 0.5800 |
| 7  | 395.00 | 99.15 | 57.52 | 282.68 | 0.4503 |
| 10 | 392.20 | 119.10 | 76.90 | 240.64 | 0.3590 |

## 4) Per-step generation timing (s)

| spec tokens | Step 2 | Step 3 | Step 4 | Step 5 |
|---:|---:|---:|---:|---:|
| 1  | 48.72 | 46.71 | 48.27 | 47.73 |
| 2  | 41.12 | 28.08 | 41.56 | 42.03 |
| 4  | 44.54 | 40.83 | 45.03 | 46.22 |
| 7  | 63.44 | 42.92 | 64.07 | 59.66 |
| 10 | 81.10 | 60.82 | 82.27 | 83.40 |

Per-step token acceptance rate (TAR):

| spec tokens | Step 1 TAR | Step 2 TAR | Step 3 TAR | Step 4 TAR | Step 5 TAR |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.8467 | 0.7931 | 0.8202 | 0.7869 | 0.8157 |
| 2  | 0.7599 | 0.7195 | 0.7530 | 0.6962 | 0.7173 |
| 4  | 0.6513 | 0.5675 | 0.6234 | 0.5555 | 0.5923 |
| 7  | 0.5080 | 0.4259 | 0.5077 | 0.4265 | 0.4746 |
| 10 | 0.4342 | 0.3588 | 0.3913 | 0.3333 | 0.3678 |

## 5) Step-1 cold-start context

| spec tokens | Step 1 total (s) | Step 1 generation (s) | Step 1 E2E tok/s | Step 1 TAR |
|---:|---:|---:|---:|---:|
| 1  | 44.58 | 16.04 | 187.44 | 0.8467 |
| 2  | 45.80 | 17.93 | 184.80 | 0.7599 |
| 4  | 45.10 | 17.30 | 191.96 | 0.6513 |
| 7  | 52.35 | 25.79 | 188.55 | 0.5080 |
| 10 | 52.61 | 26.25 | 165.65 | 0.4342 |

## 6) Comparison vs prior temp=0 non-thinking sweep (same 0.6B draft family)

Reference report:
- `docs/specdec-window-sweep-temp0-0p6b-grpo-5steps-4k-2026-02-20.md`

Steady-state (steps 2-5) comparison:

| spec tokens | thinking avg gen (s) | prior temp=0 avg gen (s) | gen ratio (thinking/prior) | thinking avg E2E tok/s | prior temp=0 avg E2E tok/s | E2E ratio (thinking/prior) | thinking TAR | prior temp=0 TAR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 47.86 | 40.91 | 1.170x | 325.23 | 397.96 | 0.817x | 0.8019 | 0.9986 |
| 2  | 38.20 | 30.95 | 1.234x | 345.87 | 460.14 | 0.752x | 0.7179 | 0.9978 |
| 4  | 44.16 | 28.85 | 1.531x | 330.11 | 465.75 | 0.709x | 0.5800 | 0.9948 |
| 7  | 57.52 | 30.82 | 1.866x | 282.68 | 460.89 | 0.613x | 0.4503 | 0.9897 |
| 10 | 76.90 | 33.13 | 2.321x | 240.64 | 448.29 | 0.537x | 0.3590 | 0.9817 |

## 7) Findings

1. Best window in this thinking-mode sweep is `s=2`.
2. Acceptance rate drops monotonically as window size grows (`0.8019 -> 0.7179 -> 0.5800 -> 0.4503 -> 0.3590`).
3. Throughput and generation latency both degrade past `s=2`; large windows (`s=7`, `s=10`) are substantially slower.
4. Compared with the prior temp=0 non-thinking run, this thinking-mode setting is slower at every tested window and has much lower TAR.

## 8) Matched no-spec baseline (completed)

Matched no-spec run (same thinking/sampling knobs, `speculative_config=null`) was completed and parsed.

Steady-state baseline summary (steps 2-5):

| mode | setup (s) | avg total step (s) | avg generation (s) | avg E2E tok/s | token acceptance rate |
|---|---:|---:|---:|---:|---:|
| no-spec | 339.70 | 97.15 | 55.38 | 294.02 | n/a |

Step-1 cold-start baseline:

| mode | Step 1 total (s) | Step 1 generation (s) | Step 1 E2E tok/s |
|---|---:|---:|---:|
| no-spec | 43.90 | 18.16 | 187.65 |

Direct comparison vs matched no-spec baseline (steady-state: steps 2-5):

| spec tokens | spec avg gen (s) | no-spec avg gen (s) | gen ratio (spec/no-spec) | spec avg E2E tok/s | no-spec avg E2E tok/s | E2E ratio (spec/no-spec) | spec TAR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 47.86 | 55.38 | 0.864x | 325.23 | 294.02 | 1.106x | 0.8019 |
| 2  | 38.20 | 55.38 | 0.690x | 345.87 | 294.02 | 1.176x | 0.7179 |
| 4  | 44.16 | 55.38 | 0.797x | 330.11 | 294.02 | 1.123x | 0.5800 |
| 7  | 57.52 | 55.38 | 1.039x | 282.68 | 294.02 | 0.961x | 0.4503 |
| 10 | 76.90 | 55.38 | 1.389x | 240.64 | 294.02 | 0.818x | 0.3590 |

Takeaways vs matched no-spec:

1. Best point remains `s=2`.
2. `s=1`, `s=2`, and `s=4` beat no-spec on both generation latency and E2E throughput.
3. `s=7` and `s=10` regress below no-spec.
4. Larger speculative windows continue to trade off acceptance for speed and are net negative in this setting.


## 9) Notes

1. Non-fatal teardown noise appears in sweep/no-spec logs (`.nfs*` cleanup traceback and NCCL destroy warnings) but all runs finished with `rc=0`.
2. This sweep used `VLLM_DISABLE_COMPILE_CACHE=1`; setup times include cold-start and compile effects.
