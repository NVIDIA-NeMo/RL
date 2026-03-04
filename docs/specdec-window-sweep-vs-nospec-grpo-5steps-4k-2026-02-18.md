# SpecDec Window Sweep vs No-Spec (GRPO 5-step, 4k) Report

Date: 2026-02-18

## 1) Experimental setup

1. Goal: quantify 4k decode behavior for a smaller draft model and test if speculative window size changes generation time materially.
2. Hardware: 1 node, 8 GPUs.
3. Target model: Qwen3-32B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
4. Draft model (spec runs): Qwen3-1.7B snapshot
   `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
5. Shared settings:
   - `grpo.max_num_steps=5`
   - `policy.generation.max_new_tokens=4096`
   - `temperature=0.0`, `top_p=1.0`, `top_k=null`
   - `policy.generation.vllm_cfg.load_format=auto`
   - `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
   - `policy.dtensor_cfg.activation_checkpointing=true`
   - Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
6. Sweep values: `num_speculative_tokens in {2, 7, 10}`.
7. Baseline: non-spec (`speculative_config=null`).

## 2) Canonical artifacts

- Sweep status file:
  `/home/scratch.shaunakj_other/logs/spec-window-sweep-2-7-10-2026-02-18-175434.status.txt`
- Spec logs:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s2-l4096-actckpt-loadfmtauto-sweep-2026-02-18-175434/run-5steps-spec1p7b-s2-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s7-l4096-actckpt-loadfmtauto-sweep-2026-02-18-180546/run-5steps-spec1p7b-s7-l4096-actckpt-flashattn-loadfmtauto.log`
  - `/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s10-l4096-actckpt-loadfmtauto-sweep-2026-02-18-181751/run-5steps-spec1p7b-s10-l4096-actckpt-flashattn-loadfmtauto.log`
- No-spec log:
  - `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-l4096-actckpt-loadfmtauto-2026-02-18-183726/run-5steps-nospec-l4096-actckpt-flashattn-loadfmtauto.log`

All four runs completed with `rc=0`.

## 3) Primary results (steady-state: steps 2-5)

| Run | Setup (s) | Wall time (s) | Avg total step (s) | Avg generation (s) | Avg E2E tok/s (group) | TAR (steps 2-5) |
|---|---:|---:|---:|---:|---:|---:|
| No-spec | 350.9 | 842.4 | 99.68 | 56.30 | 335.95 | n/a |
| Spec s=2 | 295.3 | 668.4 | 74.23 | 32.22 | 451.45 | 99.82% |
| Spec s=7 | 335.5 | 719.9 | 73.48 | 32.09 | 455.95 | 99.39% |
| Spec s=10 | 298.2 | 680.5 | 74.41 | 33.25 | 450.10 | 98.87% |

## 4) Direct speedups vs no-spec (steps 2-5)

| Spec run | Generation speedup | E2E tok/s speedup | Total step-time speedup |
|---|---:|---:|---:|
| s=2 | 1.747x | 1.344x | 1.343x |
| s=7 | 1.754x | 1.357x | 1.357x |
| s=10 | 1.693x | 1.340x | 1.340x |

Interpretation:

1. The large gain is from enabling speculative decoding at all (spec vs no-spec).
2. Window-only changes (`2 -> 7 -> 10`) are second-order in this regime.
3. `s=10` is worse than `s=2/s=7` on generation time and TAR.

## 5) Why larger windows can hurt here

1. Higher `num_speculative_tokens` increases per-cycle draft work and verifier span.
2. Benefit requires long accepted streaks; larger windows increase probability of at least one mismatch in a window.
3. Rejected draft segments become wasted work, reducing efficiency.
4. Data matches this trend:
   - TAR drops with larger windows: `99.82% -> 99.39% -> 98.87%` (steps 2-5)
   - Generation time worsens at `s=10`: `32.22s (s=2)`, `32.09s (s=7)`, `33.25s (s=10)`.

## 6) Step-1 cold-start behavior

Step 1 is not representative of steady-state and shows much lower TAR at larger windows:

| Spec run | Step 1 TAR |
|---|---:|
| s=2 | 0.8488 |
| s=7 | 0.6584 |
| s=10 | 0.5674 |

Use steps 2-5 for fair throughput and generation-time comparison.

## 7) Practical conclusion

For this 4k GRPO setup with 32B target + 1.7B draft:

1. Spec decode gives substantial generation and E2E improvement over no-spec.
2. Window size has a local optimum; larger is not always better.
3. `s=2` and `s=7` are both strong; `s=10` is beyond the sweet spot.

