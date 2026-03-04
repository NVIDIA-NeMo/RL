# SpecDec vs Non-Spec GRPO Report (5-step benchmark)

Date: 2026-02-18

## 1) Experimental setup

1. Codebase: `RL` repo, with patch in `nemo_rl/models/generation/__init__.py` so CLI `load_format` override is respected in train runs.
2. Goal: compare speculative decoding vs non-spec baseline on the same GRPO setup.
3. Hardware/parallelism: 1 node, 8 GPUs (`cluster.gpus_per_node=8`), colocated vLLM + policy workers.
4. Target model: Qwen3-32B snapshot:
   `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
5. Draft model (spec run only): Qwen3-14B snapshot:
   `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
6. Shared run settings:
   - `grpo.max_num_steps=5`
   - `policy.generation.max_new_tokens=256`
   - `temperature=0.0`, `top_p=1.0`, `top_k=null`
   - `policy.generation.vllm_cfg.load_format=auto`
   - `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`
   - Dataset: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
   - No validation (`val_period=0`, `val_at_start=false`, `val_at_end=false`)
7. Spec-only settings:
   - `speculative_config.model=<14B>`
   - `num_speculative_tokens=2`
   - `draft_tensor_parallel_size=1`
8. Non-spec settings:
   - `speculative_config=null`

## 2) Run artifacts

- Spec run log:
  `/home/scratch.shaunakj_other/logs/grpo-32b-spec14b-s2-l256-loadfmtauto-2026-02-18/run-5steps-spec14b-flashattn-scratchpaths-policyvenv-loadfmtauto.log`
- Non-spec run log:
  `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-l256-loadfmtauto-2026-02-18/run-5steps-nospec-flashattn-scratchpaths-policyvenv-loadfmtauto.log`

## 3) Step-by-step results

### 3.1 Spec run (32B + 14B draft, spec=2)

#### Timing (seconds)

| Step | Total | Generation | Prep-for-gen total | Logprob prep | Policy+Ref logprobs | Policy train | Training prep | Transfer/update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 272.00 | 173.78 | 17.07 | 4.03 | 36.35 | 39.89 | 0.52 | 2.24 |
| 2 | 68.44 | 4.12 | 42.35 | 4.57 | 1.77 | 3.74 | 11.61 | 0.79 |
| 3 | 66.09 | 4.10 | 39.17 | 4.38 | 2.07 | 3.46 | 12.64 | 0.91 |
| 4 | 64.29 | 4.01 | 36.88 | 4.42 | 1.62 | 3.95 | 13.09 | 1.10 |
| 5 | 64.86 | 3.91 | 36.13 | 4.34 | 1.62 | 4.02 | 14.51 | 0.94 |

#### Throughput + TAR

| Step | E2E tok/s (group) | E2E tok/s/gpu | Gen worker tok/s (group) | Gen worker tok/s/gpu | Train worker tok/s (group) | Train worker tok/s/gpu | Policy train tok/s/gpu | Policy+Ref tok/s/gpu | TAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.37 | 1.30 | 16.23 | 2.03 | 36.99 | 4.62 | 8.84 | 9.70 | 0.9302 (1332/1432) |
| 2 | 41.03 | 5.13 | 682.02 | 85.25 | 510.12 | 63.77 | 93.90 | 198.68 | 0.9893 (1296/1310) |
| 3 | 40.43 | 5.05 | 651.50 | 81.44 | 483.20 | 60.40 | 96.64 | 161.07 | 0.9833 (1294/1316) |
| 4 | 44.12 | 5.51 | 706.37 | 88.30 | 508.97 | 63.62 | 89.71 | 218.80 | 0.9940 (1326/1334) |
| 5 | 42.06 | 5.26 | 697.77 | 87.22 | 484.25 | 60.53 | 84.89 | 210.96 | 0.9832 (1288/1310) |

Aggregate TAR over 5 steps: `0.9752 = 6536/6702`

### 3.2 Non-spec run (32B only)

#### Timing (seconds)

| Step | Total | Generation | Prep-for-gen total | Logprob prep | Policy+Ref logprobs | Policy train | Training prep | Transfer/update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 172.81 | 73.16 | 17.22 | 4.34 | 33.77 | 43.38 | 0.52 | 2.26 |
| 2 | 70.06 | 4.75 | 43.08 | 4.60 | 1.51 | 2.80 | 12.99 | 1.03 |
| 3 | 67.17 | 4.74 | 38.60 | 3.82 | 1.52 | 4.26 | 13.92 | 0.83 |
| 4 | 62.58 | 4.62 | 36.41 | 4.42 | 1.51 | 2.69 | 12.63 | 0.86 |
| 5 | 63.13 | 4.63 | 35.13 | 4.47 | 1.62 | 2.42 | 14.61 | 0.82 |

#### Throughput

| Step | E2E tok/s (group) | E2E tok/s/gpu | Gen worker tok/s (group) | Gen worker tok/s/gpu | Train worker tok/s (group) | Train worker tok/s/gpu | Policy train tok/s/gpu | Policy+Ref tok/s/gpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.32 | 2.04 | 38.55 | 4.82 | 36.55 | 4.57 | 8.13 | 10.44 |
| 2 | 40.08 | 5.01 | 591.67 | 73.96 | 650.46 | 81.31 | 125.15 | 232.09 |
| 3 | 39.78 | 4.97 | 564.27 | 70.53 | 462.70 | 57.84 | 78.46 | 220.04 |
| 4 | 45.32 | 5.67 | 613.81 | 76.73 | 675.95 | 84.49 | 131.96 | 234.91 |
| 5 | 43.21 | 5.40 | 589.02 | 73.63 | 676.01 | 84.50 | 141.04 | 210.80 |

TAR is not applicable in non-spec mode.

## 4) Direct speed comparison (spec vs non-spec)

Per-step generation speedup (`non_spec_generation_time / spec_generation_time`):

- Step 1: `0.421x` (spec slower on first step)
- Step 2: `1.153x`
- Step 3: `1.156x`
- Step 4: `1.152x`
- Step 5: `1.184x`

Steady-state (steps 2-5):

- Generation time speedup: `1.161x`
- Generation worker tok/s speedup: `1.161x`
- E2E tok/s speedup: `0.996x` (near parity)
- Total step-time speedup: `0.997x` (near parity)

## 5) Setup-time note

- Spec setup total: `512.9s` (`vLLM 230.5s`, `policy 30.6s`, `other 251.8s`)
- Non-spec setup total: `603.9s` (`vLLM 321.8s`, `policy 28.8s`, `other 253.1s`)

## 6) Summary

Spec decoding with 32B target + 14B draft (`num_speculative_tokens=2`) is working correctly (high TAR: `97.52%` aggregate) and gives a clear generation-phase speedup (`~1.16x` steady-state). In this 5-step GRPO run, end-to-end step throughput remains near parity because non-generation phases dominate much of each step.
