# FusedAttention Experiment Log — B200

## H100 Reference Summary (from FUSEDATTN_TEST_REPORT.md)
- Best known working config on H100: RL branch sj/gpt-oss-cudnn, TP=2 EP=8, SP=true, sequence_packing=true, moe_permute_fusion=true
- Container: tk-vllm-v5 with pip cuDNN 9.19
- Known failures: sequence_packing hang at TP=2 EP=8 on some configs; seq_parallel must be true for alltoall+MoE+TP
- Key differences on B200: sm100 compute capability, nightly container has TE 2.12.0 + cuDNN 9.18.1 built-in

## Container Compatibility (nightly image)
- Transformer Engine: 2.12.0 (when venvs rebuilt with NRL_FORCE_REBUILD_VENVS=true)
- cuDNN: 9.18.1 (installed at /tmp/cudnn-linux-x86_64-9.18.1.3_cuda12-archive)
- Compute Capability: sm100 (B200)
- WARNING: Old cached venvs have TE 2.11.0 which does NOT support softmax_type=learnable + qkv_format=thd

[STATUS] CONTAINER-CHECK | RESULT: SUCCESS | NOTE: TE=2.12.0 (requires venv rebuild), cuDNN=9.18.1, sm100

---

## Attempt #1 — 2026-03-08 00:28 UTC
- **Job IDs:** 952874 (20B 2node), 952875 (20B 4node), 952876 (120B 2node), 952877 (120B 4node)
- **Model:** GPT-OSS 20B and 120B
- **Config:** alltoall dispatcher, moe_permute_fusion=true, sequence_packing=true
- **Container:** nemo_rl_nightly.sqsh (NRL_FORCE_REBUILD_VENVS=true)
- **FusedAttention confirmed:**
  - 20B: NO — ValueError: MoE+TP requires sequence_parallel=true (YAML default is false)
  - 120B: YES (sub-backend 1) — sequence_parallel already true in YAML
- **Result:**
  - 952874 (20B 2n): FAILED — sequence_parallel=false error
  - 952875 (20B 4n): FAILED — sequence_parallel=false error
  - 952876 (120B 2n): SUCCESS — 3 steps, FusedAttention (sub-backend 1), TE 2.12.0
  - 952877 (120B 4n): SUCCESS — 3 steps, FusedAttention (sub-backend 1), TE 2.12.0
- **vs H100:** Same behavior. sequence_parallel requirement is model-level, not GPU-arch dependent.
- **Error:** `ValueError: During training, performance may degrade if MoE and tensor parallelism are enabled without also enabling sequence parallelism.`
- **Next hypothesis:** Add sequence_parallel=true override for 20B configs

---

## Attempt #2 — 2026-03-08 00:58 UTC
- **Job ID:** 952887 (20B 2node)
- **Model:** GPT-OSS 20B
- **Config:** alltoall, moe_permute_fusion=true, sequence_packing=true, sequence_parallel=true
- **Container:** nemo_rl_nightly.sqsh (NRL_FORCE_REBUILD_VENVS=false — MISTAKE)
- **FusedAttention confirmed:** NO — TE 2.11.0 from old venv
- **Result:** FAILURE
- **vs H100:** Different — H100 used TE 2.12.0. This failure is B200-specific venv cache issue.
- **Error:** `Disabling FusedAttention for softmax_type = learnable and qkv_format = thd` / `Available backends = {FlashAttention=False, FusedAttention=False, UnfusedDotProductAttention=False}`
- **Root cause:** NRL_FORCE_REBUILD_VENVS=false reused old venv with TE 2.11.0. TE 2.12.0 added support for learnable+thd.
- **Next hypothesis:** Always use NRL_FORCE_REBUILD_VENVS=true to get TE 2.12.0

---

## Attempt #3 — 2026-03-08 01:10 UTC
- **Job IDs:** 952901 (20B 2node), 952889 (120B 4node 20-step — CANCELLED)
- **Model:** GPT-OSS 20B (3 steps) and 120B (20 steps)
- **Config:** alltoall, moe_permute_fusion=true, sequence_packing=true, sequence_parallel=true
- **Container:** nemo_rl_nightly.sqsh
  - 952901: NRL_FORCE_REBUILD_VENVS=true (correct)
  - 952889: NRL_FORCE_REBUILD_VENVS=false (MISTAKE — cancelled, same TE 2.11.0 issue)
- **FusedAttention confirmed:**
  - 952901: TBD (still running, venv rebuilding)
  - 952889: CANCELLED — detected TE 2.11.0 before crash
- **Result:** 952901 IN PROGRESS, 952889 CANCELLED
- **Next hypothesis:** Resubmit 120B with NRL_FORCE_REBUILD_VENVS=true

---

## Attempt #4 — 2026-03-08 01:14 UTC
- **Job ID:** 952949 (120B 4node 20-step)
- **Model:** GPT-OSS 120B
- **Config:** alltoall, moe_permute_fusion=true, sequence_packing=true, sequence_parallel=true (from YAML)
- **Container:** nemo_rl_nightly.sqsh (NRL_FORCE_REBUILD_VENVS=true)
- **FusedAttention confirmed:** TBD
- **Result:** IN PROGRESS
- **Next hypothesis:** If both 952901 and 952949 pass, submit 20-step 20B run

## Attempt #3 UPDATE — 2026-03-08 01:22 UTC
- **Job 952901 (20B 2node):** FusedAttention CONFIRMED!
  - TE 2.12.0, cuDNN 9.18.1, sm100 (B200)
  - `Selected backend = FusedAttention (sub-backend 1)` on ALL 16 workers
  - 4576+ FusedAttention calls across cluster (logprob phase)
  - `qkv_layout: thd_thd_thd`, `softmax_type: learnable`, `attn_mask_type: padding_causal`
  - Step 1/3 in progress (logprob + training forward)

## Attempt #3 FINAL — 2026-03-08 01:25 UTC
- **Job 952901 (20B 2node 3-step): SUCCESS!**
  - All 3 steps completed. FusedAttention (sub-backend 1) confirmed on all 16 workers.
  - TE 2.12.0, cuDNN 9.18.1, sm100 (B200)
  - Step 1: KL=0.1881, Step 2: KL=0.1960, Step 3: KL=0.1755
  - WandB: https://wandb.ai/nvidia/sync-grpo-b200-gptoss-exp/runs/3pdc5p2a

---

## Attempt #5 — 2026-03-08 01:25 UTC
- **Job ID:** 952952 (20B 2node 20-step)
- **Model:** GPT-OSS 20B
- **Config:** alltoall, moe_permute_fusion=true, sequence_packing=true, sequence_parallel=true
- **Container:** nemo_rl_nightly.sqsh (NRL_FORCE_REBUILD_VENVS=true)
- **FusedAttention confirmed:** Expected YES (same config as 952901 which passed)
- **Result:** IN PROGRESS
- **Next hypothesis:** Should complete 20 steps based on 3-step validation

[STATUS] GPT-OSS-20B-VALIDATE | RESULT: SUCCESS | NOTE: Job 952901, 3/3 steps, FusedAttention sub-backend 1, TE=2.12.0, sm100
[STATUS] GPT-OSS-20B-20STEP | RESULT: IN_PROGRESS | NOTE: Job 952952, WandB=https://wandb.ai/nvidia/sync-grpo-b200-gptoss-exp/runs/gn3gxe4l
[STATUS] GPT-OSS-120B-20STEP | RESULT: IN_PROGRESS | NOTE: Job 952949, FusedAttention CONFIRMED! Step 1/20, TE=2.12.0, sm100. WandB=https://wandb.ai/nvidia/sync-grpo-b200-gptoss-exp/runs/av2a9gbv
