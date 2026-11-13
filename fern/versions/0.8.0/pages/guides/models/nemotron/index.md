---
title: "Nemotron"
---
This is the landing page for Nemotron model guidance in NeMo RL. It links to
model-specific subpages that cover the released post-training recipes, launch
instructions, and known issues for each variant.

For the full list of supported models, see
[Model Support](/about/model-support).

## Model Guides

- **[Nemotron 3 Nano](/guides/model-guides/nemotron/nemotron-3-nano)** — GRPO math post-training for the
  30B-A3B hybrid Mamba MoE model, including data preparation and Slurm launch
  scripts.
- **[Nemotron 3 Nano Omni](/guides/model-guides/nemotron/nemotron-3-nano-omni)** — GRPO for the Nano Omni
  vision-language model on the AutoModel and Megatron backends
  (CLEVR-CoGenT and MMPR-Tiny recipes).
- **[Nemotron 3 Super](/guides/model-guides/nemotron/nemotron-3-super)** — the multi-stage Nemotron 3
  Super post-training recipe (RLVR, SWE, and RLHF stages).
- **[Nemotron 3 Super Omni Image MOPD](/guides/model-guides/nemotron/nemotron-3-super-omni-mopd)** — image
  on-policy distillation for the Super Omni vision-language model with a
  non-colocated teacher (10-node production recipe plus a 4-node smoke).
- **[Nemotron 3 Ultra](/guides/model-guides/nemotron/nemotron-3-ultra)** — RLVR, teacher training, and
  MOPD stages on GB200 NVL72 hardware.
- **[Nemotron 3.5 Lightning](/guides/model-guides/nemotron/nemotron-3.5-lightning)** — RLVR with NeMo Gym
  on GB200, plus a compact 4-node DAPO math recipe on the Automodel
  backend.
