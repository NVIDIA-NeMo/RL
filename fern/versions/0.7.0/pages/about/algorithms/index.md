---
title: "Algorithms"
---
NeMo RL supports multiple training algorithms for post-training large language models.

## Support Matrix

| Algorithms | Single Node | Multi-node |
|------------|-------------|------------|
| [GRPO](/about/algorithms/grpo) | [GRPO Single Node](/about/algorithms/grpo#grpo-single-node) | [GRPO Multi-node](/about/algorithms/grpo#grpo-multi-node): [GRPO Qwen2.5-32B](/about/algorithms/grpo#grpo-qwen25-32b), [GRPO Multi-Turn](/about/algorithms/grpo#grpo-multi-turn) |
| [DAPO](/about/algorithms/dapo) | [DAPO Single Node](/about/algorithms/dapo#dapo-single-node) | [DAPO Multi-node](/about/algorithms/dapo#dapo-multi-node) |
| [CISPO](/about/algorithms/cispo) | [CISPO Configuration](/about/algorithms/cispo#configuration) | [CISPO Async Lag-1 Recipe](/about/algorithms/cispo#async-lag-1-recipe) |
| [On-policy Distillation](/about/algorithms/on-policy-distillation) | [Distillation Single Node](/about/algorithms/on-policy-distillation#on-policy-distillation-single-node) | [Distillation Multi-node](/about/algorithms/on-policy-distillation#on-policy-distillation-multi-node) |
| [Multi-Teacher On-Policy Distillation (MOPD)](/about/algorithms/mopd) | — | [MOPD Configuration](/about/algorithms/mopd#configuration) |
| [Supervised Fine-Tuning (SFT)](/about/algorithms/sft) | [SFT Single Node](/about/algorithms/sft#sft-single-node) | [SFT Multi-node](/about/algorithms/sft#sft-multi-node) |
| [DPO](/about/algorithms/dpo) | [DPO Single Node](/about/algorithms/dpo#dpo-single-node) | [DPO Multi-node](/about/algorithms/dpo#dpo-multi-node) |
| [PPO](/about/algorithms/ppo) | [PPO Single Node](/about/algorithms/ppo#ppo-single-node) | [PPO Multi-node](/about/algorithms/ppo#ppo-multi-node) |
| [RM](/about/algorithms/rm) | [RM Single Node](/about/algorithms/rm#rm-single-node) | [RM Multi-node](/about/algorithms/rm#rm-multi-node) |

On-policy distillation is also supported in the PyTorch DTensor path.
