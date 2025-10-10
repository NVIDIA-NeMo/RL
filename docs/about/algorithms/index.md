# Algorithms

NeMo RL supports multiple training algorithms for post-training large language models.

## Support Matrix

| Algorithms | Single Node | Multi-node |
|------------|-------------|------------|
| [GRPO](grpo.md) | [GRPO Single Node](grpo.md#grpo-single-node) | [GRPO Multi-node](grpo.md#grpo-multi-node): [GRPO Qwen2.5-32B](grpo.md#grpo-qwen25-32b), [GRPO Multi-Turn](grpo.md#grpo-multi-turn) |
| [On-policy Distillation](on-policy-distillation.md) | [Distillation Single Node](on-policy-distillation.md#on-policy-distillation-single-node) | [Distillation Multi-node](on-policy-distillation.md#on-policy-distillation-multi-node) |
| [Supervised Fine-Tuning (SFT)](sft.md) | [SFT Single Node](sft.md#sft-single-node) | [SFT Multi-node](sft.md#sft-multi-node) |
| [DPO](dpo.md) | [DPO Single Node](dpo.md#dpo-single-node) | [DPO Multi-node](dpo.md#dpo-multi-node) |
| [RM](rm.md) | [RM Single Node](rm.md#rm-single-node) | [RM Multi-node](rm.md#rm-multi-node) |

```{toctree}
:maxdepth: 2
:hidden:

grpo
on-policy-distillation
sft
dpo
rm
```

