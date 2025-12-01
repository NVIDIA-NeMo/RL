---
description: "Step-by-step guide to running Direct Preference Optimization (DPO) with NeMo RL"
categories: ["getting-started"]
tags: ["dpo", "quickstart", "preference-learning", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-dpo)=

# Get Started with DPO

**Direct Preference Optimization (DPO)** aligns models to human preferences without needing a separate reward model. It learns directly from "A is better than B" data.

:::{card}

**Goal**: Align a model to preferences using chosen/rejected pairs.

^^^

**In this tutorial, you will**:

1. Run DPO with the default configuration
2. Customize KL penalty and loss weights
3. Understand the required data format

:::

:::{button-ref} sft
:color: secondary
:outline:
:ref-type: doc

← Previous: SFT Quickstart
:::

:::{tip}
**Going deeper**: For comprehensive coverage of **reference models**, **loss functions**, and **training stability**, refer to the [DPO Guide](../guides/dpo.md).
:::

---

## Prerequisites

Ensure you have completed the [Quickstart Installation](index.md). You should have:
*   NeMo RL installed via `uv`
*   Environment variables set (`HF_HOME`, etc.)

---

## Running DPO

### 1. Basic Command
Run the default DPO example:

```bash
uv run python examples/run_dpo.py
```

This uses `examples/configs/dpo.yaml` and trains on a preference dataset.

### 2. Customizing Training

```bash
# Change KL Penalty (controls drift from reference)
uv run python examples/run_dpo.py \
  loss_fn.reference_policy_kl_penalty=0.1

# Change Loss Weights
uv run python examples/run_dpo.py \
  loss_fn.preference_loss_weight=1.0 \
  loss_fn.sft_loss_weight=0.1
```

---

## Data Format
DPO requires triplets:
*   **Prompt**: The input.
*   **Chosen**: The preferred response.
*   **Rejected**: The dis-preferred response.

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

---

## Next Steps

Now that you've aligned your model with preferences, you're ready for advanced reinforcement learning.

:::{button-ref} grpo
:color: primary
:ref-type: doc

Next: GRPO Quickstart →
:::
