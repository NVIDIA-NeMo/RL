---
description: "Step-by-step guide to running Supervised Fine-Tuning (SFT) with NeMo RL"
categories: ["getting-started"]
tags: ["sft", "quickstart", "supervised-fine-tuning", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-sft)=

# Get Started with SFT

**Supervised Fine-Tuning (SFT)** is the standard first step in aligning language models. It trains the model to follow instructions using a dataset of prompt-response pairs.

:::{card}

**Goal**: Train a model to follow instructions using a supervised dataset.

^^^

**In this tutorial, you will**:

1. Run SFT with the default configuration
2. Scale up with the Megatron Core backend
3. Customize model and training parameters
4. Understand the SFT training process

:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Previous: Quickstart Guide
:::

:::{tip}
**Going deeper**: For comprehensive coverage of **custom datasets**, **packing strategies**, and **advanced hyperparameters**, refer to the [SFT Guide](../guides/sft.md).
:::

---

## Prerequisites

Ensure you have completed the [Quickstart Installation](index.md). You should have:
*   NeMo RL installed via `uv`
*   Environment variables set (`HF_HOME`, etc.)

---

## Running SFT

### 1. Basic Command
Run the default SFT example:

```bash
uv run python examples/run_sft.py
```

This uses the default configuration in `examples/configs/sft.yaml`.

### 2. Using Megatron Core Backend
For large-scale training:

```bash
uv run python examples/run_sft.py \
  --config examples/configs/sft_openmathinstruct2_megatron.yaml
```

### 3. Common Customizations

```bash
# Change Model
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Llama-3.2-1B-Instruct"

# Adjust Hyperparameters
uv run python examples/run_sft.py \
  sft.max_num_epochs=3 \
  sft.learning_rate=2e-5
```

---

## How it Works
SFT is standard language modeling on specific data:
1.  **Input**: Instruction + Desired Response.
2.  **Loss**: Cross-entropy loss on the response tokens.
3.  **Goal**: Maximize probability of the desired response given the instruction.

---

## Next Steps

You've successfully fine-tuned a model! Now, choose how you want to deploy or scale your training.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Local Deployment
:link: local-workstation
:link-type: doc

Manage GPUs and run concurrent experiments on your workstation.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Cluster Setup
:link: cluster
:link-type: doc

Scale your training to multi-node Slurm or Kubernetes clusters.
:::

::::
