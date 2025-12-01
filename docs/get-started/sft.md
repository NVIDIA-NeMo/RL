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

**Supervised Fine-Tuning (SFT)** is the standard first step in aligning language models. In this tutorial, you will learn the end-to-end workflow of fine-tuning a model to follow instructions.

:::{card}
**Goal**: Train a basic model (Llama-3.2-1B) on a sample dataset using your local machine.

^^^

**Steps**:

1. **Prepare Data**: Understand the input format.
2. **Configure**: Set hyperparameters in the YAML configuration.
3. **Train**: Run the `run_sft.py` script.
4. **Verify**: Check logs and model artifacts.
:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Previous: Quickstart Guide
:::

---

## Prerequisites

Ensure you have completed the [Quickstart Installation](index.md). You should have:

* NeMo RL installed via `uv`
* Environment variables set (`HF_HOME`, etc.)

---

## Step 1: Prepare Your Data

SFT requires a dataset of "Instruction" and "Response" pairs. The goal is to teach the model to generate the target response given the instruction.

By default, the example script uses the **SQuAD** dataset, but you can use your own data.

### The Data Format

NeMo RL supports standard JSONL files where each line is a training example.

```json
{"input": "What is the capital of France?", "output": "The capital of France is Paris."}
{"input": "Write a python function to add two numbers.", "output": "def add(a, b):\n    return a + b"}
```

You can specify your data in the configuration file (see Step 2).

---

## Step 2: Configure the Job

NeMo RL uses **Hydra** for configuration, allowing you to manage parameters in structured YAML files. The default configuration is at `examples/configs/sft.yaml`.

Key parameters you might want to change:

* **`policy.model_name`**: The base model to start from (e.g., `meta-llama/Llama-3.2-1B`).
* **`data.train_data_path`**: Path to your training `.jsonl` file.
* **`sft.max_num_epochs`**: How many times to iterate over the dataset.
* **`policy.optimizer.kwargs.lr`**: The step size for the optimizer.

:::{tip}
You don't need to edit the YAML file directly. You can override any parameter from the command line (shown in Step 3).
:::

---

## Step 3: Run the Training

We will use the `examples/run_sft.py` script to start the training.

::::{tab-set}

:::{tab-item} Native PyTorch (DTensor)
```bash
uv run python examples/run_sft.py \
  sft.max_num_epochs=1 \
  sft.max_num_steps=100 \
  policy.model_name="meta-llama/Llama-3.2-1B"
```
:::

:::{tab-item} Megatron Core
```bash
uv run python examples/run_sft.py \
  sft.max_num_epochs=1 \
  sft.max_num_steps=100 \
  policy.model_name="meta-llama/Llama-3.2-1B" \
  policy.megatron_cfg.enabled=true \
  policy.dtensor_cfg.enabled=false
```
:::

::::

**What's happening?**

* `uv run`: Ensures the script runs in the managed environment.
* `sft.max_num_epochs=1`: Overrides the configuration to run for just 1 epoch.
* `sft.max_num_steps=100`: Limits the run to 100 steps (good for a quick test).

---

## Step 4: Monitor and Verify

Once the script starts, it will print logs to your console. Watch for these key indicators:

1. **Data Loading**: Look for "Training and validation datasets loaded".
2. **Loss**: You should see the loss value printed periodically. Ideally, this number should decrease over time.

### Output Artifacts

By default, NeMo RL saves results to the `results/sft` directory (defined in `checkpointing.checkpoint_dir`).

* **Checkpoints**: Saved model weights (e.g., `results/sft/checkpoints/`).
* **Logs**: Training metrics (e.g., `logs/`).

To verify your training was successful, check that a checkpoint file exists in the results directory.

---

## Scaling Up

Once you are comfortable with the basic workflow, you can scale up to larger models and multi-GPU training.

### Using Megatron Core

For high-performance training on large models, switch to a Megatron-compatible configuration:

```bash
uv run python examples/run_sft.py \
  --config examples/configs/sft_openmathinstruct2_megatron.yaml
```

### Multi-Node Training

To run on a cluster (Slurm or Kubernetes), refer to the [Cluster Setup](cluster.md) guide.
