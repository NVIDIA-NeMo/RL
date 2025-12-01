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
**Goal**: Align a fine-tuned model to preferences using chosen/rejected pairs.

^^^

**Steps**:

1. **Prepare Data**: Format your preference pairs.
2. **Configure**: Set the reference model and hyperparameters.
3. **Train**: Run the `run_dpo.py` script.
4. **Verify**: Monitor the preference loss.
:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Previous: Quickstart Guide
:::

---

## Prerequisites

Before running DPO, you typically need:

* A **Supervised Fine-Tuned (SFT)** model checkpoint (or a base instruct model) to use as the starting policy.
* NeMo RL installed via `uv`.

---

## Step 1: Prepare Your Data

DPO requires a dataset of "Preference Triplets". Each example contains a prompt and two potential responses: one "chosen" (preferred) and one "rejected".

### The Data Format

NeMo RL expects a JSONL file with the following structure:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I think it's London."
}
```

* **prompt**: The instruction given to the model.
* **chosen**: The better response.
* **rejected**: The worse response.

---

## Step 2: Configure the Job

The default configuration is at `examples/configs/dpo.yaml`.

Key parameters to tune:

* **`policy.model_name`**: Your starting model (e.g., the SFT checkpoint you created).
* **`data.dataset_name`**: Set to `BinaryPreferenceDataset` to use the custom JSONL format described above.
* **`data.train_data_path`**: Path to your preference `.jsonl` file.
* **`dpo.reference_policy_kl_penalty`**: Controls how much the model stays close to the original behavior (preventing "reward hacking").
* **`dpo.preference_loss_weight`**: The strength of the preference signal.

:::{tip}
For a local test, stick to small models like `meta-llama/Llama-3.2-1B-Instruct` to avoid OOM errors.
:::

---

## Step 3: Run the Training

Run the `examples/run_dpo.py` script. You can override the model and data path directly from the CLI.

```bash
uv run python examples/run_dpo.py \
  dpo.max_num_epochs=1 \
  dpo.reference_policy_kl_penalty=0.1 \
  policy.train_global_batch_size=32 \
  data.dataset_name=BinaryPreferenceDataset \
  data.train_data_path=path/to/your/data.jsonl
```

**What's happening?**

* The script loads the **Reference Policy** (frozen) and the **Active Policy** (trainable).
* It calculates the likelihood of "chosen" vs "rejected" responses for both models.
* It updates the Active Policy to increase the margin between chosen and rejected likelihoods.

---

## Step 4: Monitor and Verify

Watch the logs for these metrics:

1. **`preference_loss`**: Should decrease, indicating the model is learning to rank "chosen" higher than "rejected".
2. **`chosen_reward` vs `rejected_reward`**: Ideally, the "reward" (implicit) for chosen responses should go up, and rejected should go down.

### Output Artifacts

Results are saved to `results/dpo`:
* **Checkpoints**: The aligned model weights.

---

## Scaling Up

### Custom Loss Functions

NeMo RL supports variations of preference loss. You can change weights to balance SFT loss (maintaining instruction following) and Preference loss:

```bash
uv run python examples/run_dpo.py \
  dpo.preference_loss_weight=1.0 \
  dpo.sft_loss_weight=0.1
```

### Multi-Node Training

For large-scale alignment, enable the Megatron backend in `examples/configs/dpo.yaml` by setting `policy.megatron_cfg.enabled=true` and configuring the distributed parameters, or use a specific Megatron-compatible configuration file if available.
