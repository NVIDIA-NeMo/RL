# Reward Model Training in NeMo RL

This document explains how to train reward models (RM) within NeMo RL. Currently, only Bradley-Terry reward models are supported on the DTensor backend. Megatron backend support is tracked [here](https://github.com/NVIDIA-NeMo/RL/issues/720).

## Launch a Training Job

The script, [examples/run_rm.py](../../examples/run_rm.py), is used to train a Bradley-Terry reward model. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a training job is as follows:

```bash
uv run examples/run_rm.py

# Can also add overrides on CLI, like changing the config or changing the model
uv run examples/run_rm.py --config examples/configs/rm.yaml policy.model_name=Qwen/Qwen2.5-1.5B
```

The default YAML config shares the same base template as the SFT config but includes a new `reward_model_cfg` section with `enabled: true` to load the model as a Reward Model. You can find an example RM config file at [examples/configs/rm.yaml](../../examples/configs/rm.yaml).

**Reminder**: Set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). Make sure to log in using `huggingface-cli` if you're working with Llama models.

## Datasets

Each RM dataset class is expected to have the following attributes:
1. `formatted_ds`: The dictionary of formatted datasets, where each dataset should be formatted like
```json
{
  "context": [], // list of dicts - The prompt message (including previous turns, if any)
  "completions": [ // list of dicts — The list of completions
    {
      "rank": 0, // int — The rank of the completion (lower rank is preferred)
      "completion": [] // list of dicts — The completion message(s)
    },
    {
      "rank": 1, // int — The rank of the completion (lower rank is preferred)
      "completion": [] // list of dicts — The completion message(s)
    }
  ]
}
```
2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

Currently, RM training supports only two completions (where the lowest rank is preferred and the highest one is rejected), with each completion being a single response. For example:
```json
{
    "context": [
        {
            "role": "user",
            "content": "What's the capital of France?"
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris."
        },
        {
            "role": "user",
            "content": "Thanks! And what's the capital of Germany?"
        }
    ],
    "completions": [
        {
            "rank": 0,
            "completion": [
                {
                    "role": "assistant",
                    "content": "The capital of Germany is Berlin."
                }
            ]
        },
        {
            "rank": 1,
            "completion": [
                {
                    "role": "assistant",
                    "content": "The capital of Germany is Munich."
                }
            ]
        }
    ]
}
```

By default, NeMo RL has support for [HelpSteer3](../../nemo_rl/data/datasets/preference_datasets/helpsteer3.py) and [Tulu3Preference](../../nemo_rl/data/datasets/preference_datasets/tulu3.py) datasets. Both of these datasets are downloaded from HuggingFace and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

We provide a [PreferenceDataset](../../nemo_rl/data/datasets/preference_datasets/preference_dataset.py) class that is compatible with jsonl-formatted preference datasets for loading datasets from local path or HuggingFace.. You can modify your config as follows to use such a custom preference dataset:
```yaml
data:
  dataset_name: PreferenceDataset
  train_data_path: <PathToTrainingDataset>  # e.g., /path/to/local/dataset.jsonl or hf_org/hf_dataset_name (HuggingFace)
  # multiple validation sets is supported
  val_data_paths:
    <NameOfValidationDataset>: <PathToValidationDataset1>
    <NameOfValidationDataset2>: <PathToValidationDataset2>
  train_split: <TrainSplit>, default is None  # used for HuggingFace datasets
  val_split: <ValSplit>, default is None  # used for HuggingFace datasets
```

We also provide a [BinaryPreferenceDataset](../../nemo_rl/data/datasets/preference_datasets/binary_preference_dataset.py) class, which is a simplified version of PreferenceDataset for pairwise ranked preference with single turn completions. You can use `prompt_key`, `chosen_key` and `rejected_key` to specify which fields in your data correspond to the question, chosen answer and rejected answer respectively. Here's an example configuration:
```yaml
data:
  dataset_name: BinaryPreferenceDataset
  train_data_path: <PathToTrainingDataset>  # e.g., /path/to/local/dataset.jsonl or hf_org/hf_dataset_name (HuggingFace)
  val_data_path: <PathToValidationDataset>
  prompt_key: <PromptKey>, default is "prompt"
  chosen_key: <ChosenKey>, default is "chosen"
  rejected_key: <RejectedKey>, default is "rejected"
  train_split: <TrainSplit>, default is None  # used for HuggingFace datasets
  val_split: <ValSplit>, default is None  # used for HuggingFace datasets
```

Please note:
- If you are using a logger, the prefix used for each validation set will be `validation-<NameOfValidationDataset>`. The total validation time, summed across all validation sets, is reported under `timing/validation/total_validation_time`.
- If you are doing checkpointing, the `metric_name` value in your `checkpointing` config should reflect the metric and validation set to be tracked. For example, `validation-<NameOfValidationDataset1>_loss`.

## Using Reward Models as Environments

Trained reward models can be used as environments in GRPO training for reinforcement learning from human feedback (RLHF). This allows you to use your trained reward model to provide rewards during policy optimization.

### Reward Model Environment

The Reward Model Environment provides a standardized interface for using trained reward models in RL training:

```python
from nemo_rl.environments.reward_model_environment import RewardModelEnvironment

env_config = {
    "enabled": True,
    "model_name": "path/to/your/trained/reward/model",
    "tokenizer": {"name": "path/to/your/trained/reward/model"},
    "precision": "bfloat16",
    "batch_size": 32,
    "resources": {"gpus_per_node": 1, "num_nodes": 1},
    "reward_model_cfg": {
        "enabled": True,
        "reward_model_type": "bradley_terry",
    },
}

reward_env = RewardModelEnvironment.remote(env_config)
```

### Integration with GRPO

To use your trained reward model with GRPO, you can use the [examples/run_grpo_rm.py](../../examples/run_grpo_rm.py) script:

```bash
# Run GRPO training with your trained reward model
uv run examples/run_grpo_rm.py --config examples/configs/grpo_rm_1B.yaml
```

### Configuration

In your GRPO configuration, specify the reward model environment:

```yaml
env:
  reward_model:
    enabled: true
    model_name: "path/to/your/trained/reward/model"
    tokenizer:
      name: "path/to/your/trained/reward/model"
    precision: "bfloat16"
    batch_size: 32
    resources:
      gpus_per_node: 1
      num_nodes: 1
    reward_model_cfg:
      enabled: true
      reward_model_type: "bradley_terry"
```

