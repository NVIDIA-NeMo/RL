# Direct Preference Optimization in NeMo RL

[Direct Preference Optimization (DPO)](https://arxiv.org/pdf/2305.18290) is an RL-free alignment algorithm that operates on preference data. Given a prompt and a pair of chosen and rejected responses, DPO aims
to increase the probability of the chosen response and decrease the probability of the rejected response relative to a frozen reference model. The actor is initialized using the reference model. For more details, refer to the
[DPO paper](https://arxiv.org/pdf/2305.18290).

## Other Objectives

- [Identity Preference Optimization (IPO)](https://arxiv.org/pdf/2310.12036)
- [Reward-Aware Preference Optimization (RPO) with backward KL divergence](https://arxiv.org/pdf/2406.11704), [forward KL divergence, and squared distance](https://arxiv.org/pdf/2502.00203)

## Launch a DPO Run

The script [examples/run_dpo.py](../../examples/run_dpo.py) can be used to launch a DPO experiment. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a DPO job is as follows:
```bash
uv run examples/run_dpo.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```
If not specified, `config` will default to [examples/configs/dpo.yaml](../../examples/configs/dpo.yaml).

## Configuration

NeMo RL allows users to configure DPO experiments using `yaml` config files. An example DPO configuration file can be found [here](../../examples/configs/dpo.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_dpo.py \
    cluster.gpus_per_node=8 \
    dpo.sft_loss_weight=0.1 \
    dpo.preference_average_log_probs=True \
    logger.wandb.name="dpo-dev-8-gpu"
```

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

Each DPO dataset class is expected to have the following attributes:
1. `formatted_ds`: The dictionary of formatted datasets, where each dataset should be formatted like
```json
{
  "context": [], // list of dicts - The prompt message (including previous turns, if any)
  "completions": [ // list of dicts — The list of completions
    {
      "rank": 0, // int — The rank of the completion (lower rank is preferred)
      "completion": [], // list of dicts — The completion message(s)
      "reward": 10.0, // Optional, float - The ground truth reward of the completion (required for rpo)
    },
    {
      "rank": 1, // int — The rank of the completion (lower rank is preferred)
      "completion": [], // list of dicts — The completion message(s)
      "reward": 0.0, // Optional, float - The ground truth reward of the completion (required for rpo)
    }
  ]
}
```
2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

DPO training supports only two completions (where the lowest rank is preferred and the highest one is rejected), with each completion being a single response. For example:
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
            ],
            "reward": 10.0 // required for rpo
        },
        {
            "rank": 1,
            "completion": [
                {
                    "role": "assistant",
                    "content": "The capital of Germany is Munich."
                }
            ],
            "reward": 0.0 // required for rpo
        }
    ]
}
```

By default, NeMo RL has support for [HelpSteer3](../../nemo_rl/data/datasets/preference_datasets/helpsteer3.py) and [Tulu3Preference](../../nemo_rl/data/datasets/preference_datasets/tulu3.py) datasets. Both of these datasets are downloaded from HuggingFace and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

We provide a [PreferenceDataset](../../nemo_rl/data/datasets/preference_datasets/preference_dataset.py) class that is compatible with jsonl-formatted preference datasets for loading datasets from local path or HuggingFace. You can modify your config as follows to use such a custom preference dataset:
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

## DPO-Specific Parameters

The DPO implementation in NeMo RL supports several key parameters that can be adjusted:

- `dpo.reference_policy_kl_penalty`: Controls the strength of the KL penalty term
- `dpo.preference_loss_weight`: Weight for the preference loss
- `dpo.sft_loss_weight`: Weight for the auxiliary SFT loss
- `dpo.preference_average_log_probs`: Whether to average log probabilities over tokens in the preference loss term
- `dpo.sft_average_log_probs`: Whether to average log probabilities over tokens in the SFT loss term
- `dpo.preference_loss`: Preference-based objective to use (choose from dpo, ipo, rpo_sq, rpo_fwd_kl, rpo_bwd_kl)
- `dpo.gt_reward_scale`: Reward scale for ground-truth rewards, only used in RPO

These parameters can be adjusted in the config file or via command-line overrides to optimize training for your specific use case.

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.
