# Direct Preference Optimization in NeMo RL

[Direct Preference Optimization (DPO)](https://arxiv.org/pdf/2305.18290) is an RL-free alignment algorithm that operates on preference data. Given a prompt and a pair of chosen and rejected responses, DPO aims
to increase the probability of the chosen response and decrease the probability of the rejected response relative to a frozen reference model. The actor is initialized using the reference model. For more details, refer to the
[DPO paper](https://arxiv.org/pdf/2305.18290).

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
```
{
    "context": list of dicts, # The prompt message (including previous turns, if any)
    "completions": list of dicts, # The list of completions
        {
            "rank": int, # The rank of the completion (lower rank is preferred)
            "completion": list of dicts, # The completion message(s)
        }
}
```
2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

Currently, DPO training supports only two completions (where the lowest rank is preferred and the highest one is rejected), with each completion being a single response. For example:
```
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

NeMo RL supports the `HelpSteer3` dataset. This dataset is downloaded from Hugging Face and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

We also provide a [PreferenceDataset](../../nemo_rl/data/hf_datasets/preference_dataset.py) class that is compatible with JSONL-formatted preference datasets. You can modify your config as follows to use such a custom preference dataset:
```
data:
  dataset_name: PreferenceDataset
  train_data_path: <LocalPathToTrainingDataset>
  val_data_path: <LocalPathToValidationDataset>
```
with support for multiple validation sets achieved with:
```
data:
  dataset_name: PreferenceDataset
  train_data_path: <LocalPathToTrainingDataset>
  val_data_paths:
      <NameOfValidationDataset1>: <LocalPathToValidationDataset1>
      <NameOfValidationDataset2>: <LocalPathToValidationDataset2>
```
If you are using a logger, the prefix used for each validation set will be `validation-<NameOfValidationDataset>`.

The older [DPODataset](../../nemo_rl/data/hf_datasets/dpo.py) class is deprecated. This class is also compatible with JSONL-formatted preference datsets. It assumes train and validation datasets have been split and processed into the expected format offline. The JSONL files should consist of examples with `prompt`, `chosen_response`, and `rejected_response` keys.

## DPO-Specific Parameters

The DPO implementation in NeMo RL supports several key parameters that can be adjusted:

- `dpo.reference_policy_kl_penalty`: Controls the strength of the KL penalty term
- `dpo.preference_loss_weight`: Weight for the preference loss
- `dpo.sft_loss_weight`: Weight for the auxiliary SFT loss
- `dpo.preference_average_log_probs`: Whether to average log probabilities over tokens in the preference loss term
- `dpo.sft_average_log_probs`: Whether to average log probabilities over tokens in the SFT loss term

These parameters can be adjusted in the config file or via command-line overrides to optimize training for your specific use case.

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.
