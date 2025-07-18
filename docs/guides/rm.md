# Reward Model Training in NeMo RL

This document explains how to train reward models (RM) within NeMo RL. Currently, only Bradley-Terry reward models are supported.

## Launch a Training Job

The script, [examples/run_rm.py](../../examples/run_rm.py), is used to train a Bradley-Terry reward model. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a training job is as follows:

```bash
uv run examples/run_rm.py --config examples/configs/rm.yaml

# Can also add overrides on CLI, like changing the model
uv run examples/run_rm.py --config examples/configs/rm.yaml policy.model_name=Qwen/Qwen2.5-1.5B
```

You must specify the YAML config. It shares the same base template as the SFT config but adds a new `reward_model_type` key to trigger RM training. You can find an example RM config file at [examples/configs/rm.yaml](../../examples/configs/rm.yaml).

**Reminder**: Set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). Make sure to log in using `huggingface-cli` if you're working with Llama models.

## Datasets

By default, NeMo RL supports the `HelpSteer3` dataset. This dataset is downloaded from Hugging Face and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

You can also configure custom preference datasets (for training and/or validation) as follows:
```
data:
  dataset_name: "PreferenceData:<NameOfDataset>:<LocalPathToDataset>"
  val_dataset_name: ["PreferenceData:<NameOfValidationDataset>:<LocalPathToValidationDataset>"]
```
Note:
- The name of any custom preference dataset must not contain `:`.
- If you are using a custom preference dataset for training, you must specify a custom preference dataset for validation.
- If you are using a logger, the prefix used for the custom validation preference dataset will be `validation-<NameOfValidationDataset>`.

When using `HelpSteer3` as the training dataset, the default validation set is also used and logged under the prefix `validation`. You can replace it with a custom preference dataset as follows:
```
data:
  dataset_name: "HelpSteer3"
  val_dataset_name: ["PreferenceData:validation:<LocalPathToValidationDataset>"]
```

Each custom preference dataset should be a JSONL file, with each line containing a valid JSON object formatted like this:
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

Currently, RM training supports only two completions (where rank 0 is preferred and rank 1 is rejected), with each completion being a single response. For example:
```
{"context": [{"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}, {"role": "user", "content": "Thanks! And what's the capital of Germany?"}], "completions": [{"rank": 0, "completion": [{"role": "assistant", "content": "The capital of Germany is Berlin."}]}, {"rank": 1, "completion": [{"role": "assistant", "content": "The capital of Germany is Munich."}]}]}
```

NeMo RL supports using multiple custom validation preference datasets during RM training:
```
data:
  dataset_name: "PreferenceData:<NameOfDataset>:<LocalPathToDataset>"
  val_dataset_name: [
    "PreferenceData:<NameOfValidationDataset1>:<LocalPathToValidationDataset1>",
    "PreferenceData:<NameOfValidationDataset2>:<LocalPathToValidationDataset2>",
   ]
```