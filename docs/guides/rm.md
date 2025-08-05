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

By default, NeMo RL supports the `HelpSteer3` dataset. This dataset is downloaded from Hugging Face and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.
