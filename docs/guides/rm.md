# Reward Model Training in NeMo RL

This document explains how to train reward models (RM) within NeMo RL. Currently, only Bradley-Terry reward models are supported.

## Launch a Training Job

The script, [examples/run_sft.py](../../examples/run_sft.py), is used to train a Bradley-Terry reward model. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a training job is as follows:

```bash
uv run examples/run_sft.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

The YAML config must be specified. It uses the same base template as the SFT config but includes a new `reward_model_type` key that triggers Reward Model training. An example RM config file can be found at [examples/configs/rm.yaml](../../examples/configs/rm.yaml).

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

By default, NeMo RL supports the `HelpSteer3` dataset. This dataset is downloaded from Hugging Face and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.
