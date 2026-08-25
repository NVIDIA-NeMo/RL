# Offline Advantage-weighted Partition-function Loss (OAPL) in NeMo RL

OAPL is an offline RL algorithm that trains on a fixed dataset of `(x, y, r, log pi_ref(y|x))`
tuples, where `x` is a prompt, `y` is a full agent trajectory (including any
tool calls and their results), and `r` is the final reward obtained for that
trajectory. Given `n` generations `y_1, ..., y_n` collected offline for the
same prompt `x`, OAPL regresses the policy towards:

```
L(theta) = (beta * (log pi_theta(y|x) - log pi_ref(y|x) + log Z(x)) - r(x, y))^2
```

where `beta` is a fixed temperature and

```
Z(x) = (1/n) * sum_i exp(r(x, y_i) / beta)
```

is the partition function estimated from the `n` generations collected for
`x`. Because `log pi_ref(y|x)` and `log Z(x)` are precomputed offline as part
of the dataset, OAPL does not require a live reference-policy forward pass or
in-batch grouping during training: each `(x, y)` pair is trained on
independently.

## Launch an OAPL Run

The script [examples/run_oapl.py](../../examples/run_oapl.py) can be used to launch an OAPL experiment. This script can either be launched locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch an OAPL job is as follows:
```bash
uv run examples/run_oapl.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```
If not specified, `config` will default to [examples/configs/oapl.yaml](../../examples/configs/oapl.yaml).

## Configuration

NeMo RL allows users to configure OAPL experiments using `yaml` config files. An example OAPL configuration file can be found [here](../../examples/configs/oapl.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_oapl.py \
    cluster.gpus_per_node=8 \
    oapl.beta=0.05 \
    logger.wandb.name="oapl-dev-8-gpu"
```

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

OAPL has no built-in public dataset — you supply your own JSONL file (local
path or `hf_org/hf_dataset_name`) using the
[OAPLDataset](../../nemo_rl/data/datasets/oapl_datasets/oapl_dataset.py)
class. Data is grouped by prompt so that `log Z(x)` can be computed from the
group's rewards. Each line/example must be formatted like this:

```json
{
  "context": [], // list of dicts - the prompt message x (including previous turns, if any)
  "completions": [ // list of dicts - the n generations y_1..y_n collected for this prompt
    {
      "completion": [], // list of dicts - the completion message(s) y_i, including any tool-call turns
      "reward": 0.0, // float - r(x, y_i), the final reward for this trajectory
      "reference_logprob": 0.0 // float - log pi_ref(y_i | x), precomputed under the reference policy
    }
    // ... at least 2 completions are required per prompt to estimate Z(x)
  ]
}
```

For example:
```json
{
  "context": [
    {
      "role": "user",
      "content": "What's 17 * 24?"
    }
  ],
  "completions": [
    {
      "completion": [
        {
          "role": "assistant",
          "content": "17 * 24 = 408."
        }
      ],
      "reward": 1.0,
      "reference_logprob": -8.21
    },
    {
      "completion": [
        {
          "role": "assistant",
          "content": "17 * 24 = 388."
        }
      ],
      "reward": 0.0,
      "reference_logprob": -7.94
    }
  ]
}
```

You can modify your config as follows to point at such a dataset:
```yaml
data:
  # other data settings, see `examples/configs/oapl.yaml` for more details
  ...
  # dataset settings
  train:
    dataset_name: OAPLDataset
    data_path: /path/to/local/train_dataset.jsonl  # local file or hf_org/hf_dataset_name (HuggingFace)
    # beta must match `oapl.beta`, since log Z(x) is precomputed here at data-load time
    beta: ${oapl.beta}
    subset: null  # used for HuggingFace datasets
    split: train  # used for HuggingFace datasets
  validation:
    dataset_name: OAPLDataset
    data_path: /path/to/local/val_dataset.jsonl
    beta: ${oapl.beta}
```

**Note:** `data.train.beta` (and `data.validation.beta`) must match `oapl.beta`
used by the loss function, since `log Z(x)` is baked into the dataset at load
time using this value.

### Agentic Trajectories with Tool Calls

`completion` may be a multi-turn agentic trajectory, with `"assistant"`
tool-call turns interleaved with `"tool"`-role turns carrying the tool's
result, for example:

```json
{
  "completion": [
    {"role": "assistant", "content": null, "tool_calls": [{"name": "calculator", "arguments": {"expression": "17 * 24"}}]},
    {"role": "tool", "content": "408"},
    {"role": "assistant", "content": "17 * 24 = 408."}
  ],
  "reward": 1.0,
  "reference_logprob": -8.21
}
```

`log pi(y|x)` only sums log-probabilities over `"assistant"`-role tokens
(both turns in the example above); `"tool"`-role tokens (the tool's output)
are masked out and never contribute to the loss, since they were not
generated by the policy. `reference_logprob` should be computed the same
way -- summing `log pi_ref` over the `"assistant"` turns only -- so that it
is comparable to the training-time `log pi_theta(y|x)`.

If your data is not already in this format, write a preprocessing script to
convert it. An example implementation of a dataset class following this
convention can be found in
[oapl_datasets/oapl_dataset.py](../../nemo_rl/data/datasets/oapl_datasets/oapl_dataset.py).
