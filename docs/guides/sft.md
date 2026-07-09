# Supervised Fine-Tuning in NeMo RL

This document explains how to perform SFT within NeMo RL. It outlines key operations, including initiating SFT runs, managing experiment configurations using YAML, and integrating custom datasets that conform to the required structure and attributes.

## Launch an SFT Run

The script, [examples/run_sft.py](../../examples/run_sft.py), can be used to launch an experiment. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch an SFT job is as follows:

```bash
uv run examples/run_sft.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

If not specified, `config` will default to [examples/configs/sft.yaml](../../examples/configs/sft.yaml).

## Example Configuration File

NeMo RL allows users to configure experiments using `yaml` config files. An example SFT configuration file can be found [here](../../examples/configs/sft.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_sft.py \
    cluster.gpus_per_node=1 \
    logger.wandb.name="sft-dev-1-gpu"
```

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

SFT datasets in NeMo RL are encapsulated using classes. Each SFT data class is expected to have the following attributes:
  1. `dataset`: A dictionary containing the formatted datasets. Each example in the dataset must conform to the format described below.
  2. `task_name`: A string identifier that uniquely identifies the dataset.

SFT datasets are expected to follow the HuggingFace chat format. Refer to the [chat dataset document](../design-docs/chat-datasets.md) for details. If your data is not in the correct format, simply write a preprocessing script to convert the data into this format. [response_datasets/squad.py](../../nemo_rl/data/datasets/response_datasets/squad.py) has an example:

**Note:** The `task_name` field is required in each formatted example.

```python
def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": data["context"],
            },
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["answers"]["text"][0],
            },
        ],
        "task_name": self.task_name,
    }
```

NeMo RL SFT uses Hugging Face chat templates to format the individual examples. Three types of chat templates are supported, which can be configured using the `tokenizer.chat_template` in your YAML config (see [sft.yaml](../../examples/configs/sft.yaml) for an example):

1. Apply the tokenizer's default chat template. To use the tokenizer's default, either omit `tokenizer.chat_template` from the config altogether, or set `tokenizer.chat_template="default"`.
2. Use a "passthrough" template which simply concatenates all messages. This is desirable if the chat template has been applied to your dataset as an offline preprocessing step. In this case, you should set `tokenizer.chat_template` to None as follows:
    ```yaml
    tokenizer:
      chat_template: NULL
    ```
3. Use a custom template: If you would like to use a custom template, create a string template in [Jinja format](https://huggingface.co/docs/transformers/en/chat_templating_writing), and add that string to the config. For example,

    ```yaml
    tokenizer:
    custom_template: "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer: '}}{%- elif message['role'] == 'assistant'  %}{{message['content'].strip()}}{%- endif %}{% endfor %}"
    ```

By default, NeMo RL has some built-in supported datasets (e.g., [OpenAssistant](../../nemo_rl/data/datasets/response_datasets/oasst.py), [OpenMathInstruct-2](../../nemo_rl/data/datasets/response_datasets/openmathinstruct2.py), [Squad](../../nemo_rl/data/datasets/response_datasets/squad.py), etc.), you can see the full list [here](../../nemo_rl/data/datasets/response_datasets/__init__.py).
All of these datasets are downloaded from HuggingFace and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

We provide a [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py) class that is compatible with JSONL-formatted response datasets for loading datasets from local path or Hugging Face. You can use `input_key`, `output_key` to specify which fields in your data correspond to the question and answer respectively. Here's an example configuration:
```yaml
data:
  # other data settings, see `examples/configs/sft.yaml` for more details
  ...
  # dataset settings
  train:
    # this dataset will override input_key and use the default values for other vars
    data_path: /path/to/local/train_dataset.jsonl  # local file or hf_org/hf_dataset_name (HuggingFace)
    input_key: question
    subset: null  # used for HuggingFace datasets
    split: train  # used for HuggingFace datasets
    split_validation_size: 0.05  # use 5% of the training data as validation data
    seed: 42  # seed for train/validation split when split_validation_size > 0
  validation:
    # this dataset will use the default values for other vars except data_path
    data_path: /path/to/local/val_dataset.jsonl
  default:
    # will use below vars as default values if dataset doesn't specify it
    dataset_name: ResponseDataset
    input_key: input
    output_key: output
    prompt_file: null
    system_prompt_file: null
    processor: "sft_processor"
```

Your JSONL files should contain one JSON object per line with the following structure:

```json
{
  "input": "Hello",     // <input_key>: <input_content>
  "output": "Hi there!" // <output_key>: <output_content>
}
```

We support using multiple datasets for train and validation. You can refer to `examples/configs/grpo_multiple_datasets.yaml` for a full configuration example. Here's an example configuration:
```yaml
data:
  _override_: true # override the data config instead of merging with it
  # other data settings, see `examples/configs/sft.yaml` for more details
  ...
  # dataset settings
  train:
    # train dataset 1
    - dataset_name: OpenMathInstruct-2
      split_validation_size: 0.05 # use 5% of the training data as validation data
      seed: 42  # seed for train/validation split when split_validation_size > 0
    # train dataset 2
    - dataset_name: DeepScaler
  validation:
    # validation dataset 1
    - dataset_name: AIME2024
      repeat: 16
    # validation dataset 2
    - dataset_name: DAPOMathAIME2024
  # default settings for all datasets
  default:
    ...
```

`AIME2025` and `AIME2026` are registered alongside `AIME2024` and accept the same config keys (e.g. `repeat`), so any of them can drop into the `validation:` list above.

### Custom datasets defined outside NeMo RL

If you want to plug in a dataset class that lives outside the `nemo_rl`
package (so you don't have to edit the built-in registry), set
`dataset_name` to a fully qualified dotted import path. The dispatcher
will import the module and resolve the class. The class must accept the
same kwargs as the built-in datasets (i.e. the full data config) and
implement `set_task_spec` and `set_processor`.

```yaml
data:
  default:
    dataset_name: my_pkg.my_module.MyDataset  # importable from PYTHONPATH
```

The class must be importable — install it as a package or add its
parent directory to `PYTHONPATH` before launching training.

We support using a single dataset for both train and validation by using `split_validation_size` to set the ratio of validation.
[OpenAssistant](../../nemo_rl/data/datasets/response_datasets/oasst.py), [OpenMathInstruct-2](../../nemo_rl/data/datasets/response_datasets/openmathinstruct2.py), [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py), [Tulu3SftMixtureDataset](../../nemo_rl/data/datasets/response_datasets/tulu3.py) are supported for this feature.
If you want to support this feature for your custom datasets or other built-in datasets, you can simply add the code to the dataset like [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py).
```python
# `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
self.val_dataset = None
self.split_train_validation(split_validation_size, seed)
```

### OpenAI Format Datasets (with Tool Calling Support)

NeMo RL also supports datasets in the OpenAI conversation format, which is commonly used for chat models and function calling. This format is particularly useful for training models with tool-use capabilities.

#### Basic Usage

To use an OpenAI format dataset, configure your YAML as follows:

```yaml
data:
  train:
    dataset_name: openai_format
    data_path: <PathToTrainingDataset>       # Path to training data
    chat_key: "messages"                     # Key for messages in the data (default: "messages")
    system_key: null                         # Key for system message in the data (optional)
    system_prompt: null                      # Default system prompt if not in data (optional)
    tool_key: "tools"                        # Key for tools in the data (default: "tools")
    use_preserving_dataset: false            # Set to true for heterogeneous tool schemas (see below)
  validation:
    ...
```

#### Data Format

Your JSONL files should contain one JSON object per line following the [OpenAI Chat Completions function calling format](https://platform.openai.com/docs/guides/function-calling):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": {"city": "Paris", "unit": "celsius"}
        }
      }
    ]},
    {"role": "tool", "content": "22°C, sunny", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "The weather in Paris is currently 22°C and sunny."}
  ],
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "City name"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
      }
    }
  ]
}
```

> [!NOTE]
> NeMo RL passes `messages` and `tools` directly to the tokenizer's `apply_chat_template()`, so correct tool call rendering also depends on the model's chat template supporting this format.

#### Tool Calling with Heterogeneous Schemas

When your dataset contains tools with different argument structures (heterogeneous schemas), you should enable `use_preserving_dataset: true` to avoid data corruption:

```yaml
data:
  dataset_name: openai_format
  ...
  use_preserving_dataset: true  # IMPORTANT: Enable this for tool calling datasets
```

**Why this matters:** Standard HuggingFace dataset loading enforces uniform schemas by adding `None` values for missing keys. For example:
- Tool A has arguments: `{"query": "search term"}`
- Tool B has arguments: `{"expression": "2+2", "precision": 2}`

Without `use_preserving_dataset: true`, the loader would incorrectly add:
- Tool A becomes: `{"query": "search term", "expression": None, "precision": None}`
- Tool B becomes: `{"query": None, "expression": "2+2", "precision": 2}`

This corrupts your training data and can lead to models generating invalid tool calls. The `PreservingDataset` mode maintains the exact structure of each tool call.


## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.


## LoRA Configuration

NeMo RL supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of SFT models, including Nano‑v3 models, on both the DTensor and Megatron backends. To enable LoRA for SFT on the default DTensor backend:

```bash
uv run examples/run_sft.py policy.dtensor_cfg.lora_cfg.enabled=true
```

For the full reference — backend support, the DTensor vs Megatron schema comparison, config examples, parameter details, example recipes, and Hugging Face export — see the dedicated [LoRA guide](lora.md).

## Optimizations

### Chunked Linear Cross-Entropy Fusion Loss

During standard SFT training the model materializes a full logit tensor of shape `[batch_size, seq_length, vocab_size]`, which can cause out-of-memory (OOM) errors for long sequences or large vocabularies. The **chunked linear cross-entropy fusion loss** avoids this by computing the loss directly from the hidden states: it chunks the sequence dimension, projects each chunk to logits on the fly, computes per-token log probabilities, and discards the logits before moving to the next chunk.

**Benefits:**

- Extends the maximum trainable sequence length significantly (e.g. from <65K to >100K tokens) by eliminating the large logit tensor from GPU memory.
- Produces numerically equivalent loss values to the standard path.

**How to enable:**

Add the following to your Megatron config in your YAML file:

```yaml
policy:
  megatron_cfg:
    enabled: true
    use_fused_linear_logprobs: true
    fused_linear_logprobs_chunk_size: 256  # tokens per chunk; smaller = less memory, larger = more throughput
```

**Notes:**

- This optimization applies to SFT training with `NLLLoss` and DPO training. See the [DPO guide](dpo.md#chunked-fused-linear-logprobs) for DPO-specific details.
- Context parallelism is not supported when fused linear logprobs are enabled.
- The `fused_linear_logprobs_chunk_size` parameter controls the trade-off between memory savings and compute throughput. The default value of 256 is a good starting point.