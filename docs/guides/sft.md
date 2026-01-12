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
  1. `formatted_ds`: The dictionary of formatted datasets. This dictionary should contain `train` and `validation` splits, and each split should conform to the format described below.
  2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

SFT datasets are expected to follow the HuggingFace chat format. Refer to the [chat dataset document](../design-docs/chat-datasets.md) for details. If your data is not in the correct format, simply write a preprocessing script to convert the data into this format. [response_datasets/squad.py](../../nemo_rl/data/datasets/response_datasets/squad.py) has an example:

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
        ]
    }
```

NeMo RL SFT uses Hugging Face chat templates to format the individual examples. Three types of chat templates are supported, which can be configured using the `tokenizer.chat_template` in your YAML config (see [sft.yaml](../../examples/configs/sft.yaml) for an example):

1. Apply the tokenizer's default chat template. To use the tokenizer's default, either omit `tokenizer.chat_template` from the config altogether, or set `tokenizer.chat_template="default"`.
2. Use a "passthrough" template which simply concatenates all messages. This is desirable if the chat template has been applied to your dataset as an offline preprocessing step. In this case, you should set `tokenizer.chat_template` to None as follows:
    ```yaml
    tokenizer:
      chat_template: NULL
    ```
3. Use a custom template: If you would like to use a custom template, create a string template in [Jinja format](https://huggingface.co/docs/transformers/v4.34.0/en/chat_templating#how-do-i-create-a-chat-template), and add that string to the config. For example,

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

Your JSONL files should contain one JSON object per line with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
      {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    ]},
    {"role": "tool", "content": "22°C, sunny", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "The weather in Paris is currently 22°C and sunny."}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      }
    }
  ]
}
```

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


Adding a new dataset is a straightforward process.
As long as your custom dataset has the `formatted_ds` and `task_spec` attributes described above, it can serve as a drop-in replacement for Squad and OpenAssistant.

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.


## LoRA Configuration

NeMo RL supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. LoRA reduces trainable parameters by using low-rank matrices for weight updates while keeping the base model frozen.

Notes:
- LoRA is supported with DTensor v2 and Megatron backends. Uses the DTensor backend by default. DTensor v1 does not support LoRA (ensure `policy.dtensor_cfg._v2=true` when using DTensor).
- Triton kernels are only used in the DTensor v2 path. For `tensor_parallel_size > 1`, Automodel currently does not support Triton kernels (see note below).

### DTensor Configuration Parameters

The LoRA configuration is specified under the `policy.dtensor_cfg.lora_cfg` section:

```yaml
policy:
  dtensor_cfg:
    lora_cfg:
      enabled: False            # Set to True to enable LoRA fine-tuning
      target_modules: []        # List of module names to apply LoRA
      exclude_modules: []       # List of module names to exclude from LoRA
      match_all_linear: true    # Apply LoRA to all linear layers
      dim: 8                    # LoRA rank (r): controls adaptation capacity
      alpha: 32                 # LoRA scaling factor (effective lr = alpha/dim)
      dropout: 0.0              # Dropout probability for LoRA layers
      dropout_position: "post"  # Dropout position: "pre" or "post"
      lora_A_init: "xavier"     # Initialization method: "xavier" or "uniform"
      use_triton: true          # Use Triton-optimized kernels (DTensor v2 path)
```

### DTensor (Automodel) Parameter Details
- **`enabled`** (bool): Whether to enable LoRA training
- **`target_modules`** (list): Specific module names to apply LoRA. Empty with `match_all_linear=true` applies to all linear layers
- **`exclude_modules`** (list): Module names to exclude from LoRA
- **`match_all_linear`** (bool): When `true`, applies LoRA to all linear layers (overrides `target_modules`)
- **`dim`** (int): LoRA rank (r). Lower values = fewer parameters but less capacity. Typical: 4, 8, 16, 32, 64
- **`alpha`** (int): LoRA scaling factor. Effective learning rate multiplier = `alpha/dim`. Typical: 16, 32, 64
- **`dropout`** (float): Dropout probability for regularization
- **`dropout_position`** (str): Apply dropout before ("pre") or after ("post") LoRA
- **`lora_A_init`** (str): Initialization method for LoRA A matrix
- **`use_triton`** (bool): Use Triton-optimized kernels for better performance. Used for DTensor v2 only. **Note**: [Automodel does not support Triton for TP > 1](https://github.com/NVIDIA-NeMo/Automodel/blob/b2db55eee98dfe81a8bfe5e23ac4e57afd8ab261/nemo_automodel/recipes/llm/train_ft.py#L199). Set to `false` when `tensor_parallel_size > 1` to avoid compatibility issues

### DTensor Example Usage

```bash
uv run examples/run_sft.py policy.dtensor_cfg.lora_cfg.enabled=true
```

### Megatron Configuration Parameters

The LoRA configuration is specified under the `policy.megatron_cfg.peft` section:

```yaml
policy:
  megatron_cfg:
    peft:
      enabled: false                # Set to True to enable LoRA fine-tuning
      target_modules: []            # List of module names to apply LoRA, defaults to all linear layers
      exclude_modules: []           # List of module names not to apply LoRa.
      dim: 32                       # LoRA rank (r): controls adaptation capacity
      alpha: 32                     # LoRA scaling factor (effective lr = alpha/dim)
      dropout: 0.0                  # Dropout probability for LoRA layers
      dropout_position: "pre"       # Dropout position: "pre" or "post"
      lora_A_init_method: "xavier"  # Initialization method for lora A: "xavier" or "uniform"
      lora_B_init_method: "zero"    # Initialization method for lora B: "zero"
      a2a_experimental: false       # Enables the experimental All-to-All (A2A) communication strategy.
      lora_dtype: None              # Weight's dtype
```

### Megatron Parameter Details
- **`enabled`** (bool): Whether to enable LoRA training
- **`target_modules`** (list): Specific module names to apply LoRA. Defaults to all linear layers if the list is left empty. Example: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
  - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections in self-attention.
  - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
  - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
  - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
  Target modules can also contain wildcards. For example, you can specify target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv on the first two layers.
- **`exclude_modules`** (List[str], optional): A list of module names not to apply LoRa. It will match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in exclude_modules. If used, will require target_modules to be empty list or None.
- **`dim`** (int): LoRA rank (r). Lower values = fewer parameters but less capacity. Typical: 4, 8, 16, 32, 64
- **`alpha`** (int): LoRA scaling factor. Effective learning rate multiplier = `alpha/dim`. Typical: 16, 32, 64
- **`dropout`** (float): Dropout probability for regularization, defaults to 0.0
- **`dropout_position`** (str): Apply dropout before ("pre") or after ("post") LoRA
- **`lora_A_init`** (str): Initialization method for lora_A (choices: ['xavier', 'uniform']), defaults to xavier.
- **`lora_B_init`** (str): Initialization method for the low-rank matrix B. Defaults to "zero".
- **`a2a_experimental`** (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
- **`lora_dtype`** (torch.dtype): Weight's dtype, by default will use orig_linear's but if they are quantized weights (e.g. 4bit) needs to be specified explicitly.

### Megatron Example Usage
The config uses DTensor by default, so the megatron backend needs to be explicitly enabled. 
```sh
uv run examples/run_sft.py \
  --config examples/configs/sft.yaml \
  policy.dtensor_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.peft.enabled=true
```

For more details on LoRA, see [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).