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
      chat_template: "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer: '}}{%- elif message['role'] == 'assistant'  %}{{message['content'].strip()}}{%- endif %}{% endfor %}"
    ```

By default, NeMo RL has some built-in supported datasets (e.g., [OpenAssistant](../../nemo_rl/data/datasets/response_datasets/oasst.py), [NuminaMath-1.5](../../nemo_rl/data/datasets/response_datasets/numinamath.py), [OpenMathInstruct-2](../../nemo_rl/data/datasets/response_datasets/openmathinstruct2.py), [Squad](../../nemo_rl/data/datasets/response_datasets/squad.py), etc.), you can see the full list [here](../../nemo_rl/data/datasets/response_datasets/__init__.py).
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
This works for any dataset class that calls `split_train_validation` in its `__init__` — which today includes most built-in datasets, among them [OpenAssistant](../../nemo_rl/data/datasets/response_datasets/oasst.py), [OpenMathInstruct-2](../../nemo_rl/data/datasets/response_datasets/openmathinstruct2.py), [OpenR1-Math-220k](../../nemo_rl/data/datasets/response_datasets/openr1_math.py), [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py), and [Tulu3SftMixtureDataset](../../nemo_rl/data/datasets/response_datasets/tulu3.py).
A dataset class that does not call it ignores `split_validation_size`; the dataset dispatcher emits a warning in that case, so a misconfigured run is not silent.
If you want to support this feature for your custom datasets or other built-in datasets, you can simply add the code to the dataset like [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py).
```python
# `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
self.val_dataset = None
self.split_train_validation(split_validation_size, seed)
```

### Energon Multimodal Datasets

The optional Energon SFT backend reads prepared WebDataset shards while the existing Hugging Face backend remains the default. `megatron-energon` ships in the `mcore` extra, and the Megatron policy workers pick that environment up from `ACTOR_ENVIRONMENT_REGISTRY`, so the driver runs under a plain `uv run`:

```bash
uv run examples/run_sft_v2.py \
  --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml
```

Each shard sample contains a JSON payload and its media members. The JSON payload uses one complete conversation as the sampling unit:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "media_index": 0},
      {"type": "text", "text": "Compare with the next image."},
      {"type": "image", "media_index": 1}
    ]},
    {"role": "assistant", "content": "The second image is brighter."}
  ],
  "media": [
    {"type": "image", "member": "000001.first.jpg"},
    {"type": "image", "member": "000001.second.jpg"}
  ],
  "tools": null
}
```

Media references must appear exactly once and in manifest order. Multi-turn conversations, assistant tool calls, matching tool results, and the final assistant response remain in the same sample. Tool results must reference an earlier tool-call ID.

Configure the prepared dataset path and split as follows:

```yaml
sft:
  # SFTv2 has no validation loop; the exemplar defaults (val_period: 10,
  # val_at_start: true) are rejected at startup.
  val_period: 0
  val_at_start: false
  val_at_end: false
data:
  _override_: true              # replace the exemplar's HF data block wholesale
  backend: energon
  max_input_seq_length: ${policy.max_total_sequence_length}
  shuffle: true
  energon:
    model_family: qwen          # required, no default: "qwen" or "nemotron"
    num_workers: 8
    shuffle_buffer_size: 1000
    processor_adapter: hf_multimodal
    packing_buffer_size: null
  train:
    path: /path/to/prepared/energon/dataset
    split: train
    virtual_epoch_length: 1000  # batches per virtual epoch
  validation: null              # SFTv2 builds a train loader only
data_plane:                     # required by run_sft_v2.py
  enabled: true
  impl: local
  max_partitions: 2
```

`model_family` and the top-level `data_plane` block have no defaults and are not
supplied by any exemplar config, so both must be set explicitly. The `sft`
overrides are needed for a different reason: SFTv2 runs no validation pass, so it
rejects `val_period`, `val_at_start` or `val_at_end` left at their exemplar
values, and `data.validation` must be null — the loader is always built with
`split_role="train"`.

The processor runs inside Energon loader workers and returns the same tokenized `message_log` representation as the Hugging Face path, including model inputs such as Qwen3-VL grid metadata or Nano Omni image sizes and frame counts. `prepare_sft_batch` creates the assistant loss mask, flattens the messages, and pads the batch without checking which loader produced it.

The v1 `SFTProcessorAdapter` and `HFMultimodalSFTProcessorAdapter` are narrow integration interfaces. They are planned to be replaced by a more comprehensive modular processor implementation; dataset loading and the policy-facing batch shape should remain stable through that change.

Sequence packing is unavailable in this path, on both sides: `packing_buffer_size` and `max_samples_per_sequence` are typed null-only, and `policy.sequence_packing` (like `policy.dynamic_batching`) is rejected at startup with `SFTv2 requires fixed NeMo-RL batching.` Packing is deferred to a later stage of the Energon integration. Energon does not provide a separate offline sequence-packing pipeline either; offline preparation may store length and media-cost metadata, but should not pre-concatenate multimodal conversations.

Training dataloader checkpoints include the Energon worker state plus a fingerprint of the source, loader, and processor settings. Restore must occur before the first iteration, and a changed fingerprint fails instead of silently continuing with a different stream. SFTv2 accepts a single train source; use an Energon metadataset to blend prepared sources.

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
