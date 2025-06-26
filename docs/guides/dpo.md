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

Each class representing a NeMo RL DPO dataset is expected to have the following attributes:
1. `formatted_ds`: The dictionary of formatted datasets. This dictionary should contain `train` and `validation` splits, and each split should conform to the format described below.
2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

DPO datasets are expected to follow a specific format with three key fields:
- `prompt`: The input prompt/context
- `chosen_response`: The preferred/winning response
- `rejected_response`: The non-preferred/losing response

[data/hf_datasets/helpsteer3.py](../../nemo_rl/data/hf_datasets/helpsteer3.py) provides an example of how to format data for DPO:

```python
def format_helpsteer3(data):
    response_1 = data["response1"]
    response_2 = data["response2"]
    overall_preference = data["overall_preference"]

    if overall_preference < 0:
        chosen = response_1
        rejected = response_2
    elif overall_preference == 0:
        chosen = response_1
        rejected = response_1
    else:
        chosen = response_2
        rejected = response_1

    return {
        "prompt": data["context"],
        "chosen_response": chosen,
        "rejected_response": rejected,
    }
```

We also provide a [DPODataset](../../nemo_rl/data/hf_datasets/dpo.py) class that is compatible with jsonl-formatted preference datsets. This class assumes train and validation datasets have been split and processed into the expected format offline. The jsonl files should consist of examples with `prompt`, `chosen_response`, and `rejected_response` keys.

## Adding Custom DPO Datasets

Adding a new DPO dataset is straightforward. Your custom dataset class should:
1. Implement the required format conversion in the constructor
2. Set up the appropriate `task_spec`

Here's a minimal example which simply re-keys an existing jsonl dataset:

```{testcode}
from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from docs.helpers import make_dpo_dataset

class CustomDPODataset:
    def preprocess_dataset(
        self,
        data,
        prompt_key: str = "context",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected"
    ):
        return {
            "prompt": data[prompt_key],
            "chosen_response": data[chosen_key],
            "rejected_response": data[rejected_key],
        }
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        prompt_key: str,
        chosen_key: str,
        rejected_key: str,
    ):
        # Load and format your dataset
        fn_kwargs={
                "prompt_key": prompt_key, 
                "chosen_key": chosen_key, 
                "rejected_key": rejected_key
            }
        formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path, split="train").map(
                self.preprocess_dataset, 
                fn_kwargs=fn_kwargs,
            ),
            "validation": load_dataset("json", data_files=val_data_path, split="train").map(
                self.preprocess_dataset, 
                fn_kwargs=fn_kwargs,
            ),
        }
        
        # Initialize task spec with dataset name
        self.task_spec = TaskDataSpec(
            task_name="custom_dpo",
        )
        self.formatted_ds = formatted_ds

# Create temporary files using helper function
train_file, val_file = make_dpo_dataset()

# Initialize dataset
dataset = CustomDPODataset(
    train_data_path=train_file.name,
    val_data_path=val_file.name,
    prompt_key="context",
    chosen_key="chosen",
    rejected_key="rejected"
)

# Test dataset properties
print(f"Task name: {dataset.task_spec.task_name}")
print(f"Train examples: {len(dataset.formatted_ds['train'])}")
print(f"Validation examples: {len(dataset.formatted_ds['validation'])}")
print(f"First train example prompt: {dataset.formatted_ds['train'][0]['prompt']}")
print(f"First train example chosen response: {dataset.formatted_ds['train'][0]['chosen_response']}")
print(f"First train example rejected response: {dataset.formatted_ds['train'][0]['rejected_response']}")
```

```{testoutput}
Task name: custom_dpo
Train examples: 2
Validation examples: 2
First train example prompt: What is 2+2?
First train example chosen response: 4
First train example rejected response: 5
```

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
