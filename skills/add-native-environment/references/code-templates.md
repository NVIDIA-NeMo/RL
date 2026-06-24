# Code Templates for Native Environments

## Data Processor Template

```python
def my_task_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    # 1. Extract the problem/prompt from datum_dict
    problem = datum_dict["problem"]

    # 2. Extract ground truth or env metadata for verification
    extra_env_info = {"ground_truth": datum_dict["answer"]}

    # 3. Build the message log with tokenized content
    message_list = []
    if task_data_spec.system_prompt:
        message_list.append({"role": "system", "content": task_data_spec.system_prompt})
    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    message_list.append({"role": "user", "content": formatted_content})

    message: str = tokenizer.apply_chat_template(
        message_list,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = tokenizer(message, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    message_log: LLMMessageLogType = [
        {"role": "user", "content": message, "token_ids": token_ids}
    ]

    # 4. Handle overlength
    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    # 5. Return DatumSpec
    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "my_task"),
    }
```

Register in `PROCESSOR_REGISTRY` at the bottom of `processors.py`:

```python
PROCESSOR_REGISTRY: Dict[str, TaskDataProcessFnCallable] = cast(
    Dict[str, TaskDataProcessFnCallable],
    {
        ...existing entries...
        "my_task_data_processor": my_task_data_processor,
    },
)
```

## Single-Turn Environment Template

```python
import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

@ray.remote
class MyTaskEnvironment(EnvironmentInterface[dict]):
    def __init__(self, config: dict):
        self.config = config

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn[dict]:
        rewards = []
        answers = []
        for message_log, meta in zip(message_log_batch, metadata):
            response = message_log[-1]["content"]
            ground_truth = meta["ground_truth"]
            extracted_answer = self._extract_answer(response)
            reward = 1.0 if self._verify(extracted_answer, ground_truth) else 0.0
            rewards.append(reward)
            answers.append(extracted_answer)

        batch_size = len(message_log_batch)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}] * batch_size,
            metadata=[None] * batch_size,
            next_stop_strings=[None] * batch_size,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(batch_size, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get("total_reward", torch.tensor([0.0]))
        accuracy = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"accuracy": accuracy}

    def _extract_answer(self, response: str) -> str:
        ...

    def _verify(self, extracted: str, ground_truth: str) -> bool:
        ...
```

## Config Templates

### Single-turn config (`examples/configs/grpo_<task_name>.yaml`)

```yaml
defaults: "grpo_math_1B.yaml"

grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_rollout_turns: 1
  max_num_steps: 200
  seed: 42

policy:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  max_total_sequence_length: 2048
  dtensor_cfg:
    enabled: true
    cpu_offload: true
    activation_checkpointing: true
    sequence_parallel: true
  generation:
    backend: "vllm"
    max_new_tokens: 1024
    temperature: 1.0
    top_p: 0.999
    top_k: 10000
    vllm_cfg:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.6
      max_model_len: ${policy.max_total_sequence_length}

data:
  train:
    dataset_name: "<dataset_name>"
    split: "train"
  default:
    processor: "<processor_name>"
    env_name: "<env_name>"

env:
  <env_name>:
    num_workers: 2

logger:
  log_dir: "logs"
  wandb_enabled: true
  wandb:
    project: "nemo-rl-native-env"
    name: "grpo-<task_name>"
```

### Multi-turn config

```yaml
defaults: "grpo_math_1B.yaml"

grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_rollout_turns: 30
  max_num_steps: 200

data:
  add_system_prompt: false
  shuffle: false

env:
  <task_name>:
    cfg:
      max_moves: 30

logger:
  log_dir: "logs"
  wandb_enabled: true
  wandb:
    project: "nemo-rl-native-env"
    name: "grpo-<task_name>"
```

### RLHF reward model config additions

```yaml
data:
  default:
    env_name: "reward_model"

env:
  reward_model:
    enabled: true
    model_name: "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    precision: "bfloat16"
    batch_size: 4
    reward_model_cfg:
      enabled: true
      reward_model_type: "bradley_terry"
    dtensor_cfg:
      enabled: true
      cpu_offload: false
      activation_checkpointing: false
    resources:
      gpus_per_node: 1
      num_nodes: 1
    dynamic_batching:
      enabled: false
    sequence_packing:
      enabled: false
```

### Scale to 8 GPU

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

policy:
  dtensor_cfg:
    enabled: true
  generation:
    vllm_cfg:
      tensor_parallel_size: 1

grpo:
  num_prompts_per_step: 128
  num_generations_per_prompt: 16

env:
  <env_name>:
    num_workers: 8
```
