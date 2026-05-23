---
name: add-native-environment
description: Interactive skill for building a new native RL environment in NeMo-RL. Guides through creating the full stack — dataset, data processor, environment, config, and training script — then validates end-to-end with a GRPO training run.
when_to_use: "add environment", "new RL environment", "create a native environment", "build a training task", "add a new game", "add a new reward function", "new GRPO task", "add native env".
allowed-tools: Bash Read Grep Glob Edit Write Agent AskUserQuestion
---

# Add Native Environment — interactive skill

This skill walks through building a **native RL environment** in NeMo-RL: a Ray actor implementing `EnvironmentInterface`, paired with a dataset and data processor, wired together with a GRPO training config. At the end, it validates the whole stack by training on 1 GPU and confirming reward increases.

**Native environments** run in-process with the training loop (as Ray actors). They are different from NeMo-Gym environments (external HTTP microservices) — for those, use the `add-benchmark` skill instead.

## Prerequisite knowledge

Before starting, read these files to understand the interfaces:

| What | File |
|------|------|
| `EnvironmentInterface`, `EnvironmentReturn` | `nemo_rl/environments/interfaces.py` |
| `ENV_REGISTRY`, `register_env()`, `create_env()` | `nemo_rl/environments/utils.py` |
| `DatumSpec`, `TaskDataSpec`, `TaskDataProcessFnCallable` | `nemo_rl/data/interfaces.py` |
| `PROCESSOR_REGISTRY`, `register_processor()` | `nemo_rl/data/processors.py` |
| `DATASET_REGISTRY`, `load_response_dataset()` | `nemo_rl/data/datasets/response_datasets/__init__.py` |

Reference implementations to study:

| Pattern | Environment | Data processor | Dataset | Config | Training script |
|---------|-------------|----------------|---------|--------|-----------------|
| **Single-turn** (model answers once, env scores) | `nemo_rl/environments/math_environment.py` | `math_hf_data_processor` in `processors.py` | `nemo_rl/data/datasets/response_datasets/openmathinstruct2.py` | `examples/configs/grpo_math_1B.yaml` | `examples/run_grpo.py` |
| **Multi-turn** (model + env interact over multiple steps) | `nemo_rl/environments/games/sliding_puzzle.py` | Inline in training script | `IterablePuzzleDataset` in `examples/run_grpo_sliding_puzzle.py` | `examples/configs/grpo_sliding_puzzle.yaml` | `examples/run_grpo_sliding_puzzle.py` |

## Workflow

Follow these stages in order. Use `AskUserQuestion` at decision points.

### Stage 1: Task Discovery

Ask the user what RL task they want to build. Gather:

1. **Task description**: What should the model learn to do? (e.g., "solve 24 game puzzles", "answer trivia questions", "play tic-tac-toe")
2. **Single-turn or multi-turn?**
   - Single-turn: model produces one response, environment scores it (like math). Set `max_rollout_turns: 1`.
   - Multi-turn: model and environment exchange messages across multiple turns with persistent state (like sliding puzzle). Set `max_rollout_turns` to the max number of turns.
3. **Reward signal**: How is correctness determined? (exact match, expression evaluation, game win condition, code execution, LLM-as-judge, etc.)

Based on the answers, select the reference pattern:
- Single-turn → follow the math environment pattern
- Multi-turn → follow the sliding puzzle pattern

**Critical: calibrate task difficulty for the model size.** Small models (0.6B-1.5B) need tasks they can partially solve from the start. If the model gets 0% reward for the first 20+ steps, GRPO cannot learn (there's no signal to reinforce). Guidelines:
- For games/puzzles: start with the easiest variant (small word list, small board, few constraints). You can increase difficulty later.
- For multi-turn tasks: reward partial progress (e.g., 0.5 for getting close, not just 1.0 for perfect).
- Test: run 5 steps and check if `mean_reward > 0.0`. If not, simplify the task before continuing.
- Example: Wordle with 200 words and a 0.6B model gets 0% reward — too hard. Use 20-30 very common words, or add partial rewards for correct letters.

### Stage 2: Model Selection

Ask the user to choose a model. Recommend starting with a small model for fast iteration:
- `Qwen/Qwen3-0.6B` (fastest, good for debugging)
- `Qwen/Qwen2.5-1.5B-Instruct` (good balance)
- `Qwen/Qwen2.5-3B-Instruct` (more capable)

Surface the **base vs. instruct model tradeoff**:
- **Instruct models**: follow instructions well out of the box, but may have collapsed entropy (low exploration), making RL less effective.
- **Base models**: higher entropy (better exploration), but may need SFT warm-up first since they haven't been fine-tuned for the task format.

Recommend instruct for initial validation. Note that if entropy is too low during training, switching to a base model + SFT may help.

### Stage 3: Dataset

Determine the data source:

**Option A: Off-the-shelf HuggingFace dataset**

1. Identify the dataset (e.g., `nvidia/OpenMathInstruct-2`, a custom HF dataset).
2. Create a dataset class in `nemo_rl/data/datasets/response_datasets/<name>.py` following the `OpenMathInstruct2Dataset` pattern:
   - Subclass `RawDataset`
   - Implement `__init__` to load via `datasets.load_dataset()`
   - Implement a `.map()` transform to normalize the schema to `{"messages": [...], "task_name": "..."}` or whatever fields the processor expects
   - Optionally implement train/val split
3. Register in `DATASET_REGISTRY` in `nemo_rl/data/datasets/response_datasets/__init__.py`.

**Option B: Procedurally generated data (games, puzzles)**

1. Create an `IterableDataset` that generates data on-the-fly (following `IterablePuzzleDataset` in `examples/run_grpo_sliding_puzzle.py`).
2. The `__iter__` method yields `DatumSpec` directly (no separate processor needed — processing is inline).
3. The `__len__` method returns a virtual length (total samples across all steps).
4. Wire the dataset in a custom training script (not via config `data.train.dataset_name`).

**Option C: Static JSONL / existing dataset**

1. Use `dataset_name: "openai_format"` or `dataset_name: "ResponseDataset"` in the config.
2. Point `data_path` to the JSONL file.
3. Write a matching processor (Stage 4).

### Stage 4: Data Processor

Skip this stage if using Option B (procedurally generated data with inline processing).

Create a data processor function in `nemo_rl/data/processors.py`. The processor converts a raw dataset row into a `DatumSpec`:

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

Register the processor in `PROCESSOR_REGISTRY` at the bottom of `processors.py`:

```python
PROCESSOR_REGISTRY: Dict[str, TaskDataProcessFnCallable] = cast(
    Dict[str, TaskDataProcessFnCallable],
    {
        ...existing entries...
        "my_task_data_processor": my_task_data_processor,
    },
)
```

### Stage 5: Environment

Create the environment class. The structure depends on single-turn vs. multi-turn.

#### Single-turn environment

Create `nemo_rl/environments/<task_name>_environment.py`:

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
            # Extract model's response (last assistant message)
            response = message_log[-1]["content"]

            # Extract ground truth from metadata
            ground_truth = meta["ground_truth"]

            # Verify and compute reward
            extracted_answer = self._extract_answer(response)
            reward = 1.0 if self._verify(extracted_answer, ground_truth) else 0.0

            rewards.append(reward)
            answers.append(extracted_answer)

        batch_size = len(message_log_batch)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}] * batch_size,
            metadata=[None] * batch_size,  # single-turn: no state to carry
            next_stop_strings=[None] * batch_size,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(batch_size, dtype=torch.bool),  # always done
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get("total_reward", torch.tensor([0.0]))
        accuracy = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"accuracy": accuracy}

    def _extract_answer(self, response: str) -> str:
        # Task-specific answer extraction logic
        ...

    def _verify(self, extracted: str, ground_truth: str) -> bool:
        # Task-specific verification logic
        ...
```

#### Multi-turn environment

Follow the sliding puzzle pattern in `nemo_rl/environments/games/sliding_puzzle.py`:

1. Define a `TypedDict` for metadata (game state, move count, max moves).
2. Create game logic class with static methods: `generate()`, `step()`, `render()`.
3. Create a runner class with `process_turn()` that parses actions from `<action>...</action>` tags.
4. Create the Ray actor `@ray.remote` class implementing `EnvironmentInterface`.
5. Use `stop_strings: ["</action>"]` in the datum to delimit action boundaries.

Key multi-turn differences:
- `metadata` carries state between turns (not `None`).
- `terminateds` is `False` until the episode ends.
- `observations` contain rendered state for the next turn.
- `next_stop_strings` tells the model when to stop generating.
- Set `max_rollout_turns` in the GRPO config to at least the max number of turns in the environment (e.g., 50 for sliding puzzle, 6 for Wordle).

For complex multi-turn environments (e.g., tool use, multi-agent), see `nemo_rl/environments/tau_bench_environment.py` (PR #2479) as a reference:
- Use Ray workers (`chunk_list_to_workers`) for parallel execution across the batch.
- Track per-episode state in the metadata `TypedDict` (episode ID, step count, etc.).
- Override `obs_use_chat_template() -> True` if observations use standard chat roles ("user", "tool") rather than the custom "environment" role.
- For environments that need external LLM calls (user simulators, judges), include mock modes for fast iteration without API dependencies.

#### Register the environment

Add to `ENV_REGISTRY` in `nemo_rl/environments/utils.py`:

```python
ENV_REGISTRY: Dict[str, EnvRegistryEntry] = {
    ...existing entries...
    "my_task": {
        "actor_class_fqn": "nemo_rl.environments.my_task_environment.MyTaskEnvironment",
    },
}
```

### Stage 6: Config

Create a YAML config at `examples/configs/grpo_<task_name>.yaml`.

**For single-turn tasks** (HF dataset path), inherit from a base config:

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

**For multi-turn tasks** (procedural data), inherit from the sliding puzzle config:

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
      # task-specific config
      max_moves: 30

logger:
  log_dir: "logs"
  wandb_enabled: true
  wandb:
    project: "nemo-rl-native-env"
    name: "grpo-<task_name>"
```

### Stage 7: Training Script

**For single-turn tasks** using a registered dataset + processor, use the standard entry point:

```bash
uv run python examples/run_grpo.py --config examples/configs/grpo_<task_name>.yaml
```

**For multi-turn / procedural tasks**, create a custom training script at `examples/run_grpo_<task_name>.py` following `examples/run_grpo_sliding_puzzle.py`:

1. Parse args and load config.
2. Initialize Ray and tokenizer.
3. Set up dataset (IterableDataset) and environment (Ray actor).

**Important**: After `config = MasterConfig(**config)`, use **attribute access** for top-level config sections: `config.grpo["seed"]`, `config.policy["tokenizer"]`, `config.env`, etc. Do **not** use `config["grpo"]` — `MasterConfig` wraps sub-dicts as attributes.
4. Call `setup()` and `grpo_train()` from `nemo_rl.algorithms.grpo`.

### Stage 8: Validate on 1 GPU

**Important 1-GPU config constraints** — when running on a single GPU, you must set:
```yaml
policy:
  dtensor_cfg:
    cpu_offload: false  # CPUOffload does NOT work on single GPU with AutoModel
```
If you see `NotImplementedError: CPUOffload doesn't work on single GPU for AutoModel`, this is why.

Also ensure `train_global_batch_size` is consistent with `num_prompts_per_step * num_generations_per_prompt`. A mismatch causes `AssertionError: Total batch size (X) is not a multiple of batch_size (Y)`.

Run the training loop on a single GPU:

```bash
# Single-turn (standard entry point)
uv run python examples/run_grpo.py --config examples/configs/grpo_<task_name>.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1

# Multi-turn (custom script)
uv run python examples/run_grpo_<task_name>.py --config examples/configs/grpo_<task_name>.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1
```

**What to check:**
- Training starts without errors.
- `train:mean_reward` (or equivalent metric) in the logs/wandb increases over steps.
- The environment correctly rewards good answers and penalizes bad ones (inspect logged samples if available).

If reward is flat at 0.0:
- The task may be too hard for the model. Try an easier variant or a more capable model.
- The reward function may have a bug. Test it manually with known-good and known-bad answers.
- The data processor may not be formatting prompts correctly. Inspect the tokenized output.
- For RLHF with reward models: the reward model may not differentiate well at this quality level. Try a different reward model or add reward scaling.

If reward is flat at 1.0:
- The task is too easy or the reward function always returns 1.0. Verify the check logic.

If reward oscillates or doesn't converge:
- Reduce learning rate (try `5e-7` instead of `5e-6`).
- Increase `num_generations_per_prompt` for better advantage estimation (16 or 32).
- Enable `use_leave_one_out_baseline: true` for variance reduction.
- For multi-turn tasks: check if the environment returns intermediate rewards properly and that metadata state is carried correctly between turns.
- Consider increasing `reference_policy_kl_penalty` to prevent the policy from diverging too far from the reference.

### Stage 8b: Using an Existing Reward Model (RLHF)

NeMo-RL has a built-in `reward_model` environment for RLHF training. Instead of writing a custom verification function, you can use a pre-trained reward model (e.g., Skywork, ArmoRM) to score model outputs.

**When to use**: The task has no deterministic verification (e.g., open-ended instruction following, helpfulness, safety alignment).

**How to configure**:

1. Use `env_name: "reward_model"` in the data config.
2. The reward model environment **requires DTensor** (`dtensor_cfg.enabled: true`). The Megatron path is **not supported**.
3. The reward model must be a Bradley-Terry model compatible with `AutoModelForSequenceClassification`.
4. Disable dynamic batching and sequence packing for the reward model.

Example config additions:
```yaml
data:
  default:
    env_name: "reward_model"

env:
  reward_model:
    enabled: true
    model_name: "Skywork/Skywork-Reward-V2-Qwen3-0.6B"  # or another HF reward model
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

**Supported reward models**: Any HuggingFace model that works with `AutoModelForSequenceClassification`, including:
- `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (small, fast)
- `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` (more capable)

**Limitation**: Only the DTensor backend is supported. Megatron-based reward models are not supported (tracked in issue #1154).

**GPU allocation**: The reward model needs its own GPU(s). With 2 GPUs, use 1 for policy + 1 for reward model. With 8 GPUs, split based on model sizes. See `examples/configs/grpo_rm_1B.yaml` for a working example.

**What to expect**: Bradley-Terry reward models output logits that can be large negative numbers (e.g., -10 to -15). A typical RLHF run starts with mean reward around -10 and should trend upward. Validated: mean reward went from -9.9 to +0.5 over 25 steps with Skywork-Reward-V2-Qwen3-0.6B on OpenMathInstruct-2.

**Troubleshooting RLHF**: If reward doesn't increase after 10+ steps:
- The reward model may not differentiate well for this task. Try a different reward model.
- Enable `reward_scaling` in the GRPO config to normalize reward range.
- Reduce learning rate to `1e-6` to prevent policy from diverging too fast.

### Stage 9: Scale to 8 GPU (optional)

Ask the user if they want to scale up. If yes, adjust the config:

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

policy:
  dtensor_cfg:
    enabled: true
  generation:
    vllm_cfg:
      tensor_parallel_size: 1  # keep TP=1 per generation worker, more workers in parallel

grpo:
  num_prompts_per_step: 128  # scale up batch size
  num_generations_per_prompt: 16

env:
  <env_name>:
    num_workers: 8  # more env workers to match throughput
```

## Checklist

Use this to verify completeness before declaring the task done:

- [ ] Environment class implements `EnvironmentInterface` with `step()` and `global_post_process_and_metrics()`
- [ ] Environment is registered in `ENV_REGISTRY` (or via `register_env()`)
- [ ] Dataset loads correctly (HF dataset class or IterableDataset)
- [ ] Data processor converts raw data to `DatumSpec` with correct `message_log`, `extra_env_info`, and tokenization
- [ ] Processor is registered in `PROCESSOR_REGISTRY` (or via `register_processor()`)
- [ ] YAML config wires dataset → processor → environment correctly
- [ ] Training script runs without errors
- [ ] `train:mean_reward` increases over training steps
- [ ] All new files have the Apache 2.0 copyright header
- [ ] Code passes `uv run --group dev pre-commit run --files <new_files>`
