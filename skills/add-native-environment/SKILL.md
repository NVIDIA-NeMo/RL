---
name: add-native-environment
license: Apache-2.0
description: Interactive skill for building a new native RL environment in NeMo-RL. Guides through creating the full stack — dataset, data processor, environment, config, and training script — then validates end-to-end with a GRPO training run.
when_to_use:
  - "add environment"
  - "new RL environment"
  - "create a native environment"
  - "build a training task"
  - "add a new game"
  - "add a new reward function"
  - "new GRPO task"
  - "add native env"
allowed-tools: Bash Read Grep Glob Edit Write Agent AskUserQuestion
---

# Add Native Environment — interactive skill

This skill walks through building a **native RL environment** in NeMo-RL: a Ray actor implementing `EnvironmentInterface`, paired with a dataset and data processor, wired together with a GRPO training config. At the end, it validates the whole stack by training on 1 GPU and confirming reward increases.

**Native environments** run in-process with the training loop (as Ray actors). They are different from NeMo-Gym environments (external HTTP microservices) — for those, use the `add-benchmark` skill instead.

**Safety:** This skill creates new Python files, config YAMLs, and training scripts, and executes shell commands including GPU training runs. Always confirm the implementation plan with the user before writing files or launching training. Do not execute training jobs without explicit user approval. Do NOT use for: deploying to Kubernetes, modifying existing environments, code review, or infrastructure tasks.

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
- For procedural puzzles: **only generate solvable instances**. If many puzzles are unsolvable, the model gets 0 reward for correct behavior (not attempting impossible puzzles), which degrades training. Example: Countdown Game with random targets 100-999 has many unsolvable instances, causing reward to decrease. Either verify solvability before presenting, or add partial rewards.
- Watch for reward *decreasing* — this means the task has negative training signal (the KL penalty outweighs the sparse reward). Simplify the task or add denser reward signal.

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

Create a data processor function in `nemo_rl/data/processors.py` that converts a raw dataset row into a `DatumSpec`. The processor must: (1) extract the prompt, (2) extract ground truth into `extra_env_info`, (3) build a tokenized `message_log`, (4) handle overlength inputs, (5) return a `DatumSpec` dict. Register it in `PROCESSOR_REGISTRY`.

See `references/code-templates.md` for the full processor template.

### Stage 5: Environment

Create the environment class. The structure depends on single-turn vs. multi-turn.

#### Single-turn environment

Create `nemo_rl/environments/<task_name>_environment.py` as a `@ray.remote` class implementing `EnvironmentInterface[dict]`. Implement `step()` to extract the model's response, verify it against ground truth from metadata, and return `EnvironmentReturn` with rewards and `terminateds=True`. Implement `global_post_process_and_metrics()` to compute accuracy.

See `references/code-templates.md` for the full single-turn environment template.

#### Tool-calling environments

For environments where the model calls tools (calculator, function calling, APIs), use the **model's native tool-calling format** rather than inventing custom tags:

1. Use `tokenizer.apply_chat_template(messages, tools=tool_definitions, ...)` to format the prompt. This embeds tool schemas in the model's trained format.
2. The model will generate tool calls in its native format (e.g., Qwen3 uses `<tool_call>{"name": ..., "arguments": ...}</tool_call>`).
3. Use `stop_strings: ["</tool_call>"]` to stop generation after each tool call.
4. Parse the tool call using the model's expected format (not custom regex).

This is better than custom tags because the model was **pre-trained** on this format — it doesn't need to learn a new output format through RL.

For production tool-calling with vLLM's native parser (structured decoding, schema validation), use the NeMo-Gym path instead (`expose_http_server: true` + `tool_parser: hermes/nemotron_json` in `vllm_cfg.http_server_serving_chat_kwargs`). See `examples/nemo_gym/` configs for reference.

#### Multi-turn environment

Follow the sliding puzzle pattern in `nemo_rl/environments/games/sliding_puzzle.py`:

1. Define a `TypedDict` for metadata (game state, move count, max moves).
2. Create game logic class with static methods: `generate()`, `step()`, `render()`.
3. Create a runner class with `process_turn()` that parses actions from the model's output (use the model's native format where possible).
4. Create the Ray actor `@ray.remote` class implementing `EnvironmentInterface`.
5. Use appropriate `stop_strings` in the datum to delimit action boundaries.

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

Create a YAML config at `examples/configs/grpo_<task_name>.yaml`. Inherit from `grpo_math_1B.yaml`. Key settings: `max_rollout_turns` (1 for single-turn, N for multi-turn), `num_prompts_per_step`, `num_generations_per_prompt`, model name, dataset/processor/env wiring.

See `references/code-templates.md` for single-turn and multi-turn config templates.

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

For tasks with no deterministic verification (open-ended instruction following, helpfulness), use the built-in `reward_model` environment with a pre-trained Bradley-Terry reward model (e.g., `Skywork/Skywork-Reward-V2-Qwen3-0.6B`). Requirements: DTensor only (no Megatron), `AutoModelForSequenceClassification`-compatible model, separate GPU for the reward model. See `references/code-templates.md` for config and `examples/configs/grpo_rm_1B.yaml` for a working example.

### Stage 9: Scale to 8 GPU (optional)

Ask the user if they want to scale up. Key changes: `gpus_per_node: 8`, increase `num_prompts_per_step` to 128, increase `num_workers` to 8. See `references/code-templates.md` for the scale-up config.

## Stage 10: Write a README

Create a `README.md` documenting: overview, data setup, training commands (1-GPU and 8-GPU), expected reward trajectory, and wandb links showing reward increasing over steps.

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
- [ ] README.md documents data setup, training commands, and results with wandb links
- [ ] For tool-calling environments: uses model's native tool format via `apply_chat_template(tools=...)`
