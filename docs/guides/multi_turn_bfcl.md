# Multi-Turn Tool Calling with BFCL v3

## Overview

[BFCL v3 (Berkeley Function Tool Calling)](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) introduces a multi-turn and multi-step function calling dataset, advancing beyond the single-turn conversations in v1 and v2. This guide explains how to use NeMo RL to implement multi-turn tool calling on BFCLv3 with Group Relative Policy Optimization (GRPO). 

We use tool APIs from [this](https://github.com/bespokelabsai/verifiers/tree/main/verifiers/tools/bfcl_tools) repository. We take in these tools and train the model to execute them based on user prompts. To do so, we train [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the BFCL v3 dataset. We then show how to use NeMo RL's evaluation scripts to evaluate the trained model on a subset of that dataset. 

## Training the Model

Follow the instructions in the [README](https://github.com/NVIDIA-NeMo/RL?tab=readme-ov-file#prerequisites) for setting up your environment. Use the following command to start training your model: 

```bash
uv run python examples/run_grpo_bfclv3.py --config examples/configs/grpo_bfclv3.yaml logger.wandb_enabled=true logger.wandb.project=nemorl_bfcl_qwen_7b_new logger.wandb.name=nemorl_bfcl_v3_all_4
```

## Configuration Parameters

The [`examples/configs/grpo_bfclv3.yaml`](../../examples/configs/grpo_bfclv3.yaml) file contains configs for multi-turn training:

**Multi-Turn Control:**
```yaml
grpo:
  max_rollout_turns: 4  # Maximum turns per episode (set to 1 for single-turn, >1 for multi-turn)
  max_num_steps: 100  # Total training steps
  max_num_epochs: 20
  num_prompts_per_step: 16  # Batch size for rollouts
  num_generations_per_prompt: 8  # Multiple generations per prompt for GRPO
```
So it will be trained for (min(max_steps, len(data_loader)))* max_num_epochs number of steps.

**Environment Configuration:**
```yaml
env:
  bfcl_multiturn:
    enable: true  # Enable BFCL multi-turn environment
    num_workers: 4  # Parallel environment workers
```

**Note:** You can also change training batch size and number of prompts generated per batch if you have compute constraints.

## Data

We downloaded the data from [here](https://github.com/bespokelabsai/verifiers/blob/main/verifiers/berkeley-function-call-leaderboard/data/BFCL_v3_multi_turn_base.json). We have done some preprocessing to convert the dataset into the desired format. Formatting involves adding a system prompt and preprocessing to add metadata/initial environment configs in the required format. You can access the preprocessed data at this link: (insert Google Drive link). 

#revise this
**Example data format:**
```json
{
  "messages": [[{
    "role": "user", 
    "content": "Create a backup folder",
    "metadata": {
      "initial_config": {"GorillaFileSystem": {...}},
      "ground_truth": [["mkdir(dir_name='backup')"], ["ls()"]],
      "user_question_bank": [["Now list the contents"]],
      "involved_classes": ["GorillaFileSystem"],
      "num_turns": 2
    }
  }]],
  "task_name": "bfcl_multiturn",
  "dataset": "bfcl_v3"
}
```
`initial_config` helps with configurations to initialize the environment. The `_initialize_episode_metadata` function in [multi_turn_tool_environment.py](/home/slikhite/slikhite-RL/nemo_rl/environments/multi_turn_tool_environment.py) takes these configs to set up episodes for both ground truth and model execution. `user_question_bank` contains the subsequent user queries that get appended to the conversation. `ground_truth` is the ground truth trajectory of function calls that need to be executed. 

### Multi turn tool calling Environment

[multi_turn_tool_environment.py](../../nemo_rl/environments/multi_turn_tool_environment.py) contains the detailed implementation of environment.

#### 1. Initialization

The BFCL v3 environment initializes with two parallel episodes:
- **Ground Truth Episode**: Executes the ground truth tool trajectory
- **Model Episode**: Executes the model's tool calling trajectory

Both environments start with identical `initial_config` state and diverge based on actions taken.

#### 2. Supported Tools

We have implemented tool APIs from the verifiers repository. You can find the tools in this directory: [./nemo_rl/environments/tools](../../nemo_rl/environments/tools). This example supports GorillaFileSystem and other tools.

#### 3. Reward Structure

The system uses a dual-reward mechanism calculated on each turn:

- **State Score (0.5 weight)**: Compares final environment states by examining all tool attributes. Intuitively, it's meant to track the environment state after finishing a turn. It receives no reward if the states don't match.
- **Call Score (0.5 weight)**: Validates tool call sequences using set-based intersection. We also provide partial rewards in case it gets some of the tools right. 

```
Turn Reward = 0.5 × state_score + 0.5 × call_score
Total Episode Reward = Turn 1 + Turn 2 + Turn 3 + ... + Turn N
```



## Training Pipeline

The multi-turn training process using GRPO follows this detailed lifecycle:

### 1. **Data Loading & Initialization**
- First turn extracted from JSONL dataset with `JsonlinesDataset` class
- Each sample contains:
  - Initial user message with system prompt applied via chat template
  - `extra_env_info` containing metadata like `initial_config`, `ground_truth`, `user_question_bank`, etc.

### 2. **Multi-Turn Rollout Process**
The core training happens in `run_multi_turn_rollout()` from `nemo_rl/experience/rollouts.py`:

#### **2.1 Episode Initialization**
- `MultiTurnToolEnvironment._initialize_episode_metadata()` creates two parallel episodes:
  - **Model Episode**: Executes model's tool calling trajectory  
  - **Ground Truth Episode**: Executes the expected ground truth trajectory
- Both episodes initialize with identical `initial_config` state and diverge based on actions taken
- Tool instances are created for both episodes using `ToolManager.initialize_tools()`

#### **2.2 Tool Execution Process**
**Model Tool Execution:**
- Model generates responses containing tool calls in `<tool>` JSON format
- `ToolManager.parse_tool_calls()` extracts tool calls from `<tool>...</tool>` tags
- Each tool call validated for proper format: `{'name': 'function_name', 'args': {...}}`
- `ToolManager.execute_tool_call()` maps function names to tool instances and executes them
- Results formatted as `"[ToolName.function_name] result_string"`

**Ground Truth Execution:**
- GT trajectory from `metadata["ground_truth"][current_turn]` contains calls like `"cd(folder='document')"`
- `_execute_gt_call()` parses these strings and executes them on GT tool instances
- Both model and GT tools maintain separate state throughout the episode

#### **2.3 Observation Collection**
- After tool execution, environment observations are generated via `_get_next_observation()`
- Tool results are formatted as: `"<tool_result> [combined_results] </tool_result>"`
- Observations are tokenized and added to `message_log` as environment messages
- Token count tracking ensures conversations don't exceed `max_seq_len`
- If truncation needed, environment observations are shortened first

#### **2.4 Turn Success Determination**
Turn success (`turn_success`) is determined by multiple criteria in `_process_turn()`:

**Success Conditions (ALL must be true):**
- Model response contains `<tool>` tags
- Tool calls can be parsed successfully from JSON
- All parsed tool calls execute without exceptions
- No execution results contain "error" (case-insensitive)
- Arguments are properly formatted as dictionaries

**Failure Scenarios:**
- No `<tool>` tags found → "Function call not found"
- JSON parsing fails → "Invalid tool command. Parsing tool calls failed"
- Tool execution throws exception → Captured and marked as failure
- Result string contains "error" → Turn marked as failed
- Malformed arguments → Turn marked as failed

#### **2.5 Next Turn Determination**
**Continuation Logic:**
- `_should_continue()` checks: `current_turn <= max_turns - 1`
- If continuing AND previous turn was successful (`turn_success=True`):
  - `_append_next_user_question()` in `rollouts.py` adds next question from `user_question_bank[current_turn-1]`
  - Next user question tokenized and appended to message log
  - `current_turn` incremented in metadata

**Termination:**
- Episode ends when `max_turns` reached OR truncation occurs
- Final rewards calculated only on episode completion

### 3. **Reward Calculation**
**Per-Turn Reward Structure:**
```
Turn Reward = 0.5 × state_score + 0.5 × call_score
Total Episode Reward = (Turn 1 + Turn 2 + ... + Turn N) / max_turns
```

**State Score (0.5 weight):**
- Compares final environment states by examining all non-private tool attributes
- Only considers tools that the model actually invoked during the turn
- Returns proportion of matching tools: `matching_tools / total_used_tools`

**Call Score (0.5 weight):**
- Set-based intersection of model calls vs ground truth calls for the turn
- Full reward (1.0) if model calls exactly match GT calls
- Partial reward: `correct_calls / total_unique_calls` (penalizes extra incorrect calls)

### 4. **Policy Optimization**
- GRPO algorithm with leave-one-out baseline for variance reduction
- Rewards accumulated across all turns and used for policy gradient updates
- Per-turn metadata stored for analysis: `state_score`, `call_score`, `turn_success`, `tool_results`

### 5. **Validation**
- Periodic evaluation on held-out data using same multi-turn rollout process
- Greedy decoding used during validation for deterministic results


## Evaluation

Evaluation scripts assess model performance on:
- Turn-by-turn tool call accuracy
- Final environment state correctness  
- Multi-turn conversation coherence