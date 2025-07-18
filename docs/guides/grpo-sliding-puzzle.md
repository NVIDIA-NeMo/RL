# Sliding Puzzle using GRPO

This guide explains how to use Nemo-RL to train a model to solve the sliding puzzle game through multi-turn interactions. This environment implements a classic **n×n sliding puzzle** where numbered tiles must be arranged in sequential order by sliding them into an empty space.

This example serves as a basic example to understand how multi-turn RL works, with a fundamental implementation of tool-calling. 

## Table of Contents

1. [Quick Start](#quick-startW) 
2. [Game Mechanics](#game-mechanics)
3. [Data Generation](#data-generation)
4. [Environment Interface](#environment-interface)
5. [Reward System](#reward-system)\
6. [Results](#results-and-performance) 

## Quick Start Guide

#### 1. Installation
```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL

git submodule update --init --recursive

pip install uv

uv venv
```

Refer to [this guide](https://docs.nvidia.com/nemo/rl/latest/index.html#prerequisites) for detailed instructions on installation.

#### 2. Training Run
```bash
# Run with default 2×2 puzzle configuration
uv run python examples/run_grpo_sliding_puzzle.py 
```

#### 3. Custom Configuration

By default, this uses the configuration in [grpo_sliding_puzzle.yaml](../../examples/configs/grpo_sliding_puzzle.yaml). You can customize parameters with command-line overrides.
```bash
# Train on 3×3 puzzles with more scrambling
python examples/run_grpo_sliding_puzzle.py \
    env.sliding_puzzle_game.cfg.game_config.size=3 \
    env.sliding_puzzle_game.cfg.game_config.shuffle_moves=10
```

#### 4. Monitor Progress
```bash
# Enable logging (optional)
python examples/run_grpo_sliding_puzzle.py \
    --config examples/configs/grpo_sliding_puzzle.yaml \
    logger.wandb_enabled=true \
    logger.tensorboard_enabled=true
```

## Game Mechanics

### Puzzle Structure

The sliding puzzle consists of:
- **Grid**: An n×n grid containing numbered tiles and one empty space
- **Tiles**: Numbered from 1 to n²-1 in sequential order
- **Empty Space**: Represented by 0, initially positioned at bottom-right corner
- **Goal State**: Sequential arrangement (1, 2, 3, ..., n²-1) with empty space at bottom-right

### Example data sample:
```
===== SLIDING PUZZLE =====
Arrange the 3x3 grid by sliding tiles into the empty space.
- The goal is to arrange numbers from 1 to 8 in order
- Use 'up', 'down', 'left', 'right' to slide in that direction
- Use 'view' to see the current state of the board

Current Board State:

  +---------+
1 | 1     3 |
2 | 4  2  5 |
3 | 7  8  6 |
  +---------+
     1  2  3 

Reach the goal state where numbers are ordered 1 through 8 with the empty space (0) at the bottom right.
Valid actions: 'up', 'down', 'left', 'right', or 'slide row col' (e.g., 'slide 1 2').
After thinking, output your chosen action on a new line starting with '<action></action>' like this:
<action>your_action</action>
If you just want to see the board, output <action>view</action>
Think carefully step-by-step before acting.

```

### Movement Rules

1. **Valid Moves**: Only tiles adjacent to the empty space can be moved
2. **Movement Direction**: Tiles slide into the empty space
3. **Grid Boundaries**: Moves cannot exceed grid boundaries
4. **Single Tile Movement**: Only one tile can move per action

All actions must be wrapped in XML tags:
```xml
<action>up</action>
<action>slide 2 1</action>
<action>view</action>
```

## Data Generation

### Configuration Parameters

The puzzle generation system uses the following parameters:

```yaml
env:
  sliding_puzzle_game:
    cfg:
      game_config:
        size: 5           
        shuffle_moves: 4     # Number of scrambling moves
      max_moves: 40          # Maximum moves allowed per episode
```

Grids are generated with sizes ranging from 2 to game_config.size. Each grid starts with a solved state and is shuffled by moving random tiles to the empty space n times, where n is a random number between 1 and shuffle_moves. The grid is shuffled using only valid moves. 
The `generate_puzzle_datum()` function in [run_grpo_sliding_puzzle.py](../../examples/run_grpo_sliding_puzzle.py) is responsible for generating the dataset. [sliding_puzzle.py](../../nemo_rl/environments/games/sliding_puzzle.py) contains the `SlidingPuzzleGameLogic` class, responsible for puzzle generation and initialization logic. The number of shuffle moves and size of the grid will control puzzle difficulty.

#### Generation Algorithm

```python
def generate_random_config(max_config: dict[str, Any]) -> dict[str, Any]:
    """Generate a random config for the sliding puzzle game."""
    shuffle_moves = random.randint(1, max_config.get("shuffle_moves"))
    if shuffle_moves % 2 == 0:
        shuffle_moves += 1  # Ensure odd number for proper scrambling
    return {
        "size": random.randint(2, max_config.get("size", 3)),
        "shuffle_moves": shuffle_moves,
    }

      game_config = generate_random_config(game_config)
      initial_game_state = SlidingPuzzleGameLogic.generate(game_config)
      initial_render = SlidingPuzzleGameLogic.render(initial_game_state)
      welcome_message = SlidingPuzzleGameLogic.init(initial_game_state)
  ```

### Dataset Size Calculation

Dataset sizes are defined based on the values in grpo_sliding_puzzle.yaml
```
Training Size = num_prompts_per_step × num_generations_per_prompt × max_num_steps
Validation Size = max_val_samples (defined in config yaml)
```

### Data Structure

Each generated datum returns a `DatumSpec`:

```python
datum: DatumSpec = {
    "message_log": message_log,              # Conversation history
    "length": len(tokenized_prompt),         # Token count
    "extra_env_info": metadata,              # Game state metadata
    "loss_multiplier": 1.0,                  # Training weight
    "idx": idx,                              # Sample index
    "task_name": task_name,                  # Task identifier
    "stop_strings": ["</action>"],           # Termination tokens
}
```

## Environment Interface

<!-- ### Architecture Flow

```
GRPO Training Pipeline:
run_grpo_sliding_puzzle.grpo_train → nemo_rl.experience.rollouts.run_multi_turn_rollouts → generate_response + calculate_reward → environments.games.sliding_puzzle.SlidingPuzzleEnv.step
``` -->

### Core Classes

[sliding_puzzle.py](../../nemo_rl/environments/games/sliding_puzzle.py) defines the environment, and the logic for interactingwith the environment. The core classes used are outlined below:

#### SlidingPuzzleEnv
Main environment class implementing Ray remote actor for distributed processing. This class uses functions from SlidingPuzzleGameLogic and SlidingPuzzleRunner class to interact with the environment.

```python
@ray.remote
class SlidingPuzzleEnv(EnvironmentInterface):
    def __init__(self, cfg: Optional[SlidingPuzzleConfig] = None):
        """Initialize environment with configuration."""
        
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[SlidingPuzzleMetadata],
    ) -> EnvironmentReturn:
        """Process batch of interactions."""
```

#### SlidingPuzzleGameLogic
Core game mechanics with static methods for puzzle operations. Includes reward calculation.

```python
class SlidingPuzzleGameLogic:
    @staticmethod
    def generate(config: dict[str, Any]) -> dict[str, Any]:
        """Generate new puzzle with specified configuration."""
        
    @staticmethod
    def init(game_state: dict[str, Any]) -> str:
        """Create welcome message with game rules."""
        
    @staticmethod
    def step(action: str, game_state: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute action and return (response, reward, terminated, new_state)."""
        
    @staticmethod
    def render(game_state: dict[str, Any]) -> str:
        """Render current puzzle state as visual grid."""
```

#### SlidingPuzzleRunner

Handles turn processing and action management:

```python
class SlidingPuzzleRunner:
    def __init__(self):
        """Initialize runner with no persistent state."""
        
    def _parse_action(self, text: str) -> Optional[str]:
        """Extract action from model response using XML tag parsing."""
        
    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: SlidingPuzzleMetadata,
    ) -> tuple[dict[str, str], float, bool, Optional[list[str]], Optional[SlidingPuzzleMetadata]]:
        """Process single turn and return (response_dict, reward, terminated, stop_strings, updated_metadata)."""
```

### Processing Pipeline

The step function creates a processing pipeline where each class handles specific responsibilities:

1. **Process Turn and parse action** (SlidingPuzzleRunner): Extract action from model response using XML tag parsing via `process_turn` method
2. **Validate Move** (SlidingPuzzleGameLogic): Check if action is valid for current game state and execute the move
3. **Execute Action** (SlidingPuzzleGameLogic): Apply move to game state using `SlidingPuzzleGameLogic.step` method
4. **Calculate Reward** (SlidingPuzzleGameLogic): Determine reward based on puzzle completion status (step function)
6. **Return Results** (SlidingPuzzleEnv): Package response as `EnvironmentReturn` object for training pipeline

## Reward System

### Reward Structure

The environment implements a sparse reward system focusing on task completion, i.e encourages learning of complete solution strategies

| Condition | Reward | Termination |
|-----------|--------|-------------|
| Valid move (non-solving) | 0.0 | False |
| Invalid move | 0.0 | False |
| Puzzle solved | 1.0 | True |
| Max moves reached | 0.0 | True |
| Invalid action format | 0.0 | False |

### Reward Calculation Logic

```python
def step(action: str, game_state: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
    """Process action and calculate reward."""
    reward = 0.0
    is_terminated = False
    
    if move_made:
        # Check if puzzle is solved
        if new_state["grid"] == new_state["solution"]:
            reward = 1.0
            is_terminated = True
        else:
            reward = 0.0  # No reward for non-solving moves
    
    return response, reward, is_terminated, new_state
```
## Results

