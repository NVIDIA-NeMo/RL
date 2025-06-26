# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class SlidingPuzzleConfig(TypedDict):
    size: int
    shuffle_moves: int


class SlidingPuzzleMetadata(TypedDict):
    game_state: dict[str, Any]  # Stores the dict returned by SlidingPuzzleGame methods
    num_moves: int
    max_moves: int


class SlidingPuzzleGameLogic:
    @staticmethod
    def generate(config: dict[str, Any]) -> dict[str, Any]:
        """Generate a new Sliding Puzzle."""
        size = config.get("size", 4)  # Default to 4x4 (15-puzzle)
        shuffle_moves = config.get(
            "shuffle_moves", 100
        )  # Number of random moves for shuffling

        # Create the solved state
        grid = [[(r * size + c + 1) for c in range(size)] for r in range(size)]
        # Set the bottom-right corner to 0 (empty space)
        grid[size - 1][size - 1] = 0

        # Save the solution
        solution = [row[:] for row in grid]

        # Find the empty space
        empty_pos = (size - 1, size - 1)

        # Shuffle the grid with valid moves
        for _ in range(shuffle_moves):
            # Get possible moves
            moves = []
            r, c = empty_pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    moves.append((nr, nc))

            # Choose a random move
            if moves:
                new_r, new_c = random.choice(moves)
                # Swap the empty space with the chosen tile
                grid[r][c], grid[new_r][new_c] = grid[new_r][new_c], grid[r][c]
                empty_pos = (new_r, new_c)

        # Create and return the game state
        return {
            "size": size,
            "grid": grid,
            "solution": solution,
            "empty_pos": empty_pos,
            "commands": {
                "up": "Slide tile below empty space up",
                "down": "Slide tile above empty space down",
                "left": "Slide tile to the right of empty space left",
                "right": "Slide tile to the left of empty space right",
                "view": "View the current state of the board",
            },
        }

    @staticmethod
    def init(game_state: dict[str, Any]) -> str:
        """Initialize Sliding Puzzle game and return welcome message."""
        size = game_state["size"]

        return (
            f"\n===== SLIDING PUZZLE =====\n"
            f"Arrange the {size}x{size} grid by sliding tiles into the empty space.\n"
            f"- The goal is to arrange numbers from 1 to {size * size - 1} in order\n"
            f"- Use 'up', 'down', 'left', 'right' to slide in that direction\n"
            f"- Use 'view' to see the current state of the board"
        )

    @staticmethod
    def step(
        action: str, game_state: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Process an action in the Sliding Puzzle game."""
        size = game_state["size"]
        grid = game_state["grid"]
        empty_r, empty_c = game_state["empty_pos"]

        # Default return values
        response = "Unknown command. Type 'help' to see available commands."
        reward = 0.0  # No penalty for invalid actions
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)

        move_made = False

        if action.startswith("slide "):
            try:
                _, r, c = action.split()
                r, c = int(r) - 1, int(c) - 1

                # Validate input
                if not (0 <= r < size and 0 <= c < size):
                    return (
                        f"Invalid position. Row/column must be between 1 and {size}.",
                        reward,
                        is_terminated,
                        new_state,
                    )

                # Check if tile is adjacent to empty space
                if abs(r - empty_r) + abs(c - empty_c) != 1:
                    return (
                        "Tile must be adjacent to the empty space.",
                        reward,
                        is_terminated,
                        new_state,
                    )

                # Slide the tile
                new_state["grid"][empty_r][empty_c] = grid[r][c]
                new_state["grid"][r][c] = 0
                new_state["empty_pos"] = (r, c)

                move_made = True
                response = f"Slid tile {grid[r][c]} into the empty space."

            except ValueError:
                return (
                    "Invalid input format. Use: slide row col",
                    reward,
                    is_terminated,
                    new_state,
                )

        elif action in ["up", "down", "left", "right"]:
            # Convert direction to row/col offset
            if action == "up":
                r, c = empty_r + 1, empty_c  # Tile below moves up
                dir_text = "up"
            elif action == "down":
                r, c = empty_r - 1, empty_c  # Tile above moves down
                dir_text = "down"
            elif action == "left":
                r, c = empty_r, empty_c + 1  # Tile to right moves left
                dir_text = "left"
            elif action == "right":
                r, c = empty_r, empty_c - 1  # Tile to left moves right
                dir_text = "right"

            # Check if the move is valid
            if 0 <= r < size and 0 <= c < size:
                # Slide the tile
                new_state["grid"][empty_r][empty_c] = grid[r][c]
                new_state["grid"][r][c] = 0
                new_state["empty_pos"] = (r, c)

                move_made = True
                response = f"Slid tile {grid[r][c]} {dir_text}."
            else:
                return f"Cannot slide {dir_text}.", reward, is_terminated, new_state

        if move_made:
            reward = 0

            # Check if puzzle is solved
            if new_state["grid"] == new_state["solution"]:
                response = "Congratulations! You've solved the puzzle!"
                reward = 1.0  # Win reward
                is_terminated = True

        return response, reward, is_terminated, new_state

    @staticmethod
    def render(game_state: dict[str, Any]) -> str:
        """Render the current Sliding Puzzle game state."""
        grid = game_state["grid"]
        size = game_state["size"]

        output = ["\n"]

        # Create a visual representation of the grid
        max_digits = len(str(size * size - 1))

        # Top border
        output.append("  " + "+" + "-" * (max_digits + 2) * size + "+")

        # Rows
        for i, row in enumerate(grid):
            row_str = f"{i + 1} |"
            for val in row:
                if val == 0:
                    # Empty space
                    row_str += " " * (max_digits + 2)
                else:
                    # Tile with number
                    row_str += f" {val:>{max_digits}} "
            row_str += "|"
            output.append(row_str)

        # Bottom border
        output.append("  " + "+" + "-" * (max_digits + 2) * size + "+")

        # Column labels
        col_labels = "    "
        for i in range(size):
            col_labels += f"{i + 1:^{max_digits + 2}}"
        output.append(col_labels)

        return "\n".join(output)


class SlidingPuzzleRunner:
    def __init__(self):
        pass  # No initialization needed as game methods are static

    def _parse_action(self, text: str) -> Optional[str]:
        """Parses the action from '<action></action>'."""
        prefix = "<action>"
        suffix = "</action>"
        # Find the prefix, case-insensitive, and potentially after some thought process
        text_lower = text.lower()
        prefix_lower = prefix.lower()
        suffix_lower = suffix.lower()

        start_idx = text_lower.rfind(prefix_lower)  # Find the last occurrence

        if start_idx != -1:
            # Find the end tag after the start tag
            end_idx = text_lower.find(suffix_lower, start_idx + len(prefix_lower))
            if end_idx != -1:
                # Extract content between tags
                action_content = text[start_idx + len(prefix) : end_idx].strip()
                return action_content
        return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: SlidingPuzzleMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[SlidingPuzzleMetadata],
    ]:
        """Processes a single turn for the sliding puzzle task."""
        game_state = metadata["game_state"]
        current_moves = metadata["num_moves"]
        max_moves = metadata["max_moves"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</action>"]
        next_metadata = metadata.copy()
        next_observation_content = ""

        # Check if max moves reached
        if current_moves >= max_moves:
            is_terminated = True
            next_observation_content = (
                f"<error>Maximum moves ({max_moves}) reached.</error>"
            )
            next_metadata = None
            return (
                {"role": "environment", "content": next_observation_content},
                0.0,
                is_terminated,
                None,
                next_metadata,
            )

        # Get last assistant message and parse action
        last_assistant_msg_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg_content = message_log[-1]["content"].strip()

        parsed_action = self._parse_action(last_assistant_msg_content)

        if parsed_action is None:
            rendered_board = SlidingPuzzleGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_board}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_board = SlidingPuzzleGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_board}\n\nViewing the board. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = (
                SlidingPuzzleGameLogic.step(parsed_action, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            next_observation_content = f"<environment>\n{step_response}\n</environment>"

            if is_terminated:
                next_metadata = None  # Clear metadata on termination

        return (
            {"role": "environment", "content": next_observation_content + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
        )


@ray.remote
class SlidingPuzzleEnv(EnvironmentInterface):
    """Sliding Puzzle environment (Ray Actor)."""

    def __init__(self, cfg: Optional[SlidingPuzzleConfig] = None):
        # cfg could contain game generation config like {'size': 3, 'shuffle_moves': 50}
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.runner = SlidingPuzzleRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[SlidingPuzzleMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of sliding puzzle interactions."""
        # Since logic is synchronous, process sequentially (can parallelize if logic becomes heavy)
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]

        # Unpack results and format according to EnvironmentReturn NamedTuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # Calculate success rate based on final reward == 1.0
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        # Could also calculate average number of moves for successful episodes, etc.
        return batch, {"sliding_puzzle_success_rate": success_rate}
