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

import random
import re
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class NumberGuessMetadata(TypedDict):
    target_number: int
    guesses_made: int
    max_guesses: int
    range_low: int
    range_high: int


class NumberGuessGameLogic:
    @staticmethod
    def generate(config: dict[str, Any]) -> dict[str, Any]:
        """Generate a new number guessing game."""
        range_low = config.get("range_low", 1)
        range_high = config.get("range_high", 50)
        target = random.randint(range_low, range_high)
        return {
            "target_number": target,
            "range_low": range_low,
            "range_high": range_high,
        }

    @staticmethod
    def step(guess: int, game_state: dict[str, Any]) -> tuple[str, float, bool]:
        """Process a guess. Returns (feedback, reward, is_terminated)."""
        target = game_state["target_number"]
        range_low = game_state["range_low"]
        range_high = game_state["range_high"]

        if guess < range_low or guess > range_high:
            return (
                f"Invalid guess. Please guess a number between {range_low} and {range_high}.",
                0.0,
                False,
            )

        if guess == target:
            return f"Correct! The number was {target}.", 1.0, True
        elif guess > target:
            return "Too high! Try a lower number.", 0.0, False
        else:
            return "Too low! Try a higher number.", 0.0, False


class NumberGuessRunner:
    def __init__(self):
        pass

    def _parse_guess(self, text: str) -> Optional[int]:
        """Parse a guess from <guess>NUMBER</guess> tags."""
        text_lower = text.lower()
        start_idx = text_lower.rfind("<guess>")
        if start_idx == -1:
            return None
        end_idx = text_lower.find("</guess>", start_idx + len("<guess>"))
        if end_idx == -1:
            # Tag was opened but not closed (stop_strings cuts at </guess>)
            content = text[start_idx + len("<guess>") :].strip()
        else:
            content = text[start_idx + len("<guess>") : end_idx].strip()

        # Extract the first integer from the content
        match = re.search(r"-?\d+", content)
        if match:
            return int(match.group())
        return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: NumberGuessMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[NumberGuessMetadata],
        Optional[list[str]],
    ]:
        """Process a single turn of the number guessing game."""
        target = metadata["target_number"]
        guesses_made = metadata["guesses_made"]
        max_guesses = metadata["max_guesses"]
        range_low = metadata["range_low"]
        range_high = metadata["range_high"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</guess>"]
        next_metadata: Optional[NumberGuessMetadata] = metadata.copy()

        # Check if max guesses already reached
        if guesses_made >= max_guesses:
            is_terminated = True
            # Partial reward: 0.3 if never guessed correctly but last state is close
            obs_content = f"Game over! You've used all {max_guesses} guesses. The number was {target}."
            return (
                {"role": "environment", "content": obs_content},
                0.0,
                True,
                None,
                None,
                None,
            )

        # Get last assistant message
        last_assistant_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_content = message_log[-1]["content"].strip()

        parsed_guess = self._parse_guess(last_assistant_content)

        if parsed_guess is None:
            obs_content = f"Invalid guess. Please guess a number between {range_low} and {range_high}."
            # Don't count invalid parses as a guess, but still terminate if needed
            next_metadata = None  # signal invalid format
            return (
                {"role": "environment", "content": obs_content + "\n"},
                0.0,
                False,
                next_stop_strings,
                None,
                None,
            )

        # Valid parse - run game logic
        game_state = {
            "target_number": target,
            "range_low": range_low,
            "range_high": range_high,
        }
        feedback, reward, game_over = NumberGuessGameLogic.step(
            parsed_guess, game_state
        )

        turn_reward = reward
        is_terminated = game_over
        new_guesses = guesses_made + 1

        if not game_over and new_guesses >= max_guesses:
            # Last guess used and not correct
            is_terminated = True
            # Partial reward if close
            distance = abs(parsed_guess - target)
            if distance <= 5:
                turn_reward = 0.3
            feedback += f" Game over! You've used all {max_guesses} guesses. The number was {target}."

        if is_terminated:
            next_metadata = None
            next_stop_strings = None
        else:
            assert next_metadata is not None
            next_metadata["guesses_made"] = new_guesses

        return (
            {"role": "environment", "content": feedback + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
            None,  # answers
        )


@ray.remote  # pragma: no cover
class NumberGuessEnv(EnvironmentInterface[NumberGuessMetadata]):
    """Number Guessing Game environment (Ray Actor)."""

    def __init__(self, cfg: Optional[dict] = None):
        self.config = cfg or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[NumberGuessMetadata],
    ) -> EnvironmentReturn[NumberGuessMetadata]:
        """Process a batch of number guessing interactions."""
        runner = NumberGuessRunner()
        results = [
            runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata)
        ]

        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []
        all_answers = []

        for obs, rew, term, stops, meta, answ in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)
            all_answers.append(answ)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards >= 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        partial_rate = (
            ((final_rewards >= 0.3) & (final_rewards < 1.0)).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {
            "number_guess_success_rate": success_rate,
            "number_guess_partial_rate": partial_rate,
        }
