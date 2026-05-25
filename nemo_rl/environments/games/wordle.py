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

import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

WORD_LIST = [
    "apple",
    "brain",
    "chair",
    "dance",
    "earth",
    "flame",
    "grape",
    "house",
    "image",
    "juice",
    "knife",
    "lemon",
    "music",
    "night",
    "ocean",
    "paint",
    "queen",
    "river",
    "stone",
    "truck",
]


class WordleMetadata(TypedDict):
    target_word: str
    guesses_remaining: int
    max_guesses: int
    previous_guesses: list[str]
    previous_feedback: list[str]


def _compute_feedback(guess: str, target: str) -> tuple[str, int, int]:
    """Compute Wordle feedback for a guess.

    Returns (feedback_string, num_green, num_yellow).
    feedback_string uses: [G]=green, [Y]=yellow, [_]=gray
    """
    guess = guess.lower()
    target = target.lower()

    result = ["_"] * 5
    target_chars = list(target)
    num_green = 0
    num_yellow = 0

    # First pass: find greens
    for i in range(5):
        if guess[i] == target_chars[i]:
            result[i] = "G"
            target_chars[i] = None  # Mark as used
            num_green += 1

    # Second pass: find yellows
    for i in range(5):
        if result[i] != "G" and guess[i] in target_chars:
            result[i] = "Y"
            target_chars[target_chars.index(guess[i])] = None
            num_yellow += 1

    feedback = " ".join(f"[{r}]{guess[i].upper()}" for i, r in enumerate(result))
    return feedback, num_green, num_yellow


def _compute_reward(num_green: int, num_yellow: int, is_correct: bool) -> float:
    """Compute partial reward based on green and yellow letters."""
    if is_correct:
        return 1.0
    return min(1.0, num_green * 0.2 + num_yellow * 0.1)


def _extract_guess(response: str) -> Optional[str]:
    """Extract word from <guess>WORD</guess> tags."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    match = re.search(r"<guess>\s*(\w+)\s*</guess>", text, re.IGNORECASE)
    if match:
        word = match.group(1).strip().lower()
        if len(word) == 5 and word.isalpha():
            return word
    return None


class WordleRunner:
    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: WordleMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[WordleMetadata],
        Optional[list[str]],
    ]:
        """Processes a single turn for the Wordle task."""
        target_word = metadata["target_word"]
        guesses_remaining = metadata["guesses_remaining"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</guess>"]
        next_metadata = metadata.copy()
        next_metadata["previous_guesses"] = list(metadata["previous_guesses"])
        next_metadata["previous_feedback"] = list(metadata["previous_feedback"])

        # Get last assistant message and parse guess
        last_assistant_msg = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg = message_log[-1]["content"].strip()

        guess = _extract_guess(last_assistant_msg)

        if guess is None:
            # Invalid format - give feedback but don't count as a guess
            observation_content = (
                "Invalid format. Please guess a 5-letter word using "
                "<guess>WORD</guess> tags."
            )
            return (
                {"role": "user", "content": observation_content},
                0.0,
                False,
                next_stop_strings,
                next_metadata,
                None,
            )

        # Valid guess - compute feedback
        is_correct = guess == target_word
        feedback, num_green, num_yellow = _compute_feedback(guess, target_word)
        turn_reward = _compute_reward(num_green, num_yellow, is_correct)

        next_metadata["guesses_remaining"] = guesses_remaining - 1
        next_metadata["previous_guesses"].append(guess)
        next_metadata["previous_feedback"].append(feedback)

        if is_correct:
            observation_content = (
                f"Guess: {guess.upper()}\n"
                f"Result: {feedback}\n\n"
                f"Correct! The word was {target_word.upper()}!"
            )
            is_terminated = True
            next_metadata = None
        elif guesses_remaining - 1 <= 0:
            observation_content = (
                f"Guess: {guess.upper()}\n"
                f"Result: {feedback}\n\n"
                f"Out of guesses! The word was {target_word.upper()}."
            )
            is_terminated = True
            next_metadata = None
        else:
            observation_content = (
                f"Guess: {guess.upper()}\n"
                f"Result: {feedback}\n"
                f"Guesses remaining: {guesses_remaining - 1}"
            )

        answers = [guess] if is_terminated else None
        return (
            {"role": "user", "content": observation_content},
            turn_reward,
            is_terminated,
            next_stop_strings if not is_terminated else None,
            next_metadata,
            answers,
        )


@ray.remote  # pragma: no cover
class WordleEnv(EnvironmentInterface[WordleMetadata]):
    """Wordle environment (Ray Actor)."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.runner = WordleRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[WordleMetadata],
    ) -> EnvironmentReturn[WordleMetadata]:
        """Processes a batch of Wordle interactions."""
        results = [
            self.runner.process_turn(log, meta)
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
        exact_match_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"wordle_exact_match_rate": exact_match_rate}
