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
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

WORDS = [
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
    target: str
    guesses_remaining: int
    max_guesses: int
    last_guess: Optional[str]


def _extract_guess(response: str) -> Optional[str]:
    """Extract a word from <guess>WORD</guess> tags."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    match = re.search(r"<guess>\s*([a-zA-Z]+)\s*</guess>", text)
    if match:
        return match.group(1).lower()
    return None


def _compute_feedback(target: str, guess: str) -> str:
    """Compute Wordle-style feedback: G=correct, Y=wrong position, X=not in word."""
    feedback: list[Optional[str]] = [None] * 5
    target_chars: list[Optional[str]] = list(target)
    guess_chars: list[Optional[str]] = list(guess)

    # First pass: mark greens
    for i in range(5):
        if guess_chars[i] == target_chars[i]:
            feedback[i] = "G"
            target_chars[i] = None
            guess_chars[i] = None

    # Second pass: mark yellows and grays
    for i in range(5):
        if feedback[i] is not None:
            continue
        if guess_chars[i] is not None and guess_chars[i] in target_chars:
            feedback[i] = "Y"
            target_chars[target_chars.index(guess_chars[i])] = None
        else:
            feedback[i] = "X"

    return "".join(f for f in feedback if f is not None)


@ray.remote  # pragma: no cover
class WordleEnv(EnvironmentInterface[WordleMetadata]):
    """Wordle environment (Ray Actor).

    Multi-turn: the model guesses a secret 5-letter word.
    Gets letter-by-letter feedback (G/Y/X) for up to 6 turns.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[WordleMetadata],
    ) -> EnvironmentReturn[WordleMetadata]:
        rewards = []
        observations = []
        terminateds = []
        new_metadata = []
        answers = []
        next_stop_strings: list[Optional[list[str]]] = []

        for message_log, meta in zip(message_log_batch, metadata):
            response = message_log[-1]["content"]
            target = meta["target"]
            guesses_remaining = meta["guesses_remaining"] - 1

            guess = _extract_guess(response)

            if guess is None or len(guess) != 5:
                obs_content = (
                    f"Invalid guess. Please use <guess>WORD</guess> format "
                    f"with a 5-letter word. You have {guesses_remaining} guesses remaining."
                )
                reward = 0.0
                done = guesses_remaining <= 0
                answer = None
            elif guess == target:
                turns_used = meta["max_guesses"] - guesses_remaining
                obs_content = f"Correct! The word was {target}. You got it in {turns_used} guesses!"
                reward = 1.0
                done = True
                answer = guess
            elif guesses_remaining <= 0:
                feedback = _compute_feedback(target, guess)
                green_count = feedback.count("G")
                reward = 0.1 * green_count
                obs_content = (
                    f"Feedback: {feedback}\n"
                    f"Out of guesses! The word was {target}. Your last guess was {guess}."
                )
                done = True
                answer = guess
            else:
                feedback = _compute_feedback(target, guess)
                obs_content = (
                    f"Feedback: {feedback}\n"
                    f"Guess {meta['max_guesses'] - guesses_remaining}/{meta['max_guesses']}. Try again."
                )
                reward = 0.0
                done = False
                answer = None

            rewards.append(reward)
            terminateds.append(done)
            observations.append(
                {
                    "role": "environment",
                    "content": f"\n<feedback>{obs_content}</feedback>\n",
                }
            )
            answers.append(answer)
            next_stop_strings.append(None if done else ["</guess>"])

            new_meta: WordleMetadata = {
                "target": target,
                "guesses_remaining": guesses_remaining,
                "max_guesses": meta["max_guesses"],
                "last_guess": guess,
            }
            new_metadata.append(new_meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, Any]]:
        final_rewards = batch.get("total_reward", torch.tensor([0.0]))
        accuracy = (
            (final_rewards >= 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        mean_reward = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"accuracy": accuracy, "mean_reward": mean_reward}
