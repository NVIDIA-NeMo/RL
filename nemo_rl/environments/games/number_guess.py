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


class NumberGuessMetadata(TypedDict):
    target: int
    guesses_remaining: int
    max_guesses: int
    last_guess: Optional[int]


def _extract_guess(response: str) -> Optional[int]:
    """Extract a number from <guess>N</guess> tags."""
    # Strip think blocks
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    match = re.search(r"<guess>\s*(\d+)\s*</guess>", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


@ray.remote  # pragma: no cover
class NumberGuessEnv(EnvironmentInterface[NumberGuessMetadata]):
    """Number Guessing Game environment (Ray Actor).

    Multi-turn: the model guesses a secret number 1-50.
    Gets "too high"/"too low" feedback for up to 7 turns.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[NumberGuessMetadata],
    ) -> EnvironmentReturn[NumberGuessMetadata]:
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

            if guess is None:
                obs_content = (
                    f"I couldn't understand your guess. Please use <guess>N</guess> format "
                    f"where N is a number between 1 and 50. You have {guesses_remaining} guesses remaining."
                )
                reward = 0.0
                done = guesses_remaining <= 0
                answer = None
            elif guess == target:
                turns_used = meta["max_guesses"] - guesses_remaining
                obs_content = f"Correct! The number was {target}. You got it in {turns_used} guesses!"
                reward = 1.0
                done = True
                answer = str(guess)
            elif guesses_remaining <= 0:
                diff = abs(guess - target)
                reward = 0.3 if diff <= 5 else 0.0
                obs_content = f"Out of guesses! The number was {target}. Your last guess was {guess}."
                done = True
                answer = str(guess)
            else:
                if guess < target:
                    obs_content = f"Too low! Try a higher number. You have {guesses_remaining} guesses remaining."
                else:
                    obs_content = f"Too high! Try a lower number. You have {guesses_remaining} guesses remaining."
                reward = 0.0
                done = False
                answer = None

            rewards.append(reward)
            terminateds.append(done)
            observations.append(
                {"role": "environment", "content": f"\n<feedback>{obs_content}</feedback>\n"}
            )
            answers.append(answer)
            next_stop_strings.append(None if done else ["</guess>"])

            new_meta: NumberGuessMetadata = {
                "target": target,
                "guesses_remaining": guesses_remaining,
                "max_guesses": meta["max_guesses"],
                "last_guess": guess,
            }
            new_metadata.append(new_meta)

        batch_size = len(message_log_batch)
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
