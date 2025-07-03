"""Simulated user environment for counting unique numbers."""

from __future__ import annotations

import random
import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class UniqueNumbersConfig(TypedDict, total=False):
    """Configuration for :class:`UniqueNumbersEnv`."""

    min_length: int
    max_length: int
    max_turns: int


class UniqueNumbersMetadata(TypedDict):
    """Metadata for a UniqueNumbersEnv episode."""

    numbers: list[int]
    unique_count: int
    turn: int
    max_turns: int

PENALTY_FOR_NO_GUESS = -0.1
PENALTY_FOR_INCORRECT_GUESS = 0.0
PENALTY_FOR_EVERY_ASK = -0.01
PENALTY_FOR_INCORRECT_FORMAT = -0.02

class _UniqueNumbersRunner:
    query_re = re.compile(r"what is number (\d+)\??$", re.IGNORECASE)
    guess_re = re.compile(r"there are (\d+) unique numbers", re.IGNORECASE)

    def process_turn(
        self, message_log: LLMMessageLogType, metadata: UniqueNumbersMetadata
    ) -> tuple[dict[str, str], float, bool, None, Optional[UniqueNumbersMetadata]]:
        turn = metadata["turn"]
        max_turns = metadata["max_turns"]

        if turn >= max_turns:
            # Out of turns
            return {"role": "user", "content": "<done>"}, PENALTY_FOR_NO_GUESS, True, None, None

        last_msg = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_msg = message_log[-1]["content"].strip()

        if not last_msg:
            # no last message from assistant, assuming done
            return {"role": "user", "content": "<done>"}, PENALTY_FOR_NO_GUESS, True, None, None

        query_match = self.query_re.search(last_msg)
        if query_match:
            k = int(query_match.group(1))
            if 1 <= k <= len(metadata["numbers"]):
                content = str(metadata["numbers"][k - 1])
            else:
                content = f"Invalid index! There are {len(metadata['numbers'])} numbers."
            next_meta = {
                "numbers": metadata["numbers"],
                "unique_count": metadata["unique_count"],
                "turn": turn + 1,
                "max_turns": max_turns,
            }
            return {"role": "user", "content": content}, PENALTY_FOR_EVERY_ASK, False, None, next_meta

        guess_match = self.guess_re.search(last_msg)
        if guess_match:
            m = int(guess_match.group(1))
            reward = 1.0 if m == metadata["unique_count"] else PENALTY_FOR_INCORRECT_GUESS
            return {"role": "user", "content": "<done>"}, reward, True, None, None

        # default response
        next_meta = {
            "numbers": metadata["numbers"],
            "unique_count": metadata["unique_count"],
            "turn": turn + 1,
            "max_turns": max_turns,
        }
        help_msg = "Please ask 'what is number k?' or say 'there are m unique numbers'."
        return {"role": "user", "content": help_msg}, PENALTY_FOR_INCORRECT_FORMAT, False, None, next_meta


@ray.remote
class UniqueNumbersEnv(EnvironmentInterface):
    """Environment where the LLM must deduce the count of unique numbers."""

    def __init__(self, cfg: Optional[UniqueNumbersConfig] = None):
        cfg = cfg or {}
        self.min_length = cfg.get("min_length", 3)
        self.max_length = cfg.get("max_length", 7)
        self.default_max_turns = cfg.get("max_turns", 10)
        self.runner = _UniqueNumbersRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[Optional[UniqueNumbersMetadata]],
    ) -> EnvironmentReturn:
        results = []
        for log, meta in zip(message_log_batch, metadata_batch):
            assert meta is not None, "Metadata must not be None for UniqueNumbersEnv."
            assert meta["numbers"] is not None, "Numbers must not be None in metadata."
            assert meta["unique_count"] > 0, "Unique count must be greater than 0 in metadata."
            results.append(self.runner.process_turn(log, meta))

        observations, rewards, terminateds, stop_strings, next_metadata = [], [], [], [], []
        for obs, rew, term, stops, meta in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            stop_strings.append(stops)
            next_metadata.append(meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )

    def shutdown(self) -> None:  # pragma: no cover
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        avg_reward = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"unique_numbers_avg_reward": avg_reward}
