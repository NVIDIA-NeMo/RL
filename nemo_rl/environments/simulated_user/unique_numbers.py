"""Simulated user environment for counting unique numbers."""

from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.simulated_user.prompt import (
    grader_instruction,
    simulated_user_instruction,
)

PENALTY_FOR_NO_GUESS = -0.2
PENALTY_FOR_INCORRECT_GUESS = 0.0
PENALTY_FOR_EVERY_ASK = 0.0
PENALTY_FOR_INCORRECT_FORMAT = 0.0

ADK_LOG_FOLDER = None  # 'logs/adk' or None to disable logging
if ADK_LOG_FOLDER is not None:
    os.makedirs(ADK_LOG_FOLDER, exist_ok=True)

GEMINI_CALL_MAX_WORKERS = (
    64  # if 1 then it will be single-threaded, otherwise it will use ThreadPoolExecutor
)


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
    simulated_user_runner: Optional[Any]
    grader_runner: Optional[Any]


class _UniqueNumbersRunner:
    query_re = re.compile(r"what is number (\d+)\??$", re.IGNORECASE)
    guess_re = re.compile(r"there are (\d+) unique number", re.IGNORECASE)

    def _maybe_dump_adk_messages_to_file(
        self,
        runner: Any,
        user_id: str,
        log_name_suffix: str = "",
        dump_folder: Optional[str] = ADK_LOG_FOLDER,
    ):
        from nemo_rl.environments.simulated_user.adk_utils import (
            extract_conversation_history,
        )

        if dump_folder is None:
            return
        session_id, messages = extract_conversation_history(
            runner, user_id, silence=True
        )
        file_name = f"{user_id}_{session_id}{log_name_suffix}.log"
        with open(os.path.join(dump_folder, file_name), "a") as f:
            for message in messages:
                f.write(f"[{message['role']}]:|||{message['content']}|||\n")

    def process_turn(
        self, message_log: LLMMessageLogType, metadata: UniqueNumbersMetadata
    ) -> tuple[dict[str, str], float, bool, None, Optional[UniqueNumbersMetadata]]:
        from nemo_rl.environments.simulated_user.adk_utils import (
            create_agent,
            run_prompt_async,
            setup_runner_async,
        )

        turn = metadata["turn"]
        max_turns = metadata["max_turns"]

        if (
            "simulated_user_runner" not in metadata
            or metadata["simulated_user_runner"] is None
        ):
            instruction = simulated_user_instruction.replace(
                "{numbers}", str(metadata["numbers"])
            )
            simulated_user_agent = create_agent(
                name="simulated_user", model="gemini-2.0-flash", instruction=instruction
            )
            metadata["simulated_user_runner"] = asyncio.run(
                setup_runner_async(
                    simulated_user_agent, "simulated_user_app", "simulated_user"
                )
            )

        if "grader_runner" not in metadata or metadata["grader_runner"] is None:
            instruction = grader_instruction.replace(
                "{numbers}", str(metadata["numbers"])
            )
            grader_agent = create_agent(
                name="grader", model="gemini-2.0-flash", instruction=instruction
            )
            metadata["grader_runner"] = asyncio.run(
                setup_runner_async(grader_agent, "grader_app", "grader")
            )

        if turn >= max_turns:
            self._maybe_dump_adk_messages_to_file(
                metadata["simulated_user_runner"], "simulated_user", "_maxturns"
            )
            return (
                {"role": "user", "content": "<done>"},
                PENALTY_FOR_NO_GUESS,
                True,
                None,
                None,
            )

        last_msg = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_msg = message_log[-1]["content"].strip()

        if not last_msg:
            # no last message from assistant, assuming done
            return (
                {"role": "user", "content": "<done>"},
                PENALTY_FOR_NO_GUESS,
                True,
                None,
                None,
            )

        # simulate user utterance via ADK
        query_match = self.query_re.search(last_msg)
        if query_match:
            simulated_content = asyncio.run(
                run_prompt_async(
                    metadata["simulated_user_runner"],
                    "simulated_user",
                    last_msg,
                    silence=True,
                )
            )
            next_meta: UniqueNumbersMetadata = {
                "numbers": metadata["numbers"],
                "unique_count": metadata["unique_count"],
                "turn": turn + 1,
                "max_turns": max_turns,
                "simulated_user_runner": metadata.get("simulated_user_runner", None),
                "grader_runner": metadata.get("grader_runner", None),
            }
            return (
                {"role": "user", "content": simulated_content},
                PENALTY_FOR_EVERY_ASK,
                False,
                None,
                next_meta,
            )

        # calculate reward if the assistant made a guess
        guess_match = self.guess_re.search(last_msg)
        if guess_match:
            m = int(guess_match.group(1))
            reward = (
                1.0 if m == metadata["unique_count"] else PENALTY_FOR_INCORRECT_GUESS
            )

            # grade the conversation via ADK grader
            if metadata["grader_runner"] is not None:
                convo_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in message_log]
                )
                grading_prompt = f"Here is the converstation \n{convo_str}\nAnd please give the score between 0 and 1."
                grading_response = asyncio.run(
                    run_prompt_async(
                        metadata["grader_runner"],
                        "grader",
                        grading_prompt,
                        silence=True,
                    )
                )
                try:
                    grade = int(re.search(r"(\d+)", grading_response).group(1))
                    reward = (reward + grade) / 2.0
                except Exception as e:
                    print(
                        f"Failed to parse grade from grader response '{grading_response}': {e}"
                    )

            self._maybe_dump_adk_messages_to_file(
                metadata["simulated_user_runner"], "simulated_user", "_stop"
            )
            self._maybe_dump_adk_messages_to_file(metadata["grader_runner"], "grader")

            return {"role": "user", "content": "<done>"}, reward, True, None, None

        # default response
        next_meta: UniqueNumbersMetadata = {
            "numbers": metadata["numbers"],
            "unique_count": metadata["unique_count"],
            "turn": turn + 1,
            "max_turns": max_turns,
            "simulated_user_runner": metadata.get("simulated_user_runner", None),
            "grader_runner": metadata.get("grader_runner", None),
        }
        help_msg = "Please ask 'what is number k?' or say 'there are m unique numbers'."
        return (
            {"role": "user", "content": help_msg},
            PENALTY_FOR_INCORRECT_FORMAT,
            False,
            None,
            next_meta,
        )


@ray.remote
class UniqueNumbersEnv(EnvironmentInterface):
    """Environment where the LLM must deduce the count of unique numbers."""

    def __init__(self, cfg: Optional[UniqueNumbersConfig] = None):
        cfg = cfg or UniqueNumbersConfig()
        self.min_length = cfg.get("min_length", 3)
        self.max_length = cfg.get("max_length", 7)
        self.default_max_turns = cfg.get("max_turns", 10)

        self.runner = _UniqueNumbersRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[Optional[UniqueNumbersMetadata]],
    ) -> EnvironmentReturn:
        args = []
        for log, meta in zip(message_log_batch, metadata):
            assert meta is not None, "Metadata must not be None for UniqueNumbersEnv."
            assert meta["numbers"] is not None, "Numbers must not be None in metadata."
            assert meta["unique_count"] > 0, (
                "Unique count must be greater than 0 in metadata."
            )
            args.append((log, meta))

        # Process either serially or in parallel
        if GEMINI_CALL_MAX_WORKERS is None or GEMINI_CALL_MAX_WORKERS <= 1:
            results = [self.runner.process_turn(log, meta) for log, meta in args]
        else:
            with ThreadPoolExecutor(max_workers=GEMINI_CALL_MAX_WORKERS) as executor:
                results = list(
                    executor.map(lambda p: self.runner.process_turn(*p), args)
                )

        observations, rewards, terminateds, stop_strings, next_metadata = (
            [],
            [],
            [],
            [],
            [],
        )
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
            answers=None,
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
