# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Custom 6x6 Sudoku data, prompts, and blank-cell reward for NeMo-RL."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from just_grpo.algorithms.block_just_grpo import align_prompt
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from .sudoku6x6_generator import generate_sudoku6x6


@dataclass(frozen=True)
class SudokuExample:
    question: str
    puzzle: list[list[int]]
    solution: list[list[int]]

    def reward(self, response: str) -> float:
        answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if len(answers) != 1 or response.count("<answer>") != 1:
            return 0.0
        if response.split("</answer>", 1)[1].replace("<|im_end|>", "").strip():
            return 0.0
        rows = [row for row in answers[0].splitlines() if row.strip()]
        if len(rows) != 6 or any(
            re.fullmatch(r"[1-6](?:[ \t]+[1-6]){5}", row.strip()) is None
            for row in rows
        ):
            return 0.0
        grid = [[int(cell) for cell in row.split()] for row in rows]
        blanks = [(r, c) for r in range(6) for c in range(6) if self.puzzle[r][c] == 0]
        return (
            sum(grid[r][c] == self.solution[r][c] for r, c in blanks) / len(blanks)
            if blanks
            else 0.0
        )


def make_examples(size: int, *, seed: int) -> list[SudokuExample]:
    return [
        SudokuExample(e["question"], e["metadata"]["puzzle"], e["metadata"]["solution"])
        for e in generate_sudoku6x6(size=size, seed=seed)
    ]


def system_prompt() -> str:
    return Path(__file__).with_name("sudoku6x6_fewshot.txt").read_text()


class SudokuDataset(Dataset):
    """Generate the reference dataset lazily; repeat complete datasets for avg@k."""

    def __init__(self, *, size: int, seed: int, repeat: int) -> None:
        self.size, self.seed, self.repeat = size, seed, repeat

    def __len__(self) -> int:
        return self.size * self.repeat

    def __getitem__(self, index: int) -> SudokuExample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return make_examples(1, seed=self.seed + index % self.size)[0]


def user_prompt(example: SudokuExample, *, style: str) -> str:
    if style == "sudoku_answer_tag":
        grid = "\n".join(
            " ".join(str(v) if v else "_" for v in row) for row in example.puzzle
        )
        return (
            f"Solve this 6x6 Sudoku puzzle:\n{grid}\n\n"
            "Put the completed grid inside <answer> </answer> tags, "
            "as rows of space-separated digits."
        )
    if style == "question":
        return (
            example.question + "\nReturn the completed grid inside <answer> </answer>."
        )
    raise ValueError(f"Unsupported prompt style: {style}")


class SudokuResponseDataset(Dataset):
    """Convert 6x6 examples into block-aligned NeMo-RL rollout messages."""

    def __init__(
        self, config: DictConfig, tokenizer: TokenizerType, *, validation: bool = False
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        split = config.data.validation if validation else config.data.train
        self.examples = SudokuDataset(
            size=split.size, seed=split.seed, repeat=split.repeat
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DatumSpec:
        example = self.examples[index]
        style = self.config.data.default.prompt_style
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(example, style=style)},
        ]
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=self.config.data.default.enable_thinking,
            **dict(self.config.policy.tokenizer.chat_template_kwargs or {}),
        )
        if len(tokens) >= self.config.data.max_input_seq_length:
            raise ValueError("Sudoku prompt exceeds data.max_input_seq_length")
        tokens = align_prompt(
            tokens,
            block_size=self.config.just_grpo.schedule.block_size,
            mask_token_id=self.config.just_grpo.schedule.mask_token_id,
        )
        # One pretokenized message preserves exactly the reference chat template
        # and block alignment. Upstream rollouts consume these token IDs directly.
        return dict(
            message_log=[
                dict(
                    role="user",
                    content=messages[-1]["content"],
                    token_ids=torch.tensor(tokens, dtype=torch.long),
                )
            ],
            length=len(tokens),
            extra_env_info={"example": example},
            loss_multiplier=1.0,
            idx=index,
            task_name=self.config.data.default.env_name,
        )


class SudokuEnvironmentImpl(EnvironmentInterface):
    """Score one response per puzzle through the standard environment interface."""

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata: list[dict[str, Any]]
    ) -> EnvironmentReturn:
        responses = [messages[-1]["content"] for messages in message_log_batch]
        rewards = torch.tensor(
            [info["example"].reward(text) for info, text in zip(metadata, responses)]
        )
        return EnvironmentReturn(
            observations=[dict(role="user", content="") for _ in responses],
            metadata=metadata,
            next_stop_strings=[None] * len(responses),
            rewards=rewards,
            terminateds=torch.ones(len(responses), dtype=torch.bool),
            answers=responses,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        rewards = batch["rewards"]
        return batch, {
            "accuracy": rewards.float().mean().item(),
            "solved_fraction": (rewards == 1).float().mean().item(),
        }

    def shutdown(self) -> None:
        pass


SudokuEnvironment = ray.remote(SudokuEnvironmentImpl)
