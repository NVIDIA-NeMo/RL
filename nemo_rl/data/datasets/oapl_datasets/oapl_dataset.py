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

import math
from typing import Any, Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


def _log_z(rewards: list[float], beta: float) -> float:
    """log( (1/n) * sum_i exp(r_i / beta) ), computed with the log-sum-exp trick."""
    scaled = [r / beta for r in rewards]
    m = max(scaled)
    return m + math.log(sum(math.exp(s - m) for s in scaled)) - math.log(len(scaled))


class OAPLDataset(RawDataset):
    """Dataset class for OAPL data which can be loaded from a JSON file.

    This class handles loading of grouped (prompt, generations) data for OAPL
    training. The input JSONL files should contain valid JSON objects
    formatted like this:
    {
        "context": list[dict],                 # The prompt message x (including previous turns, if any)
        "completions": [                       # The n generations y_1..y_n collected for this prompt
            {
                "completion": list[dict],       # The completion message(s) y_i, possibly a
                                                 # multi-turn agentic trajectory with
                                                 # interleaved "assistant" and "tool" turns
                "reward": float,                # r(x, y_i), the final reward for this trajectory
                "reference_logprob": float,     # log pi_ref(y_i | x), precomputed under the reference policy
            },
            ...
        ]
    }
    Each group must contain at least 2 completions to form a non-degenerate
    estimate of the partition function Z(x). The dataset flattens each group
    into ``n`` independent training examples, each carrying the group's
    precomputed ``log Z(x) = log( (1/n) * sum_i exp(r(x, y_i) / beta) )``.

    ``completion`` may span multiple turns (e.g. tool calls interleaved with
    tool results): ``oapl_collate_fn`` only trains on ``"assistant"``-role
    tokens, so ``"tool"``-role turns are excluded from log pi(y|x) even
    though they remain part of the trajectory context.

    Args:
        data_path: Path to the dataset JSON file
        beta: Temperature used to compute log Z(x). Must match the ``beta``
            used by ``OAPLLossConfig`` for training, since Z(x) is baked into
            the dataset at load time.
        subset: Optional subset name for the dataset, used for HuggingFace datasets
        split: Optional split name for the dataset, used for HuggingFace datasets
    """

    def __init__(
        self,
        data_path: str,
        beta: float,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.beta = beta

        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # load from local or huggingface
        grouped_dataset = load_dataset_from_path(data_path, subset, split)

        self.dataset = grouped_dataset.map(
            self._flatten_group,
            batched=True,
            remove_columns=grouped_dataset.column_names,
        )

    def _flatten_group(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        context_out: list[Any] = []
        completion_out: list[Any] = []
        reward_out: list[float] = []
        reference_logprob_out: list[float] = []
        log_z_out: list[float] = []
        task_name_out: list[str] = []

        for context, completions in zip(batch["context"], batch["completions"]):
            assert len(completions) >= 2, (
                "OAPL requires at least 2 generations per prompt to estimate "
                "the partition function Z(x)."
            )
            log_z = _log_z([c["reward"] for c in completions], self.beta)
            for completion in completions:
                context_out.append(context)
                completion_out.append(completion["completion"])
                reward_out.append(completion["reward"])
                reference_logprob_out.append(completion["reference_logprob"])
                log_z_out.append(log_z)
                task_name_out.append(self.task_name)

        return {
            "context": context_out,
            "completion": completion_out,
            "reward": reward_out,
            "reference_logprob": reference_logprob_out,
            "log_z": log_z_out,
            "task_name": task_name_out,
        }
