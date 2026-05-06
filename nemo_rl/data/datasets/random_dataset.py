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

"""Synthetic random-length dataset for benchmarking."""

from typing import Any, Callable

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.utils.sequence_length_generator import get_sequence_length_generator


class _SyntheticRandomRawDataset:
    """Indexable raw dataset that supplies task names for synthetic processing."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        return {"task_name": "random"}


class RandomDataset:
    """Synthetic dataset that generates random input sequences of varying lengths.

    This dataset is used for benchmarking purposes. It is not meant to be used for training or evaluation.

    Args:
        input_len_or_input_len_generator: Fixed input length, a callable input
            length generator, or a dict with 'mean' and 'std' for normal sampling.
        num_samples: Number of synthetic raw samples to expose.

    Returns:
        A RandomDataset object.
    """

    def __init__(
        self,
        input_len_or_input_len_generator: Callable | dict[str, Any] | int,
        num_samples: int,
    ):
        if isinstance(input_len_or_input_len_generator, dict):
            input_len_or_input_len_generator = get_sequence_length_generator(
                input_len_or_input_len_generator
            )
        self.input_len_or_input_len_generator = input_len_or_input_len_generator

        self.formatted_ds = {"train": _SyntheticRandomRawDataset(num_samples)}
        self.task_spec = TaskDataSpec(
            task_name="random",
            input_len_or_input_len_generator=self.input_len_or_input_len_generator,
        )
        self.processor = processors.random_input_len_processor
