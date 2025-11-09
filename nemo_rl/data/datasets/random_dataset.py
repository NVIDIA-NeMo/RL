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

"""Local math dataset."""

from typing import Callable

from nemo_rl.data import processors
from nemo_rl.data.datasets.response_datasets.openmathinstruct2 import (
    prepare_openinstructmath2_dataset,
)
from nemo_rl.data.interfaces import TaskDataSpec


class RandomDataset:
    """Synthetic dataset that generates random input sequences of varying lengths.

    This dataset is used for benchmarking purposes. It is not meant to be used for training or evaluation.

    Args:
        input_len_or_input_len_generator: An integer or a dictionary with keys 'mean' and 'std' for the normal distribution that samples the input length.

    Returns:
        A RandomDataset object.
    """

    def __init__(
        self,
        input_len_or_input_len_generator: Callable | int,
    ):
        self.input_len_or_input_len_generator = input_len_or_input_len_generator

        # use openmathinstruct2 dataset as iterator, the real token_ids are synthetic
        self.formatted_ds = prepare_openinstructmath2_dataset()
        self.task_spec = TaskDataSpec(
            task_name="math",
            input_len_or_input_len_generator=self.input_len_or_input_len_generator,
        )
        self.processor = processors.random_input_len_processor
