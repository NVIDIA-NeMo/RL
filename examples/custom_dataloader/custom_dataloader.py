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

from typing import Iterator

from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def example_custom_dataloader(
    data_iterators: dict[str, Iterator],
    dataloaders: dict[str, StatefulDataLoader],
    **kwargs,
) -> tuple[BatchedDataDict, dict[str, Iterator]]:
    """An example of custom dataloader function.

    This function is used to sample data from multiple dataloaders using a custom dataloader function.
    In this example, we simply sample data from each dataloader.

    When a single dataloader is exhausted, the data iterator must be reset (as demonstrated here).
    This design ensures that the MultipleDataloaderWrapper operates as an infinite iterator.

    Args:
        data_iterators: A dictionary of data iterators.
        dataloaders: A dictionary of dataloaders. It is used to reset the data iterator when it is exhausted.
        **kwargs: Additional arguments to pass to the custom dataloader function.

    Returns:
        Data from the dataloaders.
        Updated data iterators (may update if the data iterator is exhausted).
    """
    # sample data from each dataloader
    result = []
    for task_name, data_iterator in data_iterators.items():
        try:
            result.append(next(data_iterator))
        except StopIteration:
            data_iterators[task_name] = iter(dataloaders[task_name])
            result.append(next(data_iterators[task_name]))

    # merge results
    result = BatchedDataDict.from_batches(result)
    return result, data_iterators


def example_custom_dataloader_with_chosen_task(
    data_iterators: dict[str, Iterator],
    dataloaders: dict[str, StatefulDataLoader],
    chosen_task: list[str],
    expected_num_prompts: int,
    **kwargs,
) -> tuple[BatchedDataDict, dict[str, Iterator]]:
    """An example of custom dataloader function with chosen task.

    This function is used to sample data from multiple dataloaders using a custom dataloader function.
    In this example, we sample data from the chosen task.

    This function will need to call `wrapped_dataloader.set_records({"chosen_task": ..., "expected_num_prompts": ...})` to set the records in `nemo_rl/algorithms/grpo.py`.
    A usage example is shown in the test case `test_multiple_dataloader_with_records` in `tests/unit/data/test_multiple_dataloader.py`.

    When a single dataloader is exhausted, the data iterator must be reset (as demonstrated here).
    This design ensures that the MultipleDataloaderWrapper operates as an infinite iterator.

    Args:
        data_iterators: A dictionary of data iterators.
        dataloaders: A dictionary of dataloaders. It is used to reset the data iterator when it is exhausted.
        chosen_task: A list of task names to sample data from.
        expected_num_prompts: The expected number of prompts to sample.

    Returns:
        Data from the dataloaders.
        Updated data iterators (may update if the data iterator is exhausted).
    """
    # sample data from the chosen task
    result = []
    current_task_idx = 0
    current_num_prompts = 0
    while current_num_prompts < expected_num_prompts:
        task_name = chosen_task[current_task_idx]
        try:
            data = next(data_iterators[task_name])
        except StopIteration:
            data_iterators[task_name] = iter(dataloaders[task_name])
            data = next(data_iterators[task_name])

        result.append(data)
        current_num_prompts += len(data["message_log"])
        current_task_idx = (current_task_idx + 1) % len(chosen_task)

    # merge results
    result = BatchedDataDict.from_batches(result)
    return result, data_iterators
