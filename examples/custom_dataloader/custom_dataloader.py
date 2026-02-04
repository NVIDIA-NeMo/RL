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

from typing import Iterator

from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def example_custom_dataloader(
    data_iterators: dict[str, Iterator],
    dataloaders: dict[str, StatefulDataLoader],
    **kwargs,
):
    """An example of custom dataloader function.

    This function is used to sample data from multiple dataloaders using a custom dataloader function.
    In this example, we simply sample data from each dataloader.

    Args:
        dataloaders: A dictionary of dataloaders.
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
        except:
            data_iterators[task_name] = iter(dataloaders[task_name])
            result.append(next(data_iterators[task_name]))

    # merge results
    result = BatchedDataDict.from_batches(result)
    return result, data_iterators
