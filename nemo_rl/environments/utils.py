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
from typing import Any


def chunk_list_to_workers(to_chunk: list[Any], num_workers: int) -> list[list[Any]]:
    """Chunk a list into a list of lists, where each sublist is assigned to a worker. Keeps ordering of elements.

    If the list is not divisible by the number of workers, the last worker may have fewer elements.
    If there are more workers than elements, the first len(list) workers will have a single element each,
    and the remaining workers will have empty lists.

    Args:
        list: The list to be chunked.
        num_workers: The number of workers to distribute the list to.

    Returns:
        A list of lists, where each sublist contains elements assigned to a worker.

    Examples:
    ```{doctest}
    >>> from nemo_rl.environments.utils import chunk_list_to_workers
    >>> chunk_list_to_workers([1, 2, 3, 4, 5], 3)
    [[1, 2], [3, 4], [5]]
    ```
    """
    if not to_chunk:
        return [[] for _ in range(num_workers)]

    # Handle case where we have more workers than elements
    if len(to_chunk) <= num_workers:
        result = [[item] for item in to_chunk]
        # Add empty lists for remaining workers
        result.extend([[] for _ in range(num_workers - len(to_chunk))])
        return result

    # Calculate chunk size (ceiling division to ensure all elements are covered)
    chunk_size = (len(to_chunk) + num_workers - 1) // num_workers

    # Create chunks
    chunks = []
    for i in range(0, len(to_chunk), chunk_size):
        chunks.append(to_chunk[i : i + chunk_size])

    # If we somehow ended up with more chunks than workers (shouldn't happen with ceiling division)
    # merge the last chunks
    if len(chunks) > num_workers:
        chunks[num_workers - 1 :] = [sum(chunks[num_workers - 1 :], [])]

    return chunks
