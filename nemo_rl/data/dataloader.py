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

from typing import Any, Iterator

from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.weights import TaskDataloaderState, TaskName, TaskQuota
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def weighted_custom_dataloader(
    data_iterators: dict[TaskName, Iterator],
    dataloaders: dict[TaskName, StatefulDataLoader],
    **kwargs: Any,
) -> tuple[BatchedDataDict, dict[TaskName, Iterator]]:
    """Pull one batch from every task dataloader and concatenate them.

    The task mixture is already encoded in each dataloader's per-task
    ``batch_size`` (see ``nemo_rl.algorithms.grpo.setup``), so the configured
    proportions hold without any weighting logic here.

    Exhausted iterators are reset so the wrapper stays an infinite iterator.

    Args:
        data_iterators: Live iterator per task.
        dataloaders: Source dataloaders, used to reset exhausted iterators.
        **kwargs: Records forwarded from ``MultipleDataloaderWrapper.set_records``.

    Returns:
        The concatenated batch and the (possibly reset) iterators.
    """
    result = []
    for task_name, data_iterator in data_iterators.items():
        try:
            result.append(next(data_iterator))
        except StopIteration:
            data_iterators[task_name] = iter(dataloaders[task_name])
            result.append(next(data_iterators[task_name]))

    return BatchedDataDict.from_batches(result), data_iterators


class MultipleDataloaderWrapper:
    """Wrapper for multiple dataloaders.

    This wrapper is used to sample data from multiple dataloaders using a custom dataloader function.

    When a single dataloader is exhausted, the data iterator must be reset in the custom dataloader function (as demonstrated in `examples/custom_dataloader/custom_dataloader.py`).
    This design ensures that the MultipleDataloaderWrapper operates as an infinite iterator, where __next__() will not raise StopIteration and __len__() is not supported.
    """

    def __init__(
        self,
        expected_num_prompts: int,
        data_config: dict,
        dataloaders: dict[str, StatefulDataLoader],
        task_quota: TaskQuota | None = None,
    ):
        self.expected_num_prompts = expected_num_prompts
        self.data_config = data_config
        self.dataloaders = dataloaders
        # Per-task prompt counts making up one training step. Empty when the
        # datasets are mixed without weights. Async GRPO reads this to gate
        # batch release on every task's slots being filled.
        self.task_quota: TaskQuota = task_quota or {}

        # Iterators are created lazily on first __next__ rather than here.
        # Live dataloader iterators are not picklable, and async GRPO ships this
        # object to a Ray actor via AsyncTrajectoryCollector.start_collection.
        self.data_iterators: dict[TaskName, Iterator] | None = None

        # custom dataloader function to decide how to sample the data from the dataloaders
        self.custom_dataloader_func = self._load_custom_dataloader_func()
        # records to pass additional information to the custom dataloader function
        self.records = {}

    def _load_custom_dataloader_func(self):
        import sys
        from pathlib import Path

        from hydra.utils import get_method

        project_root_path = Path(__file__).absolute().parents[2]
        sys.path = [str(project_root_path)] + sys.path

        return get_method(self.data_config["custom_dataloader"])

    def _ensure_iterators(self) -> None:
        """Create the per-task iterators on first use.

        Deferred from __init__ so this object stays picklable until iteration
        actually begins.
        """
        if self.data_iterators is None:
            self.data_iterators = {
                task_name: iter(dataloader)
                for task_name, dataloader in self.dataloaders.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        self._ensure_iterators()

        # sample data from the dataloaders
        result, self.data_iterators = self.custom_dataloader_func(
            self.data_iterators, self.dataloaders, **self.records
        )

        # check if the number of prompts is expected
        assert len(result["message_log"]) == self.expected_num_prompts, (
            f"Expected {self.expected_num_prompts} prompts, but got {len(result['message_log'])}"
        )

        # reset records
        self.records = {}

        return result

    def set_records(self, records: dict):
        """Set the records for the custom dataloader.

        Records are used to pass additional information to the custom dataloader function to decide how to sample the data from the dataloaders.
        """
        self.records.update(records)

    def state_dict(self) -> TaskDataloaderState:
        """Return each task dataloader's state, keyed by task name.

        Async GRPO checkpointing reaches this through
        ``AsyncTrajectoryCollector.get_dataloader_state``, which requires the
        wrapper to expose the same interface as a plain ``StatefulDataLoader``.
        """
        return {
            task_name: dataloader.state_dict()
            for task_name, dataloader in self.dataloaders.items()
        }

    def load_state_dict(self, state: TaskDataloaderState) -> None:
        """Restore each task dataloader's state.

        A task missing from ``state`` (for example a dataset added since the
        checkpoint was written) starts from index 0 rather than inheriting an
        unrelated task's ``samples_yielded``.
        """
        for task_name, dataloader in self.dataloaders.items():
            if task_name in state:
                dataloader.load_state_dict(state[task_name])
            else:
                print(
                    f"  ⚠️ No saved dataloader state for task {task_name}; starting from the beginning",
                    flush=True,
                )

        # Force iterators to be rebuilt from the restored state on next use.
        self.data_iterators = None
