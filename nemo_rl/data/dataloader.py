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

from collections.abc import Callable, Iterator
from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader


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
    ):
        self.expected_num_prompts = expected_num_prompts
        self.data_config = data_config
        self.dataloaders = dataloaders

        # Lazily initialized so the wrapper can cross Ray actor boundaries without
        # serializing live Python iterators or Hydra-loaded callables.
        self.data_iterators: dict[str, Iterator] | None = None
        self.custom_dataloader_func: Callable | None = None
        # records to pass additional information to the custom dataloader function
        self.records: dict[str, Any] = {}

    def _load_custom_dataloader_func(self):
        import sys
        from pathlib import Path

        from hydra.utils import get_method

        project_root_path = Path(__file__).absolute().parents[2]
        if str(project_root_path) not in sys.path:
            sys.path = [str(project_root_path)] + sys.path

        return get_method(self.data_config["custom_dataloader"])

    def _ensure_initialized(self) -> None:
        if self.data_iterators is None:
            self.data_iterators = {
                task_name: iter(dataloader)
                for task_name, dataloader in self.dataloaders.items()
            }
        if self.custom_dataloader_func is None:
            self.custom_dataloader_func = self._load_custom_dataloader_func()

    def __iter__(self):
        self._ensure_initialized()
        return self

    def __next__(self):
        self._ensure_initialized()
        assert self.data_iterators is not None
        assert self.custom_dataloader_func is not None

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

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state for every wrapped dataloader."""
        return {
            "type": "multiple_dataloader",
            "dataloaders": {
                task_name: dataloader.state_dict()
                for task_name, dataloader in self.dataloaders.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state into the wrapped dataloaders."""
        dataloader_states = state_dict.get("dataloaders", state_dict)
        for task_name, dataloader_state in dataloader_states.items():
            if task_name in self.dataloaders:
                self.dataloaders[task_name].load_state_dict(dataloader_state)

        self.data_iterators = None
        self.records = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["data_iterators"] = None
        state["custom_dataloader_func"] = None
        state["records"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.data_iterators = None
        self.custom_dataloader_func = None
        self.records = {}
