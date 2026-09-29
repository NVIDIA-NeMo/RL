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

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to a JSONL file or a pre-converted Arrow/Parquet dataset.
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # Preserve JSONL records as raw strings because the NeMo-Gym processor
        # intentionally parses the nested payload later. The Hugging Face text
        # builder materializes a reusable Arrow cache instead of retaining the
        # entire source file as a Python list of strings. Pre-converted Arrow,
        # Parquet, and save_to_disk datasets are accepted as well.
        self.dataset = load_dataset_from_path(data_path, preserve_jsonl_rows=True)
        if "extra_env_info" in self.dataset.column_names:
            self.dataset = self.dataset.select_columns(["extra_env_info"])
        elif "text" in self.dataset.column_names:
            self.dataset = self.dataset.select_columns(["text"]).rename_column(
                "text", "extra_env_info"
            )
        else:
            raise ValueError(
                "A NeMo-Gym dataset must contain an 'extra_env_info' or 'text' "
                f"column, but {data_path!r} contains {self.dataset.column_names}."
            )
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
