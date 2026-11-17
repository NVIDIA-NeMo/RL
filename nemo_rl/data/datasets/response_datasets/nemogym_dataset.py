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

import hashlib
import os

import pyarrow as pa
from datasets import Dataset, Features, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to the dataset JSONL file
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # load raw line from jsonl
        # will use `json.loads` to load to dict format at `nemo_gym_data_processor` later since `Dataset` cannot handle nested structure well
        with open(data_path) as f:
            self.dataset = [raw_line for raw_line in f]

        # format the dataset
        # extra_env_info holds a whole raw JSONL line per row; Arrow's default
        # `string` type uses 32-bit offsets and silently caps a column at ~2GB
        # total ("ArrowInvalid: offset overflow while concatenating arrays"),
        # which large NemoGym shards (e.g. verbose unit-test-heavy code
        # datasets) exceed well before hitting any row-count limit.
        # `large_string` uses 64-bit offsets and has no such ceiling.
        table = pa.Table.from_pydict(
            {
                "extra_env_info": self.dataset,
                "task_name": [self.task_name] * len(self.dataset),
            },
            schema=pa.schema(
                Features(
                    {
                        "extra_env_info": Value("large_string"),
                        "task_name": Value("string"),
                    }
                ).arrow_schema
            ),
        )
        # Dataset.from_dict()/__init__() would otherwise compute a fingerprint
        # by dill-pickling the whole table and hashing the result -- for a
        # multi-GB shard that burns CPU time wildly disproportionate to what a
        # fingerprint needs. A dataset built here lives in memory only (no
        # on-disk cache_files), so nothing depends on this fingerprint being a
        # true content hash -- a cheap path+size+mtime digest identifies the
        # same shard just as well for the map()/filter() caching it feeds.
        stat = os.stat(data_path)
        fingerprint = hashlib.sha256(
            f"{data_path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()[:64]
        self.dataset = Dataset(table, fingerprint=fingerprint)

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
