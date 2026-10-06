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

import json
import os
from collections.abc import Mapping

from datasets import Dataset, Features, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import NemoGymSourceIdentity
from nemo_rl.environments.nemo_gym_shards import parse_fan_out


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to the dataset JSONL file
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(
        self,
        data_path: str,
        repeat: int = 1,
        fan_out: Mapping[str, list[str]] | None = None,
        **kwargs,
    ) -> None:
        fan_out = parse_fan_out(fan_out)
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # Keep raw lines because Dataset cannot reliably represent the nested Gym rows.
        # Record a stable source identity without parsing rows on the unsharded path.
        source_path = os.path.realpath(data_path)
        source_stat = os.stat(source_path)
        source_identity = NemoGymSourceIdentity.from_stat(source_path, source_stat)
        with open(source_path) as f:
            raw_rows = [raw_line for raw_line in f]
        source_stat_after_read = os.stat(source_path)
        if source_identity.matches(source_stat_after_read):
            self.agent_name_sources = frozenset({source_identity})
        else:
            self.agent_name_sources = None

        if fan_out:
            expanded_rows: list[str] = []
            for raw_line in raw_rows:
                row = json.loads(raw_line)
                # Explicit Environment Server routing is not agent routing and
                # therefore must never be multiplied across agent harnesses.
                if "_ng_environment_server" in row:
                    expanded_rows.append(raw_line)
                    continue
                agent_ref = row.get("agent_ref")
                agent_name = (
                    agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
                )
                route = agent_name if agent_name is not None else row.get("task_source")
                targets = fan_out.get(route) if isinstance(route, str) else None
                if targets is None:
                    expanded_rows.append(raw_line)
                    continue
                for fan_out_index, target in enumerate(targets):
                    copy = dict(row)
                    copy["agent_ref"] = {"name": target}
                    copy["_ng_fan_out_index"] = fan_out_index
                    expanded_rows.append(json.dumps(copy) + "\n")
            raw_rows = expanded_rows
            # Coverage checks must inspect the expanded in-memory rows rather
            # than the unexpanded source file recorded above.
            self.agent_name_sources = None

        # Datasets 5.0.1 combines Arrow chunks when computing the fingerprint.
        # Raw JSON columns can exceed the ~2 GiB limit of string's 32-bit offsets;
        # large_string uses 64-bit offsets so fingerprinting does not overflow.
        self.dataset = Dataset.from_dict(
            {
                "extra_env_info": raw_rows,
                "task_name": [self.task_name] * len(raw_rows),
            },
            features=Features(
                {
                    "extra_env_info": Value("large_string"),
                    "task_name": Value("string"),
                }
            ),
        )

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
