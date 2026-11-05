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

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.dataloader import MultipleDataloaderWrapper


@pytest.fixture(scope="function")
def dataloaders() -> dict[str, StatefulDataLoader]:
    dataset1 = [
        {"message_log": [{"role": "user", "content": str(x)}]} for x in range(2)
    ]
    dataset2 = [
        {"message_log": [{"role": "user", "content": str(x)}]} for x in range(2, 6)
    ]

    def collate_fn(data_batch: list[dict]) -> dict:
        return {
            "message_log": [datum["message_log"] for datum in data_batch],
        }

    dataloaders = {
        "dataloader1": StatefulDataLoader(
            dataset=dataset1, batch_size=2, shuffle=False, collate_fn=collate_fn
        ),
        "dataloader2": StatefulDataLoader(
            dataset=dataset2, batch_size=2, shuffle=False, collate_fn=collate_fn
        ),
    }

    yield dataloaders


def test_multiple_dataloader(dataloaders):
    wrapped_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_custom_dataloader"
        },
        dataloaders=dataloaders,
    )

    iter_count = 0
    for data in wrapped_dataloader:
        content = sorted([message[0]["content"] for message in data["message_log"]])

        if iter_count == 0:
            assert content == ["0", "1", "2", "3"]
        elif iter_count == 1:
            assert content == ["0", "1", "4", "5"]

        iter_count += 1
        if iter_count == 2:
            break


def test_multiple_dataloader_with_records(dataloaders):
    wrapped_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_custom_dataloader_with_chosen_task"
        },
        dataloaders=dataloaders,
    )
    # set the records to sample data from all dataloaders
    wrapped_dataloader.set_records(
        {
            "chosen_task": ["dataloader1", "dataloader2"],
            "expected_num_prompts": wrapped_dataloader.expected_num_prompts,
        }
    )

    iter_count = 0
    for data in wrapped_dataloader:
        content = sorted([message[0]["content"] for message in data["message_log"]])

        if iter_count == 0:
            assert content == ["0", "1", "2", "3"]
            # set the records to sample data from dataloader1
            wrapped_dataloader.set_records(
                {
                    "chosen_task": ["dataloader1"],
                    "expected_num_prompts": wrapped_dataloader.expected_num_prompts,
                }
            )
        elif iter_count == 1:
            assert content == ["0", "0", "1", "1"]
            # set the records to sample data from dataloader2
            wrapped_dataloader.set_records(
                {
                    "chosen_task": ["dataloader2"],
                    "expected_num_prompts": wrapped_dataloader.expected_num_prompts,
                }
            )
        elif iter_count == 2:
            assert content == ["2", "3", "4", "5"]

        iter_count += 1
        if iter_count == 3:
            break
