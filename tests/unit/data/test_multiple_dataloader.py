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

import pickle

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.dataloader import MultipleDataloaderWrapper


def collate_message_log(data_batch: list[dict]) -> dict:
    return {
        "message_log": [datum["message_log"] for datum in data_batch],
    }


def create_test_dataloaders() -> dict[str, StatefulDataLoader]:
    dataset1 = [
        {"message_log": [{"role": "user", "content": str(x)}]} for x in range(2)
    ]
    dataset2 = [
        {"message_log": [{"role": "user", "content": str(x)}]} for x in range(2, 6)
    ]

    return {
        "dataloader1": StatefulDataLoader(
            dataset=dataset1,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_message_log,
        ),
        "dataloader2": StatefulDataLoader(
            dataset=dataset2,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_message_log,
        ),
    }


@pytest.fixture(scope="function")
def dataloaders() -> dict[str, StatefulDataLoader]:
    yield create_test_dataloaders()


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


def test_multiple_dataloader_lazy_and_picklable(dataloaders):
    wrapped_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_custom_dataloader_with_chosen_task"
        },
        dataloaders=dataloaders,
    )

    assert wrapped_dataloader.data_iterators is None
    assert wrapped_dataloader.custom_dataloader_func is None

    wrapped_dataloader.set_records(
        {
            "chosen_task": ["dataloader1", "dataloader2"],
            "expected_num_prompts": wrapped_dataloader.expected_num_prompts,
        }
    )
    restored_dataloader = pickle.loads(pickle.dumps(wrapped_dataloader))

    assert restored_dataloader.data_iterators is None
    assert restored_dataloader.custom_dataloader_func is None
    assert restored_dataloader.records == {}


def test_multiple_dataloader_state_dict_restore(dataloaders):
    wrapped_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_custom_dataloader"
        },
        dataloaders=dataloaders,
    )

    data = next(wrapped_dataloader)
    assert sorted([message[0]["content"] for message in data["message_log"]]) == [
        "0",
        "1",
        "2",
        "3",
    ]

    state = wrapped_dataloader.state_dict()
    restored_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_custom_dataloader"
        },
        dataloaders=create_test_dataloaders(),
    )
    restored_dataloader.load_state_dict(state)

    data = next(restored_dataloader)
    assert sorted([message[0]["content"] for message in data["message_log"]]) == [
        "0",
        "1",
        "4",
        "5",
    ]
    assert restored_dataloader.records == {}


def test_async_target_ratio_dataloader_records():
    def create_dataloaders() -> dict[str, StatefulDataLoader]:
        return {
            "dataloader1": StatefulDataLoader(
                dataset=[
                    {"message_log": [{"role": "user", "content": f"a{x}"}]}
                    for x in range(4)
                ],
                batch_size=1,
                shuffle=False,
                collate_fn=collate_message_log,
            ),
            "dataloader2": StatefulDataLoader(
                dataset=[
                    {"message_log": [{"role": "user", "content": f"b{x}"}]}
                    for x in range(4)
                ],
                batch_size=1,
                shuffle=False,
                collate_fn=collate_message_log,
            ),
        }

    even_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_async_target_ratio_dataloader"
        },
        dataloaders=create_dataloaders(),
    )
    even_dataloader.set_records(
        {
            "target_weight_version": 2,
            "expected_num_prompts": even_dataloader.expected_num_prompts,
        }
    )
    even_batch = next(even_dataloader)
    assert [message[0]["content"] for message in even_batch["message_log"]] == [
        "a0",
        "b0",
        "a1",
        "b1",
    ]

    odd_dataloader = MultipleDataloaderWrapper(
        expected_num_prompts=4,
        data_config={
            "custom_dataloader": "examples.custom_dataloader.custom_dataloader.example_async_target_ratio_dataloader"
        },
        dataloaders=create_dataloaders(),
    )
    odd_dataloader.set_records(
        {
            "target_weight_version": 3,
            "expected_num_prompts": odd_dataloader.expected_num_prompts,
        }
    )
    odd_batch = next(odd_dataloader)
    assert [message[0]["content"] for message in odd_batch["message_log"]] == [
        "a0",
        "b0",
        "b1",
        "a1",
    ]
