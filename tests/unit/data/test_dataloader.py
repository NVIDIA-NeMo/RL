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

from collections.abc import Iterator

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.dataloader import CyclingDataLoader, shutdown_dataloader_workers


class _EpochListLoader:
    def __init__(self, sizes: list[int]) -> None:
        self.sizes = list(sizes)
        self.iter_count = 0

    def __iter__(self) -> Iterator[int]:
        size = self.sizes[min(self.iter_count, len(self.sizes) - 1)]
        self.iter_count += 1
        return iter(range(size))

    def state_dict(self) -> dict[str, int]:
        return {"iter_count": self.iter_count}


def test_cycling_dataloader_cycles_after_resume_boundary():
    dataloader = _EpochListLoader([0, 3])
    iterator = iter(CyclingDataLoader(dataloader))

    assert [next(iterator) for _ in range(3)] == [0, 1, 2]
    assert dataloader.iter_count == 2


def test_cycling_dataloader_cycles_multiple_epochs():
    dataloader = _EpochListLoader([2])
    iterator = iter(CyclingDataLoader(dataloader))

    assert [next(iterator) for _ in range(5)] == [0, 1, 0, 1, 0]
    assert dataloader.iter_count == 3


def test_cycling_dataloader_rejects_empty_dataset():
    dataloader = _EpochListLoader([0])

    with pytest.raises(RuntimeError, match="two consecutive epochs"):
        next(iter(CyclingDataLoader(dataloader)))
    assert dataloader.iter_count == 2


def test_cycling_dataloader_delegates_checkpoint_state():
    dataloader = _EpochListLoader([1])

    assert CyclingDataLoader(dataloader).state_dict() == {"iter_count": 0}


def test_shutdown_dataloader_workers_after_early_break():
    dataloader = StatefulDataLoader(list(range(8)), batch_size=2, num_workers=1)
    for _ in dataloader:
        break

    # StatefulDataLoader keeps the unfinished iterator, so its worker survives the loop.
    workers = dataloader._iterator._workers
    assert all(worker.is_alive() for worker in workers)

    shutdown_dataloader_workers(dataloader)

    assert dataloader._iterator is None
    assert not any(worker.is_alive() for worker in workers)
    # The next pass starts from the beginning, as it did before the shutdown.
    assert next(iter(dataloader)).tolist() == [0, 1]
    shutdown_dataloader_workers(dataloader)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_shutdown_dataloader_workers_is_noop_without_live_iterator(num_workers):
    dataloader = StatefulDataLoader(
        list(range(4)), batch_size=2, num_workers=num_workers
    )
    shutdown_dataloader_workers(dataloader)  # never iterated

    assert [batch.tolist() for batch in dataloader] == [[0, 1], [2, 3]]
    shutdown_dataloader_workers(dataloader)  # fully consumed

    assert dataloader._iterator is None
