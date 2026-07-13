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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("nccl")

from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup


def _make_group(rank: int = 24, world_size: int = 32) -> StatelessProcessGroup:
    group = StatelessProcessGroup.__new__(StatelessProcessGroup)
    group.rank = rank
    group.world_size = world_size
    group.nccl_communicator = MagicMock()
    group.nccl_communicator.shrink.return_value = MagicMock()
    group._retired_nccl_communicators = []
    return group


def test_shrink_replaces_communicator_and_compacts_rank(monkeypatch) -> None:
    group = _make_group()
    old_communicator = group.nccl_communicator
    synchronize = MagicMock()
    monkeypatch.setattr("torch.cuda.synchronize", synchronize)

    result = group.shrink([22, 23])

    assert result == (24, 22, 30)
    assert group.rank == 22
    assert group.world_size == 30
    assert group.nccl_communicator is old_communicator.shrink.return_value
    assert group._retired_nccl_communicators == [old_communicator]
    synchronize.assert_called_once_with()
    old_communicator.shrink.assert_called_once()
    kwargs = old_communicator.shrink.call_args.kwargs
    assert kwargs["exclude_ranks"] == [22, 23]
    assert kwargs["config"].shrink_share is True


@pytest.mark.parametrize("exclude_ranks", [[], [22, 22], [-1], [32]])
def test_shrink_validates_excluded_ranks(exclude_ranks) -> None:
    group = _make_group()

    with pytest.raises(ValueError):
        group.shrink(exclude_ranks)


def test_excluded_rank_must_not_call_shrink() -> None:
    group = _make_group(rank=22)

    with pytest.raises(ValueError, match="must not call"):
        group.shrink([22, 23])
