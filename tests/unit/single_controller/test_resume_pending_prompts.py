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
"""Resume: no deadlock or skipped prompts when the replay buffer is dropped, and
privileged-critic prefixes restored with it when it is kept."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.single_controller import (
    PENDING_PROMPTS_FILENAME,
    PRIVILEGE_PREFIXES_FILENAME,
    SingleControllerActor,
    _restorable_dispatch_index,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_dropped_replay_buffer_reseeds_the_sampler_cursor():
    # The saved cursor counted groups the dropped buffer held; restoring it would
    # leave the gate closed with nothing left to train.
    assert _restorable_dispatch_index(7, checkpoint_replay_buffer=False) is None
    assert _restorable_dispatch_index(7, checkpoint_replay_buffer=True) == 7
    assert _restorable_dispatch_index(None, checkpoint_replay_buffer=True) is None


def _collate(rows):
    return BatchedDataDict(
        {"idx": [row["idx"] for row in rows], "text": [row["text"] for row in rows]}
    )


def _controller(tmp_path, *, checkpoint_replay_buffer=False, num_prompts_per_step=2):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    dataset = [{"idx": i, "text": f"prompt-{i}"} for i in range(10)]
    ctrl._dataloader = SimpleNamespace(dataset=dataset, collate_fn=_collate)
    ctrl._async_cfg = SimpleNamespace(
        checkpoint_replay_buffer=checkpoint_replay_buffer,
        sampler=SimpleNamespace(name="ready_first"),
    )
    ctrl._algo_cfg = SimpleNamespace(num_prompts_per_step=num_prompts_per_step)
    ctrl._last_checkpoint_path = str(tmp_path)
    ctrl._trainer_version = 3
    ctrl._sampler_stamps_target_steps = False
    return ctrl


def test_restore_rehydrates_pending_prompts_from_the_dataset(tmp_path):
    torch.save([4, 7, 2], tmp_path / PENDING_PROMPTS_FILENAME)
    ctrl = _controller(tmp_path)

    asyncio.run(ctrl._maybe_restore_pending_prompts())

    assert [p["idx"] for p in ctrl._resume_pending_prompts] == [4, 7, 2]
    assert [p["text"] for p in ctrl._resume_pending_prompts] == [
        "prompt-4",
        "prompt-7",
        "prompt-2",
    ]


def test_restore_skips_pending_prompts_when_the_buffer_was_checkpointed(tmp_path):
    torch.save([4], tmp_path / PENDING_PROMPTS_FILENAME)
    ctrl = _controller(tmp_path, checkpoint_replay_buffer=True)

    asyncio.run(ctrl._maybe_restore_pending_prompts())

    assert ctrl._resume_pending_prompts == []


def test_restore_is_a_no_op_without_a_pending_file(tmp_path):
    ctrl = _controller(tmp_path)
    asyncio.run(ctrl._maybe_restore_pending_prompts())
    assert ctrl._resume_pending_prompts == []


def test_pending_prompts_are_dispatched_one_admission_per_batch(tmp_path):
    ctrl = _controller(tmp_path, num_prompts_per_step=2)
    ctrl._resume_pending_prompts = [{"idx": i} for i in (4, 7, 2)]
    admissions = []
    launched = []

    class _Sampler:
        async def admit(self, *, trainer_version_fn):
            admissions.append(trainer_version_fn())
            return None

    ctrl._sampler = _Sampler()
    ctrl._buffer = SimpleNamespace(count_for_target_step=lambda step: 0)

    async def _launch(prompt, target_step, group_id):
        launched.append((prompt["idx"], target_step, group_id))

    asyncio.run(ctrl._dispatch_resume_pending_prompts(_launch))

    # Two admissions for three prompts at two per step (the last one partial),
    # each prompt launched once, and the pending list is consumed.
    assert admissions == [3, 3]
    assert launched == [(4, None, None), (7, None, None), (2, None, None)]
    assert ctrl._resume_pending_prompts == []


def test_in_order_redispatches_only_whole_batches(tmp_path):
    ctrl = _controller(tmp_path, num_prompts_per_step=2)
    ctrl._async_cfg.sampler = SimpleNamespace(name="in_order")
    ctrl._resume_pending_prompts = [{"idx": i} for i in (4, 7, 2)]
    launched = []

    class _Sampler:
        async def admit(self, *, trainer_version_fn):
            return 0

    ctrl._sampler = _Sampler()
    ctrl._buffer = SimpleNamespace(count_for_target_step=lambda step: 0)

    async def _launch(prompt, target_step, group_id):
        launched.append(prompt["idx"])

    asyncio.run(ctrl._dispatch_resume_pending_prompts(_launch))

    # A partial in_order target batch would never fill, so the remainder is skipped.
    assert launched == [4, 7]


class _RecordingPrivilegeStore:
    def __init__(self):
        self.restored = None

    def restore(self, state, metas):
        self.restored = (state, list(metas))

    def __len__(self):
        return 1


def test_privilege_prefixes_restore_for_the_restored_replay_groups(tmp_path):
    torch.save(
        {"prefixes": {"a": torch.tensor([1])}, "stats": {}},
        tmp_path / PRIVILEGE_PREFIXES_FILENAME,
    )
    ctrl = _controller(tmp_path, checkpoint_replay_buffer=True)
    ctrl._privilege_store = _RecordingPrivilegeStore()
    restored_meta = SimpleNamespace(sample_ids=["g0_g0"])
    ctrl._buffer = SimpleNamespace(meta_list=[restored_meta, None])

    asyncio.run(ctrl._maybe_restore_privilege_prefixes(1))

    state, metas = ctrl._privilege_store.restored
    assert list(state["prefixes"]) == ["a"]
    assert metas == [restored_meta]


def test_restored_privileged_groups_without_saved_prefixes_fail_loudly(tmp_path):
    ctrl = _controller(tmp_path, checkpoint_replay_buffer=True)
    ctrl._privilege_store = _RecordingPrivilegeStore()
    ctrl._buffer = SimpleNamespace(meta_list=[SimpleNamespace(sample_ids=["g0_g0"])])

    with pytest.raises(FileNotFoundError, match=PRIVILEGE_PREFIXES_FILENAME):
        asyncio.run(ctrl._maybe_restore_privilege_prefixes(1))


def test_privilege_restore_is_a_no_op_without_restored_groups(tmp_path):
    ctrl = _controller(tmp_path, checkpoint_replay_buffer=True)
    ctrl._privilege_store = _RecordingPrivilegeStore()

    asyncio.run(ctrl._maybe_restore_privilege_prefixes(0))

    assert ctrl._privilege_store.restored is None
