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
"""SWE privileged critic prefixes are checkpointed with the replay groups that
reference them and restored for exactly those groups on resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.algorithms import swe_privileged_critic as spc
from nemo_rl.algorithms.single_controller import (
    PRIVILEGE_PREFIXES_FILENAME,
    SingleControllerActor,
    _RolloutCheckpointCut,
)
from nemo_rl.data_plane import KVBatchMeta


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)])}


class _RecordingPrivilegeStore:
    def __init__(self):
        self.restored = None

    def restore(self, state, metas):
        self.restored = (state, list(metas))

    def __len__(self):
        return 1


def _controller(checkpoint_path: Path) -> Any:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._last_checkpoint_path = str(checkpoint_path)
    return ctrl


def _store() -> spc.SwePrivilegePrefixStore:
    return spc.SwePrivilegePrefixStore(
        _CharTokenizer(), spc.SwePrivilegedCriticConfig(enabled=True)
    )


def _group_meta(group_id: str) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=[f"{group_id}_g0", f"{group_id}_g1"],
        fields=["input_ids"],
        sequence_lengths=[3, 2],
        tags=[{}, {}],
    )


def _cut(privilege_state: dict[str, Any] | None) -> _RolloutCheckpointCut:
    return _RolloutCheckpointCut(
        dataloader_state={},
        sampler_dispatch_index=0,
        next_nemo_gym_task_index=0,
        replacement_reserve=[],
        replay_metadata=None,
        privilege_state=privilege_state,
        rollout_recovery_payload=None,
        rollout_recovery_group_count=None,
        replay_row_count=0,
        staging_row_count=0,
        rolled_back_train_group_count=0,
        mutation_version=0,
        tq_save_seconds=0.0,
    )


def test_privilege_prefixes_restore_for_the_restored_replay_groups(tmp_path):
    torch.save(
        {"prefixes": {"a": torch.tensor([1])}, "stats": {}},
        tmp_path / PRIVILEGE_PREFIXES_FILENAME,
    )
    ctrl = _controller(tmp_path)
    ctrl._privilege_store = _RecordingPrivilegeStore()
    restored_meta = SimpleNamespace(sample_ids=["g0_g0"])
    ctrl._buffer = SimpleNamespace(meta_list=[restored_meta, None])

    asyncio.run(ctrl._maybe_restore_privilege_prefixes(1))

    state, metas = ctrl._privilege_store.restored
    assert list(state["prefixes"]) == ["a"]
    assert metas == [restored_meta]


def test_restored_privileged_groups_without_saved_prefixes_fail_loudly(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl._privilege_store = _RecordingPrivilegeStore()
    ctrl._buffer = SimpleNamespace(meta_list=[SimpleNamespace(sample_ids=["g0_g0"])])

    with pytest.raises(FileNotFoundError, match=PRIVILEGE_PREFIXES_FILENAME):
        asyncio.run(ctrl._maybe_restore_privilege_prefixes(1))


def test_privilege_restore_is_a_no_op_without_restored_groups(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl._privilege_store = _RecordingPrivilegeStore()

    asyncio.run(ctrl._maybe_restore_privilege_prefixes(0))

    assert ctrl._privilege_store.restored is None


def test_blind_critic_restore_is_a_no_op(tmp_path):
    ctrl = _controller(tmp_path)

    asyncio.run(ctrl._maybe_restore_privilege_prefixes(3))


@pytest.mark.parametrize("privileged", [True, False])
def test_rollout_snapshot_writes_prefixes_only_for_a_privileged_critic(
    tmp_path, privileged
):
    ctrl = _controller(tmp_path)
    ctrl._master_config = SimpleNamespace(model_dump=lambda mode: {})
    state = (
        {"prefixes": {"repo__1": torch.tensor([1, 2])}, "stats": {"repo__1": {}}}
        if privileged
        else None
    )

    asyncio.run(ctrl._write_rollout_checkpoint_sidecars(tmp_path, _cut(state)))

    path = tmp_path / PRIVILEGE_PREFIXES_FILENAME
    assert path.exists() is privileged
    if privileged:
        saved = torch.load(path, weights_only=False)
        assert torch.equal(saved["prefixes"]["repo__1"], torch.tensor([1, 2]))


def test_saved_prefixes_round_trip_into_a_fresh_store(tmp_path):
    first = _store()
    env_info = {"instance_id": "repo__1", "patch": "diff --git a/z b/z\n+fix"}
    trained = asyncio.run(first.enrich(_group_meta("g0"), env_info))
    buffered = asyncio.run(first.enrich(_group_meta("g1"), env_info))
    other = asyncio.run(
        first.enrich(
            _group_meta("g2"),
            {"instance_id": "repo__2", "gold_patch": "diff --git a/y b/y\n+fix"},
        )
    )
    # g2 trains before the save, so only repo__1 is still referenced.
    first.release([other])
    ctrl = _controller(tmp_path)
    ctrl._master_config = SimpleNamespace(model_dump=lambda mode: {})
    asyncio.run(
        ctrl._write_rollout_checkpoint_sidecars(tmp_path, _cut(first.state_dict()))
    )

    ctrl._privilege_store = _store()
    ctrl._buffer = SimpleNamespace(meta_list=[trained, None, buffered])
    asyncio.run(ctrl._maybe_restore_privilege_prefixes(2))

    restored = ctrl._privilege_store
    assert len(restored) == 1
    assert torch.equal(
        restored.prefixes_for(buffered)["repo__1"],
        first.prefixes_for(buffered)["repo__1"],
    )
    # Both restored groups hold a reference: the prefix outlives the first.
    restored.release([trained])
    assert len(restored) == 1
    restored.release([buffered])
    assert len(restored) == 0
