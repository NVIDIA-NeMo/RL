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
"""Privileged value prefixes are checkpointed with the replay groups that
reference them and restored for exactly those groups on resume."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.ppo_utils.privileged_value import PRIVILEGE_PREFIXES_FILENAME
from nemo_rl.algorithms.rollout_checkpointing.snapshots import (
    write_controller_sidecars,
)
from nemo_rl.algorithms.rollout_checkpointing.state import (
    CapturedRolloutState,
    ResumeSource,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GROUP_ID_TAG, PROMPT_IDX_TAG
from tests.unit.algorithms.test_privileged_value import _store
from tests.unit.single_controller._rollout_checkpoints import (
    attach_rollout_checkpointing,
)


def _restorer(checkpoint_path: Path, store: Any, metas: list[Any]) -> Any:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._privilege_store = store
    ctrl._buffer = SimpleNamespace(meta_list=metas)
    ctrl._resume = ResumeSource(path=str(checkpoint_path))
    attach_rollout_checkpointing(ctrl)
    return ctrl._rollout_restorer


def _group_meta(group_id: str, prompt_idx: int) -> KVBatchMeta:
    tag = {GROUP_ID_TAG: group_id, PROMPT_IDX_TAG: prompt_idx}
    return KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=[f"{group_id}_g0", f"{group_id}_g1"],
        fields=["input_ids"],
        sequence_lengths=[3, 2],
        tags=[dict(tag), dict(tag)],
    )


def _captured(privilege_state: dict[str, Any] | None) -> CapturedRolloutState:
    state: dict[str, Any] = {field.name: None for field in fields(CapturedRolloutState)}
    state.update(dataloader_state={}, privilege_state=privilege_state)
    return CapturedRolloutState(**state)


def test_restore_reads_the_prefix_cache_for_the_restored_replay_groups(tmp_path):
    state = {"prefixes": {"a": torch.tensor([1])}, "stats": {}}
    torch.save(state, tmp_path / PRIVILEGE_PREFIXES_FILENAME)
    store = MagicMock()
    restored_meta = SimpleNamespace(sample_ids=["g0_g0"])

    asyncio.run(
        _restorer(tmp_path, store, [restored_meta, None])._restore_privilege_prefixes(1)
    )

    store.restore_from_checkpoint.assert_called_once_with(
        str(tmp_path), [restored_meta]
    )


def test_restored_groups_without_saved_prefixes_fail_loudly(tmp_path):
    meta = asyncio.run(_store().enrich(_group_meta("g0", 1), {"answer": "42"}))

    with pytest.raises(FileNotFoundError, match=PRIVILEGE_PREFIXES_FILENAME):
        _store().restore_from_checkpoint(str(tmp_path), [meta])


def test_privilege_restore_is_a_no_op_without_restored_groups(tmp_path):
    store = MagicMock()

    asyncio.run(_restorer(tmp_path, store, [])._restore_privilege_prefixes(0))

    store.restore_from_checkpoint.assert_not_called()


def test_blind_critic_restore_is_a_no_op(tmp_path):
    asyncio.run(_restorer(tmp_path, None, [])._restore_privilege_prefixes(3))


@pytest.mark.parametrize("privileged", [True, False])
def test_sidecars_hold_prefixes_only_for_a_privileged_value(tmp_path, privileged):
    state = (
        {"prefixes": {"prompt1": torch.tensor([1, 2])}, "stats": {"prompt1": {}}}
        if privileged
        else None
    )

    written = asyncio.run(write_controller_sidecars(tmp_path, _captured(state)))

    path = tmp_path / PRIVILEGE_PREFIXES_FILENAME
    assert path.exists() is privileged
    assert (str(path) in written) is privileged
    if privileged:
        saved = torch.load(path, weights_only=False)
        assert torch.equal(saved["prefixes"]["prompt1"], torch.tensor([1, 2]))


def test_saved_prefixes_round_trip_into_a_fresh_store(tmp_path):
    first = _store()
    trained = asyncio.run(first.enrich(_group_meta("g0", 1), {"answer": "one"}))
    buffered = asyncio.run(first.enrich(_group_meta("g1", 1), {"answer": "one"}))
    other = asyncio.run(first.enrich(_group_meta("g2", 2), {"answer": "two"}))
    # g2 trains before the save, so only prompt 1 is still referenced.
    first.release([other])
    asyncio.run(write_controller_sidecars(tmp_path, _captured(first.state_dict())))

    restored = _store()
    asyncio.run(
        _restorer(
            tmp_path, restored, [trained, None, buffered]
        )._restore_privilege_prefixes(2)
    )

    assert restored.state_dict()["prefixes"].keys() == {"prompt1"}
    assert torch.equal(
        restored.prefixes_for(buffered)["prompt1"],
        first.prefixes_for(buffered)["prompt1"],
    )
    # Both restored groups hold a reference: the prefix outlives the first.
    restored.release([trained])
    assert restored.state_dict()["prefixes"].keys() == {"prompt1"}
    restored.release([buffered])
    assert restored.state_dict()["prefixes"] == {}
