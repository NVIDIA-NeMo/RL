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

"""Unit tests for the NeMo-Gym cohort-index passthrough (``_ng_task_index``).

These cover the pure helpers added for the GenRM cohort fix: the resume-counter
recovery math (``compute_resume_ng_task_index``) and the per-prompt-group row
stamping (``_stamp_ng_task_index``). Giving every prompt group a globally-unique
``_ng_task_index`` keeps duplicate / identical-input prompts from colliding on the
GenRM cohort key and overflowing ``num_rollouts_per_prompt``.
"""

import pytest
import torch

from nemo_rl.algorithms.async_utils import compute_resume_ng_task_index
from nemo_rl.algorithms.async_utils.trajectory_collector import (
    _NEXT_NG_TASK_INDEX_KEY,
    _NG_TASK_INDEX_KEY,
    _ROLLOUTS_STATE_FILENAME,
    _stamp_ng_task_index,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _write_rollouts_state(ckpt_dir, next_ng_task_index):
    torch.save(
        {_NEXT_NG_TASK_INDEX_KEY: next_ng_task_index},
        ckpt_dir / _ROLLOUTS_STATE_FILENAME,
    )


def _buffer_state(indices):
    """A replay-buffer ``state_dict``-like object with the given per-group indices.

    ``None`` entries model trajectories lacking the key (pre-fix checkpoints).
    """
    trajectories = []
    for idx in indices:
        traj = {"batch": None, "rollout_metrics": {}}
        if idx is not None:
            traj[_NG_TASK_INDEX_KEY] = idx
        trajectories.append(traj)
    return {"trajectories": trajectories}


# ---- compute_resume_ng_task_index ---------------------------------------


def test_resume_fresh_run_is_zero():
    assert compute_resume_ng_task_index(None, None) == 0


def test_resume_reads_saved_counter(tmp_path):
    _write_rollouts_state(tmp_path, 42)
    assert compute_resume_ng_task_index(str(tmp_path), None) == 42


def test_resume_missing_rollouts_file_uses_buffer_high_water(tmp_path):
    # Old checkpoint: no rollouts.pt, but buffered groups carry indices.
    assert compute_resume_ng_task_index(str(tmp_path), _buffer_state([3, 7, 5])) == 8


def test_resume_takes_max_of_counter_and_buffer(tmp_path):
    _write_rollouts_state(tmp_path, 42)
    # Buffer high-water (100) exceeds the saved counter -> 1 + 100.
    assert compute_resume_ng_task_index(str(tmp_path), _buffer_state([100, 10])) == 101
    _write_rollouts_state(tmp_path, 200)
    # Saved counter dominates the buffer high-water.
    assert compute_resume_ng_task_index(str(tmp_path), _buffer_state([100, 10])) == 200


def test_resume_ignores_trajectories_without_index(tmp_path):
    # Pre-fix buffer (no _ng_task_index on any group) -> start at 0.
    assert compute_resume_ng_task_index(str(tmp_path), _buffer_state([None, None])) == 0


def test_resume_handles_zero_index(tmp_path):
    # index 0 must be honored (guarded by `is not None`, not truthiness).
    assert compute_resume_ng_task_index(str(tmp_path), _buffer_state([0])) == 1


# ---- _stamp_ng_task_index ------------------------------------------------


def test_stamp_sets_same_index_on_all_rows_and_preserves_fields():
    batch = BatchedDataDict(
        {"extra_env_info": [{"a": 1, _NG_TASK_INDEX_KEY: 999}, {"a": 2}]}
    )
    _stamp_ng_task_index(batch, 5)
    rows = batch["extra_env_info"]
    # All rollouts of one prompt group share the same cohort index (overwriting
    # any non-unique index that came in on the training data), other fields kept.
    assert [r[_NG_TASK_INDEX_KEY] for r in rows] == [5, 5]
    assert [r["a"] for r in rows] == [1, 2]


def test_stamp_does_not_mutate_original_row_dicts():
    original = {"a": 1}
    batch = BatchedDataDict({"extra_env_info": [original]})
    _stamp_ng_task_index(batch, 7)
    assert _NG_TASK_INDEX_KEY not in original  # shallow-copied, not mutated in place
    assert batch["extra_env_info"][0][_NG_TASK_INDEX_KEY] == 7


def test_stamp_raises_on_non_dict_row():
    batch = BatchedDataDict({"extra_env_info": [{"a": 1}, "not-a-dict"]})
    with pytest.raises(TypeError):
        _stamp_ng_task_index(batch, 5)
