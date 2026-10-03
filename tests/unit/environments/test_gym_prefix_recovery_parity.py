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

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch


_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_parity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gym_prefix_recovery_parity", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)

_SNAPSHOT_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_snapshot.py"
)
_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "gym_prefix_recovery_snapshot", _SNAPSHOT_HELPER_PATH
)
assert _SNAPSHOT_SPEC is not None and _SNAPSHOT_SPEC.loader is not None
_SNAPSHOT_HELPER = importlib.util.module_from_spec(_SNAPSHOT_SPEC)
_SNAPSHOT_SPEC.loader.exec_module(_SNAPSHOT_HELPER)


def _row(*, token_ids: list[int] | None = None) -> dict[str, Any]:
    tokens = token_ids or [1, 2, 3]
    return {
        "sample_id": "physical-group_g0",
        "group_id": "physical-group",
        "prompt_idx": 7,
        "generation_index": 0,
        "input_length": len(tokens),
        "token_ids": tokens,
        "token_loss_mask": [0.0, 1.0, 1.0][: len(tokens)],
        "sample_loss_mask": 1.0,
        "reward": 1.0,
        "advantages": [0.0, 0.5, 0.5][: len(tokens)],
        "generation_logprobs": [-0.1, -0.2, -0.3][: len(tokens)],
        "prev_logprobs": [-0.1, -0.2, -0.3][: len(tokens)],
    }


def test_calendar_sentinel_count_matches_one_checkpointed_event() -> None:
    state = {
        "calendar": {
            "_calendar_events": {
                "columns": [
                    {"name": "event_id", "values": [1, 2]},
                    {
                        "name": "event_name",
                        "values": ["another event", "Recovery Sentinel"],
                    },
                ]
            }
        }
    }

    assert _SNAPSHOT_HELPER._calendar_sentinel_count(state, "recovery sentinel") == 1


def test_serialize_training_batch_trims_padding_and_records_identity() -> None:
    record = _HELPER.serialize_training_batch(
        train_step=3,
        chunk_index=1,
        sample_ids=["group-a_g0", "group-a_g1"],
        tags=[{"prompt_idx": 11}, {"prompt_idx": 11}],
        data={
            "input_ids": torch.tensor([[1, 2, 99], [3, 4, 5]]),
            "input_lengths": torch.tensor([2, 3]),
            "token_mask": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 0.5]),
            "total_reward": torch.tensor([1.0, 0.0]),
            "advantages": torch.tensor([[0.0, 2.0, 99.0], [0.0, 3.0, 4.0]]),
            "generation_logprobs": torch.tensor([[0.0, -0.1, 99.0], [0.0, -0.2, -0.3]]),
            "prev_logprobs": torch.tensor([[0.0, -0.1, 99.0], [0.0, -0.2, -0.3]]),
        },
        staging_sample_ids=["z", "a"],
    )

    assert record["train_step"] == 3
    assert record["staging_sample_ids"] == ["a", "z"]
    assert record["rows"][0]["group_id"] == "group-a"
    assert record["rows"][0]["prompt_idx"] == 11
    assert record["rows"][0]["generation_index"] == 0
    assert record["rows"][0]["token_ids"] == [1, 2]
    assert record["rows"][0]["advantages"] == [0.0, 2.0]


def test_compare_training_rows_is_order_independent_and_tolerates_floats() -> None:
    baseline_rows = [
        _row(),
        {
            **_row(),
            "generation_index": 1,
            "sample_id": "physical-group_g1",
        },
    ]
    recovery_rows = [
        {
            **baseline_rows[1],
            "group_id": "recovered-group",
            "sample_id": "recovered-group_g1",
            "prev_logprobs": [-0.1, -0.2, -0.300001],
        },
        {
            **baseline_rows[0],
            "group_id": "recovered-group",
            "sample_id": "recovered-group_g0",
            "advantages": [0.0, 0.500001, 0.5],
        },
    ]

    _HELPER._compare_training_rows(
        [{"train_step": 1, "rows": baseline_rows}],
        [{"train_step": 1, "rows": recovery_rows}],
        steps=1,
        prompts_per_step=1,
        generations_per_prompt=2,
        rtol=1e-4,
        atol=1e-5,
    )


def test_compare_training_rows_rejects_prefix_boundary_token_difference() -> None:
    with pytest.raises(AssertionError, match="token_ids"):
        _HELPER._compare_training_rows(
            [{"train_step": 1, "rows": [_row()]}],
            [{"train_step": 1, "rows": [_row(token_ids=[1, 2, 4])]}],
            steps=1,
            prompts_per_step=1,
            generations_per_prompt=1,
            rtol=1e-4,
            atol=1e-5,
        )


def test_prefix_completion_and_successor_checkpoint_retire_old_keys(
    tmp_path: Path,
) -> None:
    selection = {
        "model_call_id": "source-call",
        "generation_token_count": 5,
        "staging_keys": ["prefix-base", "prefix-routes"],
    }
    log = (
        "generation prefix completed: rollout_id=r model_call_id=continued "
        "source_model_call_id=source-call prefix_tokens=5 tail_tokens=7 "
        "total_generation_tokens=12"
    )
    successor = tmp_path / "step_1"
    successor.mkdir()
    (successor / "gym_checkpoint.json").write_text(
        json.dumps({"staging_keys": {"gym-0": ["new-active-key"]}})
    )

    _HELPER._assert_prefix_boundary(selection, log)
    _HELPER._assert_retired_prefix_keys(
        selection,
        [{"staging_sample_ids": ["new-active-key"], "rows": []}],
        successor,
    )


def test_prefix_completion_rejects_gap_or_overlap() -> None:
    selection = {"model_call_id": "source-call", "generation_token_count": 5}
    log = (
        "generation prefix completed: rollout_id=r model_call_id=continued "
        "source_model_call_id=source-call prefix_tokens=5 tail_tokens=7 "
        "total_generation_tokens=13"
    )

    with pytest.raises(AssertionError, match="gap or overlap"):
        _HELPER._assert_prefix_boundary(selection, log)
