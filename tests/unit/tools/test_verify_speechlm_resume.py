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

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchdata.stateful_dataloader import StatefulDataLoader

from tools.verify_speechlm_resume import (
    checkpoint_scalar_state,
    next_source_rows,
    verify,
)


def test_read_optimizer_scheduler_scalars_without_loading_moments(tmp_path):
    state = {
        "optim": {
            "state": {
                "selected.weight": {
                    "step": torch.tensor(20.0),
                    "exp_avg": torch.ones(4096),
                }
            },
            "param_groups": {"selected.weight": {"lr": 1e-6}},
        },
        "sched": {"last_epoch": 20, "_last_lr": [1e-6]},
    }
    dcp.save(state, checkpoint_id=tmp_path / "policy/optimizer/optim")
    actual = checkpoint_scalar_state(tmp_path)
    assert actual == {
        "optim.state.selected.weight.step": 20.0,
        "optim.param_groups.selected.weight.lr": 1e-6,
        "sched.last_epoch": 20,
        "sched._last_lr": [1e-6],
    }


def test_probe_restores_saved_ordered_dataloader_cursor(tmp_path):
    loader = StatefulDataLoader(
        range(200), batch_size=40, shuffle=False, drop_last=True, num_workers=1
    )
    iterator = iter(loader)
    assert next(iterator).tolist() == list(range(40))
    assert next(iterator).tolist() == list(range(40, 80))
    torch.save(loader.state_dict(), tmp_path / "train_dataloader.pt")
    assert next_source_rows(tmp_path, rows=200, batch_size=40, num_workers=1) == list(
        range(80, 120)
    )


def _native_resume_fixture(output):
    for step, batches in [(20, 2), (22, 4)]:
        checkpoint = output / "checkpoints" / f"step_{step}"
        model = checkpoint / "policy/weights/model"
        model.mkdir(parents=True)
        (model / ".metadata").touch()
        (checkpoint / "training_info.json").write_text(
            json.dumps(
                {
                    "total_steps": step,
                    "consumed_samples": step * 20,
                    "total_valid_tokens": step * 100,
                }
            )
        )
        dcp.save(
            {
                "optim": {
                    "state": {"selected.weight": {"step": torch.tensor(float(step))}},
                    "param_groups": {"selected.weight": {"lr": 1e-6}},
                },
                "sched": {"last_epoch": step},
            },
            checkpoint_id=checkpoint / "policy/optimizer/optim",
        )
        loader = StatefulDataLoader(
            range(20000), batch_size=40, shuffle=False, drop_last=True, num_workers=1
        )
        iterator = iter(loader)
        for _ in range(batches):
            next(iterator)
        torch.save(loader.state_dict(), checkpoint / "train_dataloader.pt")
    (output / "resume2.log").write_text(
        "=" * 25
        + " Step 21/22 "
        + "=" * 25
        + "\n"
        + "=" * 25
        + " Step 22/22 "
        + "=" * 25
        + "\n"
    )
    log = output / "logs/exp_002/train_data_step21.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"source_idx": [80], "prompt_group_ids": [[0]]}) + "\n")
    return log


def test_verify_20_plus_2_reads_real_checkpoint_layout(tmp_path):
    _native_resume_fixture(tmp_path)
    result = verify(tmp_path, initial_only=False)
    assert result["resumed_update_steps"] == [21, 22]
    assert result["initial"]["next_source_rows"] == list(range(80, 120))
    assert result["resumed"]["next_source_rows"] == list(range(160, 200))
    assert result["first_resumed_source_rows"] == [80]


@pytest.mark.parametrize("defect", ["reset_cursor", "ambiguous_log"])
def test_resume_verifier_rejects_reset_and_ambiguous_log(tmp_path, defect):
    log = _native_resume_fixture(tmp_path)
    if defect == "reset_cursor":
        checkpoint = tmp_path / "checkpoints"
        (checkpoint / "step_22/train_dataloader.pt").write_bytes(
            (checkpoint / "step_20/train_dataloader.pt").read_bytes()
        )
    else:
        other = tmp_path / "logs/exp_003/train_data_step21.jsonl"
        other.parent.mkdir(parents=True)
        other.write_text(log.read_text())
    with pytest.raises(AssertionError):
        verify(tmp_path, initial_only=False)
