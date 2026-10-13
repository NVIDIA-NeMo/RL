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

import gzip
import json
from types import SimpleNamespace

from nemo_rl.experience.rollout_dump import load_dump_config


def _row(reward, valid=True):
    return SimpleNamespace(
        reward=reward,
        valid=valid,
        rejection_reason=None if valid else "no_records",
        token_ids=[1, 2, 3, 4] if valid else [],
        token_mask=[0.0, 0.0, 1.0, 1.0] if valid else [],
        logprobs=[0.0, 0.0, -1.0, -3.0] if valid else [],
        prompt_len=1 if valid else 0,
    )


def test_disabled_dump_is_none():
    assert load_dump_config(None) is None
    assert load_dump_config({"enabled": False, "dir": "x", "tokenizer": "t"}) is None


def test_dump_writes_one_record_per_group(tmp_path):
    dumper = load_dump_config(
        {
            "enabled": True,
            "dir": str(tmp_path),
            "tokenizer": "t",
            "full_rows_per_group": 2,
        }
    )
    dumper._tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
    rows = [_row(0.5), _row(1.0), _row(0.0), _row(0.0, valid=False)]
    dumper.dump(
        version=3,
        agent_name="agent",
        group_id="g0",
        prompt_idx=7,
        rows=rows,
        raw_rewards=[0.5, 1.0, 0.0, 0.0],
        mask_sample=[False, False, False, True],
    )

    (path,) = (tmp_path / "v0003").iterdir()
    with gzip.open(path, "rt") as f:
        (record,) = [json.loads(line) for line in f]
    assert (record["agent"], record["group_id"], record["prompt_idx"]) == (
        "agent",
        "g0",
        7,
    )
    assert record["prompt"] == [[False, "1"]]
    rollouts = record["rollouts"]
    # Best and worst valid rollouts get decoded segments.
    assert [i for i, r in enumerate(rollouts) if "segments" in r] == [1, 2]
    assert rollouts[1]["segments"] == [[False, "2"], [True, "3 4"]]
    assert rollouts[1]["gen_nll"] == 2.0
    assert rollouts[3]["masked"] and rollouts[3]["gen_nll"] is None
