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
"""Driver-side packing must not require the optional Megatron backend.

The driver packs rows, and plans shared-prefix forwards, for every backend.
This module deliberately has no Megatron import and no ``mcore`` marker, so it
runs in the default lane where the guarantee matters.
"""

import subprocess
import sys
from pathlib import Path


def test_driver_packing_does_not_import_megatron() -> None:
    repo = Path(__file__).resolve().parents[4]
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise AssertionError("Driver-side packing unexpectedly loaded Megatron")
sys.meta_path.insert(0, NoMegatron())
from nemo_rl.data.packing import get_packer
from nemo_rl.data.packing.shared_prefix.row_metadata import (
    PROMPT_HASH_TAG,
    PROMPT_LENGTH_TAG,
    SHARED_PREFIX_TREE_IDS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GROUP_ID_TAG
from nemo_rl.models.policy.tq_policy import TQPolicy
bins = get_packer("first_fit_decreasing", bin_capacity=8).pack([4, 4])
assert len(bins) == 1 and sorted(bins[0]) == [0, 1]
meta = KVBatchMeta(
    partition_id="train",
    task_name="train",
    sample_ids=["a", "b"],
    sequence_lengths=[6, 7],
    tags=[{GROUP_ID_TAG: "group", PROMPT_LENGTH_TAG: 4, PROMPT_HASH_TAG: 1}] * 2,
)
(shard,) = shard_meta_for_dp(
    meta,
    dp_world=1,
    sequence_packing_args={
        "max_tokens_per_microbatch": 16,
        "sequence_length_pad_multiple": 1,
        "algorithm": "first_fit_decreasing",
        "share_prefixes": True,
        "bound_by_full_length": False,
    },
)
assert shard.extra_info[SHARED_PREFIX_TREE_IDS] == [0, 0]
assert not any(k == "megatron" or k.startswith("megatron.") for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(repo)], check=True)
