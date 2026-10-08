# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeMo compatibility exports delegate to the canonical Megatron implementation."""

import subprocess
import sys
from pathlib import Path

from megatron.rl.shared_prefix_packing import SharedPrefixRow
from megatron.rl.shared_prefix_tensors import materialize_shared_prefix_layout

from nemo_rl.data import packing
from nemo_rl.data.packing import shared_prefix, shared_prefix_tensors
from nemo_rl.data.packing.shared_prefix_metadata import make_repeated_group_ids


def test_layout_and_tensor_exports_are_canonical_objects():
    assert packing.SharedPrefixRow is shared_prefix.SharedPrefixRow is SharedPrefixRow
    assert (
        shared_prefix_tensors.materialize_shared_prefix_layout
        is materialize_shared_prefix_layout
    )
    assert packing.materialize_shared_prefix_layout is materialize_shared_prefix_layout


def test_single_completion_metadata_uses_canonical_ppo_compatible_planner():
    assert make_repeated_group_ids(num_rows=2, group_size=1, namespace="ppo") == [
        "ppo:0",
        "ppo:1",
    ]


def test_dense_neMo_imports_do_not_require_optional_megatron_backend():
    repo = Path(__file__).resolve().parents[4]
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise AssertionError("Dense-only import unexpectedly loaded Megatron")
sys.meta_path.insert(0, NoMegatron())
from nemo_rl.data.packing import get_packer
from nemo_rl.models.policy import get_shared_prefix_training_config
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.data.packing.shared_prefix_cost import with_prompt_length_tags
assert get_shared_prefix_training_config({}).mode == "disabled"
assert with_prompt_length_tags(None, prompt_lengths=[2], sequence_lengths=[4])
bins = get_packer("first_fit_decreasing", bin_capacity=8).pack([4, 4])
assert len(bins) == 1 and sorted(bins[0]) == [0, 1]
assert not any(k == "megatron" or k.startswith("megatron.") for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(repo)], check=True)
