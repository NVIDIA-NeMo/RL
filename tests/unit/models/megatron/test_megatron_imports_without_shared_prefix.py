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
"""The default Megatron path must not need Megatron-LM's shared-prefix modules.

``megatron.rl.shared_prefix_*`` and ``megatron.rl.tree_layout`` exist only in
Megatron-LM builds that contain MLM #7914, and the pinned Megatron-Bridge
Megatron-LM does not package ``megatron.rl`` at all. Every Megatron run imports
the modules below, so they must import without those modules and run their
default (feature-off) path. Only shared-prefix entry points may need them, and
they must then fail with an ImportError that names the missing module.

Each scenario runs in a fresh interpreter whose import system refuses the
blocked modules, so modules already imported by this pytest process cannot
mask an eager import.

Cross-module assumptions: shared-prefix imports stay inside the shared-prefix
code paths of ``nemo_rl.models.megatron.{data,train}`` and the Megatron policy
worker; ``get_microbatch_iterator`` and ``plan_shared_prefix_execution_units``
keep their signatures.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

pytestmark = pytest.mark.mcore

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PROBE = r"""
import fnmatch
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
BLOCKED = tuple(sys.argv[2].split(","))


class _WithoutSharedPrefixMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fnmatch.fnmatchcase(fullname, pattern)
            or fullname.startswith(pattern + ".")
            for pattern in BLOCKED
        ):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, _WithoutSharedPrefixMegatron())

import torch

import nemo_rl.models.megatron.data as data
import nemo_rl.models.megatron.train as train  # noqa: F401
import nemo_rl.models.policy.workers.megatron_policy_worker  # noqa: F401
import nemo_rl.models.value.workers.megatron_value_worker  # noqa: F401
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

batch = BatchedDataDict(
    {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 0]]),
        "input_lengths": torch.tensor([4, 3]),
        "shared_prefix_prompt_lengths": torch.tensor([2, 2]),
        "shared_prefix_group_id": ["g", "g"],
        "_shared_prefix_execution_slot": [0, 0],
    }
)
cfg = {
    "make_sequence_length_divisible_by": 1,
    "sequence_packing": {"enabled": False},
    "dynamic_batching": {"enabled": False},
    "megatron_cfg": {
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    },
}

# Default path: the conventional iterator is built without shared-prefix code.
_, num_microbatches, *_ = data.get_microbatch_iterator(
    batch, cfg, mbs=2, straggler_timer=None
)
assert num_microbatches == 1, num_microbatches

# Shared-prefix entry points fail loudly, naming the missing Megatron module.
shared_cfg = {
    **cfg,
    "sequence_packing": {
        "enabled": True,
        "algorithm": "modified_first_fit_decreasing",
    },
    "shared_prefix_training": {"mode": "train"},
}
entry_points = {
    "plan_shared_prefix_execution_units": lambda: (
        data.plan_shared_prefix_execution_units(batch, cfg=shared_cfg, bin_capacity=64)
    ),
    "get_microbatch_iterator": lambda: data.get_microbatch_iterator(
        batch,
        shared_cfg,
        mbs=2,
        straggler_timer=None,
        shared_prefix_bin_capacity=64,
        shared_prefix_execution_units=(),
    ),
}
for name, entry_point in entry_points.items():
    try:
        entry_point()
    except ImportError as error:
        assert "megatron.rl" in str(error), (name, error)
    else:
        raise AssertionError(f"{name} ran shared-prefix mode without megatron.rl")
print("SHARED_PREFIX_IMPORT_PROBE_OK")
"""


@pytest.mark.parametrize(
    "blocked",
    [
        # Megatron-LM before MLM #7914: megatron.rl lacks the shared-prefix modules.
        "megatron.rl.shared_prefix_*,megatron.rl.tree_layout",
        # Installed megatron-core wheels: no megatron.rl package at all.
        "megatron.rl",
    ],
    ids=["without_shared_prefix_modules", "without_megatron_rl"],
)
def test_default_megatron_path_imports_without_shared_prefix_modules(blocked):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(_REPO_ROOT), blocked],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=900,
    )
    assert result.returncode == 0 and "SHARED_PREFIX_IMPORT_PROBE_OK" in (
        result.stdout
    ), f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-6000:]}"
