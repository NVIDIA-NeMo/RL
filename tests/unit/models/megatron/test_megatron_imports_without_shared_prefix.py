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
"""The default dense path must not import Megatron-LM's shared-prefix package.

Only a run that sets ``policy.sequence_packing.share_prefixes`` needs
``megatron.core.datasets.shared_prefix``. The probe imports the modules that
every Megatron run imports and builds a dense microbatch iterator. During the
probe no NeMo-RL module may import that package, or
``nemo_rl.models.megatron.shared_prefix.setup``, which imports it at module top.

The check looks at the importing module, because Megatron-LM's own model code
may import its package. The probe runs in a fresh interpreter, so modules that
this pytest process already imported cannot hide an import.
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
import builtins
import sys

sys.path.insert(0, sys.argv[1])
SHARED_PREFIX_ONLY = (
    "megatron.core.datasets.shared_prefix",
    "nemo_rl.models.megatron.shared_prefix.setup",
)
import_module = builtins.__import__


def is_shared_prefix_only(name):
    return any(
        name == module or name.startswith(module + ".") for module in SHARED_PREFIX_ONLY
    )


def import_and_check(name, globals=None, locals=None, fromlist=(), level=0):
    importer = (globals or {}).get("__name__", "")
    imported = [name, *(f"{name}.{item}" for item in fromlist or ())]
    if importer.startswith("nemo_rl.") and any(map(is_shared_prefix_only, imported)):
        raise AssertionError(f"{importer} imports {name} on the dense path")
    return import_module(name, globals, locals, fromlist, level)


builtins.__import__ = import_and_check

import torch

import nemo_rl.models.megatron.data as data
import nemo_rl.models.megatron.train  # noqa: F401
import nemo_rl.models.policy.workers.megatron_policy_worker  # noqa: F401
import nemo_rl.models.value.workers.megatron_value_worker  # noqa: F401
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

batch = BatchedDataDict(
    {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 0]]),
        "input_lengths": torch.tensor([4, 3]),
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
_, num_microbatches, *_ = data.get_microbatch_iterator(
    batch, cfg, mbs=2, straggler_timer=None
)
assert num_microbatches == 1, num_microbatches
print("DENSE_PATH_IMPORT_PROBE_OK")
"""


def test_default_dense_path_does_not_import_megatron_shared_prefix() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=900,
    )
    assert result.returncode == 0 and "DENSE_PATH_IMPORT_PROBE_OK" in result.stdout, (
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-6000:]}"
    )
