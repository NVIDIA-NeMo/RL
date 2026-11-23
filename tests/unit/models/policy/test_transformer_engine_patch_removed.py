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

"""Regression tests for NVIDIA-NeMo/RL#3307.

The local source patch for Transformer Engine's Triton 3.5 permute-kernel
crash (NVIDIA/TransformerEngine#2286) was removed: the fix shipped upstream
in Transformer Engine 2.10, and every Transformer Engine version NeMo-RL
pins is >= 2.14.1, so the patch was a no-op that only rewrote an installed
file at worker startup. These tests guard the removal so the patch cannot
silently come back.
"""

import ast
from pathlib import Path

# Worker modules that used to call apply_transformer_engine_patch().
WORKER_MODULES = (
    "nemo_rl/models/policy/workers/automodel_policy_worker.py",
    "nemo_rl/models/policy/workers/megatron_policy_worker.py",
    "nemo_rl/models/value/workers/automodel_value_worker.py",
    "nemo_rl/models/value/workers/megatron_value_worker.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_transformer_engine_patch_module_removed():
    """The module that rewrote the installed TE source must not exist.

    Checked on disk rather than by importing it: importing the parent
    package pulls in torch/ray, which the CPU unit-test lane does not need
    for a removal guard.
    """
    assert not (_repo_root() / "nemo_rl/models/policy/workers/patches.py").exists()


def test_worker_modules_do_not_reference_patch():
    """No worker module may import or call the removed patch.

    The worker modules are heavy to import (torch/ray/megatron), so this
    checks their source with ast, following the precedent in
    test_megatron_worker.py.
    """
    repo_root = _repo_root()

    for relative_path in WORKER_MODULES:
        tree = ast.parse((repo_root / relative_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "nemo_rl.models.policy.workers.patches", (
                    relative_path
                )
            if isinstance(node, ast.Name):
                assert node.id != "apply_transformer_engine_patch", relative_path
