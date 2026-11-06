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
"""Merged stop strings must have the same order in every process.

The merge used ``list(set(...))``, whose order follows the string hash seed, so two
processes (e.g. two engine ranks, or a run and its resume) passed differently ordered
stop lists to the engine. Each case runs the merge under several ``PYTHONHASHSEED``
values in fresh interpreters.
"""

import os
import subprocess
import sys

import pytest

CONFIG_STOP_STRINGS = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
BATCH_STOP_STRINGS = [["x", "yy"], None, ["zzz", "a"]]
EXPECTED = sorted(set(CONFIG_STOP_STRINGS) | {"x", "yy", "zzz"})

MEGATRON_SNIPPET = f"""
from types import SimpleNamespace
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin
worker = SimpleNamespace(cfg={{"generation": {{"stop_strings": {CONFIG_STOP_STRINGS!r}}}}})
print(MegatronGenerationMixin._merge_stop_strings(worker, {BATCH_STOP_STRINGS!r}))
"""

VLLM_SNIPPET = f"""
from types import SimpleNamespace
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
worker = SimpleNamespace(cfg={{"stop_strings": {CONFIG_STOP_STRINGS!r}}})
print(BaseVllmGenerationWorker._merge_stop_strings(worker, {BATCH_STOP_STRINGS!r}))
"""


def _run(snippet: str, seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "snippet", [MEGATRON_SNIPPET, VLLM_SNIPPET], ids=["megatron", "vllm"]
)
def test_merged_stop_strings_do_not_depend_on_the_hash_seed(snippet):
    results = {_run(snippet, seed) for seed in ("0", "1", "2")}
    assert results == {str(EXPECTED)}
