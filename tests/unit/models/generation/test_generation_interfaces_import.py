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

import os
import subprocess
import sys
from pathlib import Path


def test_generation_interfaces_import_in_fresh_process_without_weight_sync_cycle() -> (
    None
):
    repository = Path(__file__).parents[4]
    environment = dict(os.environ, PYTHONPATH=str(repository))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nemo_rl.models.generation.interfaces import GenerationInterface",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
