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

"""Guard on the Alloy config `ray.sub` ships to Loki.

Nothing else parses `alloy/config.alloy`: a bad attribute name, a component type
that does not exist, or a `forward_to` pointing at a component that was renamed
all start Alloy, fail inside it, and surface only as an empty Loki on a real
multi-node job. `alloy validate` catches all three before merge.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
ALLOY_CONFIG = REPO_ROOT / "alloy" / "config.alloy"
# Where docker/Dockerfile puts it, which is also the path ray.sub runs.
ALLOY_BIN = "/usr/local/bin/alloy"


def test_alloy_config_validates():
    alloy = shutil.which(ALLOY_BIN) or shutil.which("alloy")
    if alloy is None:
        # The release image installs Alloy, so CI sets NEMO_RL_REQUIRE_ALLOY=1 to
        # turn this skip into a failure — otherwise dropping the Dockerfile COPY
        # would silently drop this coverage and still go green.
        if os.environ.get("NEMO_RL_REQUIRE_ALLOY") == "1":
            raise RuntimeError(f"{ALLOY_BIN} not found (docker/Dockerfile installs it)")
        pytest.skip(
            f"{ALLOY_BIN} not found — skipping Alloy config validation "
            "(set NEMO_RL_REQUIRE_ALLOY=1 to fail loud)"
        )

    # No environment needed: the config reads its endpoint and certificates via
    # sys.env, and validate accepts those unset.
    result = subprocess.run(
        [alloy, "validate", str(ALLOY_CONFIG)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"alloy validate {ALLOY_CONFIG} failed:\n{result.stdout}\n{result.stderr}"
    )
