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

"""Wrapper around Megatron-Bridge's quantization/export.py for QARL checkpoints.

This keeps the NeMo RL example entry point next to the QARL recipes while
delegating to ``Megatron-Bridge/examples/quantization/export.py`` unchanged.
All CLI arguments pass through.
"""

import runpy
import sys
from pathlib import Path

UPSTREAM_EXPORT = (
    Path(__file__).resolve().parents[2]
    / "3rdparty"
    / "Megatron-Bridge-workspace"
    / "Megatron-Bridge"
    / "examples"
    / "quantization"
    / "export.py"
)


def main() -> None:
    if not UPSTREAM_EXPORT.is_file():
        raise FileNotFoundError(
            f"Megatron-Bridge export script not found at {UPSTREAM_EXPORT}. "
            "Ensure the Megatron-Bridge submodule is initialized."
        )
    sys.argv[0] = str(UPSTREAM_EXPORT)
    runpy.run_path(str(UPSTREAM_EXPORT), run_name="__main__")


if __name__ == "__main__":
    main()
