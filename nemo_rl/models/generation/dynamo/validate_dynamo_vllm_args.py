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

"""Validate a resolved ``dynamo.vllm`` argv without starting an engine."""

import json
import os
import sys


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_dynamo_vllm_args.py '<json argv>'")

    # Parser construction evaluates vLLM device defaults. Image builds and
    # unit preflights may run on CPU-only Slurm nodes, where a CUDA wheel leaves
    # current_platform unspecified even though no engine will be constructed.
    # Match Dynamo frontend's parser-only fallback without changing GPU runs.
    import vllm.platforms

    if vllm.platforms.current_platform.device_type == "":
        from vllm.platforms.cpu import CpuPlatform

        vllm.platforms.current_platform = CpuPlatform()

    from dynamo.vllm.args import parse_args

    argv = json.loads(sys.argv[1])
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise TypeError("resolved Dynamo argv must be a JSON list of strings")
    parse_args(argv)


if __name__ == "__main__":
    main()
