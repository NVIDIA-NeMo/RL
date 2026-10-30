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

"""Record runtime packages and source locations used by a benchmark."""

import argparse
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path

import logra
import nemo_automodel
import torch

import nemo_rl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for name in [
        "torch",
        "transformers",
        "vllm",
        "ray",
        "nemo-automodel",
        "numpy",
        "tensorboard",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = dict(
        python=sys.version,
        platform=platform.platform(),
        cuda=torch.version.cuda,
        packages=versions,
        source_paths={
            m.__name__: inspect.getfile(m) for m in [logra, nemo_rl, nemo_automodel]
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
