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

"""Make a checkpoint view: CONFIG becomes the view's config.json, everything else in SRC is symlinked.

Rebuilds a checkpoint view (e.g. an Automodel text-only view whose config.json carries
quantization_config) against a different copy of the same checkpoint.

    python make_config_view.py SRC DST CONFIG
"""

import os
import shutil
import sys


def main(src: str, dst: str, config: str) -> None:
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    os.makedirs(dst, exist_ok=False)
    for name in sorted(os.listdir(src)):
        if name != "config.json":
            os.symlink(os.path.join(src, name), os.path.join(dst, name))
    shutil.copyfile(config, os.path.join(dst, "config.json"))
    print(f"{dst}: config from {config}, {len(os.listdir(dst)) - 1} links")


if __name__ == "__main__":
    main(*sys.argv[1:])
