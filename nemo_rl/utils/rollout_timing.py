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

import json
import os
import time


def log_rollout_timing(rec: dict) -> None:
    """Append one JSON line to $NRL_ROLLOUT_TIMING_FILE (no-op when unset)."""
    path = os.environ.get("NRL_ROLLOUT_TIMING_FILE")
    if not path:
        return
    rec.setdefault("wall", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
